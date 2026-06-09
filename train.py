"""
train.py
========
MAE-AST Fine-tune 訓練主程式

功能：
- argparse 參數控制
- AMP mixed precision (torch.cuda.amp)
- AdamW optimizer
- Cosine LR scheduler（含 warmup）
- Validation loop（每 epoch）
- Save best checkpoint（依 val mAP）
- Resume checkpoint
- TensorBoard logging
- tqdm progress bar
- BCEWithLogitsLoss（multi-label）
- 每 epoch 輸出：train loss / val loss / per-class AP / overall mAP

使用方式：
    python train.py
    python train.py --freeze_encoder
    python train.py --resume checkpoints/best.pt
    python train.py --batch_size 16 --epochs 50 --lr 5e-5
"""

import os
import sys
import math
import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp   import GradScaler, autocast   # 新版 API，避免 FutureWarning
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ── 把 mae_finetune/ 加入 path（讓其他模組可以 import）──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as CFG
from dataset import build_dataloader, build_label_map
from model   import MAEASTFineTune
from metrics import compute_metrics_from_tensors

logging.basicConfig(
    level   = logging.INFO,
    format  = "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("train")


# ============================================================
#  Argument Parser
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="MAE-AST Fine-tune Trainer")

    # 路徑
    parser.add_argument("--pretrained_ckpt",  default=CFG.PRETRAINED_CKPT)
    parser.add_argument("--mae_ast_root",     default=CFG.MAE_AST_PROJECT_ROOT)
    parser.add_argument("--spectrogram_dir",  default=CFG.SPECTROGRAM_DIR)
    parser.add_argument("--train_json",       default=CFG.TRAIN_JSON)
    parser.add_argument("--val_json",         default=CFG.VAL_JSON)
    parser.add_argument("--label_csv",        default=CFG.LABEL_CSV)
    parser.add_argument("--checkpoint_dir",   default=CFG.CHECKPOINT_DIR)
    parser.add_argument("--tensorboard_dir",  default=CFG.TENSORBOARD_DIR)

    # 訓練超參
    parser.add_argument("--epochs",       type=int,   default=CFG.EPOCHS)
    parser.add_argument("--batch_size",   type=int,   default=CFG.BATCH_SIZE)
    parser.add_argument("--lr",           type=float, default=CFG.LR)
    parser.add_argument("--weight_decay", type=float, default=CFG.WEIGHT_DECAY)
    parser.add_argument("--warmup_epochs",type=int,   default=CFG.WARMUP_EPOCHS)
    parser.add_argument("--num_workers",  type=int,   default=CFG.NUM_WORKERS)
    parser.add_argument("--num_classes",  type=int,   default=None,
                        help="若指定則覆蓋 label_csv 計算的類別數")

    # 模式
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="凍結 encoder，只訓練分類頭")
    parser.add_argument("--no_amp",         action="store_true",
                        help="停用 AMP mixed precision")
    parser.add_argument("--resume",         default=CFG.RESUME_CKPT,
                        help="Resume checkpoint 路徑")

    # 穩定訓練用超參
    parser.add_argument("--llrd_decay",   type=float, default=getattr(CFG, 'LLRD_DECAY', 0.85),
                        help="Layer-wise LR Decay 倍率（預設 0.85）")
    parser.add_argument("--label_smooth", type=float, default=getattr(CFG, 'LABEL_SMOOTH', 0.1),
                        help="Label smoothing epsilon（0=無，0.1=標準）")

    return parser.parse_args()


# ============================================================
#  Cosine LR Scheduler with Warmup
# ============================================================

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing with linear warmup."""

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        epoch = self.last_epoch
        if epoch < self.warmup_epochs:
            # Linear warmup
            factor = (epoch + 1) / max(1, self.warmup_epochs)
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            factor   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [base_lr * factor for base_lr in self.base_lrs]


# ============================================================
#  Label Smoothing BCE Loss
# ============================================================

class SmoothBCEWithLogitsLoss(nn.Module):
    """
    BCEWithLogitsLoss + label smoothing。
    把 hard label（0/1）軟化為：
        正樣本 1 → 1 - eps/2
        負樣本 0 → eps/2
    防止模型把 logit 推到極端大小，降低對閾値 0.5 的敏感性。
    """
    def __init__(self, eps: float = 0.1):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_s = targets * (1.0 - self.eps) + self.eps * 0.5
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, targets_s)


# ============================================================
#  Layer-wise LR Decay Optimizer
# ============================================================

def build_llrd_optimizer(
    model,
    base_lr: float,
    weight_decay: float,
    decay_rate: float = 0.85,
    betas: tuple = (0.9, 0.999),
):
    """
    Layer-wise Learning Rate Decay for MAE-AST fine-tune.

    LR 分配規則：
        classifier           : base_lr × 1.00
        encoder layer[N-1]   : base_lr × decay^1
        encoder layer[N-2]   : base_lr × decay^2
        ...
        encoder layer[0]     : base_lr × decay^N
        encoder stem (others): base_lr × decay^(N+1)

    防止底層預訓練特徵被大幅更新而震盪。
    """
    import re

    # ― 分類器頭
    classifier_params = [p for p in model.classifier.parameters() if p.requires_grad]

    # ― Encoder 參數依層索分組
    layer_params: dict = {}
    stem_params   = []

    for name, param in model.encoder.named_parameters():
        if not param.requires_grad:
            continue
        m = re.search(r'\.(?:layers|blocks)\.(\d+)\.', name)
        if m:
            idx = int(m.group(1))
            layer_params.setdefault(idx, []).append(param)
        else:
            stem_params.append(param)

    n_layers = max(layer_params.keys()) + 1 if layer_params else 0
    logger.info(
        f"[LLRD] {n_layers} encoder layers | "
        f"{len(stem_params)} stem params | "
        f"base_lr={base_lr:.1e} | decay={decay_rate}"
    )

    param_groups = [
        {"params": classifier_params, "lr": base_lr,
         "weight_decay": weight_decay, "name": "classifier"},
    ]

    for idx in sorted(layer_params.keys(), reverse=True):
        depth  = n_layers - idx
        lr_i   = base_lr * (decay_rate ** depth)
        param_groups.append({
            "params": layer_params[idx], "lr": lr_i,
            "weight_decay": weight_decay, "name": f"enc_layer_{idx:02d}",
        })
        logger.info(f"  enc_layer_{idx:02d}: lr={lr_i:.2e}")

    if stem_params:
        stem_lr = base_lr * (decay_rate ** (n_layers + 1))
        param_groups.append({
            "params": stem_params, "lr": stem_lr,
            "weight_decay": weight_decay, "name": "enc_stem",
        })
        logger.info(f"  enc_stem      : lr={stem_lr:.2e}")

    return AdamW(param_groups, betas=betas)


# ============================================================
#  Train One Epoch
# ============================================================

def train_one_epoch(
    model, loader, optimizer, criterion, scaler, device, epoch, use_amp, writer
):
    model.train()
    total_loss  = 0.0
    num_batches = 0
    start_time  = time.time()

    pbar = tqdm(loader, desc=f"[Train] Epoch {epoch}", leave=False, dynamic_ncols=True)

    for fbank, labels in pbar:
        fbank  = fbank.to(device,  non_blocking=True)   # [B, 1024, 128]
        labels = labels.to(device, non_blocking=True)   # [B, num_classes]

        optimizer.zero_grad()

        with autocast('cuda', enabled=use_amp):
            logits = model(fbank)                        # [B, num_classes]
            loss   = criterion(logits, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        batch_loss = loss.item()
        total_loss += batch_loss
        num_batches += 1

        pbar.set_postfix(loss=f"{batch_loss:.4f}")

    avg_loss   = total_loss / max(1, num_batches)
    elapsed    = time.time() - start_time
    step       = epoch  # 以 epoch 為 TensorBoard step

    logger.info(f"[Train] Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")
    writer.add_scalar("Loss/train", avg_loss, step)
    writer.add_scalar("LR", optimizer.param_groups[0]["lr"], step)

    return avg_loss


# ============================================================
#  Validation
# ============================================================

@torch.no_grad()
def validate(model, loader, criterion, device, epoch, label_names, use_amp, writer):
    model.eval()
    total_loss   = 0.0
    num_batches  = 0
    all_targets  = []
    all_scores   = []

    pbar = tqdm(loader, desc=f"[Val]   Epoch {epoch}", leave=False, dynamic_ncols=True)

    for fbank, labels in pbar:
        fbank  = fbank.to(device,  non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast('cuda', enabled=use_amp):
            logits = model(fbank)
            loss   = criterion(logits, labels)

        total_loss  += loss.item()
        num_batches += 1

        # 收集預測與 ground truth（移回 CPU）
        scores = torch.sigmoid(logits).cpu()
        # 二值化 GT（驗證時 soft label → 0/1）
        binary_labels = (labels > 0).float().cpu()

        all_scores.append(scores.numpy())
        all_targets.append(binary_labels.numpy())

    avg_loss = total_loss / max(1, num_batches)

    # 合併所有 batch
    targets_np = np.concatenate(all_targets, axis=0)  # [N, C]
    scores_np  = np.concatenate(all_scores,  axis=0)  # [N, C]

    metrics_dict, per_class_ap = compute_metrics_from_tensors(
        targets_np, scores_np,
        label_names = label_names,
        verbose     = True,
    )

    mAP = metrics_dict["mAP"]

    logger.info(f"[Val]   Epoch {epoch:3d} | Loss: {avg_loss:.4f} | mAP: {mAP:.4f}")
    writer.add_scalar("Loss/val", avg_loss, epoch)
    writer.add_scalar("mAP/val",  mAP,     epoch)
    
    # 記錄新增的指標
    writer.add_scalar("Metrics/Micro_Precision", metrics_dict["micro_precision"], epoch)
    writer.add_scalar("Metrics/Macro_Precision", metrics_dict["macro_precision"], epoch)
    writer.add_scalar("Metrics/Micro_Recall",    metrics_dict["micro_recall"],    epoch)
    writer.add_scalar("Metrics/Macro_Recall",    metrics_dict["macro_recall"],    epoch)
    writer.add_scalar("Metrics/Micro_F1",        metrics_dict["micro_f1"],        epoch)
    writer.add_scalar("Metrics/Macro_F1",        metrics_dict["macro_f1"],        epoch)

    # 記錄 per-class AP（過濾 nan）
    for name, ap in per_class_ap.items():
        if not np.isnan(ap):
            writer.add_scalar(f"AP/{name}", ap, epoch)

    # 產生混淆矩陣與詳細指標 CSV（每 5 epoch 才寫一次，減少 NTFS 磁碟 I/O 阻塞）
    if epoch % 5 == 0:
        from metrics import compute_confusion_stats
        import config as CFG
        compute_confusion_stats(
            targets_np, scores_np, label_names,
            save_dir=os.path.join(CFG.TENSORBOARD_DIR, "confusion_stats"),
            epoch=epoch
        )

    return avg_loss, mAP, per_class_ap


# ============================================================
#  Main
# ============================================================

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = CFG.USE_AMP and not args.no_amp and device.type == "cuda"

    logger.info(f"Device   : {device}")
    logger.info(f"Use AMP  : {use_amp}")
    logger.info(f"Epochs   : {args.epochs}")
    logger.info(f"Batch    : {args.batch_size}")
    logger.info(f"LR       : {args.lr}")
    logger.info(f"Freeze   : {args.freeze_encoder}")

    # ── Label map ─────────────────────────────────────────
    from dataset import build_label_map
    label_map, num_classes = build_label_map(args.label_csv)
    if args.num_classes is not None:
        num_classes = args.num_classes
        logger.info(f"Overriding num_classes → {num_classes}")
    label_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
    logger.info(f"num_classes = {num_classes}")

    # ── DataLoaders ───────────────────────────────────────
    train_loader = build_dataloader(
        json_path       = args.train_json,
        label_map       = label_map,
        num_classes     = num_classes,
        spectrogram_dir = args.spectrogram_dir,
        batch_size      = args.batch_size,
        num_workers     = args.num_workers,
        pin_memory      = CFG.PIN_MEMORY and device.type == "cuda",
        is_train        = True,
        cache_to_ram    = CFG.CACHE_TO_RAM,
    )
    val_loader = build_dataloader(
        json_path       = args.val_json,
        label_map       = label_map,
        num_classes     = num_classes,
        spectrogram_dir = args.spectrogram_dir,
        batch_size      = args.batch_size,
        num_workers     = args.num_workers,
        pin_memory      = CFG.PIN_MEMORY and device.type == "cuda",
        is_train        = False,
        cache_to_ram    = CFG.CACHE_TO_RAM,
    )

    # ── Model ─────────────────────────────────────────────
    model = MAEASTFineTune(
        pretrained_ckpt = args.pretrained_ckpt,
        mae_ast_root    = args.mae_ast_root,
        num_classes     = num_classes,
        freeze_encoder  = args.freeze_encoder,
    ).to(device)

    # ── Optimizer & Scheduler ─────────────────────────────
    if args.freeze_encoder:
        # Freeze 模式：分類頭參數用單一 LR
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr           = args.lr,
            weight_decay = args.weight_decay,
            betas        = (0.9, 0.999),
        )
        logger.info(f"[Optimizer] Single LR={args.lr:.1e} (freeze mode)")
    else:
        # Full fine-tune：使用 Layer-wise LR Decay
        optimizer = build_llrd_optimizer(
            model,
            base_lr      = args.lr,
            weight_decay = args.weight_decay,
            decay_rate   = args.llrd_decay,
        )
        logger.info(f"[Optimizer] LLRD base_lr={args.lr:.1e} decay={args.llrd_decay}")

    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs = args.warmup_epochs,
        total_epochs  = args.epochs,
    )
    scaler    = GradScaler('cuda', enabled=use_amp)  # 新版 API
    criterion = SmoothBCEWithLogitsLoss(eps=args.label_smooth)
    logger.info(f"[Loss] SmoothBCE eps={args.label_smooth}")

    # ── Resume ────────────────────────────────────────────
    start_epoch = 0
    best_map    = 0.0

    if args.resume and os.path.exists(args.resume):
        logger.info(f"Resuming from: {args.resume}")
        state = torch.load(args.resume, map_location=device, weights_only=False)

        # ― Model weights（必定載入）
        model.load_state_dict(state["model_state"])
        logger.info("[Resume] Model weights loaded.")

        # ― Optimizer（若不符則跳過）
        # last.pt 可能是舊版 optimizer（無 LLRD，1 param group）存的
        # 新版 LLRD optimizer 有 14 param groups，load_state_dict 會 RuntimeError
        optimizer_ok = False
        try:
            optimizer.load_state_dict(state["optimizer"])
            optimizer_ok = True
            logger.info("[Resume] Optimizer state loaded.")
        except (ValueError, RuntimeError, KeyError) as e:
            logger.warning(
                f"[Resume] ⚠️  Optimizer state 不符（{e}）。"
                f"使用新版 LLRD optimizer，LR 從正確 epoch 重新 warmup。"
            )

        # ― Scheduler
        if optimizer_ok and state.get("scheduler"):
            try:
                scheduler.load_state_dict(state["scheduler"])
                logger.info("[Resume] Scheduler state loaded.")
            except Exception:
                logger.warning("[Resume] Scheduler state 不符，將快轉到正確 epoch。")
                optimizer_ok = False

        # ― 若 optimizer/scheduler 都是新的，快轉 scheduler 到正確 epoch
        start_epoch = state.get("epoch", 0) + 1
        if not optimizer_ok:
            for _ in range(start_epoch):
                scheduler.step()
            logger.info(f"[Resume] Scheduler fast-forwarded to epoch {start_epoch}.")

        best_map = state.get("best_map", 0.0)
        logger.info(f"[Resume] 從 epoch {start_epoch} 繼續，best_mAP={best_map:.4f}")

    # ── TensorBoard ───────────────────────────────────────
    os.makedirs(args.tensorboard_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir,  exist_ok=True)
    writer = SummaryWriter(log_dir=args.tensorboard_dir)
    logger.info(f"TensorBoard log → {args.tensorboard_dir}")

    # ── Training Loop ─────────────────────────────────────
    best_ckpt = os.path.join(args.checkpoint_dir, "best.pt")
    last_ckpt = os.path.join(args.checkpoint_dir, "last.pt")

    logger.info("=" * 60)
    logger.info("Start Training")
    logger.info("=" * 60)

    for epoch in range(start_epoch, args.epochs):

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion,
            scaler, device, epoch, use_amp, writer
        )

        # Validate
        val_loss, val_map, per_class_ap = validate(
            model, val_loader, criterion,
            device, epoch, label_names, use_amp, writer
        )

        # LR scheduler step（after validation）
        scheduler.step()

        # Summary line
        logger.info(
            f"Epoch {epoch:3d}/{args.epochs-1} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val mAP: {val_map:.4f}"
        )

        # Save best
        if val_map > best_map:
            best_map = val_map
            model.save_checkpoint(best_ckpt, epoch, optimizer, scheduler, best_map)
            logger.info(f"★ New best mAP: {best_map:.4f} → saved to {best_ckpt}")

        # Save last
        model.save_checkpoint(last_ckpt, epoch, optimizer, scheduler, best_map)

    writer.close()
    logger.info(f"Training complete. Best val mAP = {best_map:.4f}")
    logger.info(f"Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
