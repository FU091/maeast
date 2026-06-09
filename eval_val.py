"""
eval_val.py
===========
使用訓練完的 checkpoint 跑 Val Set 評估。

輸出：
  - 每類：AP / F1 / Precision / Recall
  - 總體：mAP / Macro-F1 / Micro-F1 / Macro-Precision / Macro-Recall / Micro-Precision / Micro-Recall

使用方式（在 mae_finetune_v2/ 目錄下執行）：
    python eval_val.py
    python eval_val.py --ckpt E:\\MAE_AST\\MAE_output_v2\\checkpoints\\best.pt
    python eval_val.py --ckpt path/to/best.pt --threshold 0.4
    python eval_val.py --split test   # 改用 test set
"""

import os
import sys
import argparse
import logging

import numpy as np
import torch
from torch.amp import autocast
from tqdm import tqdm

# ── 把 mae_finetune_v2/ 加入 path ─────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as CFG
from dataset import build_dataloader, build_label_map
from model import MAEASTFineTune
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_val")


# ============================================================
#  找 checkpoint（若未指定則自動搜尋）
# ============================================================

def find_checkpoint(ckpt_arg: str) -> str:
    """若使用者未指定 --ckpt，自動在 CHECKPOINT_DIR 尋找 best.pt / last.pt"""
    if ckpt_arg and os.path.exists(ckpt_arg):
        return ckpt_arg

    search_dirs = [CFG.CHECKPOINT_DIR]

    candidates = []
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for name in ("best.pt", "last.pt"):
            p = os.path.join(d, name)
            if os.path.exists(p):
                candidates.append(p)

    if not candidates:
        raise FileNotFoundError(
            f"找不到 checkpoint！\n"
            f"請用 --ckpt 指定路徑，或確認 {CFG.CHECKPOINT_DIR} 內有 best.pt / last.pt"
        )

    if len(candidates) == 1:
        logger.info(f"自動選用 checkpoint: {candidates[0]}")
        return candidates[0]

    # 有多個 → 優先選 best.pt
    best_candidates = [c for c in candidates if "best" in os.path.basename(c)]
    chosen = best_candidates[0] if best_candidates else candidates[0]
    logger.info(f"自動選用 checkpoint (best > last): {chosen}")
    return chosen


# ============================================================
#  評估核心
# ============================================================

@torch.no_grad()
def run_eval(model, loader, device, use_amp: bool):
    """跑 inference，回傳 (targets_np [N,C], scores_np [N,C])"""
    model.eval()
    all_targets = []
    all_scores  = []

    pbar = tqdm(loader, desc="Evaluating", dynamic_ncols=True)
    for fbank, labels in pbar:
        fbank  = fbank.to(device,  non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast("cuda", enabled=use_amp):
            logits = model(fbank)

        scores = torch.sigmoid(logits).cpu().numpy()
        # val/test set labels 在 dataset.__getitem__ 已經二值化（>0 → 1）
        targets = (labels > 0).float().cpu().numpy()

        all_scores.append(scores)
        all_targets.append(targets)

    targets_np = np.concatenate(all_targets, axis=0)  # [N, C]
    scores_np  = np.concatenate(all_scores,  axis=0)  # [N, C]
    return targets_np, scores_np


def compute_full_metrics(targets: np.ndarray, scores: np.ndarray,
                         label_names: list, threshold: float = 0.5):
    """
    計算並印出完整評估指標。

    Returns: dict with all metrics
    """
    C = targets.shape[1]
    preds = (scores >= threshold).astype(np.float32)

    # ── Per-class metrics ──────────────────────────────────────
    per_class = {}
    valid_aps = []

    for c in range(C):
        name = label_names[c] if c < len(label_names) else f"class_{c}"
        gt   = targets[:, c]
        pred = preds[:, c]
        sc   = scores[:, c]

        n_pos = int(gt.sum())
        if n_pos == 0:
            per_class[name] = dict(ap=float("nan"), f1=float("nan"),
                                   precision=float("nan"), recall=float("nan"),
                                   support=0)
            continue

        ap = average_precision_score(gt, sc)
        p  = precision_score(gt, pred, zero_division=0)
        r  = recall_score(gt, pred, zero_division=0)
        f1 = f1_score(gt, pred, zero_division=0)

        per_class[name] = dict(ap=ap, f1=f1, precision=p, recall=r, support=n_pos)
        valid_aps.append(ap)

    mAP = float(np.mean(valid_aps)) if valid_aps else 0.0

    # ── Overall metrics ─────────────────────────────────────────
    overall = {
        "mAP":              mAP,
        "macro_f1":         f1_score(targets,       preds, average="macro",  zero_division=0),
        "micro_f1":         f1_score(targets,       preds, average="micro",  zero_division=0),
        "macro_precision":  precision_score(targets, preds, average="macro",  zero_division=0),
        "micro_precision":  precision_score(targets, preds, average="micro",  zero_division=0),
        "macro_recall":     recall_score(targets,   preds, average="macro",  zero_division=0),
        "micro_recall":     recall_score(targets,   preds, average="micro",  zero_division=0),
    }

    return per_class, overall


def print_results(per_class: dict, overall: dict, threshold: float):
    SEP = "=" * 74
    sep = "-" * 74

    print(f"\n{SEP}")
    print(f"  VAL SET EVALUATION  (threshold={threshold:.2f})")
    print(SEP)

    # ── Per-class table ─────────────────────────────────────────
    header = f"  {'Class':<20} {'AP':>8} {'F1':>8} {'Precision':>10} {'Recall':>8} {'#Pos':>6}"
    print(header)
    print(sep)

    for name, m in per_class.items():
        if np.isnan(m["ap"]):
            print(f"  {name:<20} {'N/A':>8} {'N/A':>8} {'N/A':>10} {'N/A':>8} {m['support']:>6}  (no positive samples)")
        else:
            print(
                f"  {name:<20} {m['ap']:8.4f} {m['f1']:8.4f} "
                f"{m['precision']:10.4f} {m['recall']:8.4f} {m['support']:>6}"
            )

    print(sep)

    # ── Overall ─────────────────────────────────────────────────
    print(f"\n{'  OVERALL METRICS':}")
    print(sep)
    print(f"  mAP            = {overall['mAP']:.4f}")
    print(f"  Macro-F1       = {overall['macro_f1']:.4f}")
    print(f"  Micro-F1       = {overall['micro_f1']:.4f}")
    print(f"  Macro-Precision= {overall['macro_precision']:.4f}")
    print(f"  Micro-Precision= {overall['micro_precision']:.4f}")
    print(f"  Macro-Recall   = {overall['macro_recall']:.4f}")
    print(f"  Micro-Recall   = {overall['micro_recall']:.4f}")
    print(SEP)


def save_csv(per_class: dict, overall: dict, out_path: str):
    import csv
    rows = []
    for name, m in per_class.items():
        rows.append({
            "class":     name,
            "AP":        round(m["ap"],        4) if not np.isnan(m["ap"]) else "N/A",
            "F1":        round(m["f1"],        4) if not np.isnan(m["f1"]) else "N/A",
            "Precision": round(m["precision"], 4) if not np.isnan(m["precision"]) else "N/A",
            "Recall":    round(m["recall"],    4) if not np.isnan(m["recall"]) else "N/A",
            "support":   m["support"],
        })
    # append overall row
    rows.append({
        "class":     "OVERALL",
        "AP":        round(overall["mAP"],              4),
        "F1":        f"Macro={overall['macro_f1']:.4f} / Micro={overall['micro_f1']:.4f}",
        "Precision": f"Macro={overall['macro_precision']:.4f} / Micro={overall['micro_precision']:.4f}",
        "Recall":    f"Macro={overall['macro_recall']:.4f} / Micro={overall['micro_recall']:.4f}",
        "support":   "",
    })

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "AP", "F1", "Precision", "Recall", "support"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Results saved → {out_path}")


# ============================================================
#  Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="MAE-AST Val Set Evaluator")
    parser.add_argument("--ckpt",       default=None,
                        help="Fine-tune checkpoint 路徑（預設自動找 best.pt）")
    parser.add_argument("--split",      default="val", choices=["val", "test"],
                        help="評估 val 或 test set（預設 val）")
    parser.add_argument("--threshold",  type=float, default=0.5,
                        help="Sigmoid 二值化閾值（預設 0.5）")
    parser.add_argument("--batch_size", type=int,   default=CFG.BATCH_SIZE,
                        help="Batch size（預設使用 config.py 的值）")
    parser.add_argument("--num_workers",type=int,   default=CFG.NUM_WORKERS)
    parser.add_argument("--no_amp",     action="store_true",
                        help="停用 AMP（若 GPU 不支援 float16）")
    parser.add_argument("--save_csv",   default=None,
                        help="若指定路徑，將結果存為 CSV（例如 results/val_metrics.csv）")
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = CFG.USE_AMP and not args.no_amp and device.type == "cuda"

    logger.info(f"Device  : {device}")
    logger.info(f"Use AMP : {use_amp}")
    logger.info(f"Split   : {args.split}")
    logger.info(f"Threshold: {args.threshold}")

    # ── 找 checkpoint ──────────────────────────────────────────
    ckpt_path = find_checkpoint(args.ckpt)
    logger.info(f"Checkpoint: {ckpt_path}")

    # ── Label map ──────────────────────────────────────────────
    label_map, num_classes = build_label_map(CFG.LABEL_CSV)
    label_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
    logger.info(f"num_classes = {num_classes}, labels: {label_names}")

    # ── DataLoader ─────────────────────────────────────────────
    json_path = CFG.VAL_JSON if args.split == "val" else CFG.TEST_JSON
    logger.info(f"JSON: {json_path}")

    loader = build_dataloader(
        json_path       = json_path,
        label_map       = label_map,
        num_classes     = num_classes,
        spectrogram_dir = CFG.SPECTROGRAM_DIR,
        batch_size      = args.batch_size,
        num_workers     = args.num_workers,
        pin_memory      = CFG.PIN_MEMORY and device.type == "cuda",
        is_train        = False,
        cache_to_ram    = False,   # eval 不需要 cache
    )

    # ── 載入模型 ───────────────────────────────────────────────
    logger.info("Loading model …")
    model = MAEASTFineTune(
        pretrained_ckpt = CFG.PRETRAINED_CKPT,
        mae_ast_root    = CFG.MAE_AST_PROJECT_ROOT,
        num_classes     = num_classes,
        freeze_encoder  = False,
    ).to(device)

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    logger.info(f"[checkpoint] epoch={state.get('epoch', '?')}  best_mAP={state.get('best_map', 0):.4f}")

    # ── Inference ──────────────────────────────────────────────
    targets_np, scores_np = run_eval(model, loader, device, use_amp)
    logger.info(f"Inference done. targets={targets_np.shape}, scores={scores_np.shape}")

    # ── 計算指標 ───────────────────────────────────────────────
    per_class, overall = compute_full_metrics(
        targets_np, scores_np, label_names, threshold=args.threshold
    )

    # ── 印出結果 ───────────────────────────────────────────────
    print_results(per_class, overall, threshold=args.threshold)

    # ── 儲存 CSV（可選）──────────────────────────────────────────
    if args.save_csv:
        save_csv(per_class, overall, args.save_csv)


if __name__ == "__main__":
    main()
