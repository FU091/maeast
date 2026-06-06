"""
model.py
========
MAE-AST Fine-tune Model
- 從 chunk_patch_75_12LayerEncoder.pt 載入預訓練 encoder
- 不依賴 fairseq training pipeline，純 PyTorch
- 支援 freeze encoder / full fine-tune
- Pooling: mean pooling over patch tokens
- Classifier: Linear(768, num_classes)
- 輸入: [B, 1024, 128] float32（直接使用你的 .pt spectrogram）
- 模型內部自動完成：
    BatchNorm2D → Unfold（patchify 16×16）→ Linear(256→768) → Transformer × 12
"""

import os
import sys
import logging
from types import SimpleNamespace

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ============================================================
#  確保可以 import MAE_AST（加入官方專案路徑）
# ============================================================

def _add_mae_ast_to_path(mae_ast_root: str):
    """將 MAE-AST 官方專案根目錄加入 sys.path"""
    if mae_ast_root not in sys.path:
        sys.path.insert(0, mae_ast_root)
        logger.info(f"[model] Added to sys.path: {mae_ast_root}")


# ============================================================
#  Checkpoint Loader
# ============================================================

def load_pretrained_mae_ast(ckpt_path: str, mae_ast_root: str):
    """
    載入官方 pretrained checkpoint。

    Checkpoint 結構（由 s3prl/mae_ast/expert.py 確認）：
        checkpoint["cfg"]["model"]  → MAE_AST_Config 的 dict
        checkpoint["cfg"]["task"]   → MAE_AST_Pretraining_Config 的 dict
        checkpoint["model"]         → state_dict

    Returns
    -------
    model  : MAE_AST instance（未加分類頭）
    model_cfg : SimpleNamespace of model config
    task_cfg  : SimpleNamespace of task config
    """
    _add_mae_ast_to_path(mae_ast_root)

    from mae_ast.models.mae_ast import MAE_AST

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"[model] Checkpoint not found: {ckpt_path}")

    logger.info(f"[model] Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model_cfg = SimpleNamespace(**checkpoint["cfg"]["model"])
    task_cfg  = SimpleNamespace(**checkpoint["cfg"]["task"])

    model = MAE_AST(model_cfg, task_cfg)

    # strict=True：確保 encoder 權重完整載入
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)

    if missing:
        logger.warning(f"[model] Missing keys ({len(missing)}): {missing[:5]} ...")
    if unexpected:
        logger.warning(f"[model] Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")

    logger.info("[model] Pretrained weights loaded successfully.")
    return model, model_cfg, task_cfg


# ============================================================
#  MAE-AST Fine-tune Model
# ============================================================

class MAEASTFineTune(nn.Module):
    """
    MAE-AST encoder backbone + multi-label classification head。

    Forward 流程（模型內部）：
        [B, 1024, 128]
          ↓ unsqueeze(1) → [B, 1, 1024, 128]
          ↓ BatchNorm2D × 0.5
          ↓ Unfold(16,16) → [B, 512, 256]       (512 patches)
          ↓ Linear(256→768)
          ↓ Sinusoidal pos embed
          ↓ TransformerEncoder × 12
          ↓ → [B, 512, 768]
          ↓ mean pooling → [B, 768]
          ↓ Dropout
          ↓ Linear(768, num_classes) → [B, num_classes]

    不做 masking（mask=False），不用 decoder（features_only=True）。
    """

    def __init__(
        self,
        pretrained_ckpt: str,
        mae_ast_root: str,
        num_classes: int,
        freeze_encoder: bool = False,
        dropout_head: float = 0.3,
    ):
        """
        Parameters
        ----------
        pretrained_ckpt : chunk_patch_75_12LayerEncoder.pt 路徑
        mae_ast_root    : MAE-AST-Public-main 根目錄
        num_classes     : 分類類別數
        freeze_encoder  : True → 只訓練分類頭；False → 全 fine-tune
        dropout_head    : 分類頭 Dropout 比率
        """
        super().__init__()

        # ── 載入 encoder backbone ───────────────────────────
        self.encoder, self.model_cfg, self.task_cfg = load_pretrained_mae_ast(
            ckpt_path    = pretrained_ckpt,
            mae_ast_root = mae_ast_root,
        )

        # ── 移除 pretraining-only 模組（decoder, reconstruction head）──
        # 這些在 fine-tune 時不需要，移除後可節省記憶體並避免誤用
        self.encoder.decoder                   = None
        self.encoder.final_proj_reconstruction = None
        self.encoder.final_proj_classification = None

        # ── 凍結 encoder ──────────────────────────────────
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info("[model] Encoder frozen. Only classification head will be trained.")
        else:
            logger.info("[model] Full fine-tune mode. All parameters trainable.")

        # ── Classification Head ───────────────────────────
        embed_dim = self.model_cfg.encoder_embed_dim  # 768
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(p=dropout_head),
            nn.Linear(embed_dim, num_classes),
        )

        self.num_classes    = num_classes
        self.freeze_encoder = freeze_encoder

        self._log_param_count()

    # ── Forward ───────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [B, 1024, 128] float32

        Returns
        -------
        logits : [B, num_classes] float32（未經 sigmoid）
        """
        # MAE-AST encoder forward（不 mask，只取 encoder features）
        # features_only=True → 回傳 {"x": [B, N_patches, 768], ...}
        result = self.encoder(
            source        = x,
            padding_mask  = None,
            mask          = False,       # fine-tune 不 mask
            features_only = True,        # 不走 decoder
        )

        # [B, 512, 768]
        patch_features = result["x"]

        # Mean pooling over patch tokens → [B, 768]
        pooled = patch_features.mean(dim=1)

        # 分類頭 → [B, num_classes]
        logits = self.classifier(pooled)

        return logits

    # ── 便利方法 ──────────────────────────────────────────
    def get_patch_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        回傳 patch-level features（未 pool），用於視覺化或進階分析。
        Returns: [B, 512, 768]
        """
        result = self.encoder(
            source        = x,
            padding_mask  = None,
            mask          = False,
            features_only = True,
        )
        return result["x"]

    def _log_param_count(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"[model] Total params: {total/1e6:.1f}M | "
            f"Trainable: {trainable/1e6:.1f}M"
        )

    # ── Checkpoint 儲存 ───────────────────────────────────
    def save_checkpoint(self, path: str, epoch: int, optimizer, scheduler, best_map: float):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "epoch"         : epoch,
                "model_state"   : self.state_dict(),
                "optimizer"     : optimizer.state_dict(),
                "scheduler"     : scheduler.state_dict() if scheduler else None,
                "best_map"      : best_map,
                "num_classes"   : self.num_classes,
                "freeze_encoder": self.freeze_encoder,
            },
            path,
        )
        logger.info(f"[model] Checkpoint saved → {path}")

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str,
        pretrained_ckpt: str,
        mae_ast_root: str,
        num_classes: int,
        freeze_encoder: bool = False,
    ):
        """從 fine-tune checkpoint 恢復模型（用於 resume 或 inference）"""
        model = cls(
            pretrained_ckpt = pretrained_ckpt,
            mae_ast_root    = mae_ast_root,
            num_classes     = num_classes,
            freeze_encoder  = freeze_encoder,
        )
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state"])
        logger.info(f"[model] Restored from fine-tune checkpoint: {ckpt_path}")
        return model, state


# ============================================================
#  Quick test
# ============================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import PRETRAINED_CKPT, MAE_AST_PROJECT_ROOT, NUM_CLASSES

    logging.basicConfig(level=logging.INFO)

    model = MAEASTFineTune(
        pretrained_ckpt = PRETRAINED_CKPT,
        mae_ast_root    = MAE_AST_PROJECT_ROOT,
        num_classes     = NUM_CLASSES,
        freeze_encoder  = False,
    )

    # 模擬一個 batch
    dummy = torch.randn(2, 1024, 128)
    logits = model(dummy)
    print(f"Input  shape: {dummy.shape}")   # [2, 1024, 128]
    print(f"Output shape: {logits.shape}")  # [2, NUM_CLASSES]
    print("Model OK!")
