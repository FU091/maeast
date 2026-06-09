"""
config.py
=========
MAE-AST Fine-tune 全域設定檔
所有路徑均為 Windows 本地路徑，不依賴遠端伺服器。
"""

import os

def to_local_path(path: str) -> str:
    r"""
    動態轉換 Windows 路徑為 WSL 格式路徑（若在 Linux/WSL 環境下執行）。
    例如: C:\Users\Lin -> /mnt/c/Users/Lin
    """
    if os.name == 'posix' and path:
        import re
        # 匹配 Windows 槽區字元 (如 C:\ 或 D:\)
        match = re.match(r'^([a-zA-Z]):\\(.*)', path)
        if match:
            drive = match.group(1).lower()
            rest = match.group(2).replace('\\', '/')
            return f"/mnt/{drive}/{rest}"
        # 如果是 Windows 格式的反斜線但沒有槽區，也做替換
        return path.replace('\\', '/')
    return path

# ============================================================
#  路徑設定
# ============================================================

# MAE-AST 官方專案根目錄（用於 import MAE_AST class）
MAE_AST_PROJECT_ROOT = to_local_path(r"E:\MAE_AST\MAE_AST\MAE-AST-Public-main")

# 預訓練權重
PRETRAINED_CKPT = to_local_path(r"E:\MAE_AST\MAE_AST\chunk_patch_75_12LayerEncoder.pt")

# Spectrogram .pt 資料夾
# 已搬入 WSL 原生路徑（/home/lin/ 在 E 槽 VHD 中，讀寫速度遠快於 /mnt/e/）
if os.name == 'posix':
    SPECTROGRAM_DIR = "/home/lin/spectrogram_6s_pt_name"
else:
    SPECTROGRAM_DIR = to_local_path(r"E:\spectrogram_6s_pt_name")  # Windows fallback

# JSON 資料集索引（train / val / test）
# 若有多個 JSON 放在同一資料夾可直接改此路徑
JSON_DIR = to_local_path(r"E:\ssast_hub\all_mammal_merged\replaced")
TRAIN_JSON = os.path.join(JSON_DIR, "all_merged_train.json")
VAL_JSON   = os.path.join(JSON_DIR, "all_merged_val.json")
TEST_JSON  = os.path.join(JSON_DIR, "all_merged_test.json")

# Label CSV（含 index, display_name 欄位）
LABEL_CSV = to_local_path(r"E:\MAE_AST\MAE_AST\mae_finetune_v2\class_labels.csv")

# Checkpoint 儲存路徑與 TensorBoard log 路徑 (自動適應 WSL 原生 ext4 系統與 Windows UNC 路徑)
if os.name == 'posix':
    CHECKPOINT_DIR  = "/home/lin/MAE_output/checkpoints"
    TENSORBOARD_DIR = "/home/lin/MAE_output/runs"
else:
    CHECKPOINT_DIR  = r"E:\MAE_AST\MAE_output_v2\checkpoints"
    TENSORBOARD_DIR = r"E:\MAE_AST\MAE_output_v2\runs"

# ============================================================
#  模型設定
# ============================================================

# 分類類別數（依照你的 label_csv 決定）
# 例：Cicada, Insect, Bird, Owl, Frog, Rain, Wind, Stream,
#      Aircraft, Machine, Speech, Bat + MammalLow_* 動態類別
NUM_CLASSES = 13  # ← 執行前請先確認你的 label_csv 行數

# Encoder embed dim（MAE-AST 12-layer 固定 768）
ENCODER_EMBED_DIM = 768

# Spectrogram 固定 shape
TARGET_LENGTH = 1024   # time frames
NUM_MEL_BINS  = 128    # frequency bins

# ============================================================
#  訓練超參數
# ============================================================

# Batch size
# - freeze encoder:     建議 64（VRAM ~6 GB）
# - full fine-tune:     建議 16~32（VRAM ~10~16 GB）
BATCH_SIZE     = 16      # batch=32 會把 RTX 4060 8.6GB VRAM 打滿（attention 機制在 12 層共需 ~6GB）
NUM_WORKERS    = 4       # HDD 環境：4 workers 擐塡 CPU-GPU 流水線
CACHE_TO_RAM   = False   # 關閉 RAM 全量預載，避免堆疊對 VRAM 造成額外壓力
PIN_MEMORY     = True

# Learning rate
LR             = 5e-5    # base LR（LLRD 會對底層自動縮小）
WEIGHT_DECAY   = 1e-2    # 1e-4 → 1e-2：AdamW + Transformer 的標準正則化強度
EPOCHS         = 30

# Cosine scheduler warmup（epoch 數）
WARMUP_EPOCHS  = 6       # 4 → 6：更緩慢地把 LR 推上去，減少初期震盪

# Layer-wise LR Decay（LLRD）
# 每往下一層 LR 乘以此倍率，防止底層預訓練特徵被破壞
LLRD_DECAY     = 0.85    # 業界常用 0.75~0.90；0.85 適合 12 層 Transformer

# Label Smoothing
# 防止模型把 logit 推到極端值，降低對 0.5 閾值的過度敏感性
LABEL_SMOOTH   = 0.1     # 0 = 無，0.1 = 標準值

# 是否凍結 encoder（True = 只訓練分類頭）
FREEZE_ENCODER = False

# AMP mixed precision
USE_AMP        = True

# Resume checkpoint（None 則從頭訓練）
RESUME_CKPT    = None   # e.g. r"C:\...\checkpoints\best.pt"

# ============================================================
#  Inference 設定
# ============================================================

# Top-k 輸出類別數
TOP_K = 5

# Sigmoid 閾值（用於判斷 positive class）
THRESHOLD = 0.5
