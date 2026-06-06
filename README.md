# MAE-AST Fine-tune — 台灣生態聲景多標籤分類

## 專案結構

```
C:\Users\Lin\Desktop\MAE_AST\
├── MAE-AST-Public-main\          ← MAE-AST 官方專案
├── chunk_patch_75_12LayerEncoder.pt  ← 預訓練權重 (~663 MB)
├── 舊有工具\                      ← SSAST 舊工具
└── mae_finetune\                  ← 本 fine-tune 專案
    ├── config.py                  ← 所有路徑與超參數
    ├── dataset.py                 ← .pt spectrogram DataLoader
    ├── model.py                   ← MAE-AST encoder + 分類頭
    ├── train.py                   ← 訓練主程式
    ├── metrics.py                 ← mAP / per-class AP
    ├── inference.py               ← 推理腳本
    ├── class_labels.csv           ← 類別對照表（需更新）
    ├── json\                      ← 放 train.json / val.json / test.json
    │   ├── train.json
    │   ├── val.json
    │   └── test.json
    ├── checkpoints\               ← 訓練過程自動儲存
    │   ├── best.pt                ← 最佳 val mAP checkpoint
    │   └── last.pt                ← 最後一個 epoch checkpoint
    └── runs\                      ← TensorBoard logs
```

---

## 環境需求

### 安裝指令（Conda env: SSAST-NEW）

```bash
conda activate SSAST-NEW

# fairseq（MAE-AST 模型需要）
pip install fairseq

# 其他套件
pip install torch torchaudio torchvision
pip install scikit-learn pandas numpy tqdm
pip install tensorboard
pip install sortedcontainers
```

> ⚠️ Windows 上 fairseq 可能需要額外步驟：
> ```bash
> pip install fairseq --no-build-isolation
> # 若仍失敗，嘗試：
> conda install -c conda-forge fairseq
> ```

---

## 前置步驟

### 1. 確認 class_labels.csv

`class_labels.csv` 需要與你的 JSON 中的 `labels` 欄位完全對應。

執行以下 Python 確認實際類別：
```python
import json
with open(r"C:\path\to\train.json", "r") as f:
    data = json.load(f)["data"]

all_labels = set()
for d in data:
    all_labels.update(d.get("labels", []))
print(sorted(all_labels))
```

更新 `class_labels.csv` 後，同步更新 `config.py` 中的 `NUM_CLASSES`。

### 2. 確認 JSON 路徑設定

在 `config.py` 中設定：
```python
JSON_DIR = r"C:\Users\Lin\Desktop\MAE_AST\mae_finetune\json"
LABEL_CSV = r"C:\Users\Lin\Desktop\MAE_AST\mae_finetune\class_labels.csv"
SPECTROGRAM_DIR = r"D:\spectrogram_6s_pt_name"
```

### 3. 更新 train.json 路徑為本地路徑

原本 JSON 中的 `wav` 路徑可能含有遠端路徑（`/work/t113618009/...`）。

執行 `舊有工具/prep_6s_ft_json_Mammal_allSplit.py`，
並修改第 45 行：
```python
REMOTE_PATH = r'D:\spectrogram_6s_pt_name'  # 改為本地路徑（正斜線或雙反斜線均可）
```

---

## 訓練指令

### 基本訓練（full fine-tune）
```bash
conda activate SSAST-NEW
cd C:\Users\Lin\Desktop\MAE_AST\mae_finetune

python train.py
```

### 只訓練分類頭（freeze encoder）
```bash
python train.py --freeze_encoder
```

### 自訂超參數
```bash
python train.py \
    --batch_size 16 \
    --epochs 50 \
    --lr 5e-5 \
    --warmup_epochs 3
```

### Resume 訓練
```bash
python train.py --resume checkpoints/last.pt
```

---

## 推理指令

### 單一 .pt 檔
```bash
python inference.py \
    --input "D:\spectrogram_6s_pt_name\FS001_1.pt" \
    --checkpoint checkpoints/best.pt
```

### 批次推理整個資料夾
```bash
python inference.py \
    --input "D:\spectrogram_6s_pt_name" \
    --batch \
    --checkpoint checkpoints/best.pt \
    --output_json predictions.json
```

### Top-K 與 threshold 設定
```bash
python inference.py \
    --input "D:\spectrogram_6s_pt_name\xxx.pt" \
    --topk 3 \
    --threshold 0.4
```

---

## TensorBoard 查看

```bash
conda activate SSAST-NEW
tensorboard --logdir C:\Users\Lin\Desktop\MAE_AST\mae_finetune\runs
# 開啟瀏覽器：http://localhost:6006
```

---

## Checkpoint 儲存位置

| 檔案 | 說明 |
|------|------|
| `checkpoints/best.pt` | 最佳 val mAP checkpoint（推理使用此檔）|
| `checkpoints/last.pt` | 最後一個 epoch（用於 resume）|

每個 checkpoint 包含：
- `model_state` : 模型權重
- `optimizer`   : optimizer 狀態
- `scheduler`   : scheduler 狀態
- `epoch`       : 當前 epoch
- `best_map`    : 最佳 mAP 記錄

---

## VRAM 使用量估計

| 模式 | Batch Size | 預估 VRAM |
|------|-----------|----------|
| Freeze encoder | 64 | ~5~6 GB |
| Freeze encoder | 128 | ~8~10 GB |
| Full fine-tune | 16 | ~8 GB |
| Full fine-tune | 32 | ~12~14 GB |
| Full fine-tune | 64 | ~20 GB（需要 A100/3090）|

> **建議起始設定**：`--batch_size 32`（full fine-tune，12 GB VRAM）

---

## 快速驗證（無 GPU）

```bash
# 測試 dataset 是否能正常讀取
python dataset.py

# 測試模型是否能 forward
python model.py

# 測試 metrics
python metrics.py
```

---

## 常見問題

### fairseq ImportError
```
ImportError: cannot import name 'compute_mask_indices' from 'fairseq.data.data_utils'
```
解法：`pip install fairseq==0.12.2`（指定版本）

### num_workers > 0 在 Windows 會卡住
設定 `config.py` 中 `NUM_WORKERS = 0`（Windows DataLoader 的已知問題）

### BatchNorm 在 float16 報錯
`dataset.py` 已自動轉換：`fbank = fbank.float()`，確保 float32 輸入

### 類別數不匹配
確認 `config.py` 中 `NUM_CLASSES` == `class_labels.csv` 的行數


## 以下執行完的輸出暫存:
Viewed %E7%AC%AC%E4%BA%8C%E9%9A%8E%E6%AE%B5prompt.md:1-172

好的，開始生成全部程式碼。我會依序建立 `mae_finetune/` 資料夾下的所有檔案。

Created config.py
Created dataset.py
Created model.py
Created metrics.py
Created train.py
Created inference.py

現在生成 README 與 class_labels.csv 範本：

Created class_labels.csv
Created README.md
Created __init__.py

現在驗證整體結構：

Listed directory mae_finetune

所有檔案已生成完畢。以下是完整總結：

---

## 生成完成 ✅

### 專案結構

```
mae_finetune/
├── config.py          ← 所有路徑與超參數
├── dataset.py         ← .pt 讀取、float32轉換、multi-hot labels
├── model.py           ← MAE-AST encoder + 分類頭
├── train.py           ← 訓練主程式（AMP/AdamW/cosine/mAP）
├── metrics.py         ← per-class AP + mAP（sklearn）
├── inference.py       ← 單一/批次推理，輸出 JSON
├── class_labels.csv   ← 類別對照表範本（需確認類別數）
└── README.md          ← 完整執行說明
```

---

### 開始前必做的事（2 件）

**① 確認 `class_labels.csv` 類別數**
```python
# 執行這段確認你實際有幾類：
import json
with open(r"C:\...\train.json", "r") as f:
    data = json.load(f)["data"]
all_labels = set()
for d in data:
    all_labels.update(d.get("labels", []))
print(sorted(all_labels), len(all_labels))
```
然後更新 `config.py` 的 `NUM_CLASSES`。

**② 確認 JSON 中的 `wav` 路徑是本地路徑**（`D:\spectrogram_6s_pt_name\...`）

---

### 執行指令

```bash
conda activate SSAST-NEW
pip install fairseq scikit-learn tensorboard tqdm pandas

# 訓練
cd C:\Users\Lin\Desktop\MAE_AST\mae_finetune
python train.py

# 推理
python inference.py --input "D:\spectrogram_6s_pt_name\xxx.pt" --checkpoint checkpoints/best.pt
```

### VRAM 估計

| 模式 | Batch | VRAM |
|------|-------|------|
| Full fine-tune | 32 | ~12 GB |
| Freeze encoder | 64 | ~6 GB |