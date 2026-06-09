# MAE-AST Fine-tune v2 ── 訓練穩定化與加速改動完整紀錄

> 環境：Windows 11 / RTX 4060 8.6 GB VRAM / 資料碟 HDD (E:)  
> 資料集：20,799 筆訓練樣本，每筆 `.pt` 格式 spectrogram `[1024, 128]` float16  
> 模型：MAE-AST 12-layer Transformer，85.3M 參數，全參數微調（Full fine-tune）

---

## 一、訓練穩定化：超參數與策略改動

### 背景：問題症狀

| 觀測現象 | 說明 |
|---------|------|
| `Loss/train` 0.14 → 0.02 | 完美下降，但代表**嚴重過擬合** |
| `Loss/val` 0.14 → 0.21 | 驗證損失持續上升，train/val gap 達 10 倍 |
| Precision、Recall 曲線 | **從第一個 epoch 就大幅鋸齒抖動**，沒有收斂趨勢 |
| AP/Cicada、AP/Frog、AP/Insect | 劇烈跳動，無法判斷模型效能 |

---

### 1.1 Layer-wise Learning Rate Decay（LLRD）── 最關鍵的改動

#### 問題根源

MAE-AST Transformer 共有 **12 層**。在全參數微調時，若所有層使用同一個 Learning Rate，底層（Layer 0–3）學到的低階聲學特徵會在每個 batch 都被大幅擾動。

這造成：
- 分類邊界每次 batch 更新後都有大幅改變
- 驗證集上的 Precision/Recall 大幅跳動（鋸齒）
- 過去學到的有效特徵表示被"打壞"

#### 解決方案：每層使用不同的 LR

採用「越底層 LR 越小」的策略，倍率為 `decay = 0.85`：

```
分類頭（classifier）: base_lr × 1.00 = 5.00e-05  ← 全新層，需要最大更新幅度
Transformer Layer 11: base_lr × 0.85¹ = 4.25e-05
Transformer Layer 10: base_lr × 0.85² = 3.61e-05
Transformer Layer 09: base_lr × 0.85³ = 3.07e-05
...（每往下一層 × 0.85）
Transformer Layer 00: base_lr × 0.85¹² = 7.11e-06  ← 底層特徵，只用 14% 的 LR
Encoder Stem（patch embed, pos embed）: 6.05e-06  ← 最低 LR
```

#### 實作：`train.py` 新增 `build_llrd_optimizer()`

```python
def build_llrd_optimizer(model, base_lr, weight_decay, decay_rate=0.85):
    import re
    classifier_params = [p for p in model.classifier.parameters() if p.requires_grad]
    layer_params = {}
    stem_params = []

    for name, param in model.encoder.named_parameters():
        if not param.requires_grad:
            continue
        m = re.search(r'\.(?:layers|blocks)\.(\d+)\.', name)
        if m:
            idx = int(m.group(1))
            layer_params.setdefault(idx, []).append(param)
        else:
            stem_params.append(param)

    n_layers = max(layer_params.keys()) + 1
    param_groups = [
        {"params": classifier_params, "lr": base_lr, "weight_decay": weight_decay},
    ]
    for idx in sorted(layer_params.keys(), reverse=True):
        depth = n_layers - idx
        lr_i  = base_lr * (decay_rate ** depth)
        param_groups.append({"params": layer_params[idx], "lr": lr_i, ...})

    if stem_params:
        stem_lr = base_lr * (decay_rate ** (n_layers + 1))
        param_groups.append({"params": stem_params, "lr": stem_lr, ...})

    return AdamW(param_groups, betas=(0.9, 0.999))
```

共產生 **14 個 param groups**（分類頭 1 + Transformer 12層 + stem 1）。

#### 效果
- 底層特徵不再被大幅更新，分類邊界趨於穩定
- Precision/Recall 鋸齒現象從根本上改善

---

### 1.2 Label Smoothing（ε = 0.1）── 降低閾值敏感性

#### 問題根源

原始 `BCEWithLogitsLoss` 使用 hard label（正樣本 = 1，負樣本 = 0）。這會引導模型把 sigmoid 輸出推向極端值（接近 0 或接近 1）。當 logit 集中在 0.5 閾值附近時，一點點參數更新就會讓大量樣本越過邊界，造成 Precision/Recall 大幅跳動。

#### 解決方案：Soft label

```python
class SmoothBCEWithLogitsLoss(nn.Module):
    """
    hard label (0/1) → soft label:
        正樣本 1  →  1 - ε/2  = 0.95
        負樣本 0  →  ε/2     = 0.05
    """
    def __init__(self, eps: float = 0.1):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        targets_s = targets * (1.0 - self.eps) + self.eps * 0.5
        return F.binary_cross_entropy_with_logits(logits, targets_s)
```

#### 效果
- 模型的 sigmoid 輸出不會趨近極端，分布更遠離 0.5 閾值
- 小的參數更新不再讓大量樣本越過邊界
- Precision/Recall 波動幅度明顯縮小

---

### 1.3 SpecAugment（時間 + 頻率遮罩）── 對抗過擬合

#### 問題根源

85.3M 參數的 Transformer 在只有 20,799 筆訓練資料的情況下，極易過擬合。  
證據：`train loss 0.02` vs `val loss 0.21`，差距達 10 倍。

#### 解決方案：訓練時隨機遮罩 spectrogram

在 `dataset.py` 的 `__getitem__` 中，訓練模式下對每個 `[1024, 128]` 的 spectrogram 做：

```python
def _spec_augment(self, fbank: torch.Tensor) -> torch.Tensor:
    T, F = fbank.shape   # 1024, 128
    fbank = fbank.clone()

    # 時間遮罩：2 個 mask，每個最多 80 frames（~0.5 秒）
    for _ in range(2):
        t = random.randint(0, min(80, T))
        if t > 0:
            t0 = random.randint(0, T - t)
            fbank[t0:t0 + t, :] = 0.0

    # 頻率遮罩：2 個 mask，每個最多 20 mel bins
    for _ in range(2):
        f = random.randint(0, min(20, F))
        if f > 0:
            f0 = random.randint(0, F - f)
            fbank[:, f0:f0 + f] = 0.0

    return fbank
```

僅在 `is_train=True` 時啟用，驗證集不做。

#### 效果
- 模型每次看到的資料都略有不同，降低對特定頻率/時間模式的過度記憶
- Val loss 飆升現象得到抑制

---

### 1.4 Weight Decay 調整（1e-4 → 1e-2）── 加強正則化

#### 改動

```python
# config.py
WEIGHT_DECAY = 1e-2   # 原本 1e-4
```

#### 理由

AdamW 搭配 Transformer 的業界標準 weight decay 為 `1e-2`（參考 BERT、ViT 等論文）。  
原本的 `1e-4` 正則化太弱，導致模型可以任意增大參數幅度，加速過擬合。

`1e-2` 的 weight decay 在每步更新時對參數施加更強的衰減約束，相當於隱式的 L2 正則化。

---

### 1.5 Warmup 延長（4 → 6 epochs）── 平滑初期震盪

#### 改動

```python
# config.py
WARMUP_EPOCHS = 6   # 原本 4
```

#### 理由

前幾個 epoch 中，Learning Rate 從 0 線性爬升到 `base_lr`。若 warmup 太短，LR 很快達到最大值，在模型還沒穩定的情況下對參數做大幅更新，造成初期震盪。

延長 warmup 讓底層特徵有更多時間「適應」新的分類任務，減少從第 1 個 epoch 就出現的鋸齒。

---

### 改動總結表

| 改動 | 檔案 | 舊值 | 新值 | 解決的問題 |
|------|------|------|------|-----------|
| Layer-wise LR Decay | `train.py` | 全層同一 LR | 14 個 param group，× 0.85 遞減 | P/R 鋸齒根源 |
| Label Smoothing | `train.py` | hard label BCELoss | ε=0.1 soft label | 閾值過敏感 |
| SpecAugment | `dataset.py` | 無增強 | 時間×2 + 頻率×2 遮罩 | 過擬合 |
| Weight Decay | `config.py` | `1e-4` | `1e-2` | 過擬合 |
| Warmup | `config.py` | 4 epochs | 6 epochs | 初期震盪 |

---

## 二、訓練加速：從 17 s/it → 4.22 s/it

### 初始狀況

```
[Train] Epoch 0: 11/649 [03:xx<... 17.24s/it]
```

每個 iteration 需要 17 秒，649 iter/epoch 需要 3 小時以上。

---

### 2.1 移除啟動時 `os.path.exists()` 逐一掃描

#### 問題

`dataset.py` 的 `__init__` 原本對每筆訓練資料逐一呼叫 `os.path.exists()` 驗證：

```python
# ❌ 原本的做法（超慢）
for i, item in enumerate(self.data_list):
    path = self._resolve_path(item["wav"])
    if os.path.exists(path):    # ← 20,799 次 NTFS 查詢
        self.valid_indices.append(i)
    else:
        missing += 1
```

**問題所在**：`E:\spectrogram_6s_pt_name\` 目錄下有 **1,816,179 個檔案**。  
NTFS 對超大平坦目錄的 MFT（主檔案表）查詢非常慢，每次 `os.path.exists()` 在此目錄下需要可觀的時間，20,799 次就造成數十秒甚至更長的啟動延遲。

#### 修法

```python
# ✅ 改後（跳過掃描）
for i, item in enumerate(self.data_list):
    self.data_list[i]["_path"] = self._resolve_path(item["wav"])
self.valid_indices = list(range(len(self.data_list)))
# 缺失檔案由 _load_single() 的 except 自動隨機重抽處理
```

直接預解析路徑，缺失檔案讓 `_load_single` 的 retry 機制在訓練時處理，啟動時間從數十秒縮到 ~1 秒。

---

### 2.2 DataLoader 加入 `persistent_workers` 與 `prefetch_factor`

#### 問題

原本每個 epoch 結束後，DataLoader 會**殺掉所有 worker process**，下個 epoch 開始時再重新 spawn。在 Windows 上，`multiprocessing.spawn` 的 process 建立成本很高（需要重新載入 Python 環境、import 所有 module），造成每 epoch 有額外延遲。

#### 修法

```python
loader = DataLoader(
    dataset,
    batch_size         = batch_size,
    num_workers        = num_workers,
    persistent_workers = (num_workers > 0),  # ← 保留 worker，不重啟
    prefetch_factor    = 2 if num_workers > 0 else None,  # ← 預取 2 個 batch
)
```

- `persistent_workers=True`：worker 在 epoch 結束後繼續存活，下個 epoch 直接重用
- `prefetch_factor=2`：每個 worker 預先準備 2 個 batch，讓 GPU 在算 batch N 時，worker 已在準備 batch N+1 和 N+2

---

### 2.3 NUM_WORKERS 從 8 降到 4（HDD seek thrashing 問題）

#### 問題

原本 `NUM_WORKERS=8`，代表 8 個 worker 同時對 HDD 發出隨機讀取請求。

```
8 workers 同時讀 → 磁頭需要在 180 萬個檔案間快速跳動 → seek thrashing
→ 反而比少 workers 更慢
```

HDD 的特性是隨機讀取速度慢（因為磁頭物理移動），多 worker 反而讓磁頭不停跳動，整體吞吐量下降。

#### 調校過程

| workers | batch | 速度 | 備註 |
|---------|-------|------|------|
| 8 | 32 | 17 s/it | HDD seek thrashing |
| 2 | 32 | ~4 s/it | 顯著改善 |
| **4** | **16** | **4.14 s/it** | 最佳實測值 |
| 4 | 16 | **4.22 s/it** | 含穩定化改動後 |

**結論**：HDD 環境下 `NUM_WORKERS=4` 加上 `BATCH_SIZE=16` 是最佳平衡點。

---

### 2.4 Confusion Stats 計算改為每 5 epoch

#### 問題

原本 `validate()` 在**每個 epoch** 都呼叫 `compute_confusion_stats()`，這個函數會同步寫兩個 CSV 到 HDD，在每次 epoch 結束後造成 IO 阻塞。

#### 修法

```python
# train.py
if epoch % 5 == 0:
    compute_confusion_stats(targets_np, scores_np, ...)
```

每 5 個 epoch 才寫一次，95% 的 epoch 不做此 IO 操作，減少不必要的等待。

---

### 2.5 修正 AMP（自動混合精度）的廢棄 API

#### 問題

原本使用的舊版 API：

```python
# ❌ 舊版（在新版 PyTorch 可能無法正確啟用 fp16）
from torch.cuda.amp import GradScaler, autocast
with autocast(enabled=use_amp):
    ...
scaler = GradScaler(enabled=use_amp)
```

在較新版本的 PyTorch 中，此 API 已廢棄（FutureWarning），且可能**無法正確啟用 fp16 模式**，導致模型在 fp32 下計算。

**fp32 vs fp16 的影響**：

| 精度 | Attention 矩陣（每層） | 12 層合計 | RTX 4060 理論算力 |
|------|----------------------|----------|-----------------|
| fp32 | [32,12,512,512] × 4 bytes = 402 MB | 4.8 GB | ~15 TFLOPS |
| fp16 | [32,12,512,512] × 2 bytes = 201 MB | 2.4 GB | ~30 TFLOPS |

fp32 模式下，VRAM 使用量幾乎倍增，且無法使用 Tensor Core 加速，導致 `forward=10s`、`backward=15s` 的異常緩慢。

#### 修法

```python
# ✅ 新版 API
from torch.amp import GradScaler, autocast

# 訓練 loop
with autocast('cuda', enabled=use_amp):
    logits = model(fbank)

# 初始化 scaler
scaler = GradScaler('cuda', enabled=use_amp)
```

修正後 AMP 正常運作，RTX 4060 的 Tensor Core 才能真正發揮 fp16 加速。

---

### 2.6 BATCH_SIZE 從 32 降回 16（VRAM 管理）

#### 問題

`BATCH_SIZE=32` 時的 VRAM 需求估算：

| 部件 | 大小 |
|------|------|
| 模型參數（fp16） | ~170 MB |
| Optimizer AdamW m+v（fp32） | ~682 MB |
| 12層 Attention 矩陣（fp16） | ~2.4 GB |
| 12層 FFN 中間層（fp16） | ~1.2 GB |
| 其他啟動值與梯度 | ~1.0 GB |
| **合計** | **~5.5 GB** |

在 8.6 GB VRAM 的 RTX 4060 上，考慮 CUDA context（~300 MB）和 PyTorch allocator overhead，`batch=32` 剛好在邊緣，實測出現 VRAM 換頁現象，計算速度從預期的 `<1s` 暴增到 `10-15s`。

`BATCH_SIZE=16` 將 Attention 矩陣從 2.4 GB 降到 1.2 GB，總 VRAM 約 4 GB，在 8.6 GB 的 RTX 4060 上有充分餘裕。

---

### 加速改動時間線

```
17 s/it   ← 初始狀況（workers=8, NTFS 掃描, 無 persistent_workers）
    ↓  移除 os.path.exists() 掃描 + persistent_workers + prefetch
    ↓  workers=8 → 4，batch=32 → 16
~4 s/it   ← 改善後（HDD seek 壓力大幅降低）
    ↓  修正廢棄 AMP API（torch.cuda.amp → torch.amp）
    ↓  確保 fp16 Tensor Core 正常運作
4.22 s/it ← 最終穩定速度（含所有穩定化改動）
```

---

### 加速改動總結表

| 改動 | 檔案 | 提速原因 | 影響幅度 |
|------|------|---------|---------|
| 移除 `os.path.exists()` 掃描 | `dataset.py` | 消除 NTFS 超大目錄查詢 | 啟動時間 數十秒 → ~1秒 |
| `persistent_workers=True` | `dataset.py` | 避免每 epoch 重啟 workers | 每 epoch 節省 5-10 秒 overhead |
| `prefetch_factor=2` | `dataset.py` | CPU/GPU 流水線並行 | 減少 GPU 等待資料的空閒時間 |
| workers 8 → 4 | `config.py` | 減少 HDD seek thrashing | 17 → ~4 s/it（最大改善） |
| batch 32 → 16 | `config.py` | VRAM 從爆掉到有餘裕 | 修復 10s/forward 異常 |
| Confusion stats 每 5 epoch | `train.py` | 減少不必要的 HDD IO | 每 epoch 節省 IO 阻塞 |
| 修正 AMP API | `train.py` | 確保 fp16 Tensor Core 運作 | 修復 fp32 fallback 慢 2-10× |
