"""
dataset.py
==========
MAE-AST Fine-tune Dataset
- 讀取 train.json / val.json / test.json
- torch.load(.pt) 讀取預計算 spectrogram
- float16 → float32 轉換（避免 MAE-AST BatchNorm2D 精度問題）
- 不做 AudioSet mean/std normalization（MAE-AST 內部已有 BatchNorm）
- 多標籤 → multi-hot tensor
- 支援 soft label（weights）
"""

import os
import json
import random
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


# ============================================================
#  工具函數
# ============================================================

def build_label_map(label_csv: str) -> dict:
    """
    從 CSV 建立 label_name → index 對照表。
    CSV 需至少含以下其中一種欄位組合：
      (A) 'index', 'display_name'
      (B) 'index', 'mid'
      (C) 只有 display_name（以列序為 index）
    """
    df = pd.read_csv(label_csv, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    label_map = {}

    if "display_name" in df.columns and "index" in df.columns:
        for _, row in df.iterrows():
            idx = int(row["index"])
            name = str(row["display_name"]).strip()
            label_map[name] = idx
            # 也加入 mid（如果有）
            if "mid" in df.columns and pd.notna(row.get("mid")):
                label_map[str(row["mid"]).strip()] = idx

    elif "display_name" in df.columns:
        for i, name in enumerate(df["display_name"]):
            label_map[str(name).strip()] = i

    elif "mid" in df.columns and "index" in df.columns:
        for _, row in df.iterrows():
            label_map[str(row["mid"]).strip()] = int(row["index"])

    else:
        raise ValueError(
            f"[dataset] 無法解析 label_csv: {label_csv}\n"
            f"需要欄位 'display_name' 或 'mid'，現有欄位: {list(df.columns)}"
        )

    num_classes = len(set(label_map.values()))
    logger.info(f"[dataset] Label map loaded: {num_classes} classes, {len(label_map)} mappings")
    return label_map, num_classes


# ============================================================
#  MAEASTDataset
# ============================================================

class MAEASTDataset(Dataset):
    """
    讀取預計算好的 spectrogram .pt 檔。
    每個 .pt 檔格式：
        {"x": Tensor[1024, 128] (float16), "y": Tensor[num_classes]}

    JSON 格式：
        {"data": [
            {"wav": "D:/spectrogram_6s_pt_name/xxx.pt",
             "labels": ["Cicada", "Bird"],
             "weights": [1.0, 0.6]},
            ...
        ]}

    回傳：
        fbank  : Tensor[1024, 128] float32
        labels : Tensor[num_classes] float32 (multi-hot / soft label)
    """

    def __init__(
        self,
        json_path: str,
        label_map: dict,
        num_classes: int,
        spectrogram_dir: str = None,
        target_length: int = 1024,
        is_train: bool = True,
        cache_to_ram: bool = False,
    ):
        """
        Parameters
        ----------
        json_path       : train.json / val.json / test.json 路徑
        label_map       : {label_name: index} 對照表
        num_classes     : 分類類別總數
        spectrogram_dir : 若指定，用此資料夾 + JSON 中的檔名組合路徑；
                          若 None，直接用 JSON 內的 'wav' 完整路徑
        target_length   : time frame 長度，預設 1024（不足補零，超出截斷）
        is_train        : 訓練模式
        cache_to_ram    : True = 啟動時將所有 .pt 全部預載入系統 RAM（float16 格式儲存）
                          可徹底消除 HDD 隨機讀取瓶頸，適合 RAM >= 16GB 的環境
        """
        self.label_map       = label_map
        self.num_classes     = num_classes
        self.spectrogram_dir = spectrogram_dir
        self.target_length   = target_length
        self.is_train        = is_train
        self._ram_tensor     = None  # Tensor[N, T, F] float16，share_memory_()
        self._ram_index      = None  # {pt_path: int} 路徑 → tensor 行號

        # ── 讀取 JSON ──────────────────────────────────────
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"[dataset] JSON not found: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.data_list = raw["data"]
        logger.info(f"[dataset] Loaded {len(self.data_list)} samples from {json_path}")

        # ── 路徑預解析（跳過逐一 os.path.exists 掃描）──────
        # 原本對 20,799 個檔案呼叫 os.path.exists()，
        # 在 NTFS 超大目錄（180 萬檔）下每次掃描可能需要數十秒。
        # 改為直接預解析路徑，缺失檔案由 _load_single() 的 except 處理（自動重抽）。
        # 若需嚴格驗證，可在啟動後執行 dataset.verify_paths()。
        for i, item in enumerate(self.data_list):
            self.data_list[i]["_path"] = self._resolve_path(item["wav"])
        self.valid_indices = list(range(len(self.data_list)))

        logger.info(
            f"[dataset] Paths resolved: {len(self.valid_indices)} samples "
            f"(existence check skipped for NTFS large-directory performance)"
        )
        if len(self.valid_indices) == 0:
            raise RuntimeError(
                "[dataset] No samples found! Check SPECTROGRAM_DIR and JSON paths."
            )

        # ── RAM 全量預加載（Shared Memory 版）──────────────────
        if cache_to_ram:
            self._load_all_to_ram()

    def _load_all_to_ram(self):
        """
        將所有 .pt 以 float16 堆疊成一個大 Tensor，
        並呼叫 .share_memory_() 讓所有 DataLoader worker
        共享同一塊實體記憶體（不複製），解決 Windows OOM 問題。

        記憶體佔用：
          float16: N × 1024 × 128 × 2 bytes
          20,799 筆 ≈ 5.45 GB（只有這一份，worker 不會複製）
        """
        from tqdm import tqdm
        total = len(self.data_list)
        logger.info(
            f"[dataset] 🔄 RAM 預載啟動（shared tensor 模式）：{total} 筆，"
            f"預估 {total * 1024 * 128 * 2 / 1e9:.2f} GB（float16，worker 共享不複製）"
        )

        fbank_list = []
        index_map  = {}   # pt_path → row index in stacked tensor
        failed     = 0
        T, F       = self.target_length, 128

        for item in tqdm(self.data_list, desc="預載 RAM", unit="file", dynamic_ncols=True):
            pt_path = item["_path"]
            try:
                data_dict = torch.load(pt_path, map_location="cpu", weights_only=False)
                fbank = data_dict["x"] if "x" in data_dict else data_dict
                if isinstance(fbank, np.ndarray):
                    fbank = torch.from_numpy(fbank)
                fbank = fbank.half()   # → float16

                # 在預載時做長度對齊，__getitem__ 就不用再做
                n = fbank.shape[0]
                if n > T:
                    fbank = fbank[:T, :]
                elif n < T:
                    pad   = torch.zeros(T - n, F, dtype=torch.float16)
                    fbank = torch.cat([fbank, pad], dim=0)  # [T, F]

                index_map[pt_path] = len(fbank_list)
                fbank_list.append(fbank)

            except Exception as e:
                logger.warning(f"[dataset] 預載失敗，略過: {pt_path}: {e}")
                failed += 1

        # ― 堆疊成單一 Tensor
        self._ram_tensor = torch.stack(fbank_list)   # [N, T, F] float16

        # ⚠️ 立即釋放個別 tensor list（節省 ~5.4 GB RAM）
        # torch.stack() 已把資料複製進新 Tensor，list 可以安全刪除
        del fbank_list
        import gc; gc.collect()

        # ― 放入 Shared Memory（worker 共享，不複製）
        self._ram_tensor.share_memory_()
        self._ram_index = index_map

        ram_gb = self._ram_tensor.element_size() * self._ram_tensor.nelement() / 1e9
        logger.info(
            f"[dataset] ✅ RAM 預載完成：{len(index_map)}/{total} 筆成功"
            f"（{failed} 筆失敗），Tensor shape={tuple(self._ram_tensor.shape)}，"
            f"佔用 RAM ≈ {ram_gb:.2f} GB（shared）"
        )

        # ― Pre-warm：強制 OS 把所有 page 載入實體 RAM
        # 避免 DataLoader worker 第一次讀取時逐頁觸發 Page Fault
        # （每次 Page Fault 在 HDD 系統下可達 10+ ms，共 ~130 萬 pages）
        logger.info("[dataset] 🔥 Pre-warming shared tensor pages（約 3-10 秒）...")
        # 用 float32 累加避免 float16 溢位（float16 max=65504，N元素加總容易發生 inf）
        _ = self._ram_tensor.sum(dtype=torch.float32).item()
        del _
        logger.info("[dataset] ✅ Pre-warm 完成，training 期間不再有 Page Fault")

    # ── Path resolution ─────────────────────────────────────
    def _resolve_path(self, json_wav: str) -> str:
        """
        優先用 spectrogram_dir + filename；
        若 spectrogram_dir 為 None，直接用 json_wav 原始路徑。
        支援 Windows/WSL 雙向轉換。
        """
        from config import to_local_path
        if self.spectrogram_dir:
            filename = os.path.basename(json_wav.replace("\\", "/"))
            return os.path.join(self.spectrogram_dir, filename)
        return to_local_path(json_wav)

    # ── Internal loader ─────────────────────────────────────
    def _load_single(self, index: int, retries: int = 0):
        """讀取單一 .pt，回傳 (fbank: float32, datum: dict)"""
        real_idx = self.valid_indices[index]
        datum    = self.data_list[real_idx]
        pt_path  = datum["_path"]

        try:
            # ── Shared RAM Tensor 優先（若已預載）─────────────
            if self._ram_tensor is not None:
                row = self._ram_index.get(pt_path)
                if row is None:
                    raise FileNotFoundError(f"[dataset] Not in RAM tensor: {pt_path}")
                # shared tensor slice → float32（worker 直接讀共享記憶體，零複製）
                fbank = self._ram_tensor[row].float()   # [T, F] float32
                # 長度對齊已在預載時完成，直接回傳
                return fbank, datum
            else:
                # ── 磁碟讀取（未啟用 RAM cache）──────────
                data_dict = torch.load(pt_path, map_location="cpu", weights_only=False)
                fbank = data_dict["x"] if "x" in data_dict else data_dict
                if isinstance(fbank, np.ndarray):
                    fbank = torch.from_numpy(fbank)
                fbank = fbank.float()  # float16 → float32

            # ── 對齊長度 ─────────────────────────────────────
            n_frames = fbank.shape[0]
            if n_frames > self.target_length:
                fbank = fbank[: self.target_length, :]
            elif n_frames < self.target_length:
                pad = torch.zeros(self.target_length - n_frames, fbank.shape[1])
                fbank = torch.cat([fbank, pad], dim=0)

            return fbank, datum

        except Exception as e:
            logger.warning(f"[dataset] Error loading {pt_path}: {e}")
            if retries >= 10:
                raise RuntimeError(
                    f"[dataset] Too many missing files. Aborting! "
                    f"Please check SPECTROGRAM_DIR and files exist. Last error: {e}"
                )
            alt = random.randint(0, len(self.valid_indices) - 1)
            return self._load_single(alt, retries + 1)

    # ── Label builder ────────────────────────────────────────
    def _build_label(self, datum: dict, lam: float = 1.0) -> np.ndarray:
        """
        將 datum 中的 labels + weights 轉成 multi-hot array。
        支援 soft label（weights 值可為 0.6 等）。
        """
        label_vec = np.zeros(self.num_classes, dtype=np.float32)
        labels    = datum.get("labels", [])
        weights   = datum.get("weights", [])

        if isinstance(labels, str):
            labels = [labels]
        if not weights:
            weights = [1.0] * len(labels)
        if len(labels) != len(weights):
            min_len = min(len(labels), len(weights))
            labels  = labels[:min_len]
            weights = weights[:min_len]

        for lbl, w in zip(labels, weights):
            lbl = str(lbl).strip()
            if lbl in self.label_map:
                idx = self.label_map[lbl]
                label_vec[idx] = max(label_vec[idx], float(w) * lam)
            else:
                logger.debug(f"[dataset] Label '{lbl}' not in label_map, skipping.")

        return label_vec

    # ── SpecAugment ──────────────────────────────────────
    def _spec_augment(self, fbank: torch.Tensor) -> torch.Tensor:
        """
        SpecAugment 資料增強（僅訓練模式）。
        fbank: [T, F] = [1024, 128]

        時間遠罩：2 個 mask，每個最多 80 frames（約 0.5 秒）
        頻率遠罩：2 個 mask，每個最多 20 mel bins
        """
        T, F = fbank.shape
        fbank = fbank.clone()

        # Time masking
        for _ in range(2):
            t = random.randint(0, min(80, T))
            if t > 0:
                t0 = random.randint(0, T - t)
                fbank[t0:t0 + t, :] = 0.0

        # Frequency masking
        for _ in range(2):
            f = random.randint(0, min(20, F))
            if f > 0:
                f0 = random.randint(0, F - f)
                fbank[:, f0:f0 + f] = 0.0

        return fbank

    # ── __getitem__ ───────────────────────────────────────
    def __getitem__(self, index: int):
        fbank, datum = self._load_single(index)
        label_vec    = self._build_label(datum)

        if self.is_train:
            # SpecAugment 訓練增強（對抗過擬合）
            fbank = self._spec_augment(fbank)
        else:
            # 驗證模式：將 soft label 二値化
            label_vec = (label_vec > 0).astype(np.float32)

        return fbank, torch.FloatTensor(label_vec)

    def __len__(self):
        return len(self.valid_indices)


# ============================================================
#  DataLoader 工廠
# ============================================================

def build_dataloader(
    json_path: str,
    label_map: dict,
    num_classes: int,
    spectrogram_dir: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    is_train: bool,
    target_length: int = 1024,
    cache_to_ram: bool = False,
) -> DataLoader:
    """
    cache_to_ram=True：啟動時將所有 .pt 預載到系統 RAM（float16）。
    適用於 HDD 讀取瓶頸、RAM >= 16GB 的環境。
    預載完成後 DataLoader workers 不再讀取磁碟，只做 CPU 計算。
    """
    dataset = MAEASTDataset(
        json_path       = json_path,
        label_map       = label_map,
        num_classes     = num_classes,
        spectrogram_dir = spectrogram_dir,
        target_length   = target_length,
        is_train        = is_train,
        cache_to_ram    = cache_to_ram,
    )

    # ― share_memory_() 已解決 Windows OOM 問題：
    #   dataset._ram_tensor 是 shared tensor，worker spawn 時
    #   PyTorch 只傳遞 handle（幾個 bytes），不複製 5+ GB 資料。
    #   因此可以安全使用 num_workers > 0。
    prefetch = 2 if num_workers > 0 else None

    loader = DataLoader(
        dataset,
        batch_size         = batch_size,
        shuffle            = is_train,
        num_workers        = num_workers,
        pin_memory         = pin_memory,
        drop_last          = is_train,
        persistent_workers = (num_workers > 0),
        prefetch_factor    = prefetch,
    )
    return loader


# ============================================================
#  Quick test
# ============================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import (
        TRAIN_JSON, SPECTROGRAM_DIR, LABEL_CSV,
        BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, TARGET_LENGTH
    )

    logging.basicConfig(level=logging.INFO)

    label_map, num_classes = build_label_map(LABEL_CSV)
    print(f"num_classes = {num_classes}")

    loader = build_dataloader(
        json_path       = TRAIN_JSON,
        label_map       = label_map,
        num_classes     = num_classes,
        spectrogram_dir = SPECTROGRAM_DIR,
        batch_size      = 4,
        num_workers     = 0,
        pin_memory      = False,
        is_train        = True,
    )

    fbank, labels = next(iter(loader))
    print(f"fbank.shape  = {fbank.shape}")   # [4, 1024, 128]
    print(f"labels.shape = {labels.shape}")  # [4, num_classes]
    print(f"fbank.dtype  = {fbank.dtype}")   # float32
    print("Dataset OK!")
