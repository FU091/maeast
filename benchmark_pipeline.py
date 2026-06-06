"""
benchmark_pipeline.py
測試目前 dataset 初始化與讀取速度，再對比優化後的速度
"""
import time, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as CFG
from dataset import build_label_map, MAEASTDataset, build_dataloader
from torch.utils.data import DataLoader

lmap, nc = build_label_map(CFG.LABEL_CSV)

# ── TEST 1: Dataset __init__ time (含 os.path.exists 掃描) ──
print("=" * 55)
print("TEST 1: Dataset __init__ time (os.path.exists scan)")
print("=" * 55)
t0 = time.time()
ds = MAEASTDataset(
    json_path=CFG.TRAIN_JSON, label_map=lmap, num_classes=nc,
    spectrogram_dir=CFG.SPECTROGRAM_DIR, is_train=True
)
init_time = time.time() - t0
print(f"  Dataset init : {init_time:.2f}s  ({len(ds)} valid samples)\n")

# ── TEST 2: 現有 DataLoader 速度（無 persistent_workers）──
print("=" * 55)
print("TEST 2: DataLoader speed (num_workers=8, no persistent_workers)")
print("=" * 55)
loader_old = DataLoader(
    ds, batch_size=32, shuffle=True,
    num_workers=8, pin_memory=False, drop_last=True
)
it = iter(loader_old)
times = []
for i, (x, y) in enumerate(it):
    if i == 0:
        t0 = time.time()  # 從第0筆開始計時，避免 worker 啟動干擾
        print(f"  Batch shape : {x.shape}, dtype: {x.dtype}")
    else:
        times.append(time.time() - t0)
        t0 = time.time()
    if i >= 10:
        break

if times:
    avg = sum(times) / len(times)
    print(f"  avg s/it    : {avg:.3f}s  (over {len(times)} batches)")
    print(f"  est epoch   : {avg * 649 / 60:.1f} min\n")
