"""
benchmark_after.py
測試優化後的 dataset 初始化與讀取速度
"""
import time, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as CFG
from dataset import build_label_map, MAEASTDataset
from torch.utils.data import DataLoader

lmap, nc = build_label_map(CFG.LABEL_CSV)

# ── TEST: 優化後 Dataset __init__ time ──
print("=" * 55)
print("TEST 1: Dataset __init__ time (AFTER - no exists scan)")
print("=" * 55)
t0 = time.time()
ds = MAEASTDataset(
    json_path=CFG.TRAIN_JSON, label_map=lmap, num_classes=nc,
    spectrogram_dir=CFG.SPECTROGRAM_DIR, is_train=True
)
init_time = time.time() - t0
print(f"  Dataset init : {init_time:.2f}s  ({len(ds)} samples)\n")

# ── TEST: 優化後 DataLoader (persistent_workers=True) ──
print("=" * 55)
print("TEST 2: DataLoader speed (num_workers=8, persistent_workers=True, prefetch=2)")
print("=" * 55)
loader_new = DataLoader(
    ds, batch_size=32, shuffle=True,
    num_workers=8, pin_memory=False, drop_last=True,
    persistent_workers=True, prefetch_factor=2,
)
it = iter(loader_new)
times = []
for i, (x, y) in enumerate(it):
    if i == 0:
        t0 = time.time()
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

print("Done.")
