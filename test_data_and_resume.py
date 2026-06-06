"""
test_data_and_resume.py
=======================
訓練前快速驗證腳本：
1. 確認 .pt 資料路徑正確，可以讀取
2. 確認 checkpoint 存在且可以載入（epoch、mAP 資訊）
3. 顯示即將 resume 的起點
"""

import os, sys, random, torch
import config as CFG

print("=" * 60)
print("【Step 1】 檢查 SPECTROGRAM_DIR")
print(f"路徑: {CFG.SPECTROGRAM_DIR}")
if not os.path.exists(CFG.SPECTROGRAM_DIR):
    print("❌ 資料夾不存在！請確認 .pt 已複製完成，或路徑設定正確。")
    sys.exit(1)

all_pts = [f for f in os.listdir(CFG.SPECTROGRAM_DIR) if f.endswith(".pt")]
print(f"✅ 找到 {len(all_pts)} 個 .pt 檔案")

# 隨機抽 5 個試讀
print("\n隨機抽 5 個 .pt 試讀...")
sample = random.sample(all_pts, min(5, len(all_pts)))
for fname in sample:
    full_path = os.path.join(CFG.SPECTROGRAM_DIR, fname)
    try:
        data = torch.load(full_path, map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            shape = data.get("x", next(iter(data.values()))).shape
        else:
            shape = data.shape
        print(f"  ✅ {fname} → shape: {shape}")
    except Exception as e:
        print(f"  ❌ {fname} → 讀取失敗: {e}")

print()
print("=" * 60)
print("【Step 2】 檢查 Checkpoint")

last_ckpt = os.path.join(CFG.CHECKPOINT_DIR, "last.pt")
best_ckpt = os.path.join(CFG.CHECKPOINT_DIR, "best.pt")

for ckpt_path, label in [(last_ckpt, "last.pt"), (best_ckpt, "best.pt")]:
    if not os.path.exists(ckpt_path):
        print(f"  ⚠️  {label} 不存在: {ckpt_path}")
        continue
    size_mb = os.path.getsize(ckpt_path) / 1024 / 1024
    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        epoch    = state.get("epoch", "未知")
        best_map = state.get("best_map", 0.0)
        print(f"  ✅ {label} ({size_mb:.0f} MB) → Epoch: {epoch}, Best mAP: {best_map:.4f}")
    except Exception as e:
        print(f"  ❌ {label} 載入失敗: {e}")

print()
print("=" * 60)
print("【Step 3】 環境總結")
print(f"SPECTROGRAM_DIR : {CFG.SPECTROGRAM_DIR}")
print(f"CHECKPOINT_DIR  : {CFG.CHECKPOINT_DIR}")
print(f"TRAIN_JSON      : {CFG.TRAIN_JSON}")
print(f"VAL_JSON        : {CFG.VAL_JSON}")
print(f"NUM_WORKERS     : {CFG.NUM_WORKERS}")
print(f"BATCH_SIZE      : {CFG.BATCH_SIZE}")
print(f"LR              : {CFG.LR}")

print()
print("一切正常後，執行以下指令接續訓練：")
print(f"  python train.py --resume {last_ckpt}")
