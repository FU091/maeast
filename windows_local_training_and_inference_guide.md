# MAE-AST Windows 本地訓練與推論使用手冊 📖

> **環境需求**：請確保您已依照 `windows_local_environment_setup.md` 完成環境建立。

---

## 1. 啟用環境與進入工作目錄

每次打開 PowerShell 或 Windows Terminal 時，請先執行以下指令以啟用虛擬環境，並切換到我們建立好的工作目錄：

```powershell
# 1. 啟用我們準備好的本地環境
conda activate mae_v2

# 2. 進入本地專屬的 v2 訓練目錄
cd E:\MAE_AST\MAE_AST\mae_finetune_v2
```

---

## 2. 開始訓練 (Training)

在 `mae_finetune_v2` 目錄下，直接執行訓練主程式即可：

```powershell
python train.py
```

### 💡 訓練進階用法：
- **只訓練分類頭 (Freeze Encoder)**：如果您不想更動預訓練的 MAE-AST 權重，想省 VRAM：
  ```powershell
  python train.py --freeze_encoder
  ```
- **接續訓練 (Resume)**：如果訓練中斷，可以從最新的 Checkpoint 接續訓練：
  ```powershell
  python train.py --resume E:\MAE_AST\MAE_output_v2\checkpoints\last.pt
  ```

---

## 3. 訓練輸出結果在哪裡？

因為我們改用了純 Windows 環境，`config.py` 會自動幫您將輸出導向到全新的本地專屬目錄（遠離 WSL），避免跟以前的紀錄混淆：

- **訓練好的模型 (Checkpoints)** 會儲存在：
  👉 `E:\MAE_AST\MAE_output_v2\checkpoints\`
  *(包含 `best.pt` 與 `last.pt`)*

- **圖表與紀錄檔 (TensorBoard Logs)** 會儲存在：
  👉 `E:\MAE_AST\MAE_output_v2\runs\`

### 📊 即時監控訓練圖表
請**額外開啟一個新的 PowerShell 視窗**（不用關閉正在訓練的視窗），啟用相同的環境後執行：
```powershell
conda activate mae_v2
tensorboard --logdir E:\MAE_AST\MAE_output_v2\runs --bind_all
```
接著打開瀏覽器輸入：`http://localhost:6006`，就能即時看見 mAP 與 Loss 的變化曲線了。

---

## 4. 如何進行推論與預測 (Inference)

當您訓練出滿意的模型後（通常是 `best.pt`），您可以使用 `inference.py` 來辨識音訊特徵 (`.pt` 檔)。

### 📍 預設載入模型：
如果您不加上 `--checkpoint` 參數，它預設會去讀取我們剛剛訓練出來的最新最佳模型：`E:\MAE_AST\MAE_output_v2\checkpoints\best.pt`。

### 🔊 模式 A：單一檔案預測
在終端機直接印出一個檔案的 Top-5 預測類別與機率：
```powershell
python inference.py --input E:\spectrogram_6s_pt_name\某個音訊.pt
```

### 📁 模式 B：大量批次預測 (整個資料夾)
如果您有一整批檔案需要辨識，可以直接指定資料夾，並加上 `--batch`，它會啟用多執行緒並行預測加速：
```powershell
python inference.py --input E:\spectrogram_6s_pt_name --batch
```

### 📝 模式 C：將大量預測結果輸出成 JSON 報告
若您需要把所有預測結果存下來做後續分析：
```powershell
python inference.py --input E:\spectrogram_6s_pt_name --batch --output_json my_predictions.json
```
執行完畢後，您的 `mae_finetune_v2` 資料夾內就會多出一個 `my_predictions.json` 檔案。

---

## ⚠️ 常見注意事項
1. **Windows DataLoader 效能**：在 Windows 上，`config.py` 中的 `NUM_WORKERS=8` 有時會導致剛開始讀取資料時「稍微卡住」。如果卡超過兩分鐘沒動靜，可以把 `config.py` 的 `NUM_WORKERS` 改成 `0`。
2. **路徑問題**：目前所有腳本（`train.py` / `model.py` 等）都已經具備智慧路徑轉換功能，您可以放心使用任何 `E:\` 或 `C:\` 的本地路徑，不需要再手動改成 Linux 的 `/mnt/e/` 格式。
