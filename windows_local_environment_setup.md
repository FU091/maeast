# MAE-AST Windows 本地完整環境建立指南 🚀

> **最後更新**：2026-06-05
> **作業系統**：Windows 原生環境 (非 WSL)
> **Python**：3.9 (Conda)
> **PyTorch**：2.5.1+cu121

本指南記錄了如何從 WSL 環境轉移到純 Windows 本地端，並成功解決 `fairseq` 與 `omegaconf` 依賴衝突的地雷，順利建立 `mae_v2` 訓練環境的完整步驟。

---

## 1. 建立 Conda 虛擬環境
在 Windows Terminal 或 PowerShell 中執行：
```powershell
conda create -n mae_v2 python=3.9 -y
conda activate mae_v2
```

---

## 2. 安裝核心套件與 PyTorch (GPU)
我們使用整理過的 `requirements_v2.txt`，裡面指定了 PyTorch 2.5.1 以及 CUDA 12.1 的來源：

```powershell
pip install -r requirements_v2.txt
```
> **注意**：如果在這一步遇到 pip 卡住或報錯，可嘗試將 `requirements_v2.txt` 中的 fairseq 相關套件先移除，改為手動安裝（見步驟 4）。

---

## 3. 安裝 fairseq（避開 C++ 編譯坑）
由於 MAE-AST 依賴 `fairseq`，而官方 `fairseq` 在 Windows 上預設需要 C++ 編譯器。我們只使用其 Python 模組（如 LayerNorm、Transformer 等），所以透過專案內的腳本跳過編譯：

```powershell
# 執行專案目錄下的自訂安裝腳本
python install_fairseq_win.py
```
這支腳本會自動下載 `fairseq` 原始碼，修改 `setup.py` 把 `ext_modules` 清空，然後用 `--no-build-isolation` 順利裝上。

---

## 4. 解決 pip 依賴衝突地雷 (omegaconf 與 ruamel.yaml)
**⚠️ 這是 Windows 環境最容易卡關的地方！**

### 💣 踩坑紀錄：
當我們安裝 `fairseq 0.12.2` 後，它會要求舊版的 `omegaconf<2.1`。
若任由 pip 自動解析依賴，pip 會去下載 `omegaconf 2.0.6`，而這個版本依賴了非常古老的 `ruamel.yaml`。
在較新的 Python/pip 環境下，編譯 `ruamel.yaml` 會拋出以下錯誤：
> `NameError: name 'Str' is not defined`
> `ERROR: Failed to build 'ruamel.yaml' when getting requirements to build wheel`

### ✅ 終極解法 (`--no-deps`)：
我們發現 `fairseq` 其實可以跟新版的 `omegaconf` 和 `hydra-core` 完美配合。為了解決編譯報錯，我們必須**強迫 pip 直接安裝新版套件，並且絕對不准它往下檢查任何相依樹**，這必須使用 `--no-deps` 參數：

```powershell
pip install omegaconf==2.3.0 hydra-core==1.3.2 cython antlr4-python3-runtime PyYAML packaging --no-deps
```
這行指令能完美避開舊套件的編譯，並把執行 fairseq 所需的底層依賴全部補齊。

---

## 5. 驗證環境
依序完成後，執行以下指令驗證環境是否健康：

```powershell
python -c "import torch; from fairseq.modules import LayerNorm; print('PyTorch & Fairseq OK!')"
```
只要能印出 `PyTorch & Fairseq OK!`，就代表環境的依賴地雷已經全數掃除了。

接下來只要進入 `mae_finetune_v2` 目錄：
```powershell
cd mae_finetune_v2
python model.py
```
若能成功印出 `Pretrained weights loaded successfully.` 與 `Model OK!`，本地訓練環境就大功告成了，您可以安心開始跑 `train.py`！
