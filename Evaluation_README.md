# MAE-AST 後處理與資料視覺化工具 (Post-processing & Visualization Tools)

本文件紀錄了目前環境中可用的資料處理與圖表繪製腳本，詳細說明它們的資料來源、產出結果，以及如何調用。

## 1. 關聯性分析工具 (Pearson Correlation)

這個工具用來分析模型預測的各類別機率之間的相關性（哪些聲音容易一起出現，哪些互斥）。

- **腳本路徑**: `./generate_pearson_graph.py`
- **輸入資料來源 (Input)**: 
  - `predictions.json` (模型推論產出的結果檔，路徑: `/home/lin/MAE_output/predictions.json`)
- **產出圖表 (Output)**:
  - `pearson_heatmap_v2.png`: Pearson 相關係數熱力圖。
  - `pearson_network_graph_v2.png`: Pearson 節點關係圖（NetworkX 繪製），紅線為正相關，藍線為負相關。
- **執行方式 (WSL)**:
  ```bash
  conda activate mae_ast
  python generate_pearson_graph.py
  ```

---

## 2. 模型效能評估工具 (Support vs Recall & PR Curve)

這個工具會將**測試集的真實標籤 (Ground Truth)** 與**模型的預測結果**進行配對 (Match)，藉此繪製出精確度 (Precision) 與召回率 (Recall) 的進階評估圖表。

- **腳本路徑**: `./verify_and_plot.py`
- **輸入資料來源 (Input)**:
  1. `train.json` (訓練集標籤，用來計算各類別的 Support/樣本數，路徑: `./json/train.json`)
  2. `test.json` (測試集標籤，作為 Ground Truth 與預測結果對答案，路徑: `./json/test.json`)
  3. `predictions.json` (推論結果，超過 180 萬筆推論，程式會透過檔名自動找出對應測試集的 607 筆資料，路徑: `/home/lin/MAE_output/predictions.json`)
- **產出圖表 (Output)**:
  - `support_vs_recall_v2.png`: 散點圖。X軸為該類別在 `train.json` 的總數 (Support)，Y軸為該類別在 `test.json` 的召回率 (Threshold=0.5)。
  - `pr_curve.png`: Precision-Recall 曲線，展示各類別在不同信心門檻下的精確率與召回率變化，圖例中包含 AP (Average Precision) 分數。
- **執行方式 (WSL)**:
  ```bash
  conda activate mae_ast
  python verify_and_plot.py
  ```

---

## 3. V2 模型評估與 Pearson 相關係數繪圖工具

這個工具直接載入 V2 最佳權重 (`best.pt`) 對驗證集 (`val.json`) 進行推論，並直接讀取 `val_metrics_best.csv` 繪製 Support vs Recall 及相關性圖表。

- **腳本路徑**: `./generate_v2_plots.py`
- **輸入資料來源 (Input)**:
  1. `val.json` (驗證集標籤，路徑: `./json/val.json`)
  2. `best.pt` (V2 最佳權重)
  3. `val_metrics_best.csv` (V2 驗證指標數據)
- **產出圖表 (Output)**:
  - `support_vs_recall_v2.png`: X軸為該類別在 `train.json` 的樣本數 (Support)，Y軸為驗證集召回率 (Recall)。
  - `pearson_heatmap_v2.png`: 驗證集預測機率的 Pearson 相關係數熱力圖。
  - `pearson_network_graph_v2.png`: 驗證集預測類別的共現/互斥關係網絡圖。
- **執行方式 (WSL)**:
  ```bash
  conda activate mae_ast
  python generate_v2_plots.py
  ```

---

## 備註 (Notes)
- 繪圖時如果遇到中文字型 (例如：`MammalLow_山羌`) 無法正常顯示的問題，這是因為 Linux (WSL) 環境預設缺少支援中文的字型 (如 DejaVu Sans 不包含 CJK 字元)。這不會影響圖表的數值與結構，僅會讓類別標籤顯示為方塊。若需要完美顯示，需在 WSL 中安裝中文字型 (`sudo apt-get install fonts-noto-cjk`)，並於 matplotlib 設定中指定字型。
- 腳本的輸出圖表預設儲存在專案根目錄下。

