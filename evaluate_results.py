import os
import json
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report

import config as CFG
from dataset import build_label_map

def to_posix_path(path: str) -> str:
    """
    如果是 POSIX (WSL) 環境，且路徑為 Windows UNC 格式或 Windows 磁碟路徑，
    自動轉換為 WSL 本地路徑。
    """
    if os.name == 'posix' and path:
        # 1. 處理 WSL UNC 路徑: \\wsl.localhost\Ubuntu\home\lin\... -> /home/lin/...
        if path.startswith(r"\\wsl.localhost\Ubuntu"):
            path = path.replace(r"\\wsl.localhost\Ubuntu", "").replace("\\", "/")
            return path
        # 2. 處理一般 Windows 磁碟路徑: E:\path -> /mnt/e/path
        import re
        match = re.match(r'^([a-zA-Z]):\\(.*)', path)
        if match:
            drive = match.group(1).lower()
            rest = match.group(2).replace('\\', '/')
            return f"/mnt/{drive}/{rest}"
        # 3. 處理其他反斜線路徑
        return path.replace('\\', '/')
    return path

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MAE-AST predictions against Ground Truth")
    parser.add_argument("--pred_json", 
                        default=r"\\wsl.localhost\Ubuntu\home\lin\MAE_output\predictions.json",
                        help="Path to the predictions JSON file")
    # 預設使用 config.py 裡的 TEST_JSON，你可以視情況換成 VAL_JSON
    parser.add_argument("--gt_json", 
                        default=CFG.TEST_JSON,
                        help="Path to the Ground Truth JSON file")
    parser.add_argument("--label_csv", 
                        default=CFG.LABEL_CSV,
                        help="Path to the class labels CSV")
    parser.add_argument("--output_dir", 
                        default=os.path.dirname(os.path.abspath(__file__)),
                        help="Directory to save the heatmap and CSV report")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 自動將路徑轉換為適配當前系統 (WSL / Windows) 的格式
    args.pred_json = to_posix_path(args.pred_json)
    args.gt_json = to_posix_path(args.gt_json)
    args.label_csv = to_posix_path(args.label_csv)
    args.output_dir = to_posix_path(args.output_dir)
    
    # 建立輸出資料夾
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading Label Map from: {args.label_csv}")
    label_map, num_classes = build_label_map(args.label_csv)
    target_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
    
    # ==========================================
    # 1. 解析 Ground Truth JSON
    # ==========================================
    print(f"Loading Ground Truth from: {args.gt_json}")
    with open(args.gt_json, "r", encoding="utf-8") as f:
        gt_data = json.load(f)["data"]
        
    gt_dict = {}
    for item in gt_data:
        # 只取檔名進行比對，忽略前面的路徑 (解決不同系統路徑不一致問題)
        basename = os.path.basename(item["wav"].replace("\\", "/"))
        labels = item.get("labels", [])
        if isinstance(labels, str):
            labels = [labels]
        # 過濾不在 label_map 內的標籤
        valid_labels = [l for l in labels if l in label_map]
        gt_dict[basename] = set(valid_labels)
        
    # ==========================================
    # 2. 解析 Predictions JSON
    # ==========================================
    print(f"Loading Predictions from: {args.pred_json}")
    with open(args.pred_json, "r", encoding="utf-8") as f:
        pred_data = json.load(f)["predictions"]
        
    pred_dict = {}
    for item in pred_data:
        basename = os.path.basename(item["file"].replace("\\", "/"))
        positives = item.get("positives", [])
        pred_labels = [p["label"] for p in positives if p["label"] in label_map]
        pred_dict[basename] = set(pred_labels)
        
    # 找出共同擁有的檔案
    common_files = sorted(list(set(gt_dict.keys()).intersection(set(pred_dict.keys()))))
    print(f"Found {len(common_files)} files in both GT and Predictions.")
    
    if len(common_files) == 0:
        print("Error: No matching files found between GT and Predictions. Please check JSON paths.")
        return

    # 初始化陣列
    y_true = np.zeros((len(common_files), num_classes), dtype=int)
    y_pred = np.zeros((len(common_files), num_classes), dtype=int)
    
    error_list = []
    # 用來記錄共現次數的矩陣 N x N
    co_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    # 填補數據
    for i, fname in enumerate(common_files):
        gt_labels = gt_dict[fname]
        pr_labels = pred_dict[fname]
        
        # 轉換為 Multi-hot 編碼
        for gl in gt_labels:
            y_true[i, label_map[gl]] = 1
        for pl in pr_labels:
            y_pred[i, label_map[pl]] = 1
            
        # 方案 C：錯誤樣本明細 (Ground Truth 與預測有差異時)
        if gt_labels != pr_labels:
            error_list.append({
                "File": fname,
                "Ground Truth": ", ".join(sorted(list(gt_labels))),
                "Prediction": ", ".join(sorted(list(pr_labels))),
                "Missed (FN)": ", ".join(sorted(list(gt_labels - pr_labels))),
                "False Alarm (FP)": ", ".join(sorted(list(pr_labels - gt_labels)))
            })
            
        # 方案 A 的共現熱力圖邏輯:
        # 如果該音檔 Ground Truth 含有 gl，且被模型預測出 pl
        for gl in gt_labels:
            for pl in pr_labels:
                co_matrix[label_map[gl], label_map[pl]] += 1
                
    # ==========================================
    # 方案 B: Per-class 二值混淆矩陣表格 & 指標
    # ==========================================
    print("\n" + "="*40)
    print(" 方案 B: Per-class 混淆矩陣 & 評估報告")
    print("="*40)
    
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    for i, class_name in enumerate(target_names):
        tn, fp, fn, tp = mcm[i].ravel()
        print(f"Class: {class_name:12s} | TP:{tp:<4d} FP:{fp:<4d} FN:{fn:<4d} TN:{tn:<4d}")
        
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
    
    # ==========================================
    # 方案 A: 共現熱力圖 (Co-occurrence Heatmap)
    # ==========================================
    print("\n" + "="*40)
    print(" 方案 A: 產生共現熱力圖 (Co-occurrence Heatmap)")
    print("="*40)
    
    df_cm = pd.DataFrame(co_matrix, index=target_names, columns=target_names)
    plt.figure(figsize=(10, 8))
    # annot=True 顯示數字, cmap="YlOrRd" 使用黃橘紅漸層
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="YlOrRd")
    plt.title("Co-occurrence / Confusion Heatmap (Ground Truth vs Predicted)")
    plt.xlabel("Predicted Class")
    plt.ylabel("Ground Truth Class")
    plt.tight_layout()
    
    heatmap_path = os.path.join(args.output_dir, "confusion_heatmap.png")
    plt.savefig(heatmap_path, dpi=300)
    print(f"-> 已儲存熱力圖至: {heatmap_path}")
    
    # ==========================================
    # 方案 C: 錯誤樣本明細 (Error Analysis)
    # ==========================================
    print("\n" + "="*40)
    print(" 方案 C: 匯出錯誤樣本明細 (Error Analysis CSV)")
    print("="*40)
    
    if error_list:
        error_df = pd.DataFrame(error_list)
        csv_path = os.path.join(args.output_dir, "error_analysis_report.csv")
        # 存成 utf-8-sig 以便 Excel 開啟不亂碼
        error_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"-> 發現 {len(error_list)} 個錯誤樣本，已匯出至: {csv_path}")
    else:
        print("-> 太棒了！所有預測皆與 Ground Truth 完全吻合，沒有任何錯誤樣本。")

if __name__ == "__main__":
    main()
