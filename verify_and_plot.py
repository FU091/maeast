import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, recall_score, average_precision_score

plt.rcParams['figure.dpi'] = 150
sns.set_theme(style="whitegrid")

def get_basename(path):
    # Handle both Windows and Linux paths
    path = path.replace('\\', '/')
    return os.path.basename(path)

def main():
    train_json_path = "/mnt/e/MAE_AST/MAE_AST/mae_finetune/json/train.json"
    test_json_path = "/mnt/e/MAE_AST/MAE_AST/mae_finetune/json/test.json"
    pred_json_path = "/home/lin/MAE_output/predictions.json"
    
    print("Loading Data...")
    with open(train_json_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)["data"]
    with open(test_json_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)["data"]
    with open(pred_json_path, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)["predictions"]
        
    # 1. Calculate Train Support
    train_support = {}
    for item in train_data:
        for label in item["labels"]:
            train_support[label] = train_support.get(label, 0) + 1
            
    print("\nTrain Support (Class Counts):")
    for k, v in sorted(train_support.items(), key=lambda x: x[1], reverse=True):
        print(f"  {k}: {v}")
        
    # 2. Match Test Ground Truth with Predictions
    test_gt = {}
    all_classes = set()
    for item in test_data:
        basename = get_basename(item["wav"])
        test_gt[basename] = item["labels"]
        for label in item["labels"]:
            all_classes.add(label)
            
    matched_preds = []
    unmatched = 0
    for item in pred_data:
        basename = get_basename(item["file"])
        if basename in test_gt:
            record = {"file": basename, "true_labels": test_gt[basename]}
            for k in item["top_k"]:
                record[k["label"]] = k["prob"]
            matched_preds.append(record)
        else:
            unmatched += 1
            
    print(f"\nMatched files: {len(matched_preds)} / {len(pred_data)} predictions.")
    print(f"Test files total: {len(test_gt)}")
    
    if len(matched_preds) == 0:
        print("Error: No files matched between predictions and test set!")
        return

    all_classes = sorted(list(all_classes))
    
    # 3. Calculate Recall and PR Curve Data per class
    # Convert to arrays
    y_true_dict = {c: [] for c in all_classes}
    y_scores_dict = {c: [] for c in all_classes}
    
    for row in matched_preds:
        for c in all_classes:
            y_true_dict[c].append(1 if c in row["true_labels"] else 0)
            y_scores_dict[c].append(row.get(c, 0.0))
            
    recalls = {}
    ap_scores = {}
    
    plt.figure(figsize=(10, 8))
    
    for c in all_classes:
        y_true = np.array(y_true_dict[c])
        y_scores = np.array(y_scores_dict[c])
        
        # Binary prediction with threshold 0.5 for point Recall
        y_pred = (y_scores >= 0.5).astype(int)
        
        # Only calculate if class is present in true labels to avoid errors
        if sum(y_true) > 0:
            rec = recall_score(y_true, y_pred, zero_division=0)
            recalls[c] = rec
            
            # PR Curve
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            ap = average_precision_score(y_true, y_scores)
            ap_scores[c] = ap
            
            plt.plot(recall, precision, lw=2, label=f'{c} (AP={ap:.2f})')
        else:
            recalls[c] = 0.0
            
    # PR Curve formatting
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Test Set)')
    plt.legend(loc='lower left', bbox_to_anchor=(1.05, 0), fontsize='small')
    plt.tight_layout()
    pr_path = "/mnt/e/MAE_AST/MAE_AST/pr_curve.png"
    plt.savefig(pr_path)
    plt.close()
    print(f"\nSaved PR Curve to {pr_path}")
    
    # 4. Plot Support vs Recall
    plot_data = []
    for c in all_classes:
        if c in train_support and c in recalls:
            plot_data.append({"Class": c, "Train Support": train_support[c], "Test Recall": recalls[c]})
            
    df_plot = pd.DataFrame(plot_data)
    
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df_plot, x="Train Support", y="Test Recall", s=100, color="blue", alpha=0.7)
    
    # Annotate points
    for i in range(df_plot.shape[0]):
        plt.text(x=df_plot["Train Support"].iloc[i] + (df_plot["Train Support"].max()*0.01), 
                 y=df_plot["Test Recall"].iloc[i], 
                 s=df_plot["Class"].iloc[i], 
                 fontdict=dict(color='black', size=10))
                 
    plt.title("Train Support vs Test Recall (Threshold=0.5)")
    plt.xlabel("Number of Samples in Train Set (Support)")
    plt.ylabel("Recall on Test Set")
    plt.xscale('log') # often support is highly skewed
    plt.tight_layout()
    scatter_path = "/mnt/e/MAE_AST/MAE_AST/support_vs_recall.png"
    plt.savefig(scatter_path)
    plt.close()
    print(f"Saved Support vs Recall Scatter Plot to {scatter_path}")

if __name__ == "__main__":
    main()
