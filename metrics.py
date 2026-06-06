"""
metrics.py
==========
多標籤分類評估指標
- per-class Average Precision (AP)
- overall mean AP (mAP)
- 基於 sklearn.metrics.average_precision_score
"""

import logging
import numpy as np
from sklearn.metrics import average_precision_score

logger = logging.getLogger(__name__)


def compute_map(
    targets: np.ndarray,
    scores: np.ndarray,
    label_names: list = None,
    verbose: bool = False,
) -> tuple[float, dict]:
    """
    計算多標籤分類的 mAP 與 per-class AP。

    Parameters
    ----------
    targets     : np.ndarray [N, C]  0/1 ground truth（二值化後）
    scores      : np.ndarray [N, C]  sigmoid 後的預測機率
    label_names : list of str，長度 C（若 None 則用 class_0, class_1, ...）
    verbose     : True → 印出 per-class AP

    Returns
    -------
    mAP        : float，overall mean AP
    per_class  : dict {label_name: AP}
    """
    from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score

    assert targets.ndim == 2 and scores.ndim == 2, \
        "targets 和 scores 必須是 2D array [N, C]"
    assert targets.shape == scores.shape, \
        f"Shape mismatch: targets {targets.shape} vs scores {scores.shape}"

    num_classes = targets.shape[1]

    if label_names is None:
        label_names = [f"class_{i}" for i in range(num_classes)]

    # 針對 Precision, Recall, F1 需要給定預測的 0/1 (用 0.5 做 threshold)
    preds_binary = (scores > 0.5).astype(np.float32)

    per_class_ap = {}
    per_class_metrics = {}
    valid_aps    = []

    for c in range(num_classes):
        gt_c       = targets[:, c]
        pred_c     = scores[:, c]
        pred_bin_c = preds_binary[:, c]
        name       = label_names[c] if c < len(label_names) else f"class_{c}"

        # 若該類別在 ground truth 中完全沒有正樣本，AP 無意義
        if gt_c.sum() == 0:
            per_class_ap[name] = float("nan")
            per_class_metrics[name] = {"precision": float("nan"), "recall": float("nan"), "f1": float("nan")}
            continue

        ap = average_precision_score(gt_c, pred_c)
        per_class_ap[name] = ap
        valid_aps.append(ap)
        
        # Calculate per-class precision, recall, f1
        p = precision_score(gt_c, pred_bin_c, zero_division=0)
        r = recall_score(gt_c, pred_bin_c, zero_division=0)
        f1 = f1_score(gt_c, pred_bin_c, zero_division=0)
        per_class_metrics[name] = {"precision": p, "recall": r, "f1": f1}

    mAP = float(np.mean(valid_aps)) if valid_aps else 0.0

    # 計算 micro/macro metrics
    metrics_dict = {
        "mAP": mAP,
        "micro_precision": precision_score(targets, preds_binary, average="micro", zero_division=0),
        "macro_precision": precision_score(targets, preds_binary, average="macro", zero_division=0),
        "micro_recall": recall_score(targets, preds_binary, average="micro", zero_division=0),
        "macro_recall": recall_score(targets, preds_binary, average="macro", zero_division=0),
        "micro_f1": f1_score(targets, preds_binary, average="micro", zero_division=0),
        "macro_f1": f1_score(targets, preds_binary, average="macro", zero_division=0)
    }

    if verbose:
        logger.info(f"[Val Metrics] mAP={mAP:.4f}  Macro-F1={metrics_dict['macro_f1']:.4f}  Micro-F1={metrics_dict['micro_f1']:.4f}  Precision={metrics_dict['macro_precision']:.4f}  Recall={metrics_dict['macro_recall']:.4f}")
        logger.info(f"{'Class':<28} {'AP':>8} {'F1':>8} {'Precision':>10} {'Recall':>10}")
        logger.info(f"{'-' * 68}")
        for name, ap in per_class_ap.items():
            if np.isnan(ap):
                logger.info(f"  {name:<26} {'N/A':>8} {'N/A':>8} {'N/A':>10} {'N/A':>10} (no positive samples)")
            else:
                p  = per_class_metrics[name]['precision']
                r  = per_class_metrics[name]['recall']
                f1 = per_class_metrics[name]['f1']
                logger.info(f"  {name:<26} {ap:8.4f} {f1:8.4f} {p:10.4f} {r:10.4f}")
        logger.info(f"{'-' * 68}")

    return metrics_dict, per_class_ap


def compute_metrics_from_tensors(
    all_targets,   # list of Tensor or ndarray, each [C]
    all_scores,    # list of Tensor or ndarray, each [C]
    label_names: list = None,
    verbose: bool = False,
) -> tuple[float, dict]:
    """
    從 list of tensors 計算 mAP（訓練迴圈收集的格式）。

    Parameters
    ----------
    all_targets : list of [C] tensors（二值化，0/1）
    all_scores  : list of [C] tensors（sigmoid 機率）

    Returns
    -------
    mAP, per_class_ap (dict)
    """
    import torch

    def to_np(x):
        if isinstance(x, (list, tuple)):
            x = np.stack([to_np(i) for i in x])
        elif hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        elif not isinstance(x, np.ndarray):
            x = np.array(x)
        return x

    targets_np = to_np(all_targets)   # [N, C]
    scores_np  = to_np(all_scores)    # [N, C]

    # 確保是 2D
    if targets_np.ndim == 1:
        targets_np = targets_np[np.newaxis, :]
    if scores_np.ndim == 1:
        scores_np = scores_np[np.newaxis, :]

    # 二值化 ground truth（訓練模式下可能是 soft label）
    binary_targets = (targets_np > 0).astype(np.float32)

    return compute_map(binary_targets, scores_np, label_names=label_names, verbose=verbose)


def compute_confusion_stats(targets: np.ndarray, scores: np.ndarray, label_names: list, save_dir: str, epoch: int):
    import os
    import pandas as pd
    
    os.makedirs(save_dir, exist_ok=True)
    preds_binary = (scores > 0.5).astype(int)
    num_classes = targets.shape[1]
    
    # 1. Per-class Stats
    records = []
    for c in range(num_classes):
        gt = targets[:, c]
        pred = preds_binary[:, c]
        
        tp = int(np.sum((gt == 1) & (pred == 1)))
        fp = int(np.sum((gt == 0) & (pred == 1)))
        fn = int(np.sum((gt == 1) & (pred == 0)))
        tn = int(np.sum((gt == 0) & (pred == 0)))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        records.append({
            "Class": label_names[c],
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Precision": precision, "Recall": recall, "F1": f1
        })
    df_per_class = pd.DataFrame(records)
    df_per_class.to_csv(os.path.join(save_dir, f"epoch_{epoch}_per_class_stats.csv"), index=False, encoding="utf-8-sig")

    # 2. Co-occurrence Heatmap (GT有此類時，模型預測了什麼)
    co_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for gt_c in range(num_classes):
        # 找出 Ground Truth 包含 gt_c 的樣本 index
        idx_with_gt_c = np.where(targets[:, gt_c] == 1)[0]
        if len(idx_with_gt_c) == 0:
            continue
        # 這些樣本中，模型預測出其他類別(包含自己)的次數
        preds_for_these = preds_binary[idx_with_gt_c] # shape: (K, num_classes)
        co_matrix[gt_c] = preds_for_these.sum(axis=0)

    # 轉成 DataFrame 並存檔
    df_co = pd.DataFrame(co_matrix, index=[f"GT_{n}" for n in label_names], columns=[f"Pred_{n}" for n in label_names])
    df_co.to_csv(os.path.join(save_dir, f"epoch_{epoch}_co_occurrence.csv"), encoding="utf-8-sig")




# ============================================================
#  Quick test
# ============================================================
if __name__ == "__main__":
    np.random.seed(42)
    N, C = 100, 12

    targets = (np.random.rand(N, C) > 0.7).astype(float)
    scores  = np.clip(targets + np.random.randn(N, C) * 0.3, 0, 1)

    label_names = [
        "Cicada", "Insect", "Bird", "Owl", "Frog",
        "Rain", "Wind", "Stream", "Aircraft", "Machine", "Speech", "Bat"
    ]

    mAP, per_class = compute_map(targets, scores, label_names=label_names, verbose=True)
    print(f"mAP = {mAP:.4f}")
