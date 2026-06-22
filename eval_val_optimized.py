import os
import sys
import json
import argparse
import logging
import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
from tqdm import tqdm

# Add parent directory to system path to import config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as CFG
from dataset import build_dataloader, build_label_map
from model import MAEASTFineTune
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_val_optimized")

def compute_class_metrics(targets: np.ndarray, scores: np.ndarray, thresholds: np.ndarray, label_names: list):
    C = targets.shape[1]
    preds = (scores >= thresholds).astype(np.float32)
    
    per_class = {}
    valid_aps = []
    
    for c in range(C):
        name = label_names[c] if c < len(label_names) else f"class_{c}"
        gt = targets[:, c]
        pred = preds[:, c]
        sc = scores[:, c]
        
        n_pos = int(gt.sum())
        if n_pos == 0:
            per_class[name] = dict(ap=float("nan"), f1=float("nan"),
                                   precision=float("nan"), recall=float("nan"),
                                   support=0)
            continue
            
        ap = average_precision_score(gt, sc)
        p = precision_score(gt, pred, zero_division=0)
        r = recall_score(gt, pred, zero_division=0)
        f1 = f1_score(gt, pred, zero_division=0)
        
        per_class[name] = dict(ap=ap, f1=f1, precision=p, recall=r, support=n_pos)
        valid_aps.append(ap)
        
    mAP = float(np.mean(valid_aps)) if valid_aps else 0.0
    
    overall = {
        "mAP": mAP,
        "macro_f1": f1_score(targets, preds, average="macro", zero_division=0),
        "micro_f1": f1_score(targets, preds, average="micro", zero_division=0),
        "macro_precision": precision_score(targets, preds, average="macro", zero_division=0),
        "micro_precision": precision_score(targets, preds, average="micro", zero_division=0),
        "macro_recall": recall_score(targets, preds, average="macro", zero_division=0),
        "micro_recall": recall_score(targets, preds, average="micro", zero_division=0),
    }
    
    return per_class, overall

def main():
    parser = argparse.ArgumentParser(description="Evaluate Val set with V2 best checkpoint and optimized thresholds")
    parser.add_argument("--ckpt", default="/mnt/e/MAE_AST/MAE_output_v2/checkpoints/best.pt", help="Path to checkpoint")
    parser.add_argument("--gt_json", default="/mnt/e/ssast_hub/all_mammal_merged/replaced/all_merged_val.json", help="Path to ground truth JSON")
    parser.add_argument("--opt_json", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "optimal_thresholds_v2.json"), help="Path to optimized thresholds JSON")
    parser.add_argument("--save_csv", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "val_metrics_optimized.csv"), help="Path to save comparison CSV")
    parser.add_argument("--batch_size", type=int, default=CFG.BATCH_SIZE)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load label map
    label_map, num_classes = build_label_map(CFG.LABEL_CSV)
    target_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
    
    # Load model
    logger.info(f"Loading model from {args.ckpt}...")
    model, _ = MAEASTFineTune.from_checkpoint(
        ckpt_path=args.ckpt,
        pretrained_ckpt=CFG.PRETRAINED_CKPT,
        mae_ast_root=CFG.MAE_AST_PROJECT_ROOT,
        num_classes=num_classes,
    )
    model = model.to(device).eval()
    
    # Load dataloader
    logger.info(f"Loading dataset from {args.gt_json}...")
    loader = build_dataloader(
        json_path=args.gt_json,
        label_map=label_map,
        num_classes=num_classes,
        spectrogram_dir=CFG.SPECTROGRAM_DIR,
        batch_size=args.batch_size,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=CFG.PIN_MEMORY and device.type == "cuda",
        is_train=False,
    )
    
    # Run inference
    y_true_list = []
    y_prob_list = []
    
    use_autocast = (device.type == "cuda")
    logger.info("Running inference on validation set...")
    with torch.no_grad():
        for fbank, labels in tqdm(loader, desc="Evaluating"):
            fbank = fbank.to(device)
            if use_autocast:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(fbank)
            else:
                logits = model(fbank)
                
            probs = torch.sigmoid(logits)
            y_prob_list.append(probs.cpu().numpy())
            y_true_list.append(labels.numpy())
            
    y_true = np.concatenate(y_true_list, axis=0)
    y_prob = np.concatenate(y_prob_list, axis=0)
    
    # 1. Baseline Threshold = 0.5 for all classes
    thresholds_05 = np.full(num_classes, 0.5)
    per_class_05, overall_05 = compute_class_metrics(y_true, y_prob, thresholds_05, target_names)
    
    # 2. Optimized Thresholds
    logger.info(f"Loading optimized thresholds from {args.opt_json}...")
    with open(args.opt_json, "r", encoding="utf-8") as f:
        opt_thresh_data = json.load(f)
        
    thresholds_opt = np.zeros(num_classes)
    for i, cls in enumerate(target_names):
        thresholds_opt[i] = opt_thresh_data.get(cls, {}).get("threshold", 0.5)
        
    per_class_opt, overall_opt = compute_class_metrics(y_true, y_prob, thresholds_opt, target_names)
    
    # 3. Print side-by-side comparison table
    SEP = "=" * 96
    print(f"\n{SEP}")
    print(f"  VAL SET EVALUATION COMPARISON: BASELINE (0.50) vs OPTIMIZED THRESHOLDS")
    print(SEP)
    header = f"  {'Class':<15} | {'AP':^7} | {'Baseline (0.50)':^22} | {'Optimized':^29} | {'F1 Diff':^8}"
    print(header)
    sub_header = f"  {'':<15} | {'':^7} | {'P':^6} {'R':^6} {'F1':^6} | {'Thr':^6} {'P':^6} {'R':^6} {'F1':^6} |"
    print(sub_header)
    print("-" * 96)
    
    rows_for_csv = []
    for i, cls in enumerate(target_names):
        m05 = per_class_05[cls]
        mopt = per_class_opt[cls]
        thr_val = thresholds_opt[i]
        
        f1_diff = mopt["f1"] - m05["f1"]
        f1_diff_str = f"+{f1_diff:.4f}" if f1_diff > 0 else f"{f1_diff:.4f}"
        if f1_diff == 0:
            f1_diff_str = " 0.0000"
            
        print(f"  {cls:<15} | {m05['ap']:6.4f} | "
              f"{m05['precision']:5.3f} {m05['recall']:5.3f} {m05['f1']:5.3f} | "
              f"{thr_val:5.3f} {mopt['precision']:5.3f} {mopt['recall']:5.3f} {mopt['f1']:5.3f} | "
              f"{f1_diff_str:>8}")
              
        rows_for_csv.append({
            "Class": cls,
            "AP": m05['ap'],
            "Base_P": m05['precision'],
            "Base_R": m05['recall'],
            "Base_F1": m05['f1'],
            "Opt_Thr": thr_val,
            "Opt_P": mopt['precision'],
            "Opt_R": mopt['recall'],
            "Opt_F1": mopt['f1'],
            "F1_Diff": f1_diff
        })
        
    print("-" * 96)
    # Overall summary comparison
    print(f"  {'OVERALL':<15} | {overall_05['mAP']:6.4f} | "
          f"{overall_05['macro_precision']:5.3f} {overall_05['macro_recall']:5.3f} {overall_05['macro_f1']:5.3f} | "
          f"{'N/A':^6} {overall_opt['macro_precision']:5.3f} {overall_opt['macro_recall']:5.3f} {overall_opt['macro_f1']:5.3f} | "
          f"{overall_opt['macro_f1'] - overall_05['macro_f1']:+7.4f} (Macro)")
    print(f"  {'':<15} | {'':^7} | "
          f"{overall_05['micro_precision']:5.3f} {overall_05['micro_recall']:5.3f} {overall_05['micro_f1']:5.3f} | "
          f"{'N/A':^6} {overall_opt['micro_precision']:5.3f} {overall_opt['micro_recall']:5.3f} {overall_opt['micro_f1']:5.3f} | "
          f"{overall_opt['micro_f1'] - overall_05['micro_f1']:+7.4f} (Micro)")
    print(SEP)
    
    # Save CSV
    df = pd.DataFrame(rows_for_csv)
    df.to_csv(args.save_csv, index=False, encoding="utf-8-sig")
    logger.info(f"Saved evaluation comparison table to {args.save_csv}")

if __name__ == "__main__":
    main()
