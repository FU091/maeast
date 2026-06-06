import os
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score

import config as CFG
from dataset import build_label_map, build_dataloader
from model import MAEASTFineTune
from evaluate_results import to_posix_path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_json", default=CFG.TEST_JSON, help="Path to ground truth JSON")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    label_csv = to_posix_path(CFG.LABEL_CSV)
    gt_json = to_posix_path(args.gt_json)
    thresh_json = to_posix_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "optimal_thresholds.json"))
    
    # 1. Load label map
    label_map, num_classes = build_label_map(label_csv)
    target_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
    
    # 2. Load thresholds
    with open(thresh_json, "r", encoding="utf-8") as f:
        thresh_data = json.load(f)
    thresholds = np.array([thresh_data.get(cls, {}).get("threshold", 0.5) for cls in target_names])
    
    # 3. Load model and dataset
    ckpt_path = to_posix_path(os.path.join(CFG.CHECKPOINT_DIR, "best.pt"))
    print(f"Loading model from {ckpt_path}")
    model, _ = MAEASTFineTune.from_checkpoint(
        ckpt_path=ckpt_path,
        pretrained_ckpt=to_posix_path(CFG.PRETRAINED_CKPT),
        mae_ast_root=to_posix_path(CFG.MAE_AST_PROJECT_ROOT),
        num_classes=num_classes,
    )
    model = model.to(device).eval()
    
    print(f"Loading test dataset from {gt_json}")
    loader = build_dataloader(
        json_path=gt_json,
        label_map=label_map,
        num_classes=num_classes,
        spectrogram_dir=to_posix_path(CFG.SPECTROGRAM_DIR),
        batch_size=32,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=CFG.PIN_MEMORY,
        is_train=False,
    )
    
    y_true_list = []
    y_prob_list = []
    
    use_autocast = (device.type == "cuda")
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
    
    # Apply thresholds
    y_pred = (y_prob >= thresholds).astype(int)
    
    print("\n" + "="*50)
    print(" Performance Metrics on Test Set (with PR Thresholds)")
    print("="*50)
    print(f"{'Class':<15} | {'mAP (AP)':<8} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8}")
    print("-"*55)
    
    ap_scores = []
    precisions = []
    recalls = []
    f1s = []
    
    for i, cls in enumerate(target_names):
        # Average Precision uses probabilities, independent of threshold
        # But we still list it for reference
        if np.sum(y_true[:, i]) > 0:
            ap = average_precision_score(y_true[:, i], y_prob[:, i])
        else:
            ap = 0.0
            
        p = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
        r = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
        f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        
        ap_scores.append(ap)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        
        print(f"{cls:<15} | {ap:.4f}   | {p:.4f}    | {r:.4f}   | {f1:.4f}")
        
    print("-"*55)
    print(f"{'Macro Average':<15} | {np.mean(ap_scores):.4f}   | {np.mean(precisions):.4f}    | {np.mean(recalls):.4f}   | {np.mean(f1s):.4f}")
    print("="*50)
    
if __name__ == "__main__":
    main()
