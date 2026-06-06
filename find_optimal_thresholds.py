import os
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve

import config as CFG
from dataset import build_label_map, build_dataloader
from model import MAEASTFineTune
from evaluate_results import to_posix_path

def find_best_thresh_pr(y_true, y_prob, class_names):
    best_thresh = {}
    for i, cls in enumerate(class_names):
        precision, recall, thresholds = precision_recall_curve(
            y_true[:, i], y_prob[:, i]
        )
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_thresh[cls] = {
            'threshold': float(thresholds[best_idx]),
            'f1': float(f1_scores[best_idx]),
            'precision': float(precision[best_idx]),
            'recall': float(recall[best_idx]),
        }
    return best_thresh

def parse_args():
    parser = argparse.ArgumentParser(description="Find optimal thresholds via PR Curve")
    parser.add_argument("--gt_json", default=CFG.TEST_JSON, help="Path to evaluation JSON")
    parser.add_argument("--label_csv", default=CFG.LABEL_CSV, help="Path to class labels")
    parser.add_argument("--checkpoint", default=None, help="Model checkpoint path")
    parser.add_argument("--pretrained_ckpt", default=CFG.PRETRAINED_CKPT)
    parser.add_argument("--mae_ast_root", default=CFG.MAE_AST_PROJECT_ROOT)
    parser.add_argument("--output_json", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "optimal_thresholds.json"))
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()

def main():
    args = parse_args()
    args.gt_json = to_posix_path(args.gt_json)
    args.label_csv = to_posix_path(args.label_csv)
    args.output_json = to_posix_path(args.output_json)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    label_map, num_classes = build_label_map(args.label_csv)
    target_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
    
    ckpt_path = args.checkpoint or to_posix_path(os.path.join(CFG.CHECKPOINT_DIR, "best.pt"))
    print(f"Loading model from {ckpt_path}")
    model, _ = MAEASTFineTune.from_checkpoint(
        ckpt_path=ckpt_path,
        pretrained_ckpt=to_posix_path(args.pretrained_ckpt),
        mae_ast_root=to_posix_path(args.mae_ast_root),
        num_classes=num_classes,
    )
    model = model.to(device).eval()
    
    print(f"Loading dataset from {args.gt_json}")
    loader = build_dataloader(
        json_path=args.gt_json,
        label_map=label_map,
        num_classes=num_classes,
        spectrogram_dir=to_posix_path(CFG.SPECTROGRAM_DIR),
        batch_size=args.batch_size,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=CFG.PIN_MEMORY,
        is_train=False,
    )
    
    y_true_list = []
    y_prob_list = []
    
    use_autocast = (device.type == "cuda")
    print("Running inference on validation/test set to find optimal thresholds...")
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
    
    print("Calculating PR curve thresholds...")
    best_thresh = find_best_thresh_pr(y_true, y_prob, target_names)
    
    print("\nOptimal Thresholds:")
    for cls in target_names:
        res = best_thresh[cls]
        print(f"{cls:20s}: Threshold={res['threshold']:.4f} (F1={res['f1']:.4f}, P={res['precision']:.4f}, R={res['recall']:.4f})")
        
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(best_thresh, f, ensure_ascii=False, indent=4)
    print(f"\nSaved optimal thresholds to: {args.output_json}")

if __name__ == "__main__":
    main()
