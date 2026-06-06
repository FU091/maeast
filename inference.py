"""
inference.py
============
MAE-AST Fine-tune 推理腳本

支援：
  1. 單一 .pt 檔推理
  2. 資料夾 batch 推理（掃描所有 .pt）

輸出：
  - top-k labels + probability
  - JSON 預測結果
  - 可選：輸出 threshold 以上的所有正類別

使用方式：
  # 單一檔案
  python inference.py --input D:/spectrogram_6s_pt_name/xxx.pt

  # 整個資料夾
  python inference.py --input D:/spectrogram_6s_pt_name --batch

  # 指定 checkpoint 與設定
  python inference.py --input D:/spectrogram_6s_pt_name/xxx.pt \\
      --checkpoint checkpoints/best.pt \\
      --label_csv mae_finetune/class_labels.csv \\
      --topk 5

  # 輸出 json
  python inference.py --input D:/spectrogram_6s_pt_name \\
      --batch --output_json predictions.json
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as CFG
from dataset import build_label_map
from model   import MAEASTFineTune

logging.basicConfig(
    level  = logging.INFO,
    format = "[%(asctime)s] %(levelname)s: %(message)s",
    datefmt= "%H:%M:%S",
)
logger = logging.getLogger("inference")


# ============================================================
#  Argument Parser
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="MAE-AST Inference")

    parser.add_argument(
        "--input", required=True,
        help="單一 .pt 路徑 或 資料夾路徑（與 --batch 搭配）"
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="批次處理：掃描 --input 資料夾下所有 .pt 檔"
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help=f"Fine-tune checkpoint 路徑（預設: {CFG.CHECKPOINT_DIR}/best.pt）"
    )
    parser.add_argument(
        "--pretrained_ckpt", default=CFG.PRETRAINED_CKPT,
        help="MAE-AST 預訓練權重路徑"
    )
    parser.add_argument(
        "--mae_ast_root", default=CFG.MAE_AST_PROJECT_ROOT,
        help="MAE-AST 官方專案根目錄"
    )
    parser.add_argument(
        "--label_csv", default=CFG.LABEL_CSV,
        help="class_labels.csv 路徑"
    )
    parser.add_argument(
        "--topk", type=int, default=CFG.TOP_K,
        help="輸出 Top-K 類別（預設 5）"
    )
    parser.add_argument(
        "--threshold", type=float, default=CFG.THRESHOLD,
        help="Sigmoid 閾值，高於此值視為正類別（預設 0.5）"
    )
    parser.add_argument(
        "--thresholds_json", default=None,
        help="JSON file containing per-class optimal thresholds. If provided, overrides --threshold."
    )
    parser.add_argument(
        "--output_json", default=None,
        help="JSON 預測結果輸出路徑（不指定則只印出終端）"
    )
    parser.add_argument(
        "--num_classes", type=int, default=None,
        help="類別數（若 None 則從 label_csv 讀取）"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="批次推理時的 batch size"
    )
    parser.add_argument(
        "--save_every", type=int, default=10000,
        help="每多少筆 flush 一次 JSON，避免記憶體不足（預設 10000）"
    )
    parser.add_argument(
        "--io_workers", type=int, default=8,
        help="並行讀取 .pt 的執行緒數（預設 8，I/O 瓶頸時調高）"
    )

    return parser.parse_args()


# ============================================================
#  .pt 讀取工具
# ============================================================

def load_spectrogram(pt_path: str, target_length: int = 1024) -> torch.Tensor:
    """
    讀取單一 .pt 檔，回傳 float32 spectrogram [1024, 128]。
    """
    data = torch.load(pt_path, map_location="cpu", weights_only=False)

    if isinstance(data, dict):
        if "x" in data:
            fbank = data["x"]
        else:
            # 嘗試第一個 tensor value
            for v in data.values():
                if isinstance(v, torch.Tensor):
                    fbank = v
                    break
            else:
                raise ValueError(f"Cannot find spectrogram tensor in {pt_path}")
    elif isinstance(data, torch.Tensor):
        fbank = data
    else:
        raise ValueError(f"Unsupported .pt format in {pt_path}: {type(data)}")

    # float16 → float32
    fbank = fbank.float()

    # 對齊長度
    n_frames = fbank.shape[0]
    if n_frames > target_length:
        fbank = fbank[:target_length, :]
    elif n_frames < target_length:
        pad   = torch.zeros(target_length - n_frames, fbank.shape[1])
        fbank = torch.cat([fbank, pad], dim=0)

    return fbank  # [1024, 128]


def _load_spectrogram_safe(pt_path: str, target_length: int = 1024):
    """包裝版：回傳 (path, tensor) 或 (path, None) 供並行讀取使用。"""
    try:
        return pt_path, load_spectrogram(pt_path, target_length)
    except Exception as e:
        return pt_path, e


# ============================================================
#  單一樣本推理
# ============================================================

@torch.no_grad()
def infer_single(
    model: MAEASTFineTune,
    pt_path: str,
    label_names: list,
    topk: int,
    threshold: float,
    thresholds_dict: dict,
    device: torch.device,
) -> dict:
    """
    推理單一 .pt 檔。

    Returns
    -------
    dict：
        file      : 檔案路徑
        top_k     : [{label, prob}, ...]（按機率降序）
        positives : 所有高於 threshold 的類別
        all_probs : {label: prob} 完整輸出
    """
    fbank  = load_spectrogram(pt_path)           # [1024, 128]
    fbank  = fbank.unsqueeze(0).to(device)       # [1, 1024, 128]

    logits = model(fbank)                         # [1, num_classes]
    probs  = torch.sigmoid(logits)[0].cpu()      # [num_classes]
    probs_np = probs.numpy()

    # Top-K
    topk_indices = np.argsort(probs_np)[::-1][:topk]
    top_k_result = [
        {"label": label_names[i], "prob": float(probs_np[i])}
        for i in topk_indices
    ]

    # Threshold positives
    positives = []
    for i in range(len(label_names)):
        cls_name = label_names[i]
        thresh = thresholds_dict.get(cls_name, threshold) if thresholds_dict else threshold
        if probs_np[i] >= thresh:
            positives.append({"label": cls_name, "prob": float(probs_np[i])})
    
    positives.sort(key=lambda x: x["prob"], reverse=True)

    # All probs
    all_probs = {
        label_names[i]: float(probs_np[i])
        for i in range(len(label_names))
    }

    return {
        "file"      : os.path.abspath(pt_path),
        "top_k"     : top_k_result,
        "positives" : positives,
        "all_probs" : all_probs,
    }


# ============================================================
#  Batch 推理
# ============================================================

@torch.no_grad()
def infer_batch(
    model: MAEASTFineTune,
    pt_files: list,
    label_names: list,
    topk: int,
    threshold: float,
    thresholds_dict: dict,
    batch_size: int,
    device: torch.device,
    num_io_workers: int = 8,
    output_json: str = None,
    save_every: int = 10000,
) -> list:
    """
    批次推理所有 .pt 檔。
    - 使用 ThreadPoolExecutor 並行讀取 .pt（解決 I/O 瓶頸）
    - 使用 torch.amp.autocast 加速 GPU forward（float16）
    """
    results      = []
    first_10     = []
    use_autocast = (device.type == "cuda")
    model.eval()

    t_io_total      = 0.0
    t_forward_total = 0.0
    n_skipped       = 0
    total_processed = 0

    f_out = None
    is_first_item = True
    if output_json:
        # 開啟 Streaming JSON 寫入模式
        f_out = open(output_json, "w", encoding="utf-8")
        f_out.write('{\n  "predictions": [\n')

    pbar = tqdm(range(0, len(pt_files), batch_size), desc="Batch Inference")
    for batch_start in pbar:
        batch_paths = pt_files[batch_start: batch_start + batch_size]

        # ── 並行 I/O：多執行緒同時讀取本 batch 的 .pt ─────
        t0 = time.perf_counter()
        batch_tensors = [None] * len(batch_paths)
        valid_paths   = []

        with ThreadPoolExecutor(max_workers=min(num_io_workers, len(batch_paths))) as pool:
            future_to_idx = {
                pool.submit(_load_spectrogram_safe, p): i
                for i, p in enumerate(batch_paths)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                path, result = future.result()
                if isinstance(result, Exception):
                    logger.warning(f"Skip {path}: {result}")
                    n_skipped += 1
                else:
                    batch_tensors[idx] = (path, result)

        t_io_total += time.perf_counter() - t0

        # 過濾掉讀取失敗的
        valid_items = [(p, t) for item in batch_tensors
                       if item is not None
                       for p, t in [item]]
        if not valid_items:
            continue
        valid_paths_b   = [p for p, _ in valid_items]
        valid_tensors_b = [t for _, t in valid_items]

        # ── GPU Forward（autocast float16）─────────────────
        t0    = time.perf_counter()
        batch = torch.stack(valid_tensors_b, dim=0).to(device)  # [B, 1024, 128]

        if use_autocast:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(batch)                            # [B, C]
        else:
            logits = model(batch)

        probs = torch.sigmoid(logits).cpu().float().numpy()     # [B, C]
        t_forward_total += time.perf_counter() - t0

        pbar.set_postfix({
            "io": f"{t_io_total:.1f}s",
            "fwd": f"{t_forward_total:.1f}s",
            "skip": n_skipped,
        })

        for i, path in enumerate(valid_paths_b):
            probs_i = probs[i]

            topk_indices = np.argsort(probs_i)[::-1][:topk]
            top_k_result = [
                {"label": label_names[j], "prob": float(probs_i[j])}
                for j in topk_indices
            ]
            positives = []
            for j in range(len(label_names)):
                cls_name = label_names[j]
                thresh = thresholds_dict.get(cls_name, threshold) if thresholds_dict else threshold
                if probs_i[j] >= thresh:
                    positives.append({"label": cls_name, "prob": float(probs_i[j])})
            
            positives.sort(key=lambda x: x["prob"], reverse=True)

            all_probs = {
                label_names[j]: float(probs_i[j])
                for j in range(len(label_names))
            }

            item = {
                "file"     : os.path.abspath(path),
                "top_k"    : top_k_result,
                "positives": positives,
                "all_probs": all_probs,
            }
            
            if len(first_10) < 10:
                first_10.append(item)
                
            if f_out:
                if not is_first_item:
                    f_out.write(",\n")
                json.dump(item, f_out, ensure_ascii=False, indent=2)
                is_first_item = False
                
                total_processed += 1
                if total_processed % save_every == 0:
                    f_out.flush()
                    os.fsync(f_out.fileno())
            else:
                results.append(item)
                total_processed += 1

    if f_out:
        f_out.write('\n  ],\n')
        f_out.write(f'  "num_files": {total_processed}\n')
        f_out.write('}\n')
        f_out.close()
        results = first_10  # 回傳前 10 筆供印出即可

    logger.info(
        f"Inference done | files={total_processed} skipped={n_skipped} "
        f"| I/O={t_io_total:.1f}s  Forward={t_forward_total:.1f}s"
    )
    return results


# ============================================================
#  Print Result
# ============================================================

def print_result(result: dict):
    print(f"\n{'=' * 55}")
    print(f"File: {os.path.basename(result['file'])}")
    print(f"  Top-K predictions:")
    for item in result["top_k"]:
        bar = "█" * int(item["prob"] * 20)
        print(f"    {item['label']:<25} {item['prob']:.3f}  {bar}")
    if result["positives"]:
        print(f"  Positive (≥ threshold):")
        for item in result["positives"]:
            print(f"    ✓ {item['label']:<23} {item['prob']:.3f}")
    else:
        print(f"  (No class above threshold)")
    print(f"{'=' * 55}")


# ============================================================
#  Main
# ============================================================

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── CUDA 效能最佳化 ───────────────────────────────────
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True   # 自動選最快的 conv kernel
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} | "
                    f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

    # ── Label map ─────────────────────────────────────────
    label_map, num_classes = build_label_map(args.label_csv)
    if args.num_classes is not None:
        num_classes = args.num_classes
    label_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
    logger.info(f"num_classes = {num_classes}")

    # ── Load model ────────────────────────────────────────
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = os.path.join(CFG.CHECKPOINT_DIR, "best.pt")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Fine-tune checkpoint not found: {ckpt_path}\n"
            f"請先執行 train.py 訓練，或透過 --checkpoint 指定路徑。"
        )

    model, state = MAEASTFineTune.from_checkpoint(
        ckpt_path       = ckpt_path,
        pretrained_ckpt = args.pretrained_ckpt,
        mae_ast_root    = args.mae_ast_root,
        num_classes     = num_classes,
    )
    model = model.to(device).eval()
    logger.info(f"Model loaded from: {ckpt_path}")
    if "best_map" in state:
        logger.info(f"Best val mAP at training: {state['best_map']:.4f}")

    # Load threshold dict if provided
    thresholds_dict = None
    if args.thresholds_json and os.path.exists(args.thresholds_json):
        with open(args.thresholds_json, "r", encoding="utf-8") as f:
            data = json.load(f)
            # handle format from find_optimal_thresholds.py
            thresholds_dict = {k: v['threshold'] for k, v in data.items()}
        logger.info(f"Loaded per-class thresholds from: {args.thresholds_json}")

    # ── Inference ─────────────────────────────────────────
    input_path = args.input

    if args.batch or os.path.isdir(input_path):
        # 批次推理
        pt_files = sorted(Path(input_path).glob("**/*.pt"))
        pt_files = [str(p) for p in pt_files]
        logger.info(f"Found {len(pt_files)} .pt files in {input_path}")

        if not pt_files:
            logger.error(f"No .pt files found in {input_path}")
            return

        # ── I/O 速度診斷：先測前 3 個檔看单檔讀取時間 ────
        probe = pt_files[:3]
        t0    = time.perf_counter()
        for _p in probe:
            load_spectrogram(_p)
        t_probe = time.perf_counter() - t0
        per_file = t_probe / len(probe)
        logger.info(
            f"I/O 速度診斷：前 {len(probe)} 個檔平均 {per_file*1000:.0f} ms/檔 "
            f"（路徑: {pt_files[0][:60]}...)"
        )
        if per_file > 0.05:
            logger.warning(
                "⚠ I/O 明顯慢！建議檢查 .pt 路徑是否透過 /mnt/ 跊界存取 Windows 磁碟。"
            )

        results = infer_batch(
            model, pt_files, label_names,
            topk          = args.topk,
            threshold     = args.threshold,
            thresholds_dict = thresholds_dict,
            batch_size    = args.batch_size,
            device        = device,
            num_io_workers= args.io_workers,
            output_json   = args.output_json,
            save_every    = args.save_every,
        )
        for r in results[:10]:   # 只印前 10 筆
            print_result(r)
        if len(results) > 10:
            print(f"\n... (showing 10/{len(results)} results)")

    else:
        # 單一檔案推理
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input .pt not found: {input_path}")

        result = infer_single(
            model, input_path, label_names,
            topk      = args.topk,
            threshold = args.threshold,
            thresholds_dict = thresholds_dict,
            device    = device,
        )
        print_result(result)
        results = [result]

    # ── 輸出 JSON ─────────────────────────────────────────
    if args.output_json and not (args.batch or os.path.isdir(input_path)):
        # 如果是單一檔案推理，或者是 batch 但沒啟用 streaming (這裡 batch 已經在裡面存了)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(
                {"predictions": results, "num_files": len(results)},
                f,
                ensure_ascii=False,
                indent=2,
            )
        logger.info(f"Predictions saved → {args.output_json}")
    elif args.output_json and (args.batch or os.path.isdir(input_path)):
        # Batch 模式在 infer_batch 已經寫檔完成
        logger.info(f"Predictions saved incrementally → {args.output_json}")


if __name__ == "__main__":
    main()
