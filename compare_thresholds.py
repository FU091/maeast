import json
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from evaluate_test_metrics import build_label_map, build_dataloader, MAEASTFineTune
import torch
from tqdm import tqdm
import config as CFG

label_map, num_classes = build_label_map(CFG.LABEL_CSV)
target_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]

model, _ = MAEASTFineTune.from_checkpoint(
    ckpt_path=CFG.CHECKPOINT_DIR + '/best.pt',
    pretrained_ckpt=CFG.PRETRAINED_CKPT,
    mae_ast_root=CFG.MAE_AST_PROJECT_ROOT,
    num_classes=num_classes,
)
model = model.cuda().eval()

loader = build_dataloader(
    json_path=CFG.TEST_JSON,
    label_map=label_map,
    num_classes=num_classes,
    spectrogram_dir=CFG.SPECTROGRAM_DIR,
    batch_size=64,
    num_workers=8,
    pin_memory=True,
    is_train=False,
)

y_true_list, y_prob_list = [], []
with torch.no_grad():
    for fbank, labels in tqdm(loader):
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            logits = model(fbank.cuda())
        probs = torch.sigmoid(logits)
        y_prob_list.append(probs.cpu().numpy())
        y_true_list.append(labels.numpy())

y_true = np.concatenate(y_true_list, axis=0)
y_prob = np.concatenate(y_prob_list, axis=0)

y_pred_05 = (y_prob >= 0.5).astype(int)

with open('optimal_thresholds.json') as f:
    thresh_data = json.load(f)
thresholds = np.array([thresh_data.get(cls, {}).get('threshold', 0.5) for cls in target_names])
y_pred_opt = (y_prob >= thresholds).astype(int)

print(f'\n{"Class":<15} | {"F1 (Thr=0.5)":<15} | {"F1 (Optimized)":<15} | {"Diff":<10}')
print('-'*60)
for i, cls in enumerate(target_names):
    f1_05 = f1_score(y_true[:, i], y_pred_05[:, i], zero_division=0)
    f1_opt = f1_score(y_true[:, i], y_pred_opt[:, i], zero_division=0)
    diff = f1_opt - f1_05
    diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
    print(f'{cls:<15} | {f1_05:.4f}          | {f1_opt:.4f}          | {diff_str}')
