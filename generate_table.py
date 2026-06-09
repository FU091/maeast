import json
import os
import pandas as pd
from sklearn.metrics import recall_score
import numpy as np

def get_basename(path):
    return os.path.basename(path.replace('\\', '/'))

train_json_path = '/mnt/e/MAE_AST/MAE_AST/mae_finetune/json/train.json'
test_json_path = '/mnt/e/MAE_AST/MAE_AST/mae_finetune/json/test.json'
pred_json_path = '/home/lin/MAE_output/predictions.json'

with open(train_json_path, 'r', encoding='utf-8') as f:
    train_data = json.load(f)['data']
with open(test_json_path, 'r', encoding='utf-8') as f:
    test_data = json.load(f)['data']
with open(pred_json_path, 'r', encoding='utf-8') as f:
    pred_data = json.load(f)['predictions']

train_support = {}
for item in train_data:
    for label in item['labels']:
        train_support[label] = train_support.get(label, 0) + 1

test_gt = {}
all_classes = set()
for item in test_data:
    basename = get_basename(item['wav'])
    test_gt[basename] = item['labels']
    for label in item['labels']:
        all_classes.add(label)

matched_preds = []
for item in pred_data:
    basename = get_basename(item['file'])
    if basename in test_gt:
        record = {'file': basename, 'true_labels': test_gt[basename]}
        for k in item['top_k']:
            record[k['label']] = k['prob']
        matched_preds.append(record)

all_classes = sorted(list(all_classes))

y_true_dict = {c: [] for c in all_classes}
y_scores_dict = {c: [] for c in all_classes}

for row in matched_preds:
    for c in all_classes:
        y_true_dict[c].append(1 if c in row['true_labels'] else 0)
        y_scores_dict[c].append(row.get(c, 0.0))

results = []
for c in all_classes:
    y_true = np.array(y_true_dict[c])
    y_scores = np.array(y_scores_dict[c])
    y_pred = (y_scores >= 0.5).astype(int)
    
    if sum(y_true) > 0:
        rec = recall_score(y_true, y_pred, zero_division=0)
        support = train_support.get(c, 0)
        results.append({'Class': c, 'Train_Support': support, 'Test_Recall': round(rec, 4)})

df = pd.DataFrame(results).sort_values(by='Train_Support', ascending=False)
df.to_csv('/mnt/e/MAE_AST/MAE_AST/support_recall_table.csv', index=False)

print('| 類別 (Class) | 訓練集樣本數 (Train Support) | 測試集召回率 (Test Recall) |')
print('| --- | --- | --- |')
for _, row in df.iterrows():
    print(f"| {row['Class']} | {row['Train_Support']} | {row['Test_Recall']:.2%} |")
