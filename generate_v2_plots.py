import json
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import torch

plt.rcParams['figure.dpi'] = 150
sns.set_theme(style="whitegrid")

def get_basename(path):
    return os.path.basename(path.replace('\\', '/'))

def plot_support_vs_recall():
    print("Generating Support vs Recall Plot...")
    metrics_csv = '/mnt/e/MAE_AST/MAE_output_v2/val_metrics_best.csv'
    train_json = '/mnt/e/MAE_AST/MAE_AST/mae_finetune/json/train.json'
    
    # 1. Read Train Support
    with open(train_json, 'r', encoding='utf-8') as f:
        train_data = json.load(f)['data']
        
    train_support = {}
    for item in train_data:
        for label in item['labels']:
            train_support[label] = train_support.get(label, 0) + 1
            
    # 2. Read Val Metrics
    df_metrics = pd.read_csv(metrics_csv)
    
    plot_data = []
    for _, row in df_metrics.iterrows():
        cls = row['class']
        if cls == 'OVERALL':
            continue
        recall = float(row['Recall'])
        support = train_support.get(cls, 0)
        if support > 0:
            plot_data.append({"Class": cls, "Train Support": support, "Val Recall": recall})
            
    df_plot = pd.DataFrame(plot_data)
    
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df_plot, x="Train Support", y="Val Recall", s=100, color="blue", alpha=0.7)
    
    for i in range(df_plot.shape[0]):
        plt.text(x=df_plot["Train Support"].iloc[i] + (df_plot["Train Support"].max()*0.01), 
                 y=df_plot["Val Recall"].iloc[i], 
                 s=df_plot["Class"].iloc[i], 
                 fontdict=dict(color='black', size=10))
                 
    plt.title("Train Support vs Validation Recall (MAE_output_v2)")
    plt.xlabel("Number of Samples in Train Set (Support)")
    plt.ylabel("Recall on Validation Set")
    plt.xscale('log')
    plt.tight_layout()
    
    scatter_path = "/mnt/e/MAE_AST/MAE_AST/support_vs_recall_v2.png"
    plt.savefig(scatter_path)
    plt.close()
    print(f"Saved Support vs Recall Scatter Plot to {scatter_path}")


def plot_pearson_correlation():
    print("Generating Pearson Correlation plots...")
    sys.path.insert(0, '/mnt/e/MAE_AST/MAE_AST/mae_finetune')
    from model import MAEASTFineTune
    
    # Run inference on val.json to get probabilities
    val_json = '/mnt/e/MAE_AST/MAE_AST/mae_finetune/json/val.json'
    ckpt_path = '/mnt/e/MAE_AST/MAE_output_v2/checkpoints/best.pt'
    pretrained_ckpt = '/mnt/e/MAE_AST/MAE_AST/chunk_patch_75_12LayerEncoder.pt'
    mae_ast_root = '/mnt/e/MAE_AST/MAE_AST/MAE-AST-Public-main'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get classes from train_json
    with open(val_json, 'r', encoding='utf-8') as f:
        val_data = json.load(f)['data']
        
    all_classes = set()
    for item in val_data:
        all_classes.update(item['labels'])
    label_names = sorted(list(all_classes))
    num_classes = len(label_names)
    
    # For inference we need exact label map used in training. Let's read class_labels.csv
    label_csv = '/mnt/e/MAE_AST/MAE_AST/mae_finetune/class_labels.csv'
    df_labels = pd.read_csv(label_csv)
    label_map = dict(zip(df_labels['display_name'], df_labels['index']))
    num_classes = len(label_map)
    label_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
    
    print(f"Loading model {ckpt_path}...")
    model, _ = MAEASTFineTune.from_checkpoint(
        ckpt_path=ckpt_path,
        pretrained_ckpt=pretrained_ckpt,
        mae_ast_root=mae_ast_root,
        num_classes=num_classes
    )
    model = model.to(device).eval()
    
    def load_spec(pt_path):
        # Convert D:\\ to /mnt/d/
        pt_path = pt_path.replace('\\', '/')
        if pt_path.startswith('D:/'):
            pt_path = pt_path.replace('D:/', '/mnt/d/', 1)
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            if "x" in data:
                fbank = data["x"]
            else:
                for v in data.values():
                    if isinstance(v, torch.Tensor):
                        fbank = v
                        break
        else:
            fbank = data
        fbank = fbank.float()
        n_frames = fbank.shape[0]
        if n_frames > 1024:
            fbank = fbank[:1024, :]
        elif n_frames < 1024:
            pad = torch.zeros(1024 - n_frames, fbank.shape[1])
            fbank = torch.cat([fbank, pad], dim=0)
        return fbank

    records = []
    print(f"Running inference on {len(val_data)} validation files...")
    
    with torch.no_grad():
        for item in val_data:
            try:
                fbank = load_spec(item['wav'])
            except Exception as e:
                continue
                
            fbank = fbank.unsqueeze(0).to(device)
            logits = model(fbank)
            probs = torch.sigmoid(logits)[0].cpu().numpy()
            
            record = {"file": item['wav']}
            for i, name in enumerate(label_names):
                record[name] = float(probs[i])
            records.append(record)
            
    df = pd.DataFrame(records)
    if 'file' in df.columns:
        df.set_index("file", inplace=True)
    df.fillna(0, inplace=True)
    
    corr_matrix = df.corr(method='pearson')
    
    # 1. Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", vmin=-1, vmax=1, 
                square=True, linewidths=.5, cbar_kws={"shrink": .75})
    plt.title("Pearson Correlation Matrix (Validation Probabilities - v2)", fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    heatmap_path = "/mnt/e/MAE_AST/MAE_AST/pearson_heatmap_v2.png"
    plt.savefig(heatmap_path)
    plt.close()
    
    # 2. Network Graph
    G = nx.Graph()
    labels = corr_matrix.columns
    for label in labels:
        G.add_node(label)
        
    threshold = 0.1
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            weight = corr_matrix.iloc[i, j]
            if abs(weight) >= threshold:
                G.add_edge(labels[i], labels[j], weight=weight)
                
    plt.figure(figsize=(14, 12))
    pos = nx.spring_layout(G, k=0.5, seed=42)
    edges = G.edges(data=True)
    pos_edges = [(u, v) for u, v, d in edges if d['weight'] > 0]
    neg_edges = [(u, v) for u, v, d in edges if d['weight'] < 0]
    pos_weights = [d['weight'] * 5 for u, v, d in edges if d['weight'] > 0]
    neg_weights = [abs(d['weight']) * 5 for u, v, d in edges if d['weight'] < 0]
    
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue', edgecolors='navy', linewidths=2)
    nx.draw_networkx_edges(G, pos, edgelist=pos_edges, width=pos_weights, edge_color='red', alpha=0.6, label='Positive Correl')
    nx.draw_networkx_edges(G, pos, edgelist=neg_edges, width=neg_weights, edge_color='blue', alpha=0.6, label='Negative Correl')
    
    # Handle CJK font warning by setting a font that supports Chinese if available, or just ignore.
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', font_family='sans-serif')
    
    plt.title("Prediction Class Relationship Graph (v2, Pearson > 0.1)", fontsize=18, fontweight='bold')
    plt.axis('off')
    
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='red', lw=4, alpha=0.6),
                    Line2D([0], [0], color='blue', lw=4, alpha=0.6)]
    plt.legend(custom_lines, ['Positive Correlation', 'Negative Correlation'], loc='upper left')
    
    graph_path = "/mnt/e/MAE_AST/MAE_AST/pearson_network_graph_v2.png"
    plt.tight_layout()
    plt.savefig(graph_path, bbox_inches='tight')
    plt.close()
    
    print(f"Saved Pearson Heatmap to {heatmap_path}")
    print(f"Saved Pearson Network to {graph_path}")


if __name__ == "__main__":
    plot_support_vs_recall()
    plot_pearson_correlation()
