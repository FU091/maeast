import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os

# Set up matplotlib for better aesthetics (Korean/Chinese font might be needed if labels are CJK, but labels are English here)
plt.rcParams['figure.dpi'] = 150
sns.set_theme(style="white")

def main():
    json_path = "/home/lin/MAE_output/predictions.json"
    print(f"Loading data from {json_path}...")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    predictions = data.get("predictions", [])
    if not predictions:
        print("No predictions found in JSON.")
        return
        
    # Build a list of dicts for dataframe
    records = []
    for item in predictions:
        record = {"file": item["file"]}
        for k in item["top_k"]:
            record[k["label"]] = k["prob"]
        records.append(record)
        
    df = pd.DataFrame(records)
    df.set_index("file", inplace=True)
    df.fillna(0, inplace=True) # Fill missing probabilities with 0
    
    # Calculate Pearson Correlation
    print("Calculating Pearson Correlation...")
    corr_matrix = df.corr(method='pearson')
    
    # 1. Plot Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", vmin=-1, vmax=1, 
                square=True, linewidths=.5, cbar_kws={"shrink": .75})
    plt.title("Pearson Correlation Matrix (Predicted Probabilities)", fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    heatmap_path = "/mnt/e/MAE_AST/MAE_AST/pearson_heatmap.png"
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Saved Heatmap to {heatmap_path}")
    
    # 2. Plot Node Relationship Graph (NetworkX)
    print("Drawing Node Relationship Graph...")
    G = nx.Graph()
    
    # Add nodes
    labels = corr_matrix.columns
    for label in labels:
        G.add_node(label)
        
    # Add edges for correlation above a certain threshold to avoid clutter
    threshold = 0.1  # Only show somewhat meaningful correlations
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            weight = corr_matrix.iloc[i, j]
            if abs(weight) >= threshold:
                G.add_edge(labels[i], labels[j], weight=weight)
                
    plt.figure(figsize=(14, 12))
    
    # Position nodes using spring layout
    pos = nx.spring_layout(G, k=0.5, seed=42)
    
    # Separate edges by positive and negative correlation
    edges = G.edges(data=True)
    pos_edges = [(u, v) for u, v, d in edges if d['weight'] > 0]
    neg_edges = [(u, v) for u, v, d in edges if d['weight'] < 0]
    
    pos_weights = [d['weight'] * 5 for u, v, d in edges if d['weight'] > 0]
    neg_weights = [abs(d['weight']) * 5 for u, v, d in edges if d['weight'] < 0]
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue', edgecolors='navy', linewidths=2)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edgelist=pos_edges, width=pos_weights, edge_color='red', alpha=0.6, label='Positive Correl')
    nx.draw_networkx_edges(G, pos, edgelist=neg_edges, width=neg_weights, edge_color='blue', alpha=0.6, label='Negative Correl')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', font_family='sans-serif')
    
    plt.title("Prediction Class Relationship Graph (Pearson > 0.1)", fontsize=18, fontweight='bold')
    plt.axis('off')
    
    # Add legend manually
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='red', lw=4, alpha=0.6),
                    Line2D([0], [0], color='blue', lw=4, alpha=0.6)]
    plt.legend(custom_lines, ['Positive Correlation', 'Negative Correlation'], loc='upper left')
    
    graph_path = "/mnt/e/MAE_AST/MAE_AST/pearson_network_graph.png"
    plt.tight_layout()
    plt.savefig(graph_path, bbox_inches='tight')
    plt.close()
    print(f"Saved Network Graph to {graph_path}")

if __name__ == "__main__":
    main()
