import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DASHBOARD_DIR = os.path.join(BASE_DIR, "dashboard")
COMPARISONS_DIR = os.path.join(DASHBOARD_DIR, "comparisons")
os.makedirs(COMPARISONS_DIR, exist_ok=True)

configs = {
    "Pure (FP32)": "a_pure",
    "Quantized (INT8)": "b_quantized",
    "Quantized + DP": "c_dp"
}

# 1. Global Accuracy Comparison
plt.figure(figsize=(10, 6))
has_data = False
for label, suffix in configs.items():
    metrics_path = os.path.join(DASHBOARD_DIR, "results", suffix, f"metrics_{suffix}.csv")
    if os.path.exists(metrics_path):
        df = pd.read_csv(metrics_path)
        if 'Round' in df.columns and 'Accuracy' in df.columns:
            plt.plot(df['Round'], df['Accuracy'], marker='o', label=label)
            has_data = True
if has_data:
    plt.title("Global Accuracy Comparison")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(COMPARISONS_DIR, "accuracy_comparison.png"))
plt.close()

# 2. Global Loss Comparison
plt.figure(figsize=(10, 6))
has_data = False
for label, suffix in configs.items():
    metrics_path = os.path.join(DASHBOARD_DIR, "results", suffix, f"metrics_{suffix}.csv")
    if os.path.exists(metrics_path):
        df = pd.read_csv(metrics_path)
        if 'Round' in df.columns and 'Loss' in df.columns:
            plt.plot(df['Round'], df['Loss'], marker='x', label=label)
            has_data = True
if has_data:
    plt.title("Global Loss Comparison")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(COMPARISONS_DIR, "loss_comparison.png"))
plt.close()

# 3. Payload Size Bar Chart
plt.figure(figsize=(10, 6))
labels = []
payloads = []
for label, suffix in configs.items():
    # Load hospital CSVs
    results_dir = os.path.join(DASHBOARD_DIR, "results", suffix)
    total_payload = 0
    count = 0
    if os.path.exists(results_dir):
        for f in os.listdir(results_dir):
            if f.startswith("Hospital_") and f.endswith(".csv"):
                # Read without header to avoid multi-index issues with jagged rows (15 cols vs 9 cols)
                df = pd.read_csv(os.path.join(results_dir, f), header=None, skiprows=1)
                
                # Filter for training rows only (which have >9 columns)
                if 10 in df.columns:
                    df_train = df[df[10].notna()]
                    
                    # Col 5 = FP32 payload MB, Col 7 = INT8 payload MB
                    if "quantized" in suffix:
                        s = pd.to_numeric(df_train[7], errors='coerce')
                    else:
                        s = pd.to_numeric(df_train[5], errors='coerce')
                        
                    mean_val = s.mean()
                    if pd.notna(mean_val):
                        total_payload += mean_val
                    count += 1
    if count > 0:
        labels.append(label)
        payloads.append(total_payload / count)

if labels:
    plt.bar(labels, payloads, color=['blue', 'orange', 'green'][:len(labels)])
    plt.title("Average Communication Payload per Client")
    plt.ylabel("Payload Size (MB)")
    plt.grid(axis='y')
    plt.savefig(os.path.join(COMPARISONS_DIR, "payload_comparison.png"))
plt.close()

print(f"Comparison graphs generated in {COMPARISONS_DIR}")
