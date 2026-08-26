import os
import pandas as pd
import matplotlib.pyplot as plt

# Load environment variables
if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())

USE_QUANTIZATION = os.environ.get("USE_QUANTIZATION", "1") == "1"
USE_DP = os.environ.get("USE_DP", "0") == "1"
if USE_DP:
    SUFFIX = "c_dp"
elif USE_QUANTIZATION:
    SUFFIX = "b_quantized"
else:
    SUFFIX = "a_pure"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DASHBOARD_DIR = os.path.join(BASE_DIR, "dashboard")
RESULTS_DIR = os.path.join(DASHBOARD_DIR, "results", SUFFIX)
PLOTS_DIR = os.path.join(DASHBOARD_DIR, "plots", SUFFIX)
os.makedirs(PLOTS_DIR, exist_ok=True)

metrics_path = os.path.join(RESULTS_DIR, f"metrics_{SUFFIX}.csv")

if not os.path.exists(metrics_path):
    print(f"Metrics file not found: {metrics_path}")
    exit(1)

df = pd.read_csv(metrics_path)

if 'Round' in df.columns:
    plt.figure(figsize=(10, 6))
    if 'Accuracy' in df.columns:
        plt.plot(df['Round'], df['Accuracy'], marker='o', label='Global Accuracy')
        for i, val in enumerate(df['Accuracy']):
            plt.text(df['Round'].iloc[i], val, f'{val:.4f}', fontsize=9, ha='center', va='bottom')
    plt.title(f"Global Accuracy ({SUFFIX})")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.margins(y=0.15)
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(PLOTS_DIR, "accuracy.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    if 'Loss' in df.columns:
        plt.plot(df['Round'], df['Loss'], marker='x', color='red', label='Global Loss')
        for i, val in enumerate(df['Loss']):
            plt.text(df['Round'].iloc[i], val, f'{val:.4f}', fontsize=9, ha='center', va='bottom')
    plt.title(f"Global Loss ({SUFFIX})")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.margins(y=0.15)
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(PLOTS_DIR, "loss.png"))
    plt.close()

print(f"Generated single-run plots in {PLOTS_DIR}")
