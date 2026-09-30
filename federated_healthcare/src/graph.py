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

from pathlib import Path
import sys
# Ensure 'src' package is importable regardless of where the script is run from
sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import RESULTS_DIR, PLOTS_DIR, DASHBOARD_DIR

RESULTS_DIR_SUFFIX = RESULTS_DIR / SUFFIX
PLOTS_DIR_SUFFIX = PLOTS_DIR / SUFFIX
PLOTS_DIR_SUFFIX.mkdir(parents=True, exist_ok=True)

metrics_path = RESULTS_DIR_SUFFIX / f"metrics_{SUFFIX}.csv"

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
    plt.savefig(PLOTS_DIR_SUFFIX / "accuracy.png")
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
    plt.savefig(PLOTS_DIR_SUFFIX / "loss.png")
    plt.close()

print(f"Generated single-run plots in {PLOTS_DIR_SUFFIX}")

# ---------------------------------------------------------
# Cross-Hospital Generalization Comparison Plots
# ---------------------------------------------------------
comparison_metrics_path = RESULTS_DIR / "comparison_results.csv"

if comparison_metrics_path.exists():
    import numpy as np
    df_comp = pd.read_csv(comparison_metrics_path)
    hospitals = ["Hospital_A", "Hospital_B", "Hospital_C"]
    
    # Filter only if hospitals exist in df
    hospitals = [h for h in hospitals if h in df_comp["hospital"].values]
    
    if hospitals:
        comparison_plots_dir = PLOTS_DIR / "comparisons" / "LvF"
        comparison_plots_dir.mkdir(parents=True, exist_ok=True)
        
        metrics = [
            ("accuracy", "comparison_accuracy.png", "Accuracy Comparison"),
            ("f1", "comparison_f1.png", "F1-Score Comparison"),
            ("precision", "comparison_precision.png", "Precision Comparison"),
            ("recall", "comparison_recall.png", "Recall Comparison"),
            ("roc_auc", "comparison_roc_auc.png", "ROC-AUC Comparison"),
            ("pr_auc", "comparison_pr_auc.png", "PR-AUC Comparison")
        ]
        
        for metric_col, filename, title in metrics:
            if metric_col not in df_comp.columns:
                continue
                
            a_only_vals = []
            fed_vals = []
            
            for h in hospitals:
                a_val = df_comp[(df_comp["model"] == "A_only") & (df_comp["hospital"] == h)][metric_col].values
                f_val = df_comp[(df_comp["model"] == "Federated") & (df_comp["hospital"] == h)][metric_col].values
                
                a_only_vals.append(a_val[0] if len(a_val) > 0 and not np.isnan(a_val[0]) else 0)
                fed_vals.append(f_val[0] if len(f_val) > 0 and not np.isnan(f_val[0]) else 0)
                
            x = np.arange(len(hospitals))
            width = 0.35
            
            fig, ax = plt.subplots(figsize=(10, 6))
            rects1 = ax.bar(x - width/2, a_only_vals, width, label='A_only')
            rects2 = ax.bar(x + width/2, fed_vals, width, label='Federated')
            
            ax.set_ylabel(metric_col.replace("_", " ").title())
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels([h.replace("_", " ") for h in hospitals])
            ax.legend(loc='lower left' if metric_col in ['accuracy', 'roc_auc', 'pr_auc'] else 'best')
            ax.grid(axis='y', alpha=0.3)
            
            ax.bar_label(rects1, padding=3, fmt='%.3f')
            ax.bar_label(rects2, padding=3, fmt='%.3f')
            
            # Auto-scale y-axis to make differences visible
            min_val = min(min(a_only_vals, default=0), min(fed_vals, default=0))
            max_val = max(max(a_only_vals, default=0), max(fed_vals, default=0))
            if max_val > 0:
                y_bottom = max(0.0, min_val - 0.05)
                # Give enough headroom for the text labels on top of the bars
                y_top = max_val + (0.05 * (max_val - y_bottom)) if max_val != y_bottom else max_val * 1.1
                if max_val <= 1.0:
                    y_top = max(y_top, max_val + 0.02) # Minimum headroom for 1.0
                ax.set_ylim(bottom=y_bottom, top=y_top)
                
            fig.tight_layout()
            plt.savefig(comparison_plots_dir / filename)
            plt.close()
            
        print(f"Generated comparison plots in {comparison_plots_dir}")
