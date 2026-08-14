import os
from glob import glob

import pandas as pd
import matplotlib.pyplot as plt

# Load environment variables from .env file if present
if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                k = key.strip()
                if k not in os.environ:
                    os.environ[k] = val.strip()


USE_QUANTIZATION = os.environ.get("USE_QUANTIZATION", "1") == "1"
SUFFIX = "quantized" if USE_QUANTIZATION else "no_quantization"

import numpy as np

def load_mismatched_csv(file_path):
    rows = []
    with open(file_path, "r") as f:
        lines = f.readlines()
        
    last_seen_round = 0
    for line in lines:
        line = line.strip()
        if not line or line.startswith("client,") or line.startswith("client\t"):
            continue
            
        parts = [p.strip() for p in line.split(",")]
        if not parts or parts[0] == "":
            continue
            
        is_fit = len(parts) > 2 and parts[2] != ""
        
        current_round = int(parts[1]) if (len(parts) > 1 and parts[1]) else 0
        if is_fit:
            last_seen_round = current_round
        else:
            if current_round == 0:
                current_round = last_seen_round
                
        row = {
            "client": parts[0],
            "round": current_round,
            "epoch_time_sec": float(parts[2]) if (len(parts) > 2 and parts[2]) else np.nan,
            "training_time_sec": float(parts[3]) if (len(parts) > 3 and parts[3]) else np.nan,
            "payload_size_bytes": float(parts[4]) if (len(parts) > 4 and parts[4]) else np.nan,
            "payload_size_mb": float(parts[5]) if (len(parts) > 5 and parts[5]) else np.nan,
            "quantized_payload_bytes": np.nan,
            "quantized_payload_mb": np.nan,
            "compression_ratio": np.nan,
            "reduction_percent": np.nan,
            "accuracy": np.nan,
            "loss": np.nan,
            "device": ""
        }
        
        if is_fit:
            if len(parts) > 6 and parts[6]:
                row["quantized_payload_bytes"] = float(parts[6])
            if len(parts) > 7 and parts[7]:
                row["quantized_payload_mb"] = float(parts[7])
            if len(parts) > 8 and parts[8]:
                row["compression_ratio"] = float(parts[8])
            if len(parts) > 9 and parts[9]:
                row["reduction_percent"] = float(parts[9])
            if len(parts) > 10:
                row["device"] = parts[10]
        else:
            if len(parts) > 6 and parts[6]:
                row["accuracy"] = float(parts[6])
            if len(parts) > 7 and parts[7]:
                row["loss"] = float(parts[7])
            if len(parts) > 8:
                row["device"] = parts[8]
                
        rows.append(row)
        
    return pd.DataFrame(rows)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DASHBOARD_DIR = os.path.join(
    BASE_DIR,
    "dashboard"
)

RESULTS_DIR = os.path.join(
    DASHBOARD_DIR,
    "results"
)
os.makedirs(RESULTS_DIR, exist_ok=True)

PLOTS_DIR = os.path.join(
    DASHBOARD_DIR,
    "plots"
)

MODE_PLOTS_DIR = os.path.join(
    PLOTS_DIR,
    SUFFIX
)
os.makedirs(MODE_PLOTS_DIR, exist_ok=True)


# ============================================================
# 1. GLOBAL MODEL GRAPHS
#    Source: dashboard/metrics.csv
# ============================================================

metrics_csv = os.path.join(
    RESULTS_DIR,
    f"metrics_{SUFFIX}.csv"
)
if not os.path.exists(metrics_csv):
    metrics_csv = os.path.join(
        RESULTS_DIR,
        "metrics.csv"
    )


if os.path.exists(metrics_csv):

    print("\nGenerating global model graphs...")

    df_global = pd.read_csv(metrics_csv)

    # Convert accuracy from 0-1 to percentage
    if (
        "Accuracy" in df_global.columns
        and df_global["Accuracy"].max() <= 1
    ):
        df_global["Accuracy"] = (
            df_global["Accuracy"] * 100
        )


    # --------------------------------------------------------
    # Accuracy Graph
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        df_global["Round"],
        df_global["Accuracy"],
        marker="o",
        linewidth=2
    )

    for x, y in zip(
        df_global["Round"],
        df_global["Accuracy"]
    ):
        plt.text(
            x,
            y + 0.3,
            f"{y:.2f}%",
            ha="center",
            fontsize=8
        )

    best_idx = df_global["Accuracy"].idxmax()

    plt.scatter(
        df_global.loc[best_idx, "Round"],
        df_global.loc[best_idx, "Accuracy"],
        s=120
    )

    plt.title("Global Model Accuracy")
    plt.xlabel("Communication Round")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            MODE_PLOTS_DIR,
            "accuracy.png"
        ),
        dpi=300
    )

    plt.close()


    # --------------------------------------------------------
    # Loss Graph
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        df_global["Round"],
        df_global["Loss"],
        marker="o",
        linewidth=2
    )

    for x, y in zip(
        df_global["Round"],
        df_global["Loss"]
    ):
        plt.text(
            x,
            y + 0.01,
            f"{y:.3f}",
            ha="center",
            fontsize=8
        )

    plt.title("Global Model Loss")
    plt.xlabel("Communication Round")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            MODE_PLOTS_DIR,
            "loss.png"
        ),
        dpi=300
    )

    plt.close()




    print("Global model graphs generated.")


else:

    print(
        "Global metrics.csv not found."
    )


# ============================================================
# 2. LOAD CLIENT BASELINE CSV FILES
# ============================================================

client_csv_pattern = os.path.join(
    RESULTS_DIR,
    f"Hospital_*_{SUFFIX}.csv"
)
client_csvs = sorted(
    glob(client_csv_pattern)
)


if not client_csvs:

    print(
        "\nNo Hospital baseline CSV files found."
    )

else:

    print(
        "\nFound client files:"
    )

    for file in client_csvs:
        print(
            " -",
            os.path.basename(file)
        )


    # --------------------------------------------------------
    # Read all client CSVs
    # --------------------------------------------------------

    client_data = []

    for file in client_csvs:

        try:

            df = load_mismatched_csv(file)

            # If client column doesn't exist,
            # infer it from filename

            if "client" not in df.columns:

                client_name = os.path.basename(
                    file
                ).replace(
                    "_baseline.csv",
                    ""
                )

                df["client"] = client_name

            client_data.append(df)

        except Exception as e:

            print(
                f"Could not read {file}: {e}"
            )


    if client_data:

        df_clients = pd.concat(
            client_data,
            ignore_index=True
        )





        # ====================================================
        # 6. CLIENT ACCURACY PER ROUND
        # ====================================================

        if (
            "accuracy" in df_clients.columns
            and "round" in df_clients.columns
        ):

            accuracy = (
                df_clients
                .copy()
            )
            accuracy["accuracy"] = pd.to_numeric(accuracy["accuracy"], errors="coerce")
            accuracy = accuracy.dropna(subset=["accuracy"])

            # Convert 0-1 accuracy to percentage
            if not accuracy.empty and accuracy["accuracy"].max() <= 1:

                accuracy[
                    "accuracy"
                ] = (
                    accuracy[
                        "accuracy"
                    ] * 100
                )


            plt.figure(
                figsize=(10, 6)
            )

            for client in accuracy[
                "client"
            ].unique():

                temp = accuracy[
                    accuracy["client"] == client
                ]

                plt.plot(
                    temp["round"],
                    temp["accuracy"],
                    marker="o",
                    label=client
                )


            plt.xlabel(
                "Federated Round"
            )

            plt.ylabel(
                "Accuracy (%)"
            )

            plt.title(
                "Client Model Accuracy per Federated Round"
            )

            plt.legend()
            plt.grid(True)

            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    MODE_PLOTS_DIR,
                    "client_accuracy_roundwise.png"
                ),
                dpi=300
            )

            plt.close()


        # ====================================================
        # 7. CLIENT LOSS PER ROUND
        # ====================================================

        if (
            "loss" in df_clients.columns
            and "round" in df_clients.columns
        ):

            loss_data = (
                df_clients
                .copy()
            )
            loss_data["loss"] = pd.to_numeric(loss_data["loss"], errors="coerce")
            loss_data = loss_data.dropna(subset=["loss"])


            plt.figure(
                figsize=(10, 6)
            )

            for client in loss_data[
                "client"
            ].unique():

                temp = loss_data[
                    loss_data["client"] == client
                ]

                plt.plot(
                    temp["round"],
                    temp["loss"],
                    marker="o",
                    label=client
                )


            plt.xlabel(
                "Federated Round"
            )

            plt.ylabel(
                "Loss"
            )

            plt.title(
                "Client Model Loss per Federated Round"
            )

            plt.legend()
            plt.grid(True)

            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    MODE_PLOTS_DIR,
                    "client_loss_roundwise.png"
                ),
                dpi=300
            )

            plt.close()


        # ====================================================
        # 8. AVERAGE TRAINING TIME
        # ====================================================

        if "training_time_sec" in df_clients.columns:

            average_training = (
                df_clients
                .dropna(subset=["training_time_sec"])
                .groupby("client")["training_time_sec"]
                .mean()
            )

            print(
                "\nAverage Training Time:"
            )

            print(
                average_training
            )


        # ====================================================
        # 9. AVERAGE PAYLOAD SIZE
        # ====================================================

        if "payload_size_mb" in df_clients.columns:

            # Use quantized_payload_mb if quantized, else payload_size_mb
            col = "quantized_payload_mb" if USE_QUANTIZATION and "quantized_payload_mb" in df_clients.columns else "payload_size_mb"
            average_payload = (
                df_clients
                .dropna(subset=[col])
                .groupby("client")[col]
                .mean()
            )

            print(
                "\nAverage Payload Size:"
            )

            print(
                average_payload
            )


        # ====================================================
        # 10. AVERAGE ACCURACY
        # ====================================================

        if "accuracy" in locals():

            average_accuracy = (
                accuracy
                .groupby(
                    "client"
                )["accuracy"]
                .mean()
            )

            print(
                "\nAverage Client Accuracy:"
            )

            print(
                average_accuracy
            )


# ============================================================
# FINAL MESSAGE
# ============================================================

print(
    "\n=============================================="
)

print(
    "All available graphs generated successfully."
)

print(
    "Graphs saved in:"
)

print(
    PLOTS_DIR
)

print(
    "=============================================="
)

# ============================================================
# QUANTIZATION COMPARISON
# ============================================================

import glob


def load_experiment_files(pattern):

    files = glob.glob(pattern)

    frames = []

    for file in files:

        try:

            df = load_mismatched_csv(file)

            # Extract hospital name
            name = os.path.basename(file)

            hospital = (
                name
                .replace("_baseline.csv", "")
                .replace("_quantized.csv", "")
                .replace("_no_quantization.csv", "")
            )

            df["hospital"] = hospital

            frames.append(df)

        except Exception as e:

            print(
                f"Failed to read {file}: {e}"
            )

    if frames:

        return pd.concat(
            frames,
            ignore_index=True
        )

    return pd.DataFrame()


# ============================================================
# Load baseline
# ============================================================

baseline_df = load_experiment_files(
    os.path.join(
        RESULTS_DIR,
        "Hospital_*_no_quantization.csv"
    )
)
if baseline_df.empty:
    baseline_df = load_experiment_files(
        os.path.join(
            RESULTS_DIR,
            "Hospital_*_baseline.csv"
        )
    )

# ============================================================
# Load quantized
# ============================================================

quantized_df = load_experiment_files(
    os.path.join(
        RESULTS_DIR,
        "Hospital_*_quantized.csv"
    )
)


# ============================================================
# Payload Comparison
# ============================================================

if (
    not baseline_df.empty
    and not quantized_df.empty
):

    baseline_payload = (
        baseline_df
        .groupby("hospital")[
            "payload_size_mb"
        ]
        .mean()
    )

    quantized_payload = (
        quantized_df
        .groupby("hospital")[
            "quantized_payload_mb"
        ]
        .mean()
    )

    comparison = pd.DataFrame({
        "Baseline FP32":
            baseline_payload,

        "Quantized INT8":
            quantized_payload
    })

    comparison.plot(
        kind="bar",
        figsize=(10, 6)
    )

    plt.title(
        "Communication Payload: "
        "FP32 vs INT8"
    )

    plt.xlabel("Hospital")

    plt.ylabel(
        "Average Payload Size (MB)"
    )

    plt.xticks(
        rotation=0
    )

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.6
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "quantization_payload_comparison.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# PAYLOAD REDUCTION
# ============================================================

if (
    not baseline_df.empty
    and not quantized_df.empty
):

    baseline_avg = (
        baseline_df
        .groupby("hospital")[
            "payload_size_mb"
        ]
        .mean()
    )

    quantized_avg = (
        quantized_df
        .groupby("hospital")[
            "quantized_payload_mb"
        ]
        .mean()
    )

    reduction = (
        (
            baseline_avg -
            quantized_avg
        )
        / baseline_avg
    ) * 100

    plt.figure(
        figsize=(9, 6)
    )

    bars = plt.bar(
        reduction.index,
        reduction.values,
        color="seagreen"
    )

    for bar, value in zip(
        bars,
        reduction.values
    ):
        if pd.notna(value) and np.isfinite(value):
            plt.text(
                bar.get_x()
                + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}%",
                ha="center",
                va="bottom"
            )

    plt.title(
        "Communication Payload Reduction "
        "Using INT8 Quantization"
    )

    plt.xlabel("Hospital")

    plt.ylabel(
        "Payload Reduction (%)"
    )

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.6
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "payload_reduction.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# FINAL COMPARISON SUMMARY GENERATION
# ============================================================

def parse_classification_report(report_path):
    accuracy = np.nan
    f1_score = np.nan
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                if parts[0] == "accuracy" and len(parts) >= 2:
                    try:
                        accuracy = float(parts[1])
                    except:
                        pass
                elif len(parts) >= 5 and parts[0] == "weighted" and parts[1] == "avg":
                    try:
                        f1_score = float(parts[4])
                    except:
                        pass
    return accuracy, f1_score

# Load round metrics for both runs
round_no_quant_csv = os.path.join(RESULTS_DIR, "round_metrics_no_quantization.csv")
round_quant_csv = os.path.join(RESULTS_DIR, "round_metrics_quantized.csv")

round_no_quant = pd.read_csv(round_no_quant_csv) if os.path.exists(round_no_quant_csv) else pd.DataFrame()
round_quant = pd.read_csv(round_quant_csv) if os.path.exists(round_quant_csv) else pd.DataFrame()

# Load classification reports
report_no_quant_path = os.path.join(DASHBOARD_DIR, "classification_reports", "classification_report_no_quantization.txt")
if not os.path.exists(report_no_quant_path):
    report_no_quant_path = os.path.join(DASHBOARD_DIR, "classification_reports", "classification_report.txt")

report_quant_path = os.path.join(DASHBOARD_DIR, "classification_reports", "classification_report_quantized.txt")

acc_no_quant, f1_no_quant = parse_classification_report(report_no_quant_path)
acc_quant, f1_quant = parse_classification_report(report_quant_path)

# Metrics calculation
payload_no_quant = baseline_df["payload_size_mb"].mean() if not baseline_df.empty else np.nan
payload_quant = quantized_df["quantized_payload_mb"].mean() if not quantized_df.empty else np.nan

comm_no_quant = round_no_quant["Aggregation_Time_sec"].mean() if not round_no_quant.empty else np.nan
comm_quant = round_quant["Aggregation_Time_sec"].mean() if not round_quant.empty else np.nan

round_no_quant_val = round_no_quant["Total_Round_Time_sec"].mean() if not round_no_quant.empty else np.nan
round_quant_val = round_quant["Total_Round_Time_sec"].mean() if not round_quant.empty else np.nan

train_no_quant = baseline_df["training_time_sec"].mean() if not baseline_df.empty else np.nan
train_quant = quantized_df["training_time_sec"].mean() if not quantized_df.empty else np.nan

# Prepare comparison summary
comparison_data = [
    {"Metric": "Payload size", "Normal FL": f"{payload_no_quant:.2f} MB" if pd.notna(payload_no_quant) else "N/A", "Quantized FL": f"{payload_quant:.2f} MB" if pd.notna(payload_quant) else "N/A"},
    {"Metric": "Communication time", "Normal FL": f"{comm_no_quant:.2f} sec" if pd.notna(comm_no_quant) else "N/A", "Quantized FL": f"{comm_quant:.2f} sec" if pd.notna(comm_quant) else "N/A"},
    {"Metric": "Round time", "Normal FL": f"{round_no_quant_val:.2f} sec" if pd.notna(round_no_quant_val) else "N/A", "Quantized FL": f"{round_quant_val:.2f} sec" if pd.notna(round_quant_val) else "N/A"},
    {"Metric": "Training time", "Normal FL": f"{train_no_quant:.2f} sec" if pd.notna(train_no_quant) else "N/A", "Quantized FL": f"{train_quant:.2f} sec" if pd.notna(train_quant) else "N/A"},
    {"Metric": "Accuracy", "Normal FL": f"{acc_no_quant * 100:.2f}%" if pd.notna(acc_no_quant) else "N/A", "Quantized FL": f"{acc_quant * 100:.2f}%" if pd.notna(acc_quant) else "N/A"},
    {"Metric": "F1-score", "Normal FL": f"{f1_no_quant:.4f}" if pd.notna(f1_no_quant) else "N/A", "Quantized FL": f"{f1_quant:.4f}" if pd.notna(f1_quant) else "N/A"}
]

summary_df = pd.DataFrame(comparison_data)

# Save comparison summary to CSV
summary_df.to_csv(os.path.join(DASHBOARD_DIR, "comparison_summary.csv"), index=False)
parent_summary = os.path.join(os.path.dirname(DASHBOARD_DIR), "dashboard", "comparison_summary.csv")
if os.path.exists(os.path.dirname(parent_summary)):
    summary_df.to_csv(parent_summary, index=False)

# Write to markdown comparison summary
md_content = f"""# Federated Learning Comparison Summary

| Metric | Normal FL | Quantized FL |
| :--- | :--- | :--- |
| **Payload size** | {summary_df.loc[0, 'Normal FL']} | {summary_df.loc[0, 'Quantized FL']} |
| **Communication time** | {summary_df.loc[1, 'Normal FL']} | {summary_df.loc[1, 'Quantized FL']} |
| **Round time** | {summary_df.loc[2, 'Normal FL']} | {summary_df.loc[2, 'Quantized FL']} |
| **Training time** | {summary_df.loc[3, 'Normal FL']} | {summary_df.loc[3, 'Quantized FL']} |
| **Accuracy** | {summary_df.loc[4, 'Normal FL']} | {summary_df.loc[4, 'Quantized FL']} |
| **F1-score** | {summary_df.loc[5, 'Normal FL']} | {summary_df.loc[5, 'Quantized FL']} |
"""

with open(os.path.join(PLOTS_DIR, "comparison_summary.md"), "w") as f:
    f.write(md_content)

print("\n==============================================")
print("FINALLY COMPARE:")
print("==============================================")
print(summary_df.to_string(index=False))
print("==============================================\n")