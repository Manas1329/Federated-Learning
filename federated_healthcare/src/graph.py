import os
from glob import glob

import pandas as pd
import matplotlib.pyplot as plt


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

PLOTS_DIR = os.path.join(
    DASHBOARD_DIR,
    "plots"
)

os.makedirs(
    PLOTS_DIR,
    exist_ok=True
)


# ============================================================
# 1. GLOBAL MODEL GRAPHS
#    Source: dashboard/metrics.csv
# ============================================================

metrics_csv = os.path.join(
    DASHBOARD_DIR,
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
            PLOTS_DIR,
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
            PLOTS_DIR,
            "loss.png"
        ),
        dpi=300
    )

    plt.close()


    # --------------------------------------------------------
    # Accuracy vs Loss
    # --------------------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.plot(
        df_global["Round"],
        df_global["Accuracy"],
        marker="o",
        label="Accuracy (%)"
    )

    plt.plot(
        df_global["Round"],
        df_global["Loss"] * 100,
        marker="s",
        label="Loss ×100"
    )

    plt.title("Global Model Accuracy vs Loss")
    plt.xlabel("Communication Round")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "accuracy_loss.png"
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
    DASHBOARD_DIR,
    "Hospital_*_baseline.csv"
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

            df = pd.read_csv(file)

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
        # 3. TRAINING TIME PER ROUND
        # ====================================================

        if (
            "training_time_sec" in df_clients.columns
            and "round" in df_clients.columns
        ):

            training = (
                df_clients
                .dropna(
                    subset=[
                        "training_time_sec"
                    ]
                )
                .groupby(
                    [
                        "client",
                        "round"
                    ]
                )["training_time_sec"]
                .first()
                .reset_index()
            )


            plt.figure(
                figsize=(10, 6)
            )

            for client in training[
                "client"
            ].unique():

                temp = training[
                    training["client"] == client
                ]

                plt.plot(
                    temp["round"],
                    temp["training_time_sec"],
                    marker="o",
                    label=client
                )

            plt.xlabel(
                "Federated Round"
            )

            plt.ylabel(
                "Training Time (seconds)"
            )

            plt.title(
                "Local Training Time per Federated Round"
            )

            plt.legend()
            plt.grid(True)

            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    PLOTS_DIR,
                    "training_time_roundwise.png"
                ),
                dpi=300
            )

            plt.close()


        # ====================================================
        # 4. EPOCH-WISE TRAINING TIME
        # ====================================================

        if (
            "epoch_time_sec" in df_clients.columns
        ):

            epoch_data = (
                df_clients
                .dropna(
                    subset=[
                        "epoch_time_sec"
                    ]
                )
                .copy()
            )


            # Create epoch number based on
            # order within each client/round

            epoch_data[
                "epoch_number"
            ] = (
                epoch_data
                .groupby(
                    [
                        "client",
                        "round"
                    ]
                )
                .cumcount() + 1
            )


            plt.figure(
                figsize=(10, 6)
            )

            for client in epoch_data[
                "client"
            ].unique():

                temp = epoch_data[
                    epoch_data["client"] == client
                ]

                # Average epoch time across rounds

                epoch_avg = (
                    temp
                    .groupby(
                        "epoch_number"
                    )["epoch_time_sec"]
                    .mean()
                    .reset_index()
                )

                plt.plot(
                    epoch_avg[
                        "epoch_number"
                    ],
                    epoch_avg[
                        "epoch_time_sec"
                    ],
                    marker="o",
                    label=client
                )


            plt.xlabel(
                "Epoch Number"
            )

            plt.ylabel(
                "Average Epoch Time (seconds)"
            )

            plt.title(
                "Average Training Time per Epoch"
            )

            plt.legend()
            plt.grid(True)

            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    PLOTS_DIR,
                    "epoch_training_time.png"
                ),
                dpi=300
            )

            plt.close()


        # ====================================================
        # 5. PAYLOAD SIZE PER ROUND
        # ====================================================

        if (
            "payload_size_mb" in df_clients.columns
            and "round" in df_clients.columns
        ):

            payload = (
                df_clients
                .dropna(
                    subset=[
                        "payload_size_mb"
                    ]
                )
                .groupby(
                    [
                        "client",
                        "round"
                    ]
                )["payload_size_mb"]
                .first()
                .reset_index()
            )


            plt.figure(
                figsize=(10, 6)
            )

            for client in payload[
                "client"
            ].unique():

                temp = payload[
                    payload["client"] == client
                ]

                plt.plot(
                    temp["round"],
                    temp["payload_size_mb"],
                    marker="o",
                    label=client
                )


            plt.xlabel(
                "Federated Round"
            )

            plt.ylabel(
                "Model Payload Size (MB)"
            )

            plt.title(
                "Model Payload Size per Federated Round"
            )

            plt.legend()
            plt.grid(True)

            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    PLOTS_DIR,
                    "payload_size_roundwise.png"
                ),
                dpi=300
            )

            plt.close()


        # ====================================================
        # 6. CLIENT ACCURACY PER ROUND
        # ====================================================

        if (
            "accuracy" in df_clients.columns
            and "round" in df_clients.columns
        ):

            accuracy = (
                df_clients
                .dropna(
                    subset=[
                        "accuracy"
                    ]
                )
                .copy()
            )


            # Convert 0-1 accuracy to percentage

            if accuracy[
                "accuracy"
            ].max() <= 1:

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
                    PLOTS_DIR,
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
                .dropna(
                    subset=[
                        "loss"
                    ]
                )
            )


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
                    PLOTS_DIR,
                    "client_loss_roundwise.png"
                ),
                dpi=300
            )

            plt.close()


        # ====================================================
        # 8. AVERAGE TRAINING TIME
        # ====================================================

        if "training" in locals():

            average_training = (
                training
                .groupby(
                    "client"
                )["training_time_sec"]
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

        if "payload" in locals():

            average_payload = (
                payload
                .groupby(
                    "client"
                )["payload_size_mb"]
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
# 11. SERVER ROUND METRICS
# ============================================================

round_metrics_csv = os.path.join(
    DASHBOARD_DIR,
    "round_metrics.csv"
)

if os.path.exists(round_metrics_csv):

    print("\nGenerating server round graphs...")

    df_round = pd.read_csv(
        round_metrics_csv
    )

    # --------------------------------------------------------
    # Total Round Time
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        df_round["Round"],
        df_round["Total_Round_Time_sec"],
        marker="o",
        linewidth=2
    )

    plt.xlabel("Federated Round")
    plt.ylabel("Total Round Time (seconds)")
    plt.title("Total Federated Round Time")
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "total_round_time.png"
        ),
        dpi=300
    )

    plt.close()


    # --------------------------------------------------------
    # Aggregation Time
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        df_round["Round"],
        df_round["Aggregation_Time_sec"],
        marker="o",
        linewidth=2
    )

    plt.xlabel("Federated Round")
    plt.ylabel("Aggregation Time (seconds)")
    plt.title("Server Aggregation Time per Round")
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "aggregation_time.png"
        ),
        dpi=300
    )

    plt.close()


    # --------------------------------------------------------
    # Total Round Time vs Aggregation Time
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        df_round["Round"],
        df_round["Total_Round_Time_sec"],
        marker="o",
        label="Total Round Time"
    )

    plt.plot(
        df_round["Round"],
        df_round["Aggregation_Time_sec"],
        marker="s",
        label="Aggregation Time"
    )

    plt.xlabel("Federated Round")
    plt.ylabel("Time (seconds)")
    plt.title("Round Time vs Aggregation Time")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "round_vs_aggregation_time.png"
        ),
        dpi=300
    )

    plt.close()


    # --------------------------------------------------------
    # Client Success / Failure
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        df_round["Round"],
        df_round["Successful_Clients"],
        marker="o",
        label="Successful Clients"
    )

    plt.plot(
        df_round["Round"],
        df_round["Failed_Clients"],
        marker="s",
        label="Failed Clients"
    )

    plt.xlabel("Federated Round")
    plt.ylabel("Number of Clients")
    plt.title("Client Participation per Federated Round")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "client_participation.png"
        ),
        dpi=300
    )

    plt.close()


    # --------------------------------------------------------
    # Print averages
    # --------------------------------------------------------

    print("\nAverage Total Round Time:")
    print(
        df_round[
            "Total_Round_Time_sec"
        ].mean()
    )

    print("\nAverage Aggregation Time:")
    print(
        df_round[
            "Aggregation_Time_sec"
        ].mean()
    )

    print("\nRound Metrics:")
    print(df_round)

else:

    print(
        "\nround_metrics.csv not found."
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

            df = pd.read_csv(file)

            # Extract hospital name
            name = os.path.basename(file)

            hospital = (
                name
                .replace("_baseline.csv", "")
                .replace("_quantized.csv", "")
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
        DASHBOARD_DIR,
        "Hospital_*_baseline.csv"
    )
)


# ============================================================
# Load quantized
# ============================================================

quantized_df = load_experiment_files(
    os.path.join(
        DASHBOARD_DIR,
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