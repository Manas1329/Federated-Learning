import flwr as fl
import os
import time
import pandas as pd
import torch

from collections import OrderedDict
from flwr.common import parameters_to_ndarrays
from model import ChestCNN

from flwr.common import (
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)

from quantization import (
    dequantize_parameters,
)
from dataclasses import replace

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

METRICS_FILE = os.path.join(
    BASE_DIR,
    "dashboard",
    "metrics.csv"
)

ROUND_METRICS_FILE = os.path.join(
    BASE_DIR,
    "dashboard",
    "round_metrics.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "global_model.pth"
)


# --------------------------------------------------
# Reset old metrics
# --------------------------------------------------

if os.path.exists(METRICS_FILE):
    os.remove(METRICS_FILE)

if os.path.exists(ROUND_METRICS_FILE):
    os.remove(ROUND_METRICS_FILE)


# --------------------------------------------------
# Global round timer
# --------------------------------------------------

round_start_times = {}


# --------------------------------------------------
# Custom FedAvg Strategy
# --------------------------------------------------

class SaveModelStrategy(fl.server.strategy.FedAvg):

    def configure_fit(
        self,
        server_round,
        parameters,
        client_manager,
    ):

        # Start timer for this round
        round_start_times[server_round] = time.perf_counter()

        print("\n")
        print("=" * 60)
        print(f"Starting Federated Round {server_round}")
        print("=" * 60)

        # Send round number to clients
        config = {
            "server_round": server_round
        }

        # Get normal Flower configuration
        fit_ins = fl.common.FitIns(
            parameters,
            config
        )

        # Select clients using client_manager passed directly into function
        clients = client_manager.sample(
            num_clients=self.min_fit_clients,
            min_num_clients=self.min_fit_clients
        )

        return [
            (client, fit_ins)
            for client in clients
        ]


    # --------------------------------------------------
    # Aggregate Fit
    # --------------------------------------------------

    def aggregate_fit(
        self,
        server_round,
        results,
        failures,
    ):

        aggregation_start = time.perf_counter()

        # ==========================================================
        # DEQUANTIZE CLIENT PARAMETERS
        # ==========================================================

        dequantized_results = []

        for client_proxy, fit_res in results:

            try:

                # Convert Flower Parameters -> NumPy arrays
                quantized_parameters = parameters_to_ndarrays(
                    fit_res.parameters
                )

                # INT8 -> FP32
                fp32_parameters = dequantize_parameters(
                    quantized_parameters
                )

                # Convert FP32 NumPy arrays back to Flower Parameters
                fp32_parameters_flower = ndarrays_to_parameters(
                    fp32_parameters
                )

                # Replace the INT8 parameters with FP32 parameters
                fit_res.parameters = fp32_parameters_flower

                dequantized_results.append(
                    (
                        client_proxy,
                        fit_res
                    )
                )

            except Exception as e:

                print(
                    f"Error dequantizing client update: {e}"
                )

        # ==========================================================
        # FEDAVG
        # ==========================================================

        aggregated_parameters, aggregated_metrics = (
            super().aggregate_fit(
                server_round,
                dequantized_results,
                failures,
            )
        )

        # ==========================================================
        # AGGREGATION TIME
        # ==========================================================

        aggregation_time = (
            time.perf_counter()
            - aggregation_start
        )

        # ==========================================================
        # NUMBER OF CLIENTS
        # ==========================================================

        successful_clients = len(
            dequantized_results
        )

        failed_clients = len(
            failures
        )

        # ==========================================================
        # ROUND TIME
        # ==========================================================

        if server_round in round_start_times:

            round_time = (
                time.perf_counter()
                - round_start_times[server_round]
            )

        else:

            round_time = 0

        # ==========================================================
        # PRINT RESULTS
        # ==========================================================

        print("\n")
        print("=" * 60)
        print(
            f"Round {server_round} completed"
        )
        print("=" * 60)

        print(
            f"Successful Clients: "
            f"{successful_clients}"
        )

        print(
            f"Failed Clients: "
            f"{failed_clients}"
        )

        print(
            f"Dequantization + Aggregation Time: "
            f"{aggregation_time:.4f} sec"
        )

        print(
            f"Total Round Time: "
            f"{round_time:.4f} sec"
        )

        # ==========================================================
        # SAVE GLOBAL MODEL
        # ==========================================================

        if aggregated_parameters is not None:

            model = ChestCNN()

            params = parameters_to_ndarrays(
                aggregated_parameters
            )

            params_dict = zip(
                model.state_dict().keys(),
                params
            )

            state_dict = OrderedDict(
                {
                    k: torch.tensor(v)
                    for k, v in params_dict
                }
            )

            model.load_state_dict(
                state_dict,
                strict=True
            )

            # Save final global model
            if server_round == 10:

                torch.save(
                    model.state_dict(),
                    MODEL_PATH,
                )

                print("\n")
                print("=" * 60)
                print("Global Model Saved Successfully")
                print(MODEL_PATH)
                print("=" * 60)

        # ==========================================================
        # SAVE ROUND METRICS
        # ==========================================================

        round_data = pd.DataFrame(
            [[
                server_round,
                successful_clients,
                failed_clients,
                aggregation_time,
                round_time
            ]],
            columns=[
                "Round",
                "Successful_Clients",
                "Failed_Clients",
                "Aggregation_Time_sec",
                "Total_Round_Time_sec"
            ]
        )

        round_data.to_csv(
            ROUND_METRICS_FILE,
            mode="a",
            header=not os.path.exists(
                ROUND_METRICS_FILE
            ),
            index=False
        )

        return (
            aggregated_parameters,
            aggregated_metrics
        )


# --------------------------------------------------
# Evaluation Metrics
# --------------------------------------------------

def evaluate_metrics_aggregation_fn(metrics):

    total_examples = sum(
        num_examples
        for num_examples, _
        in metrics
    )


    weighted_accuracy = (
        sum(
            num_examples * m["accuracy"]
            for num_examples, m in metrics
        )
        / total_examples
    )


    weighted_loss = (
        sum(
            num_examples * m["loss"]
            for num_examples, m in metrics
        )
        / total_examples
    )


    # --------------------------------------------------
    # Save accuracy/loss
    # --------------------------------------------------

    if os.path.exists(METRICS_FILE):

        df_old = pd.read_csv(
            METRICS_FILE
        )

        next_round = (
            len(df_old) + 1
        )

    else:

        next_round = 1


    df = pd.DataFrame(
        [[
            next_round,
            weighted_accuracy,
            weighted_loss
        ]],
        columns=[
            "Round",
            "Accuracy",
            "Loss"
        ]
    )


    df.to_csv(
        METRICS_FILE,
        mode="a",
        header=not os.path.exists(
            METRICS_FILE
        ),
        index=False
    )


    print(
        f"Global Accuracy: "
        f"{weighted_accuracy:.4f}"
    )

    print(
        f"Global Loss: "
        f"{weighted_loss:.4f}"
    )


    return {
        "accuracy": weighted_accuracy,
        "loss": weighted_loss,
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    strategy = SaveModelStrategy(

        fraction_fit=1.0,

        fraction_evaluate=1.0,

        min_fit_clients=3,

        min_evaluate_clients=3,

        min_available_clients=3,

        evaluate_metrics_aggregation_fn=(
            evaluate_metrics_aggregation_fn
        ),
    )


    print(
        "Starting Flower Server..."
    )


    fl.server.start_server(

        server_address="0.0.0.0:8080",

        config=fl.server.ServerConfig(
            num_rounds=10
        ),

        strategy=strategy,
    )