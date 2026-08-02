# import flwr as fl
# import os
# import pandas as pd

# # Path to log metrics for Streamlit
# # METRICS_FILE = "dashboard/metrics.csv"
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# METRICS_FILE = os.path.join(BASE_DIR, "dashboard", "metrics.csv")
# if os.path.exists(METRICS_FILE):
#     os.remove(METRICS_FILE)

# # Define custom evaluation metric aggregation
# # def evaluate_metrics_aggregation_fn(metrics):
# #     accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
# #     examples = [num_examples for num_examples, _ in metrics]
    
# #     # Calculate weighted average accuracy
# #     weighted_acc = sum(accuracies) / sum(examples)
    
# #     # Determine the next round number
# #     if os.path.exists(METRICS_FILE):
# #         existing_df = pd.read_csv(METRICS_FILE)
# #         next_round = len(existing_df) + 1
# #     else:
# #         next_round = 1

# #     # Append to CSV for real-time visual tracking
# #     df = pd.DataFrame([[next_round, weighted_acc]], columns=["Round", "Accuracy"])
# #     df.to_csv(METRICS_FILE, mode='a', header=not os.path.exists(METRICS_FILE), index=False)
    
# #     return {"accuracy": weighted_acc}
# def evaluate_metrics_aggregation_fn(metrics):

#     total_examples = sum(num_examples for num_examples, _ in metrics)

#     weighted_accuracy = (
#         sum(num_examples * m["accuracy"] for num_examples, m in metrics)
#         / total_examples
#     )

#     weighted_loss = (
#         sum(num_examples * m["loss"] for num_examples, m in metrics)
#         / total_examples
#     )

#     if os.path.exists(METRICS_FILE):
#         df_old = pd.read_csv(METRICS_FILE)
#         next_round = len(df_old) + 1
#     else:
#         next_round = 1

#     df = pd.DataFrame(
#         [[next_round, weighted_accuracy, weighted_loss]],
#         columns=["Round", "Accuracy", "Loss"],
#     )

#     df.to_csv(
#         METRICS_FILE,
#         mode="a",
#         header=not os.path.exists(METRICS_FILE),
#         index=False,
#     )

#     return {
#         "accuracy": weighted_accuracy,
#         "loss": weighted_loss,
#     }

# # Start Flower Server
# if __name__ == "__main__":
#     strategy = fl.server.strategy.FedAvg(
#         fraction_fit=1.0,             # Train with all available clients every round
#         fraction_evaluate=1.0,        # Evaluate with all available clients
#         min_fit_clients=3,            # Wait for at least 3 nodes to connect
#         min_evaluate_clients=3,
#         min_available_clients=3,
#         evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
#     )
    
#     print("Starting Flower Federated Server on port 8080...")
#     fl.server.start_server(
#         server_address="0.0.0.0:8080",
#         config=fl.server.ServerConfig(num_rounds=10), # 5 global optimization rounds chnge to 10
#         strategy=strategy,
#     )
import flwr as fl
import os
import pandas as pd
import torch

from collections import OrderedDict

from flwr.common import parameters_to_ndarrays
from model import ChestCNN

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

METRICS_FILE = os.path.join(BASE_DIR, "dashboard", "metrics.csv")

MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "global_model.pth")

if os.path.exists(METRICS_FILE):
    os.remove(METRICS_FILE)


class SaveModelStrategy(fl.server.strategy.FedAvg):

    def aggregate_fit(
        self,
        server_round,
        results,
        failures,
    ):

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round,
            results,
            failures,
        )

        if aggregated_parameters is not None:

            model = ChestCNN()

            params = parameters_to_ndarrays(aggregated_parameters)

            params_dict = zip(model.state_dict().keys(), params)

            state_dict = OrderedDict(
                {
                    k: torch.tensor(v)
                    for k, v in params_dict
                }
            )

            model.load_state_dict(state_dict, strict=True)

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

        return aggregated_parameters, aggregated_metrics


def evaluate_metrics_aggregation_fn(metrics):

    total_examples = sum(num_examples for num_examples, _ in metrics)

    weighted_accuracy = (
        sum(num_examples * m["accuracy"] for num_examples, m in metrics)
        / total_examples
    )

    weighted_loss = (
        sum(num_examples * m["loss"] for num_examples, m in metrics)
        / total_examples
    )

    if os.path.exists(METRICS_FILE):

        df_old = pd.read_csv(METRICS_FILE)

        next_round = len(df_old) + 1

    else:

        next_round = 1

    df = pd.DataFrame(
        [[next_round, weighted_accuracy, weighted_loss]],
        columns=[
            "Round",
            "Accuracy",
            "Loss",
        ],
    )

    df.to_csv(
        METRICS_FILE,
        mode="a",
        header=not os.path.exists(METRICS_FILE),
        index=False,
    )

    return {
        "accuracy": weighted_accuracy,
        "loss": weighted_loss,
    }


if __name__ == "__main__":

    strategy = SaveModelStrategy(

        fraction_fit=1.0,

        fraction_evaluate=1.0,

        min_fit_clients=3,

        min_evaluate_clients=3,

        min_available_clients=3,

        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,

    )

    print("Starting Flower Server...")

    fl.server.start_server(

        server_address="0.0.0.0:8080",

        config=fl.server.ServerConfig(
            num_rounds=10,
        ),

        strategy=strategy,

    )