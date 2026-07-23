import flwr as fl
import os
import pandas as pd

# Path to log metrics for Streamlit
METRICS_FILE = "dashboard/metrics.csv"
if os.path.exists(METRICS_FILE):
    os.remove(METRICS_FILE)

# Define custom evaluation metric aggregation
def evaluate_metrics_aggregation_fn(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    # Calculate weighted average accuracy
    weighted_acc = sum(accuracies) / sum(examples)
    
    # Determine the next round number
    if os.path.exists(METRICS_FILE):
        existing_df = pd.read_csv(METRICS_FILE)
        next_round = len(existing_df) + 1
    else:
        next_round = 1

    # Append to CSV for real-time visual tracking
    df = pd.DataFrame([[next_round, weighted_acc]], columns=["Round", "Accuracy"])
    df.to_csv(METRICS_FILE, mode='a', header=not os.path.exists(METRICS_FILE), index=False)
    
    return {"accuracy": weighted_acc}

# Start Flower Server
if __name__ == "__main__":
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,             # Train with all available clients every round
        fraction_evaluate=1.0,        # Evaluate with all available clients
        min_fit_clients=3,            # Wait for at least 3 nodes to connect
        min_evaluate_clients=3,
        min_available_clients=3,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    )
    
    print("Starting Flower Federated Server on port 8080...")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5), # 5 global optimization rounds
        strategy=strategy,
    )