import sys
import os
import time
import csv

# Ensure 'src' package is importable regardless of where the script is run from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import flwr as fl
import torch
from collections import OrderedDict
from model import ChestCNN, train, test
from utils import load_hospital_data


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_PATH = os.environ.get("DATA_PATH", "./data/hospital_A")
SERVER_ADDRESS = os.environ.get("SERVER_ADDRESS", "localhost:8080")

# Get hospital/client name from environment
CLIENT_NAME = os.environ.get("CLIENT_NAME", "Hospital_A")

# CSV file for recording experiments
CSV_FILE = f"{CLIENT_NAME}_baseline.csv"


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[{CLIENT_NAME}] Using device: {device}")


# --------------------------------------------------
# Model
# --------------------------------------------------

net = ChestCNN().to(device)


# --------------------------------------------------
# Data
# --------------------------------------------------

trainloader, testloader = load_hospital_data(DATA_PATH)


# --------------------------------------------------
# CSV Setup
# --------------------------------------------------

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "client",
            "round",
            "epoch_time_sec",
            "training_time_sec",
            "payload_size_bytes",
            "payload_size_mb",
            "accuracy",
            "loss",
            "device"
        ])


# --------------------------------------------------
# Helper: Calculate model payload size
# --------------------------------------------------

def calculate_payload_size(parameters):

    total_bytes = 0

    for param in parameters:
        total_bytes += param.nbytes

    return total_bytes


# --------------------------------------------------
# Flower Client
# --------------------------------------------------

class HospitalClient(fl.client.NumPyClient):

    def get_parameters(self, config):

        return [
            val.cpu().numpy()
            for _, val in net.state_dict().items()
        ]


    def set_parameters(self, parameters):

        params_dict = zip(
            net.state_dict().keys(),
            parameters
        )

        state_dict = OrderedDict({
            k: torch.tensor(v)
            for k, v in params_dict
        })

        net.load_state_dict(
            state_dict,
            strict=True
        )


    # --------------------------------------------------
    # FIT
    # --------------------------------------------------

    def fit(self, parameters, config):

        self.set_parameters(parameters)

        # Get round number from server config
        round_number = config.get("server_round", 0)

        print("\n" + "=" * 50)
        print(f"[{CLIENT_NAME}] Federated Round {round_number}")
        print("=" * 50)

        # ----------------------------------------------
        # Total training timer
        # ----------------------------------------------

        total_training_start = time.perf_counter()

        # ----------------------------------------------
        # Epoch timing
        # ----------------------------------------------

        epoch_times = []

        # Your current training = 2 epochs
        NUM_EPOCHS = 2

        for epoch in range(NUM_EPOCHS):

            epoch_start = time.perf_counter()

            # Train for one epoch
            train(
                net,
                trainloader,
                epochs=1
            )

            epoch_end = time.perf_counter()

            epoch_time = epoch_end - epoch_start

            epoch_times.append(epoch_time)

            print(
                f"[{CLIENT_NAME}] "
                f"Epoch {epoch + 1}/{NUM_EPOCHS}: "
                f"{epoch_time:.2f} sec"
            )

        total_training_end = time.perf_counter()

        total_training_time = (
            total_training_end -
            total_training_start
        )

        # ----------------------------------------------
        # Get model parameters
        # ----------------------------------------------

        parameters = self.get_parameters(config={})

        # ----------------------------------------------
        # Calculate payload size
        # ----------------------------------------------

        payload_bytes = calculate_payload_size(parameters)

        payload_mb = payload_bytes / (1024 * 1024)

        print(
            f"[{CLIENT_NAME}] "
            f"Total Training Time: "
            f"{total_training_time:.2f} sec"
        )

        print(
            f"[{CLIENT_NAME}] "
            f"Payload Size: "
            f"{payload_mb:.2f} MB"
        )

        # ----------------------------------------------
        # Save training record
        # ----------------------------------------------

        with open(CSV_FILE, "a", newline="") as f:

            writer = csv.writer(f)

            for i, epoch_time in enumerate(epoch_times):

                writer.writerow([
                    CLIENT_NAME,
                    round_number,
                    epoch_time,
                    total_training_time,
                    payload_bytes,
                    payload_mb,
                    "",
                    "",
                    str(device)
                ])

        return (
            parameters,
            len(trainloader.dataset),
            {
                "training_time": float(total_training_time),
                "payload_size_mb": float(payload_mb)
            }
        )


    # --------------------------------------------------
    # EVALUATE
    # --------------------------------------------------

    def evaluate(self, parameters, config):

        self.set_parameters(parameters)

        round_number = config.get("server_round", 0)

        evaluation_start = time.perf_counter()

        loss, accuracy = test(
            net,
            testloader
        )

        evaluation_time = (
            time.perf_counter() -
            evaluation_start
        )

        print(
            f"[{CLIENT_NAME}] "
            f"Round {round_number} "
            f"Evaluation → "
            f"Loss: {loss:.4f}, "
            f"Accuracy: {accuracy:.4f}"
        )

        # Save evaluation result
        with open(CSV_FILE, "a", newline="") as f:

            writer = csv.writer(f)

            writer.writerow([
                CLIENT_NAME,
                round_number,
                "",
                "",
                "",
                "",
                float(accuracy),
                float(loss),
                str(device)
            ])

        return (
            float(loss),
            len(testloader.dataset),
            {
                "accuracy": float(accuracy),
                "loss": float(loss),
                "evaluation_time": float(evaluation_time)
            }
        )


# --------------------------------------------------
# Start Client
# --------------------------------------------------

if __name__ == "__main__":

    fl.client.start_numpy_client(
        server_address=SERVER_ADDRESS,
        client=HospitalClient()
    )