import sys
import os
import time
import csv
import flwr as fl
import torch
from collections import OrderedDict
from model import ChestCNN, train, train_dp, test
from utils import load_hospital_data

from quantization import (
    quantize_parameters,
)

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


# Ensure 'src' package is importable regardless of where the script is run from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_PATH = os.environ.get("DATA_PATH", "./data/hospital_A")
SERVER_ADDRESS = os.environ.get("SERVER_ADDRESS", "localhost:8080")

# Get hospital/client name from environment
CLIENT_NAME = os.environ.get("CLIENT_NAME", "Hospital_A")

USE_QUANTIZATION = os.environ.get("USE_QUANTIZATION", "1") == "1"

USE_DP = os.environ.get(
    "USE_DP", "0"
) == "1"

DP_NOISE_MULTIPLIER = float(
    os.environ.get(
        "DP_NOISE_MULTIPLIER",
        "1.0"
    )
)

DP_MAX_GRAD_NORM = float(
    os.environ.get(
        "DP_MAX_GRAD_NORM",
        "1.0"
    )
)

DP_DELTA = float(
    os.environ.get(
        "DP_DELTA",
        "1e-5"
    )
)

# CSV file for recording experiments
if USE_DP:
    SUFFIX = "c_dp"
elif USE_QUANTIZATION:
    SUFFIX = "b_quantized"
else:
    SUFFIX = "a_pure"

SRC_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(os.path.dirname(SRC_DIR), "dashboard", "results", SUFFIX)

os.makedirs(RESULTS_DIR, exist_ok=True)
CSV_FILE = os.path.join(RESULTS_DIR, f"{CLIENT_NAME}_{SUFFIX}.csv")


# --------------------------------------------------
# Device
# --------------------------------------------------

FORCE_CPU = os.environ.get("FORCE_CPU", "0") == "1"
if FORCE_CPU:
    device = torch.device("cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[{CLIENT_NAME}] Using device: {device}")


# --------------------------------------------------
# Model
# --------------------------------------------------

net = ChestCNN().to(device)


# --------------------------------------------------
# Data
# --------------------------------------------------

if USE_DP:
    trainloader, testloader = load_hospital_data(DATA_PATH, batch_size=4)
    
    from opacus import PrivacyEngine
    import torch.optim as optim
    
    global_privacy_engine = PrivacyEngine()
    global_dp_optimizer = optim.Adam(net.parameters(), lr=0.001)
    
    make_private_result = global_privacy_engine.make_private(
        module=net,
        optimizer=global_dp_optimizer,
        data_loader=trainloader,
        noise_multiplier=DP_NOISE_MULTIPLIER,
        max_grad_norm=DP_MAX_GRAD_NORM,
    )

    if len(make_private_result) == 4:
        global_private_net, global_dp_optimizer, global_dp_trainloader, _ = make_private_result
    else:
        global_private_net, global_dp_optimizer, global_dp_trainloader = make_private_result

    # ============================================================
    # MONKEY-PATCH OPACUS CONTRACT TO FIX DEVICE MISMATCH
    # ============================================================
    import opacus.optimizers.optimizer
    if not hasattr(opacus.optimizers.optimizer, '_patched_contract'):
        old_contract = opacus.optimizers.optimizer.contract
        def new_contract(*args, **kwargs):
            if len(args) >= 3:
                a1, a2 = args[1], args[2]
                if isinstance(a1, torch.Tensor) and isinstance(a2, torch.Tensor):
                    if a1.device != a2.device:
                        # Move per_sample_clip_factor to the device of grad_sample
                        args = (args[0], a1.to(a2.device), a2) + args[3:]
            return old_contract(*args, **kwargs)
        opacus.optimizers.optimizer.contract = new_contract
        opacus.optimizers.optimizer._patched_contract = True
else:
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
            k: torch.tensor(v).to(device)
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
        global net
        
        self.set_parameters(parameters)
        net.to(device)
        if USE_DP:
            global_private_net.to(device)

        # ============================================================
        # GET ROUND NUMBER
        # ============================================================

        round_number = config.get(
            "server_round",
            0
        )

        print("\n" + "=" * 60)
        print(f"[{CLIENT_NAME}] Federated Round {round_number}")
        print("=" * 60)

        # ============================================================
        # TOTAL TRAINING TIMER
        # ============================================================

        total_training_start = time.perf_counter()

        # ============================================================
        # TRAINING CONFIGURATION
        # ============================================================

        NUM_EPOCHS = 2

        epoch_times = []

        epsilon = None

        # ============================================================
        # DIFFERENTIAL PRIVACY TRAINING
        # ============================================================

        if USE_DP:

            print(
                f"[{CLIENT_NAME}] "
                f"DP-SGD enabled"
            )

            print(
                f"[{CLIENT_NAME}] "
                f"Noise Multiplier: "
                f"{DP_NOISE_MULTIPLIER}"
            )

            print(
                f"[{CLIENT_NAME}] "
                f"Max Gradient Norm: "
                f"{DP_MAX_GRAD_NORM}"
            )

            print(
                f"[{CLIENT_NAME}] "
                f"Delta: "
                f"{DP_DELTA}"
            )

            # --------------------------------------------------------
            # Train ALL local epochs in ONE DP training session
            # --------------------------------------------------------

            epoch_start = time.perf_counter()

            epsilon = train_dp(
                private_net=global_private_net,
                optimizer=global_dp_optimizer,
                trainloader=global_dp_trainloader,
                privacy_engine=global_privacy_engine,
                epochs=NUM_EPOCHS,
                delta=DP_DELTA
            )

            epoch_end = time.perf_counter()

            epoch_time = (
                epoch_end -
                epoch_start
            )

            # Since DP training currently happens as one session,
            # record the complete DP training time.
            epoch_times.append(epoch_time)

            print(
                f"[{CLIENT_NAME}] "
                f"DP Training: "
                f"{NUM_EPOCHS} epochs | "
                f"{epoch_time:.2f} sec"
            )

            print(
                f"[{CLIENT_NAME}] "
                f"Privacy Budget: "
                f"epsilon={epsilon:.4f}, "
                f"delta={DP_DELTA}"
            )

        # ============================================================
        # NORMAL TRAINING
        # ============================================================

        else:

            for epoch in range(NUM_EPOCHS):

                epoch_start = time.perf_counter()

                train(
                    net,
                    trainloader,
                    epochs=1
                )

                epoch_end = time.perf_counter()

                epoch_time = (
                    epoch_end -
                    epoch_start
                )

                epoch_times.append(
                    epoch_time
                )

                print(
                    f"[{CLIENT_NAME}] "
                    f"Epoch {epoch + 1}/{NUM_EPOCHS}: "
                    f"{epoch_time:.2f} sec"
                )

        # ============================================================
        # TOTAL TRAINING TIME
        # ============================================================

        total_training_end = time.perf_counter()

        total_training_time = (
            total_training_end -
            total_training_start
        )

        # ============================================================
        # GET FP32 PARAMETERS
        # ============================================================

        fp32_parameters = self.get_parameters(
            config={}
        )

        # ============================================================
        # CALCULATE FP32 PAYLOAD
        # ============================================================

        original_payload_bytes = calculate_payload_size(
            fp32_parameters
        )

        original_payload_mb = (
            original_payload_bytes /
            (1024 * 1024)
        )

        # ============================================================
        # INT8 QUANTIZATION
        # ============================================================

        (
            quantized_parameters,
            quantized_payload_bytes,
            quantized_payload_mb,
            compression_ratio,
            reduction_percent
        ) = quantize_parameters(
            fp32_parameters
        )

        # ============================================================
        # PRINT RESULTS
        # ============================================================

        print(
            f"[{CLIENT_NAME}] "
            f"Total Training Time: "
            f"{total_training_time:.2f} sec"
        )

        print(
            f"[{CLIENT_NAME}] "
            f"FP32 Payload Size: "
            f"{original_payload_mb:.4f} MB"
        )

        print(
            f"[{CLIENT_NAME}] "
            f"INT8 Payload Size: "
            f"{quantized_payload_mb:.4f} MB"
        )

        print(
            f"[{CLIENT_NAME}] "
            f"Compression Ratio: "
            f"{compression_ratio:.2f}x"
        )

        print(
            f"[{CLIENT_NAME}] "
            f"Payload Reduction: "
            f"{reduction_percent:.2f}%"
        )

        # ============================================================
        # SAVE TRAINING RECORD
        # ============================================================

        with open(
            CSV_FILE,
            "a",
            newline=""
        ) as f:

            writer = csv.writer(f)

            for i, epoch_time in enumerate(
                epoch_times
            ):

                writer.writerow([

                    # Client
                    CLIENT_NAME,

                    # FL round
                    round_number,

                    # Epoch/training time
                    epoch_time,

                    # Total training time
                    total_training_time,

                    # FP32 payload
                    original_payload_bytes,
                    original_payload_mb,

                    # INT8 payload
                    quantized_payload_bytes,
                    quantized_payload_mb,

                    # Quantization
                    compression_ratio,
                    reduction_percent,

                    # Device
                    str(device),

                    # DP information
                    epsilon if epsilon is not None else "",
                    DP_DELTA if USE_DP else "",
                    DP_NOISE_MULTIPLIER if USE_DP else "",
                    DP_MAX_GRAD_NORM if USE_DP else ""
                ])

        # ============================================================
        # PARAMETERS TO SEND TO SERVER
        # ============================================================

        returned_parameters = (
            quantized_parameters
            if USE_QUANTIZATION
            else fp32_parameters
        )

        # ============================================================
        # RETURN TO FLOWER SERVER
        # ============================================================

        return (

            returned_parameters,

            len(trainloader.dataset),

            {

                # Training information
                "training_time": float(
                    total_training_time
                ),
                "client_name": str(CLIENT_NAME),
                "training_duration": float(total_training_time),

                # FP32 communication
                "payload_size_mb": float(
                    original_payload_mb
                ),

                # INT8 communication
                "quantized_payload_mb": float(
                    quantized_payload_mb
                ),

                # Compression
                "compression_ratio": float(
                    compression_ratio
                ),

                "payload_reduction_percent": float(
                    reduction_percent
                ),

                # Differential Privacy
                "epsilon": float(
                    epsilon
                ) if epsilon is not None else -1.0,

                "delta": float(
                    DP_DELTA
                ) if USE_DP else -1.0,

                "dp_noise_multiplier": float(
                    DP_NOISE_MULTIPLIER
                ) if USE_DP else 0.0,

                "dp_max_grad_norm": float(
                    DP_MAX_GRAD_NORM
                ) if USE_DP else 0.0
            }
        )


    # --------------------------------------------------
    # EVALUATE
    # --------------------------------------------------

    def evaluate(self, parameters, config):
        from sklearn.metrics import precision_score, recall_score, f1_score

        self.set_parameters(parameters)

        round_number = config.get("server_round", 0)

        evaluation_start = time.perf_counter()

        # Run standard loss/accuracy test
        loss, accuracy = test(
            net,
            testloader
        )

        # --------------------------------------------------
        # Collect predictions for F1 / Precision / Recall
        # --------------------------------------------------
        net.eval()
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for images, labels in testloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        import numpy as np
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)

        precision = precision_score(
            all_labels, all_predictions,
            average="binary", zero_division=0
        )
        recall = recall_score(
            all_labels, all_predictions,
            average="binary", zero_division=0
        )
        f1 = f1_score(
            all_labels, all_predictions,
            average="binary", zero_division=0
        )

        evaluation_time = (
            time.perf_counter() -
            evaluation_start
        )

        print(
            f"[{CLIENT_NAME}] "
            f"Round {round_number} Evaluation -> "
            f"Loss: {loss:.4f}, "
            f"Accuracy: {accuracy:.4f}, "
            f"F1: {f1:.4f}, "
            f"Precision: {precision:.4f}, "
            f"Recall: {recall:.4f}"
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
                "accuracy":         float(accuracy),
                "loss":             float(loss),
                "f1":               float(f1),
                "precision":        float(precision),
                "recall":           float(recall),
                "evaluation_time":  float(evaluation_time)
            }
        )


# --------------------------------------------------
# Start Client
# --------------------------------------------------

if __name__ == "__main__":

    fl.client.start_client(
        server_address=SERVER_ADDRESS,
        client=HospitalClient().to_client(),
        grpc_max_message_length=1024*1024*1024
    )