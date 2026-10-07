"""
dacm.py
=======
Distribution-Aware Class-weighted Mitigation (DACM) Engine.

Provides fully federated, dynamic recovery after client unlearning/exclusion:
1. Dynamic surviving-client detection (no hardcoded hospital names).
2. Local class-count metadata calculation on client side (NO raw images transmitted).
3. Server-side DACM distribution audit using the logarithmic compensation formula:
       w = 1.0 + alpha * ln(tau_safe / gamma_pneumonia) if gamma < tau_safe else 1.0
4. Independent local recovery training on surviving clients using:
       criterion = nn.CrossEntropyLoss(weight=dacm_weights)
5. Federated averaging (FedAvg) aggregation of local models into the recovered global model.
"""

import os
import sys
import copy
import argparse
from pathlib import Path
from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np

# Ensure src is importable
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from paths import DATA_DIR, MODELS_DIR
from model import ChestCNN


# ----------------------------------------------------------------------
# 1. Dynamic Surviving-Client Detection
# ----------------------------------------------------------------------

def discover_clients(excluded_clients=None) -> list[str]:
    """
    Dynamically discover eligible client IDs.
    
    Checks environment variable KNOWN_CLIENTS first; if not set,
    inspects DATA_DIR for available hospital directories.
    Filters out any client in excluded_clients.
    
    Returns a sorted list of surviving client IDs.
    """
    if excluded_clients is None:
        excluded_set = set()
    elif isinstance(excluded_clients, str):
        excluded_set = {excluded_clients.strip().lower()}
    else:
        excluded_set = {str(c).strip().lower() for c in excluded_clients}

    detected = set()

    # Priority 1: Check environment variable KNOWN_CLIENTS
    env_clients = os.environ.get("KNOWN_CLIENTS")
    if env_clients:
        for c in env_clients.split(","):
            c = c.strip()
            if c:
                detected.add(c)
    else:
        # Priority 2: Auto-detect from DATA_DIR
        if DATA_DIR.exists():
            for p in DATA_DIR.iterdir():
                if p.is_dir() and (p / "train").exists():
                    name = p.name
                    parts = name.split("_")
                    if len(parts) == 2 and parts[0].lower() == "hospital":
                        canonical = f"Hospital_{parts[1].upper()}"
                    else:
                        canonical = name
                    detected.add(canonical)

    if not detected:
        # Safe fallback default
        detected = {"Hospital_A", "Hospital_B", "Hospital_C"}

    survivors = [c for c in sorted(detected) if c.strip().lower() not in excluded_set]
    return survivors


# ----------------------------------------------------------------------
# 2. Dynamic Local Class Counting (Client-Side Metadata Only)
# ----------------------------------------------------------------------

def get_client_class_counts(client_id: str, trainloader=None) -> dict:
    """
    Locally inspect the client's dataset to count training samples by class.
    
    Returns class-count metadata only:
        {"NORMAL": int, "PNEUMONIA": int, "TOTAL": int}
        
    PRIVACY GUARANTEE:
    Medical images remain strictly local on the client. Only the metadata
    counts are communicated to the server.
    """
    if trainloader is None:
        from utils import load_partitions
        trainloader, _, _ = load_partitions(client_id=client_id)

    dataset = trainloader.dataset

    # Fast O(N) integer lookup without decoding/loading image pixels
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices") and hasattr(dataset.dataset, "targets"):
        targets = [dataset.dataset.targets[i] for i in dataset.indices]
        normal_count = sum(1 for t in targets if t == 0)
        pneumonia_count = sum(1 for t in targets if t == 1)
    elif hasattr(dataset, "targets"):
        normal_count = sum(1 for t in dataset.targets if t == 0)
        pneumonia_count = sum(1 for t in dataset.targets if t == 1)
    else:
        # Robust fallback: iterate batch labels directly
        normal_count = 0
        pneumonia_count = 0
        for _, batch_labels in trainloader:
            normal_count += int(torch.sum(batch_labels == 0).item())
            pneumonia_count += int(torch.sum(batch_labels == 1).item())

    total = normal_count + pneumonia_count
    return {
        "NORMAL": int(normal_count),
        "PNEUMONIA": int(pneumonia_count),
        "TOTAL": int(total),
    }


# ----------------------------------------------------------------------
# 3. Server DACM Distribution Audit & Weight Calculation
# ----------------------------------------------------------------------

def calculate_dacm_weights(surviving_counts: dict, tau_safe: float = 0.50, alpha: float = 3.0) -> torch.Tensor:
    """
    Novelty Step 1 & 2: Audit surviving distribution and calculate compensatory weight.
    
    Aggregates metadata across surviving clients:
        total_normal = sum(...)
        total_pneumonia = sum(...)
        total = total_normal + total_pneumonia
        gamma_pneumonia = total_pneumonia / total
        
    DACM logarithmic formula:
        if gamma_pneumonia < tau_safe:
            w = 1.0 + alpha * ln(tau_safe / gamma_pneumonia)
        else:
            w = 1.0
            
    Returns:
        torch.Tensor([1.0, float(w)], dtype=torch.float32)
    """
    total_normal = sum(c["NORMAL"] for c in surviving_counts.values())
    total_pneumonia = sum(c["PNEUMONIA"] for c in surviving_counts.values())
    total = total_normal + total_pneumonia

    if total == 0:
        print("[DACM Warning] Total surviving samples is 0. Returning default weights [1.0, 1.0].")
        return torch.tensor([1.0, 1.0], dtype=torch.float32)

    gamma_pneumonia = total_pneumonia / total
    print(f"\n[DACM Audit] Surviving Normal Samples    : {total_normal}")
    print(f"[DACM Audit] Surviving Pneumonia Samples : {total_pneumonia}")
    print(f"[DACM Audit] Surviving Total Samples     : {total}")
    print(f"[DACM Audit] Surviving Pneumonia Ratio   : {gamma_pneumonia*100:.2f}% (Safety Threshold: {tau_safe*100:.2f}%)")

    if gamma_pneumonia < tau_safe:
        # The DACM Logarithmic Compensation Formula
        w = 1.0 + alpha * np.log(tau_safe / gamma_pneumonia)
        print(f"[DACM Trigger] Minority class shortage detected (< {tau_safe*100:.1f}%).")
        print(f"[DACM Trigger] Transmitting compensatory loss weight: {w:.4f}")
        return torch.tensor([1.0, float(w)], dtype=torch.float32)

    print("[DACM Audit] Surviving distribution is balanced (>= safety threshold). Weight remains 1.0.")
    return torch.tensor([1.0, 1.0], dtype=torch.float32)


# Backward compatibility alias
calculate_dacu_weights = calculate_dacm_weights


# ----------------------------------------------------------------------
# 4. Client Independent Local Recovery Training
# ----------------------------------------------------------------------

def train_local_recovery(
    local_model: nn.Module,
    trainloader,
    dacm_weights: torch.Tensor,
    epochs: int = 2,
    lr: float = 1e-3,
    device = None,
    client_id: str = "Client",
) -> tuple[nn.Module, int]:
    """
    Train a LOCAL COPY of the model for DACM recovery.
    
    Each surviving client trains independently from the SAME starting model
    using nn.CrossEntropyLoss(weight=dacm_weights) and SGD.
    
    Returns:
        (trained_local_model, n_samples)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    local_model.to(device)
    local_model.train()

    criterion = nn.CrossEntropyLoss(weight=dacm_weights.to(device))
    optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)

    n_samples = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = local_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1
            if epoch == 0:
                n_samples += labels.size(0)

        avg_loss = epoch_loss / max(batch_count, 1)
        print(f"  [DACM Local Training] {client_id:<12} | Epoch {epoch+1}/{epochs} | "
              f"avg_loss={avg_loss:.4f} | samples={n_samples}")

    return local_model, n_samples


# ----------------------------------------------------------------------
# 5. Server Weighted FedAvg Aggregation
# ----------------------------------------------------------------------

def fedavg_aggregate(
    local_models_with_sizes: list,
    base_model: nn.Module,
    device = None,
) -> nn.Module:
    """
    Federated Averaging (FedAvg) aggregation weighted by sample counts:
        theta_recovered = sum_k (n_k / N) * theta_k
        
    Args:
        local_models_with_sizes : list of (model, n_samples) tuples
        base_model              : reference model structure
        device                  : target device
        
    Returns:
        New model instance with aggregated weights.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_samples = sum(n for _, n in local_models_with_sizes)
    if total_samples == 0:
        print("[WARNING] FedAvg: total_samples=0, returning base model clone.")
        return copy.deepcopy(base_model)

    agg_state = OrderedDict()
    ref_state = base_model.state_dict()
    for key in ref_state:
        agg_state[key] = torch.zeros_like(ref_state[key], dtype=torch.float32)

    for local_model, n in local_models_with_sizes:
        weight = n / total_samples
        local_state = local_model.state_dict()
        for key in agg_state:
            agg_state[key] += weight * local_state[key].float()

    aggregated = copy.deepcopy(base_model)
    aggregated.load_state_dict(
        {k: v.to(ref_state[k].dtype) for k, v in agg_state.items()},
        strict=True,
    )
    aggregated.to(device)
    return aggregated


# ----------------------------------------------------------------------
# 6. Full Dynamic Federated DACM Recovery Orchestrator
# ----------------------------------------------------------------------

def execute_dacm_recovery(
    unlearned_model: nn.Module = None,
    unlearned_model_path: str = None,
    excluded_client: str = None,
    surviving_clients: list = None,
    active_clients: set = None,
    tau_safe: float = 0.50,
    alpha: float = 3.0,
    epochs: int = 2,
    lr: float = 1e-3,
    device = None,
    save_path: str = None,
) -> tuple[nn.Module, torch.Tensor, dict, list]:
    """
    Execute fully federated and dynamic DACM recovery.
    
    Flow:
        1. Dynamically identify surviving clients (server exclusion state).
        2. Obtain local class count metadata from each client (no raw images).
        3. Server audits distribution and calculates DACM weight.
        4. Each surviving client starts from the SAME starting model and trains
           independently with nn.CrossEntropyLoss(weight=dacm_weights).
        5. Server aggregates local updates with sample-weighted FedAvg.
        6. Recovered global model is saved.
        
    Returns:
        (recovered_model, dacm_weights, surviving_counts, surviving_clients)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Resolve starting / unlearned model
    if unlearned_model is None:
        if unlearned_model_path is None:
            # Fallback search order
            candidates = [
                MODELS_DIR / f"unlearned_{excluded_client}_b_quantized.pth" if excluded_client else None,
                MODELS_DIR / "unlearned_baseline_model.pth",
                MODELS_DIR / "global_model_b_quantized.pth",
                MODELS_DIR / "global_model_a_pure.pth",
            ]
            for c in candidates:
                if c and c.exists():
                    unlearned_model_path = str(c)
                    break

        if not unlearned_model_path or not Path(unlearned_model_path).exists():
            raise FileNotFoundError(f"Unlearned model checkpoint not found at: {unlearned_model_path}")

        print(f"[DACM] Loading starting model from: {unlearned_model_path}")
        unlearned_model = ChestCNN().to(device)
        unlearned_model.load_state_dict(
            torch.load(unlearned_model_path, map_location=device)
        )
    else:
        unlearned_model.to(device)

    # 2. Dynamic surviving-client detection
    if surviving_clients is None:
        if active_clients is not None:
            excluded_norm = {excluded_client.strip().lower()} if excluded_client else set()
            surviving_clients = [c for c in sorted(active_clients) if c.strip().lower() not in excluded_norm]
        else:
            surviving_clients = discover_clients(excluded_clients=[excluded_client] if excluded_client else None)

    if not surviving_clients:
        raise ValueError(f"No surviving clients found after excluding '{excluded_client}'!")

    print("\n" + "=" * 60)
    print("Initiating Federated Dynamic DACM Recovery")
    print("=" * 60)
    print(f"[DACM] Excluded Client    : {excluded_client or 'None'}")
    print(f"[DACM] Surviving Clients   : {surviving_clients}")
    print(f"[DACM] Target Device       : {device}")
    print(f"[DACM] Recovery Epochs     : {epochs} (independent per client)")
    print(f"[DACM] Learning Rate       : {lr} (SGD)")

    # 3. Dynamic metadata acquisition from surviving clients (NO RAW IMAGES TO SERVER)
    from utils import load_partitions

    surviving_counts = {}
    surviving_loaders = {}

    print("\n[DACM] Collecting class-count metadata from surviving clients...")
    for cid in surviving_clients:
        tr, _, _ = load_partitions(client_id=cid)
        counts = get_client_class_counts(cid, trainloader=tr)
        surviving_counts[cid] = counts
        surviving_loaders[cid] = tr
        print(f"  [Client Metadata] {cid:<12} -> NORMAL: {counts['NORMAL']:<5} "
              f"PNEUMONIA: {counts['PNEUMONIA']:<5} TOTAL: {counts['TOTAL']:<5} (images remain local)")

    # 4. Server DACM distribution audit
    dacm_weights = calculate_dacm_weights(surviving_counts, tau_safe=tau_safe, alpha=alpha)

    # 5. Independent parallel/local recovery training (NO sequential training)
    print("\n[DACM] Launching independent local recovery on surviving clients...")
    local_models_with_sizes = []

    for cid in surviving_clients:
        print(f"\n[DACM] -- Client: {cid} --")
        local_clone = copy.deepcopy(unlearned_model)
        trained_local, n_samples = train_local_recovery(
            local_model=local_clone,
            trainloader=surviving_loaders[cid],
            dacm_weights=dacm_weights,
            epochs=epochs,
            lr=lr,
            device=device,
            client_id=cid,
        )
        local_models_with_sizes.append((trained_local, n_samples))

    # 6. Server FedAvg aggregation
    print(f"\n[DACM] FedAvg aggregating updates from {len(local_models_with_sizes)} surviving clients...")
    recovered_model = fedavg_aggregate(local_models_with_sizes, unlearned_model, device=device)
    print("[DACM] Federated aggregation complete.")

    # 7. Save the recovered model
    if save_path is None:
        if excluded_client:
            save_path = str(MODELS_DIR / f"dacm_recovered_{excluded_client}.pth")
        else:
            save_path = str(MODELS_DIR / "dacm_compensated_model.pth")

    save_path_obj = Path(save_path)
    save_path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(recovered_model.state_dict(), str(save_path_obj))
    print(f"\n[SUCCESS] Recovered model saved to: {save_path_obj}")

    # Compatibility checkpoints
    if excluded_client and excluded_client.lower() == "hospital_b":
        compat_path = MODELS_DIR / "dacm_compensated_model.pth"
        torch.save(recovered_model.state_dict(), str(compat_path))
        compat_dacu_path = MODELS_DIR / "dacu_compensated_model.pth"
        torch.save(recovered_model.state_dict(), str(compat_dacu_path))
        print(f"[SUCCESS] Compatibility checkpoints updated: {compat_path}, {compat_dacu_path}")

    return recovered_model, dacm_weights, surviving_counts, surviving_clients


# Backward compatibility alias
execute_dacu_recovery = execute_dacm_recovery


# ----------------------------------------------------------------------
# CLI Entry Point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dynamic Federated DACM Recovery Protocol"
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default="Hospital_B",
        help="Client ID to exclude/unlearn (e.g. Hospital_B or Hospital_A). Default: Hospital_B.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to starting unlearned model checkpoint. Default: auto-detected.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of local recovery epochs per surviving client (default: 2).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Local recovery learning rate (default: 1e-3).",
    )
    parser.add_argument(
        "--tau-safe",
        type=float,
        default=0.50,
        help="DACM minority class safety threshold (default: 0.50).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=3.0,
        help="DACM logarithmic penalty factor (default: 3.0).",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Output checkpoint file path for the recovered model.",
    )

    args = parser.parse_args()

    execute_dacm_recovery(
        excluded_client=args.exclude,
        unlearned_model_path=args.model_path,
        tau_safe=args.tau_safe,
        alpha=args.alpha,
        epochs=args.epochs,
        lr=args.lr,
        save_path=args.save_path,
    )
