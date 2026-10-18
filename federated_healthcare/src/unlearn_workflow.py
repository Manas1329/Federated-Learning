"""
unlearn_workflow.py
===================
Post-training unlearning orchestrator.

Called by unlearn_server.py after admin approval.

Pipeline:
    1. Load global model + all data
    2. Diagnostics: evaluate BEFORE unlearning
    3. Gradient-ascent unlearning on target client (conservative defaults)
    4. Diagnostics: evaluate AFTER unlearning + parameter distance
    5. Remove target from active registry
    6. DACM audit of surviving clients (existing formula, unchanged)
    7. If shortage → proper FedAvg recovery (independent local training + aggregation)
    8. Three-stage comparison table
    9. Save recovered model

Environment variables (all optional):
    UNLEARN_LR             default 1e-4   (conservative — avoids collapse)
    UNLEARN_STEPS          default 2      (conservative)
    DACM_RECOVERY_EPOCHS   default 2
    DACM_RECOVERY_LR       default 1e-3   (SGD, matches original run_dacu.py)
    DACM_TAU_SAFE          default 0.50
    DACM_ALPHA             default 3.0
    KNOWN_CLIENTS          default Hospital_A,Hospital_B,Hospital_C
"""

import os
import sys
import copy
import torch
import numpy as np
from pathlib import Path
from collections import OrderedDict

# Ensure src is importable regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import MODELS_DIR
# Heavy imports (flwr-chain) are deferred into run_unlearn_workflow()
# so unlearn_server.py starts without touching tensorflow/protobuf.


# ------------------------------------------------------------------
# Constants — all overridable via environment variables
# ------------------------------------------------------------------
KNOWN_CLIENTS: set = {
    c.strip()
    for c in os.environ.get(
        "KNOWN_CLIENTS", "Hospital_A,Hospital_B,Hospital_C"
    ).split(",")
    if c.strip()
}

DACM_TAU_SAFE:    float = float(os.environ.get("DACM_TAU_SAFE", "0.50"))
DACM_ALPHA:       float = float(os.environ.get("DACM_ALPHA", "3.0"))
RECOVERY_EPOCHS:  int   = int(os.environ.get("DACM_RECOVERY_EPOCHS", "2"))
RECOVERY_LR:      float = float(os.environ.get("DACM_RECOVERY_LR", "1e-3"))

# Conservative defaults — avoids collapsing the global model
UNLEARN_LR:       float = float(os.environ.get("UNLEARN_LR", "1e-4"))
UNLEARN_STEPS:    int   = int(os.environ.get("UNLEARN_STEPS", "2"))

USE_DP           = os.environ.get("USE_DP", "0") == "1"
USE_QUANTIZATION = os.environ.get("USE_QUANTIZATION", "1") == "1"

if USE_DP:
    SUFFIX = "c_dp"
elif USE_QUANTIZATION:
    SUFFIX = "b_quantized"
else:
    SUFFIX = "a_pure"


# ==================================================================
# Internal helper functions
# ==================================================================

def _print_banner(title: str):
    print("\n" + "=" * 62)
    print(f"  {title}")
    print("=" * 62)


def _count_classes(trainloader) -> dict:
    """
    Count NORMAL (label=0) and PNEUMONIA (label=1) samples.
    ImageFolder sorts folders alphabetically: NORMAL=0, PNEUMONIA=1.
    """
    normal_count = 0
    pneumonia_count = 0
    for _, labels in trainloader:
        normal_count    += int(torch.sum(labels == 0).item())
        pneumonia_count += int(torch.sum(labels == 1).item())
    return {"NORMAL": normal_count, "PNEUMONIA": pneumonia_count}


def _evaluate_model_object(model, test_loaders: list, device) -> tuple:
    """
    Evaluate a model object (not a path) on a list of DataLoaders.
    Returns (accuracy, precision, recall, f1, confusion_matrix).
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix
    )

    model.eval()
    model.to(device)
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for loader in test_loaders:
            for images, labels in loader:
                images = images.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec  = recall_score(all_labels, all_preds, zero_division=0)
    f1   = f1_score(all_labels, all_preds, zero_division=0)
    cm   = confusion_matrix(all_labels, all_preds)
    return acc, prec, rec, f1, cm


def _get_test_loaders(client_ids: list) -> list:
    """Return a list of test DataLoaders for the given client IDs."""
    from utils import load_partitions
    loaders = []
    for cid in client_ids:
        try:
            _, testloader, _ = load_partitions(client_id=cid)
            loaders.append(testloader)
        except Exception as exc:
            print(f"[WARNING] Could not load test data for {cid}: {exc}")
    return loaders


def _print_eval_results(label: str, acc, prec, rec, f1, cm):
    print(f"\n  {'─'*52}")
    print(f"  {label}")
    print(f"  {'─'*52}")
    print(f"  Accuracy   : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision  : {prec:.4f}")
    print(f"  Recall     : {rec:.4f}")
    print(f"  F1 Score   : {f1:.4f}")
    print(f"  Confusion Matrix:\n{cm}")


def _compute_param_distance(state_a: dict, state_b: dict) -> float:
    """L2 norm of (state_a − state_b) across all parameters."""
    total = 0.0
    for key in state_a:
        diff = state_a[key].float() - state_b[key].float()
        total += diff.norm().item() ** 2
    return total ** 0.5


def _clone_model(source_model) -> "ChestCNN":
    """Deep-copy a model (weights + architecture) without touching the original."""
    return copy.deepcopy(source_model)


def _train_local_recovery(
    local_model,
    trainloader,
    weight_tensor: torch.Tensor,
    epochs: int,
    lr: float,
    device,
    client_id: str,
) -> tuple:
    """
    Train a LOCAL COPY of the model for DACM recovery.

    Uses SGD + weighted CrossEntropyLoss (matching original run_dacu.py).
    Returns (trained_local_model, n_train_samples).
    """
    import torch.nn as nn

    local_model.to(device)
    local_model.train()

    criterion = nn.CrossEntropyLoss(weight=weight_tensor.to(device))
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
        print(f"  [DACM] {client_id}  Epoch {epoch+1}/{epochs}  "
              f"avg_loss={avg_loss:.4f}  samples={n_samples}")

    return local_model, n_samples


def _fedavg_aggregate(local_models_with_sizes: list, base_model, device) -> "ChestCNN":
    """
    Weighted FedAvg aggregation.

    Args:
        local_models_with_sizes : list of (model, n_samples) tuples
        base_model              : used only for structure (state_dict keys)
        device                  : target device

    Returns a new model with the aggregated state dict.
    """
    total_samples = sum(n for _, n in local_models_with_sizes)
    if total_samples == 0:
        print("[WARNING] FedAvg: total_samples=0, returning first model unchanged.")
        return local_models_with_sizes[0][0]

    # Initialise aggregated state dict to zeros
    agg_state = OrderedDict()
    ref_state  = base_model.state_dict()
    for key in ref_state:
        agg_state[key] = torch.zeros_like(ref_state[key], dtype=torch.float32)

    # Weighted sum
    for local_model, n in local_models_with_sizes:
        weight = n / total_samples
        local_state = local_model.state_dict()
        for key in agg_state:
            agg_state[key] += weight * local_state[key].float()

    # Load into a fresh copy of the base model
    aggregated = _clone_model(base_model)
    aggregated.load_state_dict(
        {k: v.to(ref_state[k].dtype) for k, v in agg_state.items()},
        strict=True,
    )
    return aggregated


def _run_dacm_recovery(
    unlearned_model,
    surviving_trainloaders: dict,
    surviving_counts: dict,
    dacu_weight_tensor: torch.Tensor,
    device,
    wc: float,
) -> "ChestCNN":
    """
    Proper federated DACM recovery:

    1. Deep-copy unlearned_model for EACH surviving client.
    2. Each client trains its own LOCAL copy independently.
    3. Server aggregates with weighted FedAvg (n_samples weighting).

    This prevents sequential training bias where the last client dominates.
    """
    _print_banner("STEP 4 — DACM Federated Recovery Training")

    print(f"[DACM] Recovery configuration:")
    print(f"  DACM_RECOVERY_EPOCHS = {RECOVERY_EPOCHS}")
    print(f"  DACM_RECOVERY_LR     = {RECOVERY_LR}  (SGD)")
    print(f"  Loss weights         : [NORMAL=1.0, PNEUMONIA={wc:.4f}]")
    print(f"  Aggregation          : Weighted FedAvg by n_train_samples\n")

    # Verify weight order: index 0 = NORMAL, index 1 = PNEUMONIA
    assert float(dacu_weight_tensor[0]) == 1.0, \
        "Weight[0] (NORMAL) must be 1.0 — check class ordering!"
    print(f"[DACM] Weight tensor confirmed: [NORMAL={float(dacu_weight_tensor[0]):.4f}, "
          f"PNEUMONIA={float(dacu_weight_tensor[1]):.4f}]")

    local_models_with_sizes = []

    for cid, trainloader in surviving_trainloaders.items():
        print(f"\n[DACM] ── {cid} ──")
        print(f"  NORMAL={surviving_counts[cid]['NORMAL']}, "
              f"PNEUMONIA={surviving_counts[cid]['PNEUMONIA']}")

        # Independent local copy — not the shared model
        local_model = _clone_model(unlearned_model)

        trained_local, n_samples = _train_local_recovery(
            local_model=local_model,
            trainloader=trainloader,
            weight_tensor=dacu_weight_tensor,
            epochs=RECOVERY_EPOCHS,
            lr=RECOVERY_LR,
            device=device,
            client_id=cid,
        )
        local_models_with_sizes.append((trained_local, n_samples))

    print(f"\n[DACM] FedAvg aggregating {len(local_models_with_sizes)} local models ...")
    aggregated_model = _fedavg_aggregate(local_models_with_sizes, unlearned_model, device)
    print(f"[DACM] FedAvg aggregation complete.")
    return aggregated_model


# ==================================================================
# Main workflow entry-point
# ==================================================================

def run_unlearn_workflow(
    target_client: str,
    suffix: str = None,
    active_clients: set = None,
) -> bool:
    """
    Execute the full post-approval unlearning + DACM recovery pipeline.

    Args:
        target_client  : e.g. "Hospital_B"
        suffix         : model suffix (auto-detected from .env if None)
        active_clients : mutable set; target is removed from it on success

    Returns True on success, False on non-fatal failure.
    """

    # ------------------------------------------------------------------
    # Lazy imports — avoid flwr/tensorflow/protobuf at server startup
    # ------------------------------------------------------------------
    from model import ChestCNN                                    # noqa
    from utils import load_partitions                             # noqa
    from unlearn_baseline import execute_baseline_unlearning      # noqa
    from dacu import calculate_dacu_weights                       # noqa

    effective_suffix = suffix or SUFFIX
    if active_clients is None:
        active_clients = set(KNOWN_CLIENTS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model_path = MODELS_DIR / f"global_model_{effective_suffix}.pth"

    _print_banner(f"UNLEARNING WORKFLOW — Target: {target_client}")
    print(f"[UNLEARNING] Suffix         : {effective_suffix}")
    print(f"[UNLEARNING] Global model   : {global_model_path}")
    print(f"[UNLEARNING] Device         : {device}")
    print(f"[UNLEARNING] Active clients : {sorted(active_clients)}")
    print(f"[UNLEARNING] UNLEARN_LR     : {UNLEARN_LR}  (conservative default)")
    print(f"[UNLEARNING] UNLEARN_STEPS  : {UNLEARN_STEPS}")

    # ------------------------------------------------------------------
    # Guard: global model must exist
    # ------------------------------------------------------------------
    if not global_model_path.exists():
        print(f"[ERROR] Global model not found: {global_model_path}")
        print("[ERROR] Run FL training first.")
        return False

    # ------------------------------------------------------------------
    # Load global model object for in-memory diagnostics
    # ------------------------------------------------------------------
    global_model = ChestCNN().to(device)
    global_model.load_state_dict(
        torch.load(str(global_model_path), map_location=device)
    )
    global_model_state_snapshot = copy.deepcopy(global_model.state_dict())

    # ------------------------------------------------------------------
    # Load data: target + surviving
    # ------------------------------------------------------------------
    _print_banner("STEP 1 — Loading all client data")

    try:
        target_trainloader, target_testloader, _ = load_partitions(client_id=target_client)
    except Exception as exc:
        print(f"[ERROR] Could not load data for {target_client}: {exc}")
        return False

    surviving_clients  = sorted(active_clients - {target_client})
    surviving_counts   = {}
    surviving_trainloaders = {}
    surviving_testloaders  = []

    if not surviving_clients:
        print("[ERROR] No surviving clients. Aborting.")
        return False

    for cid in surviving_clients:
        try:
            tr, te, _ = load_partitions(client_id=cid)
            counts = _count_classes(tr)
            surviving_counts[cid]      = counts
            surviving_trainloaders[cid] = tr
            surviving_testloaders.append(te)
            print(f"  {cid} — NORMAL: {counts['NORMAL']}, PNEUMONIA: {counts['PNEUMONIA']}")
        except Exception as exc:
            print(f"[WARNING] Could not load {cid}: {exc}. Skipping.")

    if not surviving_counts:
        print("[ERROR] No surviving client data available. Aborting.")
        return False

    # ------------------------------------------------------------------
    # DIAGNOSTICS — STAGE 1: Global model BEFORE unlearning
    # ------------------------------------------------------------------
    _print_banner("DIAGNOSTICS — Global Model (Before Unlearning)")

    pre_surv_acc, pre_surv_prec, pre_surv_rec, pre_surv_f1, pre_surv_cm = \
        _evaluate_model_object(global_model, surviving_testloaders, device)

    pre_tgt_acc, pre_tgt_prec, pre_tgt_rec, pre_tgt_f1, pre_tgt_cm = \
        _evaluate_model_object(global_model, [target_testloader], device)

    _print_eval_results(
        f"Global model on SURVIVING clients {surviving_clients}",
        pre_surv_acc, pre_surv_prec, pre_surv_rec, pre_surv_f1, pre_surv_cm
    )
    _print_eval_results(
        f"Global model on TARGET client [{target_client}]",
        pre_tgt_acc, pre_tgt_prec, pre_tgt_rec, pre_tgt_f1, pre_tgt_cm
    )

    # ------------------------------------------------------------------
    # STEP 2 — Gradient Ascent Unlearning
    # ------------------------------------------------------------------
    _print_banner("STEP 2 — Gradient Ascent Unlearning")
    print(f"[UNLEARNING] lr={UNLEARN_LR}  epochs={UNLEARN_STEPS}  finetune=0")
    print(f"[UNLEARNING] Conservative defaults prevent model collapse.")
    print(f"[UNLEARNING] Override with: UNLEARN_LR=... UNLEARN_STEPS=...")

    try:
        unlearned_model = execute_baseline_unlearning(
            global_model_path=str(global_model_path),
            target_loader=target_trainloader,
            target_client=target_client,
            surviving_loaders=[],   # pure erasure — recovery handled by FedAvg
            lr=UNLEARN_LR,
            unlearn_epochs=UNLEARN_STEPS,
            finetune_epochs=0,
        )
    except Exception as exc:
        print(f"[ERROR] Unlearning failed: {exc}")
        import traceback; traceback.print_exc()
        return False

    # Save unlearned checkpoint
    unlearned_path = MODELS_DIR / f"unlearned_{target_client}_{effective_suffix}.pth"
    torch.save(unlearned_model.state_dict(), unlearned_path)
    print(f"\n[UNLEARNING] Unlearned model saved → {unlearned_path}")

    # ------------------------------------------------------------------
    # DIAGNOSTICS — STAGE 2: Unlearned model
    # ------------------------------------------------------------------
    _print_banner("DIAGNOSTICS — Unlearned Model (After Gradient Ascent)")

    param_dist = _compute_param_distance(
        global_model_state_snapshot,
        unlearned_model.state_dict()
    )
    print(f"\n[UNLEARNING DIAGNOSTICS]")
    print(f"  Parameter change L2 norm  : {param_dist:.4f}")
    if param_dist > 50.0:
        print(f"  [WARNING] Very large parameter change — model may have collapsed.")
    elif param_dist < 0.5:
        print(f"  [WARNING] Very small parameter change — unlearning may be insufficient.")
    else:
        print(f"  [OK] Parameter change is in a reasonable range.")

    post_unlearn_surv_acc, post_unlearn_surv_prec, post_unlearn_surv_rec, \
        post_unlearn_surv_f1, post_unlearn_surv_cm = \
        _evaluate_model_object(unlearned_model, surviving_testloaders, device)

    post_unlearn_tgt_acc, post_unlearn_tgt_prec, post_unlearn_tgt_rec, \
        post_unlearn_tgt_f1, post_unlearn_tgt_cm = \
        _evaluate_model_object(unlearned_model, [target_testloader], device)

    _print_eval_results(
        f"Unlearned model on SURVIVING clients",
        post_unlearn_surv_acc, post_unlearn_surv_prec,
        post_unlearn_surv_rec, post_unlearn_surv_f1, post_unlearn_surv_cm
    )
    _print_eval_results(
        f"Unlearned model on TARGET client [{target_client}]",
        post_unlearn_tgt_acc, post_unlearn_tgt_prec,
        post_unlearn_tgt_rec, post_unlearn_tgt_f1, post_unlearn_tgt_cm
    )

    if post_unlearn_surv_rec == 0.0:
        print(f"\n  [WARNING] Model has COLLAPSED after unlearning — recall=0 on survivors.")
        print(f"  [WARNING] Consider reducing UNLEARN_LR or UNLEARN_STEPS.")

    # ------------------------------------------------------------------
    # Remove target from active registry
    # ------------------------------------------------------------------
    active_clients.discard(target_client)
    print(f"\n[UNLEARNING] {target_client} removed from active registry.")
    print(f"[UNLEARNING] Surviving clients: {', '.join(surviving_clients)}")

    # ------------------------------------------------------------------
    # STEP 3 — DACM Distribution Audit
    # ------------------------------------------------------------------
    _print_banner("STEP 3 — DACM Distribution Audit")

    total_normal    = sum(c["NORMAL"]    for c in surviving_counts.values())
    total_pneumonia = sum(c["PNEUMONIA"] for c in surviving_counts.values())
    total           = total_normal + total_pneumonia

    if total == 0:
        print("[ERROR] Total sample count = 0. Cannot compute gamma_c. Aborting.")
        return False

    gamma_c = total_pneumonia / total
    print(f"[DACM] Surviving distribution:")
    for cid, c in surviving_counts.items():
        print(f"  {cid}: NORMAL={c['NORMAL']}, PNEUMONIA={c['PNEUMONIA']}")
    print(f"\n[DACM] total_NORMAL    = {total_normal}")
    print(f"[DACM] total_PNEUMONIA = {total_pneumonia}")
    print(f"[DACM] gamma_c         = {gamma_c:.4f}")
    print(f"[DACM] tau_safe        = {DACM_TAU_SAFE}")

    # Call existing DACM formula (unchanged)
    dacu_weight_tensor = calculate_dacu_weights(
        surviving_counts,
        tau_safe=DACM_TAU_SAFE,
        alpha=DACM_ALPHA,
    )
    wc = float(dacu_weight_tensor[1])

    # ------------------------------------------------------------------
    # STEP 4 — Federated Recovery (only if shortage detected)
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unlearned_model.to(device)

    if gamma_c < DACM_TAU_SAFE:
        recovered_model = _run_dacm_recovery(
            unlearned_model=unlearned_model,
            surviving_trainloaders=surviving_trainloaders,
            surviving_counts=surviving_counts,
            dacu_weight_tensor=dacu_weight_tensor,
            device=device,
            wc=wc,
        )
    else:
        print(f"\n[DACM] No recovery required — gamma_c ({gamma_c:.4f}) >= tau_safe ({DACM_TAU_SAFE}).")
        recovered_model = unlearned_model

    # ------------------------------------------------------------------
    # Save recovered model
    # ------------------------------------------------------------------
    recovered_path = MODELS_DIR / f"dacm_recovered_{target_client}_{effective_suffix}.pth"
    torch.save(recovered_model.state_dict(), recovered_path)
    print(f"\n[DACM] Recovered model saved → {recovered_path}")

    # ------------------------------------------------------------------
    # STEP 5 — Three-Stage Evaluation
    # ------------------------------------------------------------------
    _print_banner("STEP 5 — Three-Stage Evaluation (Surviving Clients)")

    print(f"[VERIFY] Evaluating all three model stages on: {surviving_clients}\n")

    # Stage 3: recovered model
    rec_acc, rec_prec, rec_rec, rec_f1, rec_cm = \
        _evaluate_model_object(recovered_model, surviving_testloaders, device)

    _print_eval_results("MODEL 1 — Original Global Model", pre_surv_acc, pre_surv_prec,
                        pre_surv_rec, pre_surv_f1, pre_surv_cm)
    _print_eval_results("MODEL 2 — After Unlearning", post_unlearn_surv_acc, post_unlearn_surv_prec,
                        post_unlearn_surv_rec, post_unlearn_surv_f1, post_unlearn_surv_cm)
    _print_eval_results("MODEL 3 — After DACM Recovery", rec_acc, rec_prec,
                        rec_rec, rec_f1, rec_cm)

    # Summary table
    _print_banner("FINAL COMPARISON TABLE (Surviving Clients)")
    print(f"  {'Model':<35} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print(f"  {'─'*73}")
    print(f"  {'Original Global Model':<35} {pre_surv_acc:>9.4f} {pre_surv_prec:>10.4f} "
          f"{pre_surv_rec:>8.4f} {pre_surv_f1:>8.4f}")
    print(f"  {'After Unlearning (' + target_client + ')':<35} {post_unlearn_surv_acc:>9.4f} "
          f"{post_unlearn_surv_prec:>10.4f} {post_unlearn_surv_rec:>8.4f} {post_unlearn_surv_f1:>8.4f}")
    print(f"  {'After DACM Recovery':<35} {rec_acc:>9.4f} {rec_prec:>10.4f} "
          f"{rec_rec:>8.4f} {rec_f1:>8.4f}")
    print(f"  {'─'*73}")

    # ------------------------------------------------------------------
    # Collapse detection / verification gate
    # ------------------------------------------------------------------
    print()
    if rec_rec == 0.0 or rec_f1 == 0.0:
        print(f"[WARNING] Model collapse detected after DACM recovery.")
        print(f"[WARNING] Recovery did not restore minority-class (Pneumonia) predictions.")
        print(f"[WARNING] Recall={rec_rec:.4f}  F1={rec_f1:.4f}")
        print(f"[WARNING] Suggestions:")
        print(f"[WARNING]   - Increase DACM_RECOVERY_EPOCHS (current={RECOVERY_EPOCHS})")
        print(f"[WARNING]   - Reduce UNLEARN_LR (current={UNLEARN_LR}) to prevent collapse")
        print(f"[WARNING]   - Reduce UNLEARN_STEPS (current={UNLEARN_STEPS})")
        print(f"[VERIFY]  Model requires further recovery / tuning.")
    else:
        if rec_rec > post_unlearn_surv_rec:
            print(f"[VERIFY]  DACM recovery improved Recall: "
                  f"{post_unlearn_surv_rec:.4f} → {rec_rec:.4f}  ✓")
        print(f"[VERIFY]  Final model accepted.")

    print(f"[VERIFY]  Recovered model path: {recovered_path}")

    _print_banner("UNLEARNING WORKFLOW COMPLETE")
    print(f"  Target removed   : {target_client}")
    print(f"  Surviving        : {', '.join(surviving_clients)}")
    print(f"  UNLEARN_LR used  : {UNLEARN_LR}")
    print(f"  UNLEARN_STEPS    : {UNLEARN_STEPS}")
    print(f"  Param distance   : {param_dist:.4f}")
    print(f"  gamma_c          : {gamma_c:.4f}")
    print(f"  wc               : {wc:.4f}")
    print(f"  Recovered model  : {recovered_path}")
    print("=" * 62)

    return True
