"""
deploy_model.py
===============
FINAL GLOBAL MODEL DEPLOYMENT + LOCAL HOSPITAL INFERENCE

ADDITIVE FEATURE ONLY.
Does NOT modify any existing FL, Trust, Dropout, DACU, DACM,
Unlearning, or Quantization behaviour.

Run this script AFTER federated training (and optional unlearning / DACM
recovery) has fully completed.

Dual-mode deployment
--------------------
--mode local   (default)
    Copies the final model into models/deployed/{suffix}/{hospital}/
    and runs local inference on this machine.
    Use this when all clients and server are on the SAME laptop.

--mode network
    Delegates to deploy_server.py which HTTP-pushes the model to each
    eligible hospital laptop running deploy_receiver.py.
    Use this for the real 4-laptop competition setup.

Eligibility rules
-----------------
Permanent exclusion is read from existing DACM artefacts:
    MODELS_DIR/dacm_recovered_{client}_{suffix}.pth
A client whose artefact exists was permanently unlearned and will NOT
receive the final model.
Temporarily dropped / straggling clients are NOT excluded.

Usage
-----
    cd federated_healthcare/src

    # LOCAL mode (default)
    python deploy_model.py
    python deploy_model.py --suffix b_quantized
    python deploy_model.py --no-inference

    # NETWORK mode
    python deploy_model.py --mode network \\
        --hospital-ips Hospital_A=192.168.1.101,Hospital_B=192.168.1.102,Hospital_C=192.168.1.103 \\
        --receiver-port 9765 \\
        --serve-port 9764

Options
-------
    --mode              local|network   (default: local)
    --suffix SUFFIX     Model suffix (auto-detected from .env)
    --model-path PATH   Explicit final model .pth path
    --known-clients LIST  Comma-separated known hospital names
    --excluded LIST     Comma-separated extra hospitals to exclude
    --deploy-dir DIR    Where to copy deployed model files (local mode)
    --no-inference      Skip local inference (local mode only)
    --hospital-ips MAP  Hospital_A=IP,Hospital_B=IP  (network mode)
    --receiver-port N   Port hospital deploy_receiver.py listens on (network, default 9765)
    --serve-port N      Port this server exposes GET /model on (network, default 9764)
    --no-push           Network mode: serve only, don't push to hospitals
    --no-serve          Network mode: push only, don't start HTTP server
"""

from __future__ import annotations

import os
import sys
import csv
import shutil
import argparse
import time as _time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure src is importable regardless of cwd
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Load .env if present (mirrors server.py behaviour exactly)
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                if _k.strip() not in os.environ:
                    os.environ[_k.strip()] = _v.strip()

_root_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
if _root_env_path.exists() and _root_env_path != _env_path:
    with open(_root_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                if _k.strip() not in os.environ:
                    os.environ[_k.strip()] = _v.strip()

from paths import MODELS_DIR, DATA_DIR  # noqa: E402

import torch
import numpy as np


# ===========================================================================
# Banner helper
# ===========================================================================

def _banner(title: str, width: int = 60):
    print("\n" + "=" * width)
    print("  " + title)
    print("=" * width)


# ===========================================================================
# Step 1 – Determine suffix
# ===========================================================================

def _resolve_suffix(cli_suffix):
    """Mirror server.py suffix detection exactly. Does not modify anything."""
    if cli_suffix:
        return cli_suffix
    if "EXPERIMENT_NAME" in os.environ:
        return os.environ["EXPERIMENT_NAME"]
    use_dp    = os.environ.get("USE_DP",           "0") == "1"
    use_quant = os.environ.get("USE_QUANTIZATION", "1") == "1"
    if use_dp:
        return "c_dp"
    elif use_quant:
        return "b_quantized"
    else:
        return "a_pure"


# ===========================================================================
# Step 2 – Identify final authoritative model (READ-ONLY)
# ===========================================================================

def _find_final_model(suffix, cli_model_path):
    """
    Priority:
      1. Explicit --model-path
      2. Most-recently-modified dacm_recovered_*_{suffix}.pth
      3. global_model_{suffix}.pth
    Returns (path, description_string).
    """
    if cli_model_path:
        p = Path(cli_model_path)
        if not p.exists():
            raise FileNotFoundError(
                "[ERROR] --model-path does not exist: {}".format(p)
            )
        return p, "Explicitly specified model"

    recovered_candidates = sorted(
        MODELS_DIR.glob("dacm_recovered_*_{}.pth".format(suffix)),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    if recovered_candidates:
        chosen = recovered_candidates[0]
        return (
            chosen,
            "Recovered model after federated unlearning + DACM recovery ({})".format(
                chosen.name
            ),
        )

    normal_path = MODELS_DIR / "global_model_{}.pth".format(suffix)
    if normal_path.exists():
        return normal_path, "Normal final global model (no unlearning)"

    raise FileNotFoundError(
        "[ERROR] No final model found for suffix '{}'.\n"
        "  Looked for: dacm_recovered_*_{}.pth  and  {}\n"
        "  Run FL training first, or specify --model-path.".format(
            suffix, suffix, normal_path
        )
    )


# ===========================================================================
# Step 3 – Resolve eligible / excluded clients (READ-ONLY)
# ===========================================================================

def _resolve_clients(cli_known, cli_excluded, suffix):
    """
    Determine eligible and excluded hospitals.
    Permanent exclusion is read EXCLUSIVELY from existing DACM artefacts.
    Returns (eligible_list, excluded_list).
    """
    if cli_known:
        known = [c.strip() for c in cli_known.split(",") if c.strip()]
    else:
        env_known = os.environ.get(
            "KNOWN_CLIENTS", "Hospital_A,Hospital_B,Hospital_C"
        )
        known = [c.strip() for c in env_known.split(",") if c.strip()]

    if not known:
        raise ValueError("[ERROR] No known clients could be determined.")

    # Auto-detect excluded from existing DACM artefacts (READ-ONLY)
    auto_excluded = set()
    prefix        = "dacm_recovered_"
    suffix_part   = "_{}".format(suffix)
    for rp in MODELS_DIR.glob("dacm_recovered_*_{}.pth".format(suffix)):
        stem = rp.stem
        if stem.startswith(prefix) and stem.endswith(suffix_part):
            client_name = stem[len(prefix): len(stem) - len(suffix_part)]
            if client_name in known:
                auto_excluded.add(client_name)

    explicit_excluded = set()
    if cli_excluded:
        explicit_excluded = {c.strip() for c in cli_excluded.split(",") if c.strip()}

    all_excluded = auto_excluded | explicit_excluded
    eligible     = sorted([c for c in known if c not in all_excluded])
    excluded     = sorted(all_excluded)
    return eligible, excluded


# ===========================================================================
# Step 4 – Deploy (copy) model to each eligible hospital folder (LOCAL mode)
# ===========================================================================

def _deploy_to_hospitals(final_model_path, eligible_clients, excluded_clients,
                         deploy_dir):
    """
    Copy the final model file into per-hospital sub-directories.
    The server remains the authoritative owner. Patient data is never touched.
    Returns {client_name: deployed_path}.
    """
    deploy_dir.mkdir(parents=True, exist_ok=True)
    deployment_status = {}

    _banner("FINAL GLOBAL MODEL DEPLOYMENT")
    print("\n  Final Model  : {}".format(final_model_path))
    print("\n  Eligible Clients:")
    for c in eligible_clients:
        print("    - {}".format(c))
    if excluded_clients:
        print("\n  Excluded Clients (permanent exclusion):")
        for c in excluded_clients:
            print("    - {}".format(c))
    print("\n  Deployment Status:")

    for client in eligible_clients:
        client_dir = deploy_dir / client
        client_dir.mkdir(parents=True, exist_ok=True)
        dest = client_dir / "deployed_model.pth"
        try:
            shutil.copy2(str(final_model_path), str(dest))
            print("    {:20s} -> SENT / AVAILABLE  [{}]".format(client, dest))
            deployment_status[client] = dest
        except Exception as exc:
            print("    {:20s} -> ERROR copying model: {}".format(client, exc))

    for client in excluded_clients:
        print(
            "    {:20s} -> EXCLUDED / NOT SENT "
            "(existing permanent exclusion)".format(client)
        )
    print()
    return deployment_status


# ===========================================================================
# Step 5 – Local inference per eligible hospital (LOCAL mode)
# ===========================================================================

def _run_local_inference(client_name, deployed_model_path, device):
    """
    Load the model and run local inference for one hospital.
    PATIENT DATA STAYS LOCAL — no patient data is sent anywhere.
    Returns result dict or None on failure.
    """
    from model import ChestCNN, test        # existing architecture + test fn
    from utils import load_hospital_data    # existing data loader

    local_data_path = DATA_DIR / client_name.lower()
    if not local_data_path.exists():
        print(
            "  [{}] WARNING: Local data path not found: {}".format(
                client_name, local_data_path
            )
        )
        print("  [{}] Skipping inference - no local data available.".format(client_name))
        return None

    try:
        _trainloader, testloader = load_hospital_data(str(local_data_path))
    except Exception as exc:
        print("  [{}] WARNING: Could not load local data: {}".format(client_name, exc))
        return None

    model = ChestCNN().to(device)
    try:
        state_dict = torch.load(str(deployed_model_path), map_location=device)
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        print("  [{}] ERROR loading model: {}".format(client_name, exc))
        return None
    model.eval()

    try:
        loss, accuracy = test(model, testloader)
    except Exception as exc:
        print("  [{}] ERROR during inference: {}".format(client_name, exc))
        return None

    all_preds  = []
    all_labels = []
    try:
        with torch.no_grad():
            for images, labels in testloader:
                images = images.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy().tolist())
                all_labels.extend(labels.numpy().tolist())
    except Exception as exc:
        print("  [{}] WARNING: Detailed prediction collection failed: {}".format(
            client_name, exc
        ))

    preds_arr           = np.array(all_preds)
    normal_predicted    = int(np.sum(preds_arr == 0))
    pneumonia_predicted = int(np.sum(preds_arr == 1))
    total               = len(preds_arr)

    return {
        "client_name":         client_name,
        "local_data_path":     str(local_data_path),
        "total_samples":       total,
        "normal_predicted":    normal_predicted,
        "pneumonia_predicted": pneumonia_predicted,
        "loss":                loss,
        "accuracy":            accuracy,
    }


# ===========================================================================
# Step 6 – Print inference summary banner (LOCAL mode)
# ===========================================================================

def _print_inference_summary(results, eligible_clients):
    _banner("LOCAL INFERENCE STATUS")
    print()
    for client, result in zip(eligible_clients, results):
        if result is None:
            print(
                "  {:20s} -> MODEL LOAD ERROR / DATA MISSING - INFERENCE SKIPPED".format(
                    client
                )
            )
        else:
            acc_pct = result["accuracy"] * 100
            print("  {:20s} -> MODEL LOADED -> INFERENCE READY".format(client))
            print("    {:<26}: {}".format("Local data path", result["local_data_path"]))
            print("    {:<26}: {}".format("Test samples",    result["total_samples"]))
            print("    {:<26}: {:.4f}".format("Loss",        result["loss"]))
            print("    {:<26}: {:.2f}%".format("Accuracy",   acc_pct))
            print("    {:<26}: {}".format("NORMAL predictions",    result["normal_predicted"]))
            print("    {:<26}: {}".format("PNEUMONIA predictions", result["pneumonia_predicted"]))
            print()
    print("=" * 60)


# ===========================================================================
# Step 7 – Save deployment log CSV (LOCAL mode)
# ===========================================================================

def _save_deployment_log(final_model_path, model_type, eligible_clients,
                         excluded_clients, inference_results, deploy_dir):
    """
    Append a deployment record to deploy_dir/deployment_log.csv.
    NEW file — does NOT modify any existing CSV or dashboard output.
    """
    log_path     = deploy_dir / "deployment_log.csv"
    timestamp    = _time.strftime("%Y-%m-%d %H:%M:%S")
    write_header = not log_path.exists()

    try:
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "Timestamp", "Client", "Status",
                    "ModelPath", "ModelType", "LocalDataPath",
                    "TotalSamples", "Loss", "Accuracy",
                    "NormalPredicted", "PneumoniaPredicted",
                ])

            result_map = {}
            for r in inference_results:
                if r is not None:
                    result_map[r["client_name"]] = r

            for client in eligible_clients:
                r = result_map.get(client)
                if r:
                    writer.writerow([
                        timestamp, client, "DEPLOYED_INFERRED",
                        str(final_model_path),
                        model_type.replace("\n", " "),
                        r["local_data_path"], r["total_samples"],
                        "{:.4f}".format(r["loss"]),
                        "{:.4f}".format(r["accuracy"]),
                        r["normal_predicted"], r["pneumonia_predicted"],
                    ])
                else:
                    writer.writerow([
                        timestamp, client, "DEPLOYED_INFERENCE_SKIPPED",
                        str(final_model_path),
                        model_type.replace("\n", " "),
                        "", "", "", "", "", "",
                    ])

            for client in excluded_clients:
                writer.writerow([
                    timestamp, client, "EXCLUDED_NOT_DEPLOYED",
                    str(final_model_path),
                    model_type.replace("\n", " "),
                    "", "", "", "", "", "",
                ])

        print("\n[DEPLOYMENT] Log written -> {}".format(log_path))
    except Exception as exc:
        print("[DEPLOYMENT] WARNING: Could not write deployment log: {}".format(exc))


# ===========================================================================
# LOCAL mode orchestrator
# ===========================================================================

def run_deployment(suffix=None, model_path=None, known_clients=None,
                   excluded=None, deploy_dir=None, no_inference=False):
    """
    Execute the full LOCAL deployment + local inference pipeline.

    ADDITIVE ONLY - reads existing state, does not modify any training,
    trust, dropout, quantization, or unlearning logic.

    Returns True on success, False on failure.
    """
    try:
        # 1. Suffix
        effective_suffix = _resolve_suffix(suffix)

        # 2. Final authoritative model
        final_model_path, model_type = _find_final_model(effective_suffix, model_path)

        _banner("FINAL GLOBAL MODEL DEPLOYMENT PIPELINE  [LOCAL MODE]")
        print("\n  Mode        : LOCAL")
        print("  Suffix      : {}".format(effective_suffix))
        print("  Final Model : {}".format(final_model_path))
        print("  Model Type  : {}".format(model_type))

        # 3. Eligible / excluded clients
        eligible_clients, excluded_clients = _resolve_clients(
            known_clients, excluded, effective_suffix
        )
        if not eligible_clients:
            print(
                "\n[DEPLOYMENT] WARNING: No eligible clients found. Nothing to deploy."
            )
            return False

        # 4. Deploy directory
        effective_deploy_dir = (
            Path(deploy_dir)
            if deploy_dir
            else MODELS_DIR / "deployed" / effective_suffix
        )

        # 5. Deploy (copy) model files
        deployment_status = _deploy_to_hospitals(
            final_model_path = final_model_path,
            eligible_clients = eligible_clients,
            excluded_clients = excluded_clients,
            deploy_dir       = effective_deploy_dir,
        )

        # 6. Local inference
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("\n[DEPLOYMENT] Inference device: {}".format(device))

        inference_results = []

        if not no_inference:
            _banner("LOCAL HOSPITAL INFERENCE")
            print(
                "  PATIENT DATA STAYS LOCAL - only the trained model is distributed.\n"
                "  Each hospital loads its own local data for inference.\n"
            )
            for client in eligible_clients:
                deployed_path = deployment_status.get(client)
                if deployed_path is None:
                    print("  [{}] Model was not deployed - skipping inference.".format(client))
                    inference_results.append(None)
                    continue
                print("  [{}] Running local inference ...".format(client))
                result = _run_local_inference(client, deployed_path, device)
                inference_results.append(result)
            _print_inference_summary(inference_results, eligible_clients)
        else:
            print("\n[DEPLOYMENT] --no-inference flag set. Skipping local inference.")
            inference_results = [None] * len(eligible_clients)

        # 7. Write deployment log
        _save_deployment_log(
            final_model_path  = final_model_path,
            model_type        = model_type,
            eligible_clients  = eligible_clients,
            excluded_clients  = excluded_clients,
            inference_results = inference_results,
            deploy_dir        = effective_deploy_dir,
        )

        # Summary
        _banner("DEPLOYMENT COMPLETE")
        print("\n  Final Model  : {}".format(final_model_path))
        print("  Model Type   : {}".format(model_type))
        print("  Deployed To  : {}".format(effective_deploy_dir))
        print("  Eligible     : {}".format(", ".join(eligible_clients)))
        print("  Excluded     : {}".format(", ".join(excluded_clients) if excluded_clients else "None"))
        print()
        return True

    except FileNotFoundError as exc:
        print(str(exc))
        return False
    except Exception as exc:
        print("[DEPLOYMENT ERROR] {}".format(exc))
        import traceback
        traceback.print_exc()
        return False


# ===========================================================================
# NETWORK mode bridge  (delegates to deploy_server.py)
# ===========================================================================

def run_network_deployment(
    suffix=None, model_path=None, known_clients=None, excluded=None,
    hospital_ips=None, receiver_port=9765, serve_port=9764,
    no_push=False, no_serve=False,
):
    """
    NETWORK mode: delegates entirely to deploy_server.run_deploy_server().
    Does NOT copy files locally. Sends model over HTTP to hospital laptops.
    Returns True on success.
    """
    try:
        from deploy_server import run_deploy_server
    except ImportError as exc:
        print("[NETWORK MODE ERROR] Cannot import deploy_server: {}".format(exc))
        print("  Make sure deploy_server.py is in the same directory.")
        return False

    return run_deploy_server(
        suffix        = suffix,
        model_path    = model_path,
        known_clients = known_clients,
        excluded      = excluded,
        hospital_ips  = hospital_ips,
        receiver_port = receiver_port,
        serve_port    = serve_port,
        no_push       = no_push,
        no_serve      = no_serve,
    )


# ===========================================================================
# CLI entry point
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="deploy_model.py",
        description=(
            "Deploy the final global FL model to eligible hospitals. "
            "Supports LOCAL mode (single machine) and NETWORK mode (multi-laptop). "
            "ADDITIVE — does NOT modify any existing FL, Trust, Dropout, "
            "DACU, DACM, Unlearning, or Quantization behaviour."
        ),
    )

    # --- Shared options ---
    parser.add_argument("--mode", default="local", choices=["local", "network"],
        help="local: copy files on this machine (default). "
             "network: HTTP-push to hospital laptops via deploy_server.py.")
    parser.add_argument("--suffix", default=None,
        help="Model suffix (e.g. b_quantized, a_pure, c_dp). Auto-detected from .env.")
    parser.add_argument("--model-path", default=None, dest="model_path",
        help="Absolute path to final model .pth. Skips auto-detection.")
    parser.add_argument("--known-clients", default=None, dest="known_clients",
        help="Comma-separated known hospital names.")
    parser.add_argument("--excluded", default=None,
        help="Comma-separated hospitals to explicitly exclude.")

    # --- LOCAL mode options ---
    parser.add_argument("--deploy-dir", default=None, dest="deploy_dir",
        help="[LOCAL] Directory to copy deployed model files into.")
    parser.add_argument("--no-inference", action="store_true", dest="no_inference",
        help="[LOCAL] Skip local inference.")

    # --- NETWORK mode options ---
    parser.add_argument("--hospital-ips", default=None, dest="hospital_ips",
        help="[NETWORK] Hospital_A=192.168.1.101,Hospital_B=192.168.1.102 ...")
    parser.add_argument("--receiver-port", type=int, default=9765, dest="receiver_port",
        help="[NETWORK] Port where deploy_receiver.py listens (default 9765).")
    parser.add_argument("--serve-port", type=int, default=9764, dest="serve_port",
        help="[NETWORK] Port for GET /model endpoint on server (default 9764).")
    parser.add_argument("--no-push", action="store_true", dest="no_push",
        help="[NETWORK] Only serve; do not push to hospitals.")
    parser.add_argument("--no-serve", action="store_true", dest="no_serve",
        help="[NETWORK] Only push; do not start HTTP server.")

    args = parser.parse_args()

    if args.mode == "network":
        success = run_network_deployment(
            suffix        = args.suffix,
            model_path    = args.model_path,
            known_clients = args.known_clients,
            excluded      = args.excluded,
            hospital_ips  = args.hospital_ips,
            receiver_port = args.receiver_port,
            serve_port    = args.serve_port,
            no_push       = args.no_push,
            no_serve      = args.no_serve,
        )
    else:
        success = run_deployment(
            suffix        = args.suffix,
            model_path    = args.model_path,
            known_clients = args.known_clients,
            excluded      = args.excluded,
            deploy_dir    = args.deploy_dir,
            no_inference  = args.no_inference,
        )

    sys.exit(0 if success else 1)
