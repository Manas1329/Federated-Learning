import flwr as fl
import os
import sys
import time
import math
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Optional, Dict, Any, Set, Tuple
from collections import OrderedDict

# Ensure 'src' package is importable regardless of where the script is run from
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Load environment variables from .env file if present
_curr = SRC_DIR
_candidates = [
    Path.cwd() / ".env",
    _curr / ".env",
    _curr.parent / ".env",
    _curr.parent.parent / ".env",
]
for _cand in _candidates:
    if _cand.exists():
        with open(_cand) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    k = key.strip()
                    if k not in os.environ:
                        os.environ[k] = val.strip()
        break

VERBOSE_LOGGING = os.environ.get("VERBOSE_LOGGING", "0") == "1"

if not VERBOSE_LOGGING:
    import logging
    logging.getLogger("flwr").setLevel(logging.WARNING)

from flwr.common import (
    parameters_to_ndarrays,
    ndarrays_to_parameters,
    Scalar,
)

from paths import RESULTS_DIR, MODELS_DIR, APPROVED_MODELS_DIR, resolve_data_path
from model import ChestCNN, evaluate_metrics
from device_utils import get_device
from quantization import dequantize_parameters
from checkpoint_manager import (
    verify_checkpoint,
    save_checkpoint_atomic,
    load_raw_state_dict
)
from run_manager import RetrainingRunManager
from model_approval import ModelApprovalGate

# Trust / Tagging Module (Phase 1)
try:
    from trust_manager import trust_manager as _tm
    _TRUST_OK = True
except Exception as _te:
    _TRUST_OK = False
    print(f"[TrustManager] Import warning: {_te}")

# --------------------------------------------------
# Paths & Periodic Retraining Configuration
# --------------------------------------------------

USE_DP = os.environ.get("USE_DP", "0") == "1"
USE_QUANTIZATION = os.environ.get("USE_QUANTIZATION", "1") == "1"
TOTAL_ROUNDS = int(os.environ.get("NUM_ROUNDS", "10"))
TRAINING_MODE = os.environ.get("TRAINING_MODE", "warm_start").strip().lower()
ALLOW_RANDOM_FALLBACK = os.environ.get("ALLOW_RANDOM_FALLBACK", "0").strip() in ("1", "true")
INIT_MODEL_PATH_ENV = os.environ.get("INIT_MODEL_PATH", "").strip()

EARLY_STOPPING_ENABLED = os.environ.get("EARLY_STOPPING_ENABLED", "1").strip() in ("1", "true")
EARLY_STOPPING_PATIENCE = int(os.environ.get("EARLY_STOPPING_PATIENCE", "5"))
EARLY_STOPPING_MIN_DELTA = float(os.environ.get("EARLY_STOPPING_MIN_DELTA", "0.001"))
MIN_ROUNDS = int(os.environ.get("MIN_ROUNDS", "5"))

if "EXPERIMENT_NAME" in os.environ:
    SUFFIX = os.environ["EXPERIMENT_NAME"]
elif USE_DP:
    SUFFIX = "c_dp"
elif USE_QUANTIZATION:
    SUFFIX = "b_quantized"
else:
    SUFFIX = "a_pure"

RESULTS_DIR_SUFFIX = RESULTS_DIR / SUFFIX
RESULTS_DIR_SUFFIX.mkdir(parents=True, exist_ok=True)

METRICS_FILE = RESULTS_DIR_SUFFIX / f"metrics_{SUFFIX}.csv"
ROUND_METRICS_FILE = RESULTS_DIR_SUFFIX / f"round_metrics_{SUFFIX}.csv"
MODEL_DIR = MODELS_DIR
MODEL_PATH = MODEL_DIR / f"global_model_{SUFFIX}.pth"

# Initialize Run Manager (Non-destructive: does not delete historical metrics)
RUN_ID = os.environ.get("RUN_ID", None)
run_manager = RetrainingRunManager(run_id=RUN_ID)

device = get_device()
print(f"[Server] Active Run ID: {run_manager.run_id}")
print(f"[Server] Central device: {device}")
print(f"[Server] Training mode: {TRAINING_MODE} (ALLOW_RANDOM_FALLBACK={ALLOW_RANDOM_FALLBACK})")

# Set of all known federation participants — used by the auto-unlearn trigger.
KNOWN_CLIENTS: set = {
    c.strip()
    for c in os.environ.get("KNOWN_CLIENTS", "Hospital_A,Hospital_B,Hospital_C").split(",")
    if c.strip()
}

round_start_times = {}

# --------------------------------------------------
# Setup Initial Model & Warm-Start Verification
# --------------------------------------------------

def setup_initial_model(
    training_mode: str,
    init_path_str: str,
    allow_random: bool,
    manager: RetrainingRunManager,
    target_device: torch.device
) -> Tuple[ChestCNN, fl.common.Parameters, Optional[str], Optional[str]]:
    """
    Initialize global model for the periodic retraining cycle.
    In warm-start mode, strictly verifies checkpoint integrity and parameters.
    Saves initial_model.pth to the run models directory.
    """
    model = ChestCNN().to(target_device)
    parent_model_id = None
    parent_run_id = None

    if training_mode == "warm_start":
        candidate_path = None
        if init_path_str:
            candidate_path = Path(init_path_str)
        elif (APPROVED_MODELS_DIR / "global_model.pth").exists():
            candidate_path = APPROVED_MODELS_DIR / "global_model.pth"
        elif MODEL_PATH.exists():
            candidate_path = MODEL_PATH

        if not candidate_path or not candidate_path.exists():
            if not allow_random:
                configured_path = init_path_str or "<not set>"
                raise FileNotFoundError(
                    "[WARM_START_FATAL] Warm-start selected, but no valid global "
                    "checkpoint was found. "
                    f"INIT_MODEL_PATH={configured_path}; "
                    f"checked={candidate_path or '<no fallback path>'}. "
                    "For a first run, set TRAINING_MODE=fresh. "
                    "For retraining, set INIT_MODEL_PATH to an existing approved "
                    "global_model.pth. "
                    "ALLOW_RANDOM_FALLBACK=1 is available only when random "
                    "initialization is explicitly intended."
                )
            print("[WARM_START] Checkpoint missing. Falling back to fresh random weights (ALLOW_RANDOM_FALLBACK=1).")
            training_mode = "fresh"
        else:
            is_valid, err_msg, state_dict, metadata = verify_checkpoint(
                candidate_path,
                expected_model=model,
                allow_random_fallback=allow_random
            )
            if not is_valid:
                raise RuntimeError(f"[WARM_START_FATAL] Invalid checkpoint: {err_msg}")

            model.load_state_dict(state_dict)
            parent_model_id = str(candidate_path)
            parent_run_id = metadata.get("run_id") or metadata.get("promoted_from_run")
            print(f"[WARM_START] Verified and loaded warm-start checkpoint: {candidate_path}")
            if parent_run_id:
                print(f"[WARM_START] Linked parent run ID: {parent_run_id}")

    if training_mode == "fresh":
        seed = int(os.environ.get("SPLIT_SEED", "42"))
        torch.manual_seed(seed)
        model = ChestCNN().to(target_device)
        print(f"[FRESH_INIT] ChestCNN initialized with seed {seed}")

    # Convert model parameters for Flower
    ndarrays = [v.cpu().numpy() for v in model.state_dict().values()]
    initial_params = ndarrays_to_parameters(ndarrays)

    # Save initial_model.pth atomically
    init_meta = {
        "run_id": manager.run_id,
        "training_mode": training_mode,
        "parent_model_id": parent_model_id,
        "parent_run_id": parent_run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
    }
    save_checkpoint_atomic(model.state_dict(), manager.initial_model_path, metadata=init_meta)
    print(f"[Server] Saved initial model checkpoint: {manager.initial_model_path}")

    return model, initial_params, parent_model_id, parent_run_id


# --------------------------------------------------
# Initial Baseline Validation Evaluation (Before Round 1)
# --------------------------------------------------

def run_initial_baseline_evaluation(
    initial_model: ChestCNN,
    hospital_ids: list[str],
    manager: RetrainingRunManager,
    target_device: torch.device
) -> Dict[str, Any]:
    """
    Evaluate initial global model on each hospital's validation partition
    before round 1 gradient updates.
    """
    from utils import load_hospital_data

    print("\n" + "=" * 65)
    print("  INITIAL BASELINE EVALUATION (BEFORE ROUND 1)")
    print("=" * 65)

    results = []
    pooled_tp = 0
    pooled_tn = 0
    pooled_fp = 0
    pooled_fn = 0
    pooled_loss_sum = 0.0
    pooled_total = 0

    eval_start = time.perf_counter()

    for h_id in sorted(hospital_ids):
        data_path = resolve_data_path(None, h_id)
        trainloader, valloader, testloader = load_hospital_data(data_path, batch_size=32)
        h_metrics = evaluate_metrics(initial_model, valloader, device=target_device)

        pooled_tp += h_metrics["tp"]
        pooled_tn += h_metrics["tn"]
        pooled_fp += h_metrics["fp"]
        pooled_fn += h_metrics["fn"]
        pooled_loss_sum += h_metrics["loss_sum"]
        pooled_total += h_metrics["total"]

        print(
            f"  {h_id:<14} | Samples: {h_metrics['total']:<4} | "
            f"Acc: {h_metrics['accuracy']*100:.2f}% | Loss: {h_metrics['loss']:.4f} | "
            f"F1: {h_metrics['f1']:.4f} | Rec: {h_metrics['recall']:.4f}"
        )

        results.append({
            "client": h_id,
            "sample_count": h_metrics["total"],
            "correct": h_metrics["correct"],
            "incorrect": h_metrics["total"] - h_metrics["correct"],
            "tp": h_metrics["tp"],
            "tn": h_metrics["tn"],
            "fp": h_metrics["fp"],
            "fn": h_metrics["fn"],
            "loss_sum": h_metrics["loss_sum"],
            "loss": h_metrics["loss"],
            "accuracy": h_metrics["accuracy"],
            "precision": h_metrics["precision"],
            "recall": h_metrics["recall"],
            "f1": h_metrics["f1"]
        })

    eval_duration = time.perf_counter() - eval_start

    pooled_acc = (pooled_tp + pooled_tn) / pooled_total if pooled_total > 0 else 0.0
    pooled_prec = pooled_tp / (pooled_tp + pooled_fp) if (pooled_tp + pooled_fp) > 0 else 0.0
    pooled_rec = pooled_tp / (pooled_tp + pooled_fn) if (pooled_tp + pooled_fn) > 0 else 0.0
    pooled_f1 = (2 * pooled_prec * pooled_rec / (pooled_prec + pooled_rec)) if (pooled_prec + pooled_rec) > 0 else 0.0
    pooled_loss = pooled_loss_sum / pooled_total if pooled_total > 0 else 0.0

    print("-" * 65)
    print(
        f"  POOLED BASELINE | Samples: {pooled_total:<4} | "
        f"Acc: {pooled_acc*100:.2f}% | Loss: {pooled_loss:.4f} | "
        f"F1: {pooled_f1:.4f} | Rec: {pooled_rec:.4f} | Duration: {eval_duration:.2f}s"
    )
    print("=" * 65 + "\n")

    # Save to initial_metrics.csv
    init_df = pd.DataFrame(results)
    init_df.to_csv(manager.initial_metrics_path, index=False)

    return {
        "pooled": {
            "accuracy": pooled_acc,
            "loss": pooled_loss,
            "f1": pooled_f1,
            "precision": pooled_prec,
            "recall": pooled_rec,
            "total": pooled_total,
            "duration": eval_duration
        },
        "per_hospital": {r["client"]: r for r in results}
    }


# --------------------------------------------------
# Post-Training Unlearning Auto-Trigger
# --------------------------------------------------

def _trigger_post_training_unlearning(
    quarantined_clients: set,
    suffix: str,
    active_clients: set,
):
    print("\n" + "=" * 62)
    print("  [AUTO-UNLEARN] POST-TRAINING UNLEARNING TRIGGERED")
    print("=" * 62)
    print(f"  [AUTO-UNLEARN] Quarantined clients detected : {quarantined_clients}")
    print(f"  [AUTO-UNLEARN] Final checkpoint suffix      : {suffix}")
    print(f"  [AUTO-UNLEARN] Active federation members   : {sorted(active_clients)}")
    print("=" * 62)

    try:
        from unlearn_workflow import run_unlearn_workflow
    except Exception as import_err:
        print(f"[AUTO-UNLEARN] ERROR: Could not import unlearn_workflow: {import_err}")
        return

    remaining_active = set(active_clients)
    for target in sorted(quarantined_clients):
        print(f"\n[AUTO-UNLEARN] Starting post-training unlearning for: {target}")
        try:
            success = run_unlearn_workflow(
                target_client=target,
                suffix=suffix,
                active_clients=remaining_active,
            )
            if success:
                print(f"[AUTO-UNLEARN] Successfully completed unlearning for {target}.")
        except Exception as unlearn_err:
            print(f"[AUTO-UNLEARN] Exception during unlearning of {target}: {unlearn_err}")


# --------------------------------------------------
# Persistent Exclusion Helpers
# --------------------------------------------------

def _get_excluded_cids(excluded_names: set) -> set:
    if not _TRUST_OK or not excluded_names:
        return set()
    return {
        cid
        for cid, name in _tm.cid_to_name.items()
        if name in excluded_names
    }


def _check_promote_to_excluded(
    quarantined_names: Set[str],
    excluded_names: Set[str],
    rejected_names: Optional[Set[str]] = None,
) -> set:
    newly_excluded: set = set()
    if not _TRUST_OK:
        return newly_excluded

    if rejected_names is None:
        rejected_names = set()

    name_to_cid = {v: k for k, v in _tm.cid_to_name.items()}

    def _get_tag(name: str) -> Optional[str]:
        trust_info = None
        if hasattr(_tm, "client_state") and name in _tm.client_state:
            trust_info = _tm.client_state[name].get("latest_trust")
        if not trust_info:
            cid = name_to_cid.get(name)
            if cid:
                trust_info = _tm.get_trust_for_cid(cid)
        return trust_info.get("tag") if trust_info else None

    for name in quarantined_names:
        if name in excluded_names:
            continue
        tag = _get_tag(name)
        if tag in ("SUSPICIOUS", "UNTRUSTED"):
            newly_excluded.add(name)
            print(f"[EXCL] {name} -> PERMANENTLY EXCLUDED (QUARANTINE + {tag} tag).")

    for name in rejected_names:
        if name in excluded_names or name in newly_excluded:
            continue
        tag = _get_tag(name)
        if tag == "UNTRUSTED":
            newly_excluded.add(name)
            print(f"[EXCL] {name} -> PERMANENTLY EXCLUDED (REJECT + UNTRUSTED tag).")

    return newly_excluded


# --------------------------------------------------
# Custom FedAvg Strategy with Periodic Retraining Features
# --------------------------------------------------

class SaveModelStrategy(fl.server.strategy.FedAvg):

    def __init__(
        self,
        run_manager_inst: RetrainingRunManager,
        total_rounds: int = 10,
        early_stopping_enabled: bool = True,
        early_stopping_patience: int = 5,
        early_stopping_min_delta: float = 0.001,
        min_rounds: int = 5,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.run_manager = run_manager_inst
        self.total_rounds = total_rounds
        self.dacm_recovery_weight = 1.0
        self.dacu_recovery_weight = 1.0
        self.quarantined_clients: set = set()
        self.excluded_names: set = set()
        self.excluded_cids: set = set()

        self._initial_target_clients: int = int(
            self.min_fit_clients
        ) if hasattr(self, "min_fit_clients") else int(
            os.environ.get("TARGET_CLIENTS", "3")
        )

        # Early Stopping & Checkpoint Selection State
        self.early_stopping_enabled = early_stopping_enabled
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.min_rounds = min_rounds
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.best_round = 0
        self.patience_counter = 0
        self.early_stop_triggered = False
        self.early_stop_reason: Optional[str] = None

        self.current_round_parameters: Optional[fl.common.Parameters] = None
        self.current_round_number: int = 0

    def _refresh_exclusion(self) -> None:
        self.excluded_cids = _get_excluded_cids(self.excluded_names)

    def _reduce_quorum_for_exclusions(self) -> None:
        n_excluded = len(self.excluded_names)
        if n_excluded == 0:
            return
        new_target = max(1, self._initial_target_clients - n_excluded)
        new_min = max(1, new_target - 1)
        if self.min_fit_clients != new_target or self.min_available_clients > new_min:
            print(
                f"[EXCL] Adjusting Flower quorum: "
                f"min_fit_clients {self.min_fit_clients} -> {new_target}, "
                f"min_available_clients {self.min_available_clients} -> {new_min}"
            )
            self.min_fit_clients = new_target
            self.min_evaluate_clients = new_target
            self.min_available_clients = new_min

    def configure_fit(self, server_round, parameters, client_manager):
        round_start_times[server_round] = time.perf_counter()

        print("\n" + "=" * 60)
        print(f"Starting Federated Round {server_round}")
        print("=" * 60)

        self._refresh_exclusion()
        self._reduce_quorum_for_exclusions()

        if self.excluded_cids:
            all_cm_clients = client_manager.all()
            for exc_cid in list(self.excluded_cids):
                if exc_cid in all_cm_clients:
                    try:
                        client_manager.unregister(all_cm_clients[exc_cid])
                    except Exception:
                        pass

        if _TRUST_OK:
            try:
                _tm.set_global_params(parameters_to_ndarrays(parameters))
            except Exception:
                pass

        base_config = {
            "server_round": server_round,
            "use_quantization": bool(USE_QUANTIZATION),
        }

        dacm_weight = getattr(self, 'dacm_recovery_weight', getattr(self, 'dacu_recovery_weight', 1.0))
        base_config["pneumonia_weight"] = float(dacm_weight)

        available_clients = client_manager.num_available()
        sample_size = min(self.min_fit_clients, available_clients)

        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=self.min_available_clients
        )

        if self.excluded_cids:
            clients = [c for c in clients if c.cid not in self.excluded_cids]

        result = []
        for client in clients:
            config = dict(base_config)
            result.append((client, fl.common.FitIns(parameters, config)))

        return result

    def configure_evaluate(self, server_round, parameters, client_manager):
        self._refresh_exclusion()
        self._reduce_quorum_for_exclusions()

        if self.excluded_cids:
            all_cm_clients = client_manager.all()
            for exc_cid in list(self.excluded_cids):
                if exc_cid in all_cm_clients:
                    try:
                        client_manager.unregister(all_cm_clients[exc_cid])
                    except Exception:
                        pass

        config: Dict[str, Scalar] = {"server_round": server_round}
        evaluate_ins = fl.common.EvaluateIns(parameters, config)
        available_clients = client_manager.num_available()
        sample_size = min(self.min_evaluate_clients, available_clients)

        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=self.min_available_clients
        )

        if self.excluded_cids:
            clients = [c for c in clients if c.cid not in self.excluded_cids]

        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(self, server_round, results, failures):
        aggregation_start = time.perf_counter()
        dequantized_results = []

        for client_proxy, fit_res in results:
            try:
                raw_params = parameters_to_ndarrays(fit_res.parameters)
                client_metric_quant = fit_res.metrics.get("is_quantized", None)

                if client_metric_quant is True:
                    is_quant = True
                elif client_metric_quant is False:
                    is_quant = False
                else:
                    is_quant = (
                        len(raw_params) > 0
                        and len(raw_params) % 2 == 0
                        and raw_params[1].dtype == np.int8
                    )

                if is_quant:
                    fp32_parameters = dequantize_parameters(raw_params)
                    fit_res.parameters = ndarrays_to_parameters(fp32_parameters)
                    dequantized_results.append((client_proxy, fit_res))
                else:
                    dequantized_results.append((client_proxy, fit_res))
            except Exception as e:
                print(f"[Server] Error processing update from client {client_proxy.cid}: {e}")

        if _TRUST_OK:
            try:
                for _cp, _fit_res in dequantized_results:
                    _cname = _fit_res.metrics.get("client_name", f"Client_{_cp.cid[:8]}")
                    _cndarrs = parameters_to_ndarrays(_fit_res.parameters)
                    _tm.record_update(_cname, _cndarrs, _cp.cid, _fit_res.metrics)
                for _fail in failures:
                    if isinstance(_fail, tuple) and len(_fail) >= 1:
                        _fail_cp = _fail[0]
                        if hasattr(_fail_cp, "cid"):
                            _tm.record_dropout(_fail_cp.cid)
            except Exception as _e:
                print(f"[TrustManager] aggregate_fit warning: {_e}")

        # Drop updates from any reconnected permanently excluded clients
        clean_dequantized = []
        for _cp, _fit_res in dequantized_results:
            _cname = _fit_res.metrics.get("client_name", f"Client_{_cp.cid[:8]}")
            if _cname in self.excluded_names:
                self.excluded_cids.add(_cp.cid)
                continue
            clean_dequantized.append((_cp, _fit_res))
        dequantized_results = clean_dequantized

        # Pre-Aggregation Trust Gate
        accepted_results = []
        quarantined_results = []
        rejected_results = []

        if _TRUST_OK:
            try:
                _decisions = _tm.get_pre_agg_decisions()
                for _cp, _fit_res in dequantized_results:
                    _cname = _fit_res.metrics.get("client_name", f"Client_{_cp.cid[:8]}")
                    _dec = _decisions.get(_cname, "ACCEPT")

                    if _dec == "ACCEPT":
                        accepted_results.append((_cp, _fit_res))
                    elif _dec == "QUARANTINE":
                        quarantined_results.append((_cp, _fit_res))
                        self.quarantined_clients.add(_cname)
                    else:
                        rejected_results.append((_cp, _fit_res))

                _newly_quarantined_names = {
                    _fit_res.metrics.get("client_name", f"Client_{_cp.cid[:8]}")
                    for _cp, _fit_res in quarantined_results
                }
                _newly_rejected_names = {
                    _fit_res.metrics.get("client_name", f"Client_{_cp.cid[:8]}")
                    for _cp, _fit_res in rejected_results
                }
                if _newly_quarantined_names or _newly_rejected_names:
                    _newly_excluded = _check_promote_to_excluded(
                        quarantined_names=_newly_quarantined_names,
                        excluded_names=self.excluded_names,
                        rejected_names=_newly_rejected_names,
                    )
                    if _newly_excluded:
                        self.excluded_names.update(_newly_excluded)
                        self.excluded_cids = _get_excluded_cids(self.excluded_names)
                        self._reduce_quorum_for_exclusions()

                if not accepted_results:
                    accepted_results = dequantized_results
            except Exception as _ge:
                accepted_results = dequantized_results
        else:
            accepted_results = dequantized_results

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round,
            accepted_results,
            failures,
        )

        aggregation_time = (time.perf_counter() - aggregation_start)
        successful_clients = len(dequantized_results)
        failed_clients = len(failures)
        round_time = (time.perf_counter() - round_start_times[server_round]) if server_round in round_start_times else 0

        # Store current aggregated parameters and round number for exact checkpoint binding
        self.current_round_parameters = aggregated_parameters
        self.current_round_number = server_round

        # DACM Distribution Audit
        total_normal = 0
        total_pneumonia = 0
        for _, fit_res in results:
            normal_count = fit_res.metrics.get("normal_count", 0)
            pneumonia_count = fit_res.metrics.get("pneumonia_count", 0)
            if isinstance(normal_count, (int, float)):
                total_normal += normal_count
            if isinstance(pneumonia_count, (int, float)):
                total_pneumonia += pneumonia_count

        total = total_normal + total_pneumonia
        if total > 0:
            gamma = total_pneumonia / total
            tau_safe = 0.50
            if gamma < tau_safe:
                alpha = 3.0
                self.dacm_recovery_weight = 1.0 + alpha * math.log(tau_safe / gamma)
            else:
                self.dacm_recovery_weight = 1.0

        # Round timing record
        round_data = pd.DataFrame([[
            server_round,
            successful_clients,
            failed_clients,
            aggregation_time,
            round_time
        ]], columns=[
            "Round", "Successful_Clients", "Failed_Clients",
            "Aggregation_Time_sec", "Total_Round_Time_sec"
        ])
        round_data.to_csv(
            ROUND_METRICS_FILE,
            mode="a",
            header=not os.path.exists(ROUND_METRICS_FILE),
            index=False
        )

        return (aggregated_parameters, aggregated_metrics)

    def aggregate_evaluate(self, server_round, results, failures):
        """
        Aggregate validation evaluation results from hospital clients.
        Performs exact checkpoint selection and checks early-stopping criteria.
        """
        if not results:
            return None, {}

        metrics_list = [(res.num_examples, res.metrics) for _, res in results]
        total_examples = sum(num_examples for num_examples, _ in metrics_list)

        def metric_float(metrics: Dict[str, Scalar], key: str, default: float = 0.0) -> float:
            value = metrics.get(key, default)
            return float(value) if isinstance(value, (int, float)) else default

        pooled_tp = sum(metric_float(m, "tp") for _, m in metrics_list)
        pooled_tn = sum(metric_float(m, "tn") for _, m in metrics_list)
        pooled_fp = sum(metric_float(m, "fp") for _, m in metrics_list)
        pooled_fn = sum(metric_float(m, "fn") for _, m in metrics_list)
        pooled_loss_sum = sum(
            metric_float(m, "loss_sum", metric_float(m, "loss") * num_examples)
            for num_examples, m in metrics_list
        )

        weighted_accuracy = (pooled_tp + pooled_tn) / total_examples if total_examples > 0 else 0.0
        weighted_loss = pooled_loss_sum / total_examples if total_examples > 0 else 0.0
        weighted_precision = pooled_tp / (pooled_tp + pooled_fp) if (pooled_tp + pooled_fp) > 0 else 0.0
        weighted_recall = pooled_tp / (pooled_tp + pooled_fn) if (pooled_tp + pooled_fn) > 0 else 0.0
        weighted_f1 = (2 * weighted_precision * weighted_recall / (weighted_precision + weighted_recall)) if (weighted_precision + weighted_recall) > 0 else 0.0

        print("\n" + "=" * 56)
        print(f"  GLOBAL ROUND {server_round} VALIDATION RESULTS")
        print("=" * 56)
        print(f"  Clients Evaluated   : {len(results)}")
        print(f"  Total Val Samples   : {total_examples}")
        print(f"  Pooled Accuracy     : {weighted_accuracy * 100:.2f}%")
        print(f"  Pooled Loss         : {weighted_loss:.4f}")
        print(f"  Pooled F1 Score     : {weighted_f1:.4f}")
        print(f"  Pooled Precision    : {weighted_precision:.4f}")
        print(f"  Pooled Recall       : {weighted_recall:.4f}")
        print("=" * 56)

        # Save to global and run-specific metrics files
        df_round = pd.DataFrame([[
            server_round, weighted_accuracy, weighted_loss,
            weighted_f1, weighted_precision, weighted_recall,
            pooled_tp, pooled_tn, pooled_fp, pooled_fn
        ]], columns=[
            "Round", "Accuracy", "Loss", "F1", "Precision", "Recall", "TP", "TN", "FP", "FN"
        ])
        df_round.to_csv(
            METRICS_FILE,
            mode="a",
            header=not os.path.exists(METRICS_FILE),
            index=False
        )
        df_round.to_csv(
            self.run_manager.round_metrics_path,
            mode="a",
            header=not os.path.exists(self.run_manager.round_metrics_path),
            index=False
        )

        # ==========================================================
        # EXACT CHECKPOINT SELECTION & ATOMIC PERSISTENCE
        # ==========================================================
        if self.current_round_parameters is not None:
            model = ChestCNN()
            params = parameters_to_ndarrays(self.current_round_parameters)
            params_dict = zip(model.state_dict().keys(), params)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)

            improvement = self.best_val_loss - weighted_loss
            if improvement >= self.early_stopping_min_delta:
                print(f"[CHECKPOINT] Validation loss improved from {self.best_val_loss:.4f} to {weighted_loss:.4f} (delta={improvement:.4f} >= {self.early_stopping_min_delta:.4f}).")
                self.best_val_loss = weighted_loss
                self.best_val_acc = weighted_accuracy
                self.best_round = server_round
                self.patience_counter = 0

                best_meta = {
                    "run_id": self.run_manager.run_id,
                    "round": server_round,
                    "validation_metrics": {
                        "loss": weighted_loss,
                        "accuracy": weighted_accuracy,
                        "f1": weighted_f1,
                        "precision": weighted_precision,
                        "recall": weighted_recall
                    }
                }
                save_checkpoint_atomic(model.state_dict(), self.run_manager.best_model_path, metadata=best_meta)
                self.run_manager.update_best(server_round, weighted_loss, weighted_accuracy)
                print(f"[CHECKPOINT] Atomically saved best model: {self.run_manager.best_model_path}")
            else:
                self.patience_counter += 1
                print(f"[EARLY STOPPING] No sufficient improvement ({improvement:.4f} < {self.early_stopping_min_delta:.4f}). Patience: {self.patience_counter}/{self.early_stopping_patience}")

                # Check Early Stopping
                if (
                    self.early_stopping_enabled
                    and server_round >= self.min_rounds
                    and self.patience_counter >= self.early_stopping_patience
                ):
                    self.early_stop_triggered = True
                    self.early_stop_reason = (
                        f"Validation loss did not improve by {self.early_stopping_min_delta} "
                        f"for {self.early_stopping_patience} consecutive rounds after round {self.min_rounds}"
                    )
                    print("\n" + "!" * 62)
                    print(f"  EARLY STOPPING TRIGGERED AT ROUND {server_round}")
                    print(f"  Reason: {self.early_stop_reason}")
                    print(f"  Best round was round {self.best_round} (Loss: {self.best_val_loss:.4f})")
                    print("!" * 62 + "\n")

            # Always save final model on last round or early stopping
            if server_round == self.total_rounds or self.early_stop_triggered:
                final_meta = {
                    "run_id": self.run_manager.run_id,
                    "final_round": server_round,
                    "best_round": self.best_round,
                    "final_metrics": {
                        "loss": weighted_loss,
                        "accuracy": weighted_accuracy,
                        "f1": weighted_f1
                    }
                }
                save_checkpoint_atomic(model.state_dict(), self.run_manager.final_model_path, metadata=final_meta)
                save_checkpoint_atomic(model.state_dict(), MODEL_PATH, metadata=final_meta)
                print(f"[CHECKPOINT] Saved final model checkpoint: {self.run_manager.final_model_path}")

                targets_to_unlearn = self.quarantined_clients | self.excluded_names
                if targets_to_unlearn:
                    _surviving_clients = KNOWN_CLIENTS - self.excluded_names
                    _trigger_post_training_unlearning(
                        quarantined_clients=targets_to_unlearn,
                        suffix=SUFFIX,
                        active_clients=_surviving_clients,
                    )

        # Trust Module recording
        if _TRUST_OK:
            try:
                for _num_ex, _m in metrics_list:
                    _cname = _m.get("client_name", "Unknown")
                    if _cname == "Unknown":
                        continue
                    _tm.record_evaluation(
                        client_name=str(_cname),
                        accuracy=float(_m.get("accuracy", 0.0)),
                        loss=float(_m.get("loss", 0.0)),
                        f1=float(_m.get("f1", 0.0)),
                        precision=float(_m.get("precision", 0.0)),
                        recall=float(_m.get("recall", 0.0)),
                        num_examples=int(_num_ex),
                    )
                _tm.finalize_round(server_round)
            except Exception as _e:
                print(f"[TrustManager] evaluate_metrics warning: {_e}")

        return weighted_loss, {
            "accuracy": weighted_accuracy,
            "loss": weighted_loss,
            "f1": weighted_f1,
            "precision": weighted_precision,
            "recall": weighted_recall,
        }


# --------------------------------------------------
# Main Server Runner
# --------------------------------------------------

if __name__ == "__main__":
    from dropout_handler import AdaptiveServer

    TARGET_CLIENTS = int(os.environ.get("TARGET_CLIENTS", "3"))
    MIN_CLIENTS = max(2, math.ceil(0.6 * TARGET_CLIENTS))

    # Initialize model (Fresh or Warm-Start with verification)
    init_model, init_params, parent_id, parent_run = setup_initial_model(
        training_mode=TRAINING_MODE,
        init_path_str=INIT_MODEL_PATH_ENV,
        allow_random=ALLOW_RANDOM_FALLBACK,
        manager=run_manager,
        target_device=device
    )

    # Initialize Run Configuration
    run_config = {
        "run_id": run_manager.run_id,
        "training_mode": TRAINING_MODE,
        "parent_model_id": parent_id,
        "parent_run_id": parent_run,
        "total_rounds": TOTAL_ROUNDS,
        "target_clients": TARGET_CLIENTS,
        "min_clients": MIN_CLIENTS,
        "use_dp": USE_DP,
        "use_quantization": USE_QUANTIZATION,
        "early_stopping_enabled": EARLY_STOPPING_ENABLED,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
        "min_rounds": MIN_ROUNDS,
        "dataset_version": os.environ.get("DATASET_VERSION", "dataset_v001")
    }
    run_manager.init_run(run_config)

    # Initial Baseline Validation Evaluation before Round 1
    init_eval_results = run_initial_baseline_evaluation(
        initial_model=init_model,
        hospital_ids=list(KNOWN_CLIENTS),
        manager=run_manager,
        target_device=device
    )

    strategy = SaveModelStrategy(
        run_manager_inst=run_manager,
        total_rounds=TOTAL_ROUNDS,
        early_stopping_enabled=EARLY_STOPPING_ENABLED,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        early_stopping_min_delta=EARLY_STOPPING_MIN_DELTA,
        min_rounds=MIN_ROUNDS,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=TARGET_CLIENTS,
        min_evaluate_clients=TARGET_CLIENTS,
        min_available_clients=MIN_CLIENTS,
        initial_parameters=init_params,
    )

    client_manager = fl.server.SimpleClientManager()

    if USE_DP:
        default_hard_deadline = 180.0
        default_round_timeout = 1800.0
    else:
        default_hard_deadline = 60.0
        default_round_timeout = 300.0

    DROPOUT_HARD_DEADLINE = float(os.environ.get("DROPOUT_HARD_DEADLINE", default_hard_deadline))
    ROUND_TIMEOUT = float(os.environ.get("ROUND_TIMEOUT", default_round_timeout))

    server = AdaptiveServer(
        client_manager=client_manager,
        strategy=strategy,
        target_clients=TARGET_CLIENTS,
        min_clients=MIN_CLIENTS,
        total_rounds=TOTAL_ROUNDS,
        hard_deadline=DROPOUT_HARD_DEADLINE,
        alpha=0.3,
        beta=0.3,
        k=1.0,
        suffix=SUFFIX,
        models_dir=str(MODEL_DIR)
    )

    FL_SERVER_BIND = os.environ.get("FL_SERVER_BIND", "0.0.0.0")
    FL_PORT = int(os.environ.get("FL_PORT", "8080"))
    bind_address = f"{FL_SERVER_BIND}:{FL_PORT}"

    print(f"Starting Periodic Retraining Federated Server...")
    print(f"  Run ID              : {run_manager.run_id}")
    print(f"  Training Mode       : {TRAINING_MODE}")
    print(f"  Binding on          : {bind_address}")
    print(f"  Target clients      : {TARGET_CLIENTS}")
    print(f"  Minimum clients     : {MIN_CLIENTS}")
    print(f"  FL rounds           : {TOTAL_ROUNDS}")
    print(f"  Early Stopping      : {EARLY_STOPPING_ENABLED} (patience={EARLY_STOPPING_PATIENCE}, min_delta={EARLY_STOPPING_MIN_DELTA}, min_rounds={MIN_ROUNDS})")
    print(f"  DP enabled          : {USE_DP}")
    print(f"  Quantization        : {USE_QUANTIZATION}")
    print()

    fl.server.start_server(
        server_address=bind_address,
        server=server,
        config=fl.server.ServerConfig(
            num_rounds=TOTAL_ROUNDS,
            round_timeout=ROUND_TIMEOUT
        ),
        grpc_max_message_length=1024*1024*1024
    )

    # --------------------------------------------------
    # POST-TRAINING FINAL INDEPENDENT TESTING
    # --------------------------------------------------
    print("\n" + "=" * 65)
    print("  FINAL INDEPENDENT TEST EVALUATION (RESERVED TEST COHORT)")
    print("=" * 65)

    gate = ModelApprovalGate(APPROVED_MODELS_DIR)
    selected_ckpt = run_manager.best_model_path if run_manager.best_model_path.exists() else run_manager.final_model_path

    test_results = gate.evaluate_checkpoint_on_data(
        checkpoint_path=selected_ckpt,
        hospital_ids=list(KNOWN_CLIENTS),
        split="test",
        device=device
    )

    test_rows = []
    for h_id, m in sorted(test_results["per_hospital"].items()):
        test_rows.append({
            "client": h_id,
            "sample_count": m["total"],
            "correct": m["correct"],
            "incorrect": m["total"] - m["correct"],
            "tp": m["tp"],
            "tn": m["tn"],
            "fp": m["fp"],
            "fn": m["fn"],
            "loss": m["loss"],
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"]
        })
        print(f"  {h_id:<14} | Samples: {m['total']:<4} | Acc: {m['accuracy']*100:.2f}% | Loss: {m['loss']:.4f} | F1: {m['f1']:.4f} | Rec: {m['recall']:.4f}")

    p_test = test_results["pooled"]
    test_rows.append({
        "client": "POOLED_GLOBAL",
        "sample_count": p_test["total"],
        "correct": "",
        "incorrect": "",
        "tp": "",
        "tn": "",
        "fp": "",
        "fn": "",
        "loss": p_test["loss"],
        "accuracy": p_test["accuracy"],
        "precision": p_test["precision"],
        "recall": p_test["recall"],
        "f1": p_test["f1"]
    })
    print("-" * 65)
    print(f"  POOLED TEST    | Samples: {p_test['total']:<4} | Acc: {p_test['accuracy']*100:.2f}% | Loss: {p_test['loss']:.4f} | F1: {p_test['f1']:.4f} | Rec: {p_test['recall']:.4f}")
    print("=" * 65 + "\n")

    test_df = pd.DataFrame(test_rows)
    test_df.to_csv(run_manager.final_test_metrics_path, index=False)

    # --------------------------------------------------
    # MODEL APPROVAL WORKFLOW
    # --------------------------------------------------
    baseline_metrics = None
    if gate.approved_model_path.exists():
        try:
            baseline_metrics = gate.evaluate_checkpoint_on_data(
                checkpoint_path=gate.approved_model_path,
                hospital_ids=list(KNOWN_CLIENTS),
                split="test",
                device=device
            )
        except Exception as bex:
            print(f"[APPROVAL] Warning: Could not evaluate existing approved baseline: {bex}")

    promoted, decision_reason = gate.decide_and_apply(
        candidate_path=selected_ckpt,
        candidate_metrics=test_results,
        run_id=run_manager.run_id,
        baseline_metrics=baseline_metrics
    )

    run_manager.finalize(
        final_round=strategy.current_round_number,
        early_stopped=strategy.early_stop_triggered,
        early_stopping_reason=strategy.early_stop_reason,
        final_test_metrics=p_test
    )
    run_manager.metadata["promotion_result"] = "PROMOTED" if promoted else "REJECTED"
    run_manager.metadata["promotion_reason"] = decision_reason
    run_manager.save_metadata()

    print(f"\n[Server] Retraining cycle finished successfully for Run {run_manager.run_id}!")
