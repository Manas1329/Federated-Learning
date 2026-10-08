import flwr as fl
import os
import time
import pandas as pd
import torch
import math
from typing import Optional, Set, Dict, List, Tuple

# Trust / Tagging Module (Phase 1)
# Wrapped in try/except: if import fails, training continues normally
try:
    from trust_manager import trust_manager as _tm
    _TRUST_OK = True
except Exception as _te:
    _TRUST_OK = False
    print(f"[TrustManager] Import warning: {_te}")

# Load environment variables from .env file if present (search cwd, src, parent, project root)
from pathlib import Path
_curr = Path(__file__).resolve().parent
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
# Paths & Config
# --------------------------------------------------

USE_DP = os.environ.get("USE_DP", "0") == "1"
USE_QUANTIZATION = os.environ.get("USE_QUANTIZATION", "1") == "1"
TOTAL_ROUNDS = int(os.environ.get("NUM_ROUNDS", "10"))

if "EXPERIMENT_NAME" in os.environ:
    SUFFIX = os.environ["EXPERIMENT_NAME"]
elif USE_DP:
    SUFFIX = "c_dp"
elif USE_QUANTIZATION:
    SUFFIX = "b_quantized"
else:
    SUFFIX = "a_pure"

# Set of all known federation participants — used by the auto-unlearn trigger.
# Overridable via environment variable: KNOWN_CLIENTS=Hospital_A,Hospital_B,Hospital_C
KNOWN_CLIENTS: set = {
    c.strip()
    for c in os.environ.get("KNOWN_CLIENTS", "Hospital_A,Hospital_B,Hospital_C").split(",")
    if c.strip()
}

from pathlib import Path
import sys
# Ensure 'src' package is importable regardless of where the script is run from
sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import RESULTS_DIR, MODELS_DIR

RESULTS_DIR_SUFFIX = RESULTS_DIR / SUFFIX
RESULTS_DIR_SUFFIX.mkdir(parents=True, exist_ok=True)

METRICS_FILE = RESULTS_DIR_SUFFIX / f"metrics_{SUFFIX}.csv"
ROUND_METRICS_FILE = RESULTS_DIR_SUFFIX / f"round_metrics_{SUFFIX}.csv"
MODEL_DIR = MODELS_DIR
MODEL_PATH = MODEL_DIR / f"global_model_{SUFFIX}.pth"


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
# Post-Training Unlearning Auto-Trigger
# --------------------------------------------------

def _trigger_post_training_unlearning(
    quarantined_clients: set,
    suffix: str,
    active_clients: set,
):
    """
    Called automatically after the final FL round checkpoint is saved.

    For each client that was quarantined at any point during training,
    this function runs the full post-training unlearning pipeline:
        1. Gradient-ascent erasure of the target client's historical contribution.
        2. DACM distribution audit on surviving clients.
        3. Compensatory FedAvg recovery retraining.
        4. Three-stage evaluation (before / after unlearn / after recovery).
        5. Saves the recovered model as dacm_recovered_<target>_<suffix>.pth

    The existing quarantine/gating logic is NOT affected — exclusion from the
    current round's FedAvg has already happened before this function is called.
    """
    print("\n" + "=" * 62)
    print("  [AUTO-UNLEARN] POST-TRAINING UNLEARNING TRIGGERED")
    print("=" * 62)
    print(f"  [AUTO-UNLEARN] Quarantined clients detected : {quarantined_clients}")
    print(f"  [AUTO-UNLEARN] Final checkpoint suffix      : {suffix}")
    print(f"  [AUTO-UNLEARN] Active federation members   : {sorted(active_clients)}")
    print("  [AUTO-UNLEARN] Reason: Historical contributions from quarantined")
    print("  [AUTO-UNLEARN] clients remain in the global model from earlier rounds.")
    print("  [AUTO-UNLEARN] Gradient-ascent unlearning + DACM recovery will now")
    print("  [AUTO-UNLEARN] remove that influence from the saved checkpoint.")
    print("=" * 62)

    # Lazy import — avoids touching flwr/tensorflow chain at server startup.
    try:
        from unlearn_workflow import run_unlearn_workflow
    except Exception as import_err:
        print(f"[AUTO-UNLEARN] ERROR: Could not import unlearn_workflow: {import_err}")
        print("[AUTO-UNLEARN] Post-training unlearning could not run.")
        return

    # Process each quarantined client sequentially.
    # active_clients is mutated in-place by run_unlearn_workflow on success,
    # so subsequent unlearning calls see the correct surviving set.
    remaining_active = set(active_clients)
    for target in sorted(quarantined_clients):
        print(f"\n[AUTO-UNLEARN] ── Starting post-training unlearning for: {target} ──")
        try:
            success = run_unlearn_workflow(
                target_client=target,
                suffix=suffix,
                active_clients=remaining_active,
            )
            if success:
                print(f"[AUTO-UNLEARN] ✓ Unlearning workflow completed for {target}.")
                print(f"[AUTO-UNLEARN]   {target} has been erased from the global model.")
                print(f"[AUTO-UNLEARN]   Remaining active clients: {sorted(remaining_active)}")
                # Record in all_global_models_registry.csv if present
                registry_file = MODELS_DIR / "all_global_models_registry.csv"
                if registry_file.exists():
                    try:
                        import datetime
                        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        rec_model_name = f"dacm_recovered_{target}_{suffix}"
                        surv_str = "|".join(sorted(remaining_active))
                        with open(registry_file, "a", newline="") as rf:
                            import csv
                            writer = csv.writer(rf)
                            writer.writerow([
                                ts,
                                rec_model_name,
                                surv_str,
                                len(remaining_active),
                                0,
                                len(remaining_active),
                                1,
                                "",
                                ""
                            ])
                        print(f"[AUTO-UNLEARN] Registered {rec_model_name} in model registry associated with {surv_str}.")
                    except Exception as reg_err:
                        print(f"[AUTO-UNLEARN] Note: Could not update registry: {reg_err}")
            else:
                print(f"[AUTO-UNLEARN] ✗ Unlearning workflow returned False for {target}.")
                print(f"[AUTO-UNLEARN]   Check logs above for the specific failure reason.")
        except Exception as unlearn_err:
            print(f"[AUTO-UNLEARN] ✗ Exception during unlearning of {target}: {unlearn_err}")
            import traceback
            traceback.print_exc()
            print(f"[AUTO-UNLEARN]   Continuing to next quarantined client (if any).")

    print("\n" + "=" * 62)
    print("  [AUTO-UNLEARN] POST-TRAINING UNLEARNING STAGE COMPLETE")
    print("=" * 62)


# --------------------------------------------------
# Persistent Exclusion Helpers
# --------------------------------------------------

def _get_excluded_cids(excluded_names: set) -> set:
    """
    Translate a set of human-readable client names into the Flower cid strings
    that the ClientProxy/ClientManager use internally.

    Reads from trust_manager.cid_to_name (populated by record_update after
    round 1).  Returns an empty set when TrustManager is unavailable or when
    no cid mapping exists yet (e.g., round 1).
    """
    if not _TRUST_OK or not excluded_names:
        return set()
    return {
        cid
        for cid, name in _tm.cid_to_name.items()
        if name in excluded_names
    }


def _check_promote_to_excluded(
    quarantined_names: set,
    excluded_names: set,
    rejected_names: set = None,
) -> set:
    """
    Examine clients with gating decisions from the current round against their
    PREVIOUS round's Trust Tag (stored in latest_trust).

    Rules:
    1. SUSPICIOUS tag + QUARANTINE decision -> Permanent Exclusion
    2. UNTRUSTED tag  + (QUARANTINE or REJECT) decision -> Permanent Exclusion

    Uses the previous-round tag because finalize_round() — which assigns the
    current-round tag — has not yet run at aggregate_fit time.

    Returns the set of names that were newly promoted this call.
    """
    newly_excluded: set = set()
    if not _TRUST_OK:
        return newly_excluded

    if rejected_names is None:
        rejected_names = set()

    # Build a reverse lookup: name -> cid (trust_manager stores cid->name)
    name_to_cid = {v: k for k, v in _tm.cid_to_name.items()}

    def _get_tag(name: str) -> Optional[str]:
        trust_info = None
        # Primary lookup: direct from client_state by client name
        if hasattr(_tm, "client_state") and name in _tm.client_state:
            trust_info = _tm.client_state[name].get("latest_trust")
        if not trust_info:
            cid = name_to_cid.get(name)
            if cid:
                trust_info = _tm.get_trust_for_cid(cid)
        return trust_info.get("tag") if trust_info else None

    # Check quarantined clients: SUSPICIOUS or UNTRUSTED
    for name in quarantined_names:
        if name in excluded_names:
            continue  # already excluded
        tag = _get_tag(name)
        if tag == "SUSPICIOUS":
            newly_excluded.add(name)
            print(
                f"\n[EXCL] {name} -> PERMANENTLY EXCLUDED"
                f" (QUARANTINE + SUSPICIOUS tag)."
                f" Will not be selected for training or evaluation"
                f" in any subsequent round."
            )
        elif tag == "UNTRUSTED":
            newly_excluded.add(name)
            print(
                f"\n[EXCL] {name} -> PERMANENTLY EXCLUDED"
                f" (QUARANTINE + UNTRUSTED tag)."
                f" Will not be selected for training or evaluation"
                f" in any subsequent round."
            )

    # Check rejected clients: UNTRUSTED only
    for name in rejected_names:
        if name in excluded_names or name in newly_excluded:
            continue  # already excluded
        tag = _get_tag(name)
        if tag == "UNTRUSTED":
            newly_excluded.add(name)
            print(
                f"\n[EXCL] {name} -> PERMANENTLY EXCLUDED"
                f" (REJECT + UNTRUSTED tag)."
                f" Will not be selected for training or evaluation"
                f" in any subsequent round."
            )

    return newly_excluded


# --------------------------------------------------
# Custom FedAvg Strategy
# --------------------------------------------------

class SaveModelStrategy(fl.server.strategy.FedAvg):

    # ---> DACM: ADD INIT TO TRACK WEIGHTS <---
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Default weight is 1.0 (normal training)
        self.dacm_recovery_weight = 1.0
        self.dacu_recovery_weight = 1.0
        # Accumulates every client name that received a QUARANTINE decision
        # across any round during training. Populated in aggregate_fit.
        # Used after the final round to auto-trigger post-training unlearning.
        self.quarantined_clients: set = set()
        # Total rounds configured for this run — used to detect the final round.
        self.total_rounds: int = TOTAL_ROUNDS
        # ── Persistent exclusion ──────────────────────────────────────────
        # Clients added here are NEVER selected for training or evaluation again.
        # Populated when a client gets QUARANTINE gate + SUSPICIOUS trust tag.
        self.excluded_names: set = set()
        # Matching Flower cid strings — derived from excluded_names via cid_to_name.
        # Refreshed at the start of configure_fit / configure_evaluate each round.
        self.excluded_cids: set  = set()
        # Original target count, stored so we can decrement min_fit_clients /
        # min_evaluate_clients / min_available_clients when clients are excluded
        # and Flower would otherwise block waiting for the excluded client.
        self._initial_target_clients: int = int(
            self.min_fit_clients
        ) if hasattr(self, "min_fit_clients") else int(
            os.environ.get("TARGET_CLIENTS", "3")
        )

    def _refresh_exclusion(self) -> None:
        """Re-derive excluded_cids from excluded_names via trust_manager's cid map.

        Safe to call every round: the cid_to_name map grows monotonically as
        clients connect, so later calls may resolve names that were unresolvable
        in earlier rounds.
        """
        self.excluded_cids = _get_excluded_cids(self.excluded_names)

    def _reduce_quorum_for_exclusions(self) -> None:
        """
        Adjust min_fit_clients, min_evaluate_clients, and min_available_clients
        to reflect the reduced federation size after permanent exclusions.

        Without this, Flower's client_manager.sample() will block indefinitely
        waiting for the excluded client to become available.

        The adjusted value is max(1, original_target - len(excluded)).
        """
        n_excluded = len(self.excluded_names)
        if n_excluded == 0:
            return
        new_target = max(1, self._initial_target_clients - n_excluded)
        new_min    = max(1, new_target - 1)          # at least 1 below target
        if self.min_fit_clients != new_target or self.min_available_clients > new_min:
            print(
                f"[EXCL] Adjusting Flower quorum: "
                f"min_fit_clients {self.min_fit_clients} -> {new_target}, "
                f"min_available_clients {self.min_available_clients} -> {new_min}"
            )
            self.min_fit_clients      = new_target
            self.min_evaluate_clients = new_target
            self.min_available_clients = new_min

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

        # ── Sync exclusion cids and adjust quorum before sampling ─────────────
        self._refresh_exclusion()
        self._reduce_quorum_for_exclusions()

        if self.excluded_names:
            print(
                f"[EXCL] Permanently excluded clients (will not be selected): "
                f"{sorted(self.excluded_names)}"
            )

        # Unregister permanently excluded clients from Flower client_manager
        # so they can never be sampled by client_manager
        if self.excluded_cids:
            all_cm_clients = client_manager.all()
            for exc_cid in list(self.excluded_cids):
                if exc_cid in all_cm_clients:
                    try:
                        client_manager.unregister(all_cm_clients[exc_cid])
                    except Exception:
                        pass

        # Store global params for trust update-norm computation
        if _TRUST_OK:
            try:
                _tm.set_global_params(parameters_to_ndarrays(parameters))
            except Exception:
                pass

        base_config = {"server_round": server_round}

        # ---> DACM: INJECT THE WEIGHT INTO THE CLIENT CONFIG <---
        dacm_weight = getattr(self, 'dacm_recovery_weight', getattr(self, 'dacu_recovery_weight', 1.0))
        base_config["pneumonia_weight"] = float(dacm_weight)

        available_clients = client_manager.num_available()
        sample_size = min(self.min_fit_clients, available_clients)

        # Select clients using client_manager passed directly into function
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=self.min_available_clients
        )

        # ── Filter out permanently excluded clients ───────────────────────────
        # Flower's client_manager has no awareness of our exclusion set, so we
        # filter the sampled list here.  The quorum adjustment above ensures
        # sample_size already reflects the smaller active federation.
        if self.excluded_cids:
            before = len(clients)
            clients = [c for c in clients if c.cid not in self.excluded_cids]
            after   = len(clients)
            if before != after:
                excluded_str = ", ".join(sorted(self.excluded_names))
                print(
                    f"[EXCL] configure_fit Round {server_round}: "
                    f"removed {before - after} excluded client(s) from fit list "
                    f"[{excluded_str}]"
                )

        # Build per-client FitIns — send only operational training config (trust is server-side only)
        result = []
        for client in clients:
            config = dict(base_config)
            result.append((client, fl.common.FitIns(parameters, config)))

        return result

    def configure_evaluate(
        self,
        server_round,
        parameters,
        client_manager,
    ):
        self._refresh_exclusion()
        self._reduce_quorum_for_exclusions()

        # Unregister permanently excluded clients from Flower client_manager
        if self.excluded_cids:
            all_cm_clients = client_manager.all()
            for exc_cid in list(self.excluded_cids):
                if exc_cid in all_cm_clients:
                    try:
                        client_manager.unregister(all_cm_clients[exc_cid])
                    except Exception:
                        pass

        config = {
            "server_round": server_round
        }
        evaluate_ins = fl.common.EvaluateIns(
            parameters,
            config
        )
        available_clients = client_manager.num_available()
        sample_size = min(self.min_evaluate_clients, available_clients)

        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=self.min_available_clients
        )

        # ── Filter out permanently excluded clients from evaluation ────────────
        if self.excluded_cids:
            before = len(clients)
            clients = [c for c in clients if c.cid not in self.excluded_cids]
            if before != len(clients):
                print(
                    f"[EXCL] configure_evaluate Round {server_round}: "
                    f"removed {before - len(clients)} excluded client(s) from eval list"
                )

        return [
            (client, evaluate_ins)
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

        if USE_QUANTIZATION:
            for client_proxy, fit_res in results:
                try:
                    # Convert Flower Parameters -> NumPy arrays
                    quantized_parameters = parameters_to_ndarrays(fit_res.parameters)

                    # INT8 -> FP32
                    fp32_parameters = dequantize_parameters(quantized_parameters)

                    # Convert FP32 NumPy arrays back to Flower Parameters
                    fp32_parameters_flower = ndarrays_to_parameters(fp32_parameters)

                    # Replace the INT8 parameters with FP32 parameters
                    fit_res.parameters = fp32_parameters_flower

                    dequantized_results.append((client_proxy, fit_res))
                except Exception as e:
                    print(f"Error dequantizing client update: {e}")
        else:
            dequantized_results = results

        # ==========================================================
        # TRUST MODULE: record update behaviour per client (Stage 1)
        # Wrapped in try/except — FL training continues even if this fails
        # ==========================================================
        if _TRUST_OK:
            try:
                for _cp, _fit_res in dequantized_results:
                    _cname   = _fit_res.metrics.get("client_name", f"Client_{_cp.cid[:8]}")
                    _cndarrs = parameters_to_ndarrays(_fit_res.parameters)
                    _tm.record_update(_cname, _cndarrs, _cp.cid, _fit_res.metrics)
                for _fail in failures:
                    if isinstance(_fail, tuple) and len(_fail) >= 1:
                        _fail_cp = _fail[0]
                        if hasattr(_fail_cp, "cid"):
                            _tm.record_dropout(_fail_cp.cid)
            except Exception as _e:
                print(f"[TrustManager] aggregate_fit warning: {_e}")

        # ── Drop updates from any reconnected permanently excluded clients ────
        clean_dequantized = []
        for _cp, _fit_res in dequantized_results:
            _cname = _fit_res.metrics.get("client_name", f"Client_{_cp.cid[:8]}")
            if _cname in self.excluded_names:
                print(
                    f"\n[EXCL] RECONNECTION BLOCKED: {_cname} (CID: {_cp.cid}) is "
                    f"PERMANENTLY EXCLUDED. Discarding update from aggregation."
                )
                self.excluded_cids.add(_cp.cid)
                if _TRUST_OK:
                    _tm.cid_to_name[_cp.cid] = _cname
                continue
            clean_dequantized.append((_cp, _fit_res))
        dequantized_results = clean_dequantized

        # ==========================================================
        # PRE-AGGREGATION TRUST GATE
        # ──────────────────────────────────────────────────────────
        # After recording all client updates, apply the pre-aggregation
        # gating decision BEFORE calling FedAvg.
        #
        # WHY SEPARATE FROM FEDAVG?
        # - A quarantined/rejected client's update must NOT enter FedAvg.
        #   Passing it to super().aggregate_fit() would contaminate the
        #   global model before any unlearning/recovery can act.
        # - The existing DACM distribution audit block below this still
        #   runs on ALL clients (including quarantined) so it can detect
        #   distribution shifts across the full federation.
        # - FedAvg continues to use sample-count weighting for accepted
        #   clients — Trust Score is NOT used as a FedAvg weight.
        #
        # QUARANTINE → EXISTING DACM/UNLEARNING:
        # - Quarantined clients are excluded from FedAvg this round.
        # - The in-round DACM distribution audit (below) handles recovery
        #   weight computation for the next round.
        # - Post-training: run_unlearn_workflow() (unlearn_server.py) is
        #   available unchanged for full unlearning + DACM recovery.
        # ==========================================================
        accepted_results    = []
        quarantined_results = []
        rejected_results    = []

        if _TRUST_OK:
            try:
                _decisions = _tm.get_pre_agg_decisions()

                if _decisions:
                    print("\n" + "=" * 60)
                    print(f"  PRE-AGGREGATION TRUST GATE — Round {server_round}")
                    print("=" * 60)

                for _cp, _fit_res in dequantized_results:
                    _cname = _fit_res.metrics.get("client_name", f"Client_{_cp.cid[:8]}")
                    _dec   = _decisions.get(_cname, "ACCEPT")

                    if _dec == "ACCEPT":
                        accepted_results.append((_cp, _fit_res))
                        if _decisions:
                            print(f"  {_cname:<20} → ✅ ACCEPT   (enters FedAvg)")
                    elif _dec == "QUARANTINE":
                        quarantined_results.append((_cp, _fit_res))
                        # Accumulate for post-training unlearning (fired after final round)
                        self.quarantined_clients.add(_cname)
                        print(f"  {_cname:<20} -> QUARANTINE (excluded from FedAvg; "
                              f"scheduled for post-training unlearning after round {self.total_rounds})")
                    else:  # REJECT
                        rejected_results.append((_cp, _fit_res))
                        print(f"  {_cname:<20} -> REJECT    (excluded from aggregation)")

                if _decisions:
                    print("=" * 60)

                # ── Persistent exclusion:
                # 1. SUSPICIOUS + QUARANTINE -> Permanent Exclusion
                # 2. UNTRUSTED + (QUARANTINE or REJECT) -> Permanent Exclusion
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
                        # Immediately refresh cid mapping so subsequent rounds
                        # have the correct excluded_cids set.
                        self.excluded_cids = _get_excluded_cids(self.excluded_names)
                        # Adjust Flower quorum right away so the next round's
                        # configure_fit doesn't block on the excluded client.
                        self._reduce_quorum_for_exclusions()

                # Handle edge cases
                if not accepted_results:
                    if quarantined_results:
                        # All clients quarantined — fall back to accepting all to prevent
                        # a round with zero updates (catastrophic failure mode).
                        print("[TRUST GATE] WARNING: All clients quarantined/rejected.")
                        print("[TRUST GATE] Falling back to accepting all updates to prevent round failure.")
                        print("[TRUST GATE] Review anomaly thresholds in PRE_AGG_CONFIG.")
                        accepted_results = dequantized_results
                        quarantined_results = []
                    elif rejected_results:
                        print("[TRUST GATE] WARNING: All clients rejected — no valid updates.")
                        print("[TRUST GATE] Falling back to accepting all to prevent catastrophic failure.")
                        accepted_results = dequantized_results
                        rejected_results = []

            except Exception as _ge:
                print(f"[TRUST GATE] Warning: {_ge} — using all dequantized results")
                accepted_results    = dequantized_results
                quarantined_results = []
        else:
            # Trust module not loaded — pass all updates to FedAvg normally
            accepted_results = dequantized_results

        # Log gate summary
        if quarantined_results or rejected_results:
            _n_acc  = len(accepted_results)
            _n_quar = len(quarantined_results)
            _n_rej  = len(rejected_results)
            print(f"[TRUST GATE] FedAvg input: {_n_acc} ACCEPTED, "
                  f"{_n_quar} QUARANTINED, {_n_rej} REJECTED")

        # ==========================================================
        # FEDAVG (only accepted clients)
        # ==========================================================
        aggregated_parameters, aggregated_metrics = (
            super().aggregate_fit(
                server_round,
                accepted_results,   # ← only ACCEPTED updates enter FedAvg
                failures,
            )
        )

        # ==========================================================
        # AGGREGATION TIME & METRICS
        # ==========================================================
        aggregation_time = (time.perf_counter() - aggregation_start)
        successful_clients = len(dequantized_results)
        failed_clients = len(failures)

        if server_round in round_start_times:
            round_time = (time.perf_counter() - round_start_times[server_round])
        else:
            round_time = 0

        # ==========================================================
        # DACM: DISTRIBUTION AUDIT & DYNAMIC CALCULATION
        # ==========================================================
        total_normal = 0
        total_pneumonia = 0
        
        # Extract DP counts reported by all successful clients
        for _, fit_res in results:
            total_normal += fit_res.metrics.get("normal_count", 0)
            total_pneumonia += fit_res.metrics.get("pneumonia_count", 0)
            
        total = total_normal + total_pneumonia
        if total > 0:
            gamma = total_pneumonia / total
            print("\n" + "=" * 60)
            print(f"[DACM Server Audit] Surviving Pneumonia Ratio: {gamma*100:.2f}%")
            
            tau_safe = 0.50
            if gamma < tau_safe:
                # Calculate compensation penalty
                alpha = 3.0
                self.dacm_recovery_weight = 1.0 + alpha * math.log(tau_safe / gamma)
                print(f"[DACM ALERT] Minority class shortage detected (< {tau_safe*100:.0f}%)")
                print(f"[DACM LOGIC] Broadcasting recovery weight for next round: {self.dacm_recovery_weight:.4f}")
            else:
                self.dacm_recovery_weight = 1.0
                print(f"[DACM LOGIC] Distribution safe. Standard training continues.")
            print("=" * 60)

        # ==========================================================
        # PRINT RESULTS
        # ==========================================================
        print("\n")
        print("=" * 60)
        print(f"Round {server_round} completed")
        print("=" * 60)
        print(f"Successful Clients: {successful_clients}")
        print(f"Failed Clients: {failed_clients}")
        print(f"Dequantization + Aggregation Time: {aggregation_time:.4f} sec")
        print(f"Total Round Time: {round_time:.4f} sec")

        # ==========================================================
        # SAVE GLOBAL MODEL
        # ==========================================================
        if aggregated_parameters is not None:
            model = ChestCNN()
            params = parameters_to_ndarrays(aggregated_parameters)
            params_dict = zip(model.state_dict().keys(), params)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            
            model.load_state_dict(state_dict, strict=True)

            # Save final global model
            if server_round == TOTAL_ROUNDS:
                torch.save(model.state_dict(), MODEL_PATH)

                print("\n")
                print("=" * 60)
                print("Global Model Saved Successfully")
                print(MODEL_PATH)
                print("=" * 60)

                # ============================================================
                # AUTO-TRIGGER: Post-Training Unlearning for Excluded/Quarantined Clients
                # ============================================================
                # This fires AFTER the final-round checkpoint is saved, so the
                # unlearning pipeline always loads a clean, fully-converged model.
                #
                # Quarantine (above) only excluded the client from the CURRENT
                # round's FedAvg. The client's contributions from earlier rounds
                # are still present in the saved checkpoint. Gradient-ascent
                # unlearning + DACM recovery removes that historical influence.
                # ============================================================
                targets_to_unlearn = self.quarantined_clients | self.excluded_names
                if targets_to_unlearn:
                    # Pass the surviving active clients (KNOWN_CLIENTS minus
                    # those permanently excluded) so DACM recovery only involves
                    # clients that are genuinely still in the federation.
                    _surviving_clients = KNOWN_CLIENTS - self.excluded_names
                    _trigger_post_training_unlearning(
                        quarantined_clients=targets_to_unlearn,
                        suffix=SUFFIX,
                        active_clients=_surviving_clients,
                    )
                else:
                    print("[AUTO-UNLEARN] No quarantined or excluded clients detected — "
                          "post-training unlearning not required.")

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
            header=not os.path.exists(ROUND_METRICS_FILE),
            index=False
        )

        return (aggregated_parameters, aggregated_metrics)


# --------------------------------------------------
# Evaluation Metrics
# --------------------------------------------------

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

    # --------------------------------------------------
    # Weighted F1 / Precision / Recall (if clients sent them)
    # --------------------------------------------------
    has_f1 = all("f1" in m for _, m in metrics)

    if has_f1:
        weighted_f1 = sum(num_examples * m["f1"] for num_examples, m in metrics) / total_examples
        weighted_precision = sum(num_examples * m["precision"] for num_examples, m in metrics) / total_examples
        weighted_recall = sum(num_examples * m["recall"] for num_examples, m in metrics) / total_examples
    else:
        weighted_f1 = 0.0
        weighted_precision = 0.0
        weighted_recall = 0.0

    # --------------------------------------------------
    # Determine round number for CSV
    # --------------------------------------------------
    if os.path.exists(METRICS_FILE):
        df_old = pd.read_csv(METRICS_FILE)
        next_round = len(df_old) + 1
    else:
        next_round = 1

    # --------------------------------------------------
    # Save accuracy / loss
    # --------------------------------------------------
    df = pd.DataFrame(
        [[next_round, weighted_accuracy, weighted_loss]],
        columns=["Round", "Accuracy", "Loss"]
    )

    df.to_csv(
        METRICS_FILE,
        mode="a",
        header=not os.path.exists(METRICS_FILE),
        index=False
    )
    # --------------------------------------------------
    # Demo-friendly round results banner
    # --------------------------------------------------
    print("\n")
    print("=" * 56)
    print(f"  GLOBAL ROUND {next_round} RESULTS")
    print("=" * 56)
    print(f"  Clients Evaluated   : {len(metrics)}")
    print(f"  Global Accuracy     : {weighted_accuracy * 100:.2f}%")
    print(f"  Global Loss         : {weighted_loss:.4f}")
    if has_f1:
        print(f"  F1 Score            : {weighted_f1:.4f}")
        print(f"  Precision           : {weighted_precision:.4f}")
        print(f"  Recall              : {weighted_recall:.4f}")
    else:
        print("  F1 / Precision / Recall : N/A (metrics not returned by clients)")
    print("=" * 56)

    # ==========================================================
    # TRUST MODULE: record evaluation and finalise trust scores
    # Wrapped in try/except — does NOT affect FL accuracy/loss return
    # ==========================================================
    if _TRUST_OK:
        try:
            for _num_ex, _m in metrics:
                _cname = _m.get("client_name", "Unknown")
                if _cname == "Unknown":
                    continue
                _tm.record_evaluation(
                    client_name  = _cname,
                    accuracy     = float(_m.get("accuracy",  0.0)),
                    loss         = float(_m.get("loss",      0.0)),
                    f1           = float(_m.get("f1",        0.0)),
                    precision    = float(_m.get("precision", 0.0)),
                    recall       = float(_m.get("recall",    0.0)),
                    num_examples = int(_num_ex),
                )
            _tm.finalize_round(next_round)
        except Exception as _e:
            print(f"[TrustManager] evaluate_metrics warning: {_e}")

    return {
        "accuracy":  weighted_accuracy,
        "loss":      weighted_loss,
        "f1":        weighted_f1,
        "precision": weighted_precision,
        "recall":    weighted_recall,
    }
# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    import math
    from federated_healthcare.src.dropout_handler import AdaptiveServer
    
    # Configure TARGET and MIN clients
    TARGET_CLIENTS = int(os.environ.get("TARGET_CLIENTS", "3"))
    MIN_CLIENTS = max(2, math.ceil(0.6 * TARGET_CLIENTS))

    strategy = SaveModelStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=TARGET_CLIENTS,
        min_evaluate_clients=TARGET_CLIENTS,
        min_available_clients=MIN_CLIENTS,
        evaluate_metrics_aggregation_fn=(
            evaluate_metrics_aggregation_fn
        ),
    )

    client_manager = fl.server.SimpleClientManager()
    
    # DP-SGD is much slower, so we need much larger timeouts
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
        models_dir=MODEL_DIR
    )

    # Bind address: default 0.0.0.0 (all network interfaces, reachable by remote clients).
    # Override with FL_SERVER_BIND if you need to bind to a specific interface.
    # Override with FL_PORT if you need a non-default port.
    # Do NOT set this to 'localhost' or '127.0.0.1' in a real multi-device deployment —
    # remote hospital clients would not be able to connect.
    FL_SERVER_BIND = os.environ.get("FL_SERVER_BIND", "0.0.0.0")
    FL_PORT = int(os.environ.get("FL_PORT", "8080"))
    bind_address = f"{FL_SERVER_BIND}:{FL_PORT}"

    print(f"Starting Adaptive Flower Server with Dropout Handling...")
    print(f"  Binding on          : {bind_address}")
    print(f"  Target clients      : {TARGET_CLIENTS}")
    print(f"  Minimum clients     : {MIN_CLIENTS}")
    print(f"  FL rounds           : {TOTAL_ROUNDS}")
    print(f"  Round timeout       : {ROUND_TIMEOUT} sec")
    print(f"  Dropout hard limit  : {DROPOUT_HARD_DEADLINE} sec")
    print(f"  DP enabled          : {USE_DP}")
    print(f"  Quantization        : {USE_QUANTIZATION}")
    print()
    print(f"  Remote hospital clients should connect with:")
    print(f"    SERVER_ADDRESS=<this-machine-LAN-IP>:{FL_PORT}")
    print(f"    python federated_healthcare/src/client.py")
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