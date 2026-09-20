"""
trust_manager.py
================
Phase-1 Client Trust / Tagging Module for Federated Healthcare AI.

Design:
  - Each client is compared ONLY to its OWN history  →  Non-IID safe
  - Trust changes gradually via EMA                  →  one bad round != untrusted
  - Failure-safe (every external call should be try/except)
  - Does NOT modify aggregation, training, DP, or quantization

Trust Score =
    0.40 x Update Behaviour Score   (L2 norm anomaly vs client's OWN history)
  + 0.30 x Training Behaviour Score (acc/loss trend per client's OWN baseline)
  + 0.20 x Historical Reputation    (EMA of past trust scores)
  + 0.10 x Participation Reliability (rounds succeeded / rounds total)

Tagging thresholds (configurable in TRUST_CONFIG):
    80-100  ->  TRUSTED
    50-79   ->  SUSPICIOUS
    0-49    ->  UNTRUSTED

Usage in server.py:
    from trust_manager import trust_manager as tm
    # configure_fit    : tm.set_global_params(...)
    # aggregate_fit    : tm.record_update(...),  tm.record_dropout(...)
    # evaluate_agg_fn  : tm.record_evaluation(...), tm.finalize_round(round)
"""

import csv
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

# =============================================================
# CONFIGURABLE CONSTANTS
# Change these values to tune the trust system behaviour.
# =============================================================

TRUST_CONFIG: Dict = {
    # Weighted combination of the four component scores
    "weights": {
        "update_behaviour":          0.40,
        "training_behaviour":        0.30,
        "historical_reputation":     0.20,
        "participation_reliability": 0.10,
    },

    # Tag boundaries (0-100 scale)
    # score >= TRUSTED    -> TRUSTED
    # score >= SUSPICIOUS -> SUSPICIOUS
    # below SUSPICIOUS    -> UNTRUSTED
    "thresholds": {
        "TRUSTED":    80.0,
        "SUSPICIOUS": 50.0,
    },

    # EMA decay: New_hist = (1-alpha)*Old_hist + alpha*CurrentScore
    # alpha=0.30 -> one bad round shifts trust by at most 30% of the gap
    "historical_ema_alpha": 0.30,

    # Minimum rounds of norm history before anomaly detection starts.
    # Before this, the update score returns neutral_score (no penalty).
    "min_history_for_anomaly": 3,

    # Default score used when there is insufficient history / missing data
    "neutral_score": 75.0,

    # Initial historical trust for any brand-new client (neutral, not penalised)
    "initial_historical_trust": 75.0,
}

# Emoji labels for tags
TAG_DISPLAY = {
    "TRUSTED":    "TRUSTED",
    "SUSPICIOUS": "SUSPICIOUS",
    "UNTRUSTED":  "UNTRUSTED",
}

TAG_EMOJI = {
    "TRUSTED":    "🟢 TRUSTED",
    "SUSPICIOUS": "🟡 SUSPICIOUS",
    "UNTRUSTED":  "🔴 UNTRUSTED",
}


# =============================================================
# Path helpers
# =============================================================

_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.dirname(_SRC_DIR)


def _get_results_dir() -> Tuple[str, str]:
    """Return (results_dir, suffix) based on current environment."""
    use_dp    = os.environ.get("USE_DP",           "0") == "1"
    use_quant = os.environ.get("USE_QUANTIZATION", "1") == "1"
    suffix    = "c_dp" if use_dp else ("b_quantized" if use_quant else "a_pure")
    path      = os.path.join(_BASE_DIR, "dashboard", "results", suffix)
    os.makedirs(path, exist_ok=True)
    return path, suffix


# =============================================================
# TrustManager class
# =============================================================

class TrustManager:
    """
    Singleton that maintains per-client trust state across FL rounds.

    Flower's server callbacks are sequential within a round:
        configure_fit -> aggregate_fit -> configure_evaluate -> aggregate_evaluate
    so we do not need internal locks.
    """

    def __init__(self):
        # Per-client persistent state
        # {client_name: {norm_history, acc_history, loss_history,
        #                historical_trust, participation, trust_rounds,
        #                current_norm, current_acc, current_loss, latest_trust}}
        self.client_state: Dict[str, Dict] = {}

        # Previous round's global parameters (list of float64 ndarrays)
        self._global_params: Optional[List[np.ndarray]] = None

        # CID to client name mapping (built from fit_res.metrics["client_name"])
        self.cid_to_name: Dict[str, str] = {}

        # Sets tracking participation in the CURRENT round
        self._round_participants: set = set()   # submitted update successfully
        self._round_dropouts:     set = set()   # failed / timed out

        # Last completed round results (used by configure_fit)
        self.last_round_results: Dict[str, Dict] = {}

        # CSV path (lazy-init)
        self._csv_path: Optional[str] = None

    # ----------------------------------------------------------
    # Client state initialisation
    # ----------------------------------------------------------

    def _init_client(self, name: str) -> None:
        """Initialise state for a new client (idempotent)."""
        if name not in self.client_state:
            self.client_state[name] = {
                # Per-round history vectors (grow each round)
                "norm_history":  [],   # L2 update norms
                "acc_history":   [],   # local evaluation accuracy
                "loss_history":  [],   # local evaluation loss

                # Historical trust (EMA updated after each round)
                "historical_trust": float(TRUST_CONFIG["initial_historical_trust"]),

                # Participation counter
                "participation": {"success": 0, "total": 0},

                # Full trust history: [(round, score, tag), ...]
                "trust_rounds": [],

                # Latest trust result dict (used by configure_fit)
                "latest_trust": None,
            }

    # ----------------------------------------------------------
    # Called from configure_fit — store current global params
    # ----------------------------------------------------------

    def set_global_params(self, params_ndarrays: List[np.ndarray]) -> None:
        """Store the current global model parameters for update-norm computation."""
        self._global_params = [p.astype(np.float64) for p in params_ndarrays]

    # ----------------------------------------------------------
    # Called from aggregate_fit for each successful client
    # ----------------------------------------------------------

    def record_update(
        self,
        client_name:     str,
        client_ndarrays: List[np.ndarray],
        cid:             str,
        fit_metrics:     Optional[Dict] = None,
    ) -> None:
        """Record a client's model update for trust computation."""
        self._init_client(client_name)
        self.cid_to_name[cid] = client_name
        self._round_participants.add(client_name)

        # Compute L2 norm of the update (client_params - global_params)
        norm = self._compute_update_norm(client_ndarrays)
        self.client_state[client_name]["current_norm"] = norm

    # ----------------------------------------------------------
    # Called from aggregate_fit for each failed client
    # ----------------------------------------------------------

    def record_dropout(self, cid_or_name: str) -> None:
        """Record a client dropout/failure."""
        # Resolve CID -> name if possible
        name = self.cid_to_name.get(cid_or_name, cid_or_name)
        self._round_dropouts.add(name)

    # ----------------------------------------------------------
    # Called from evaluate_metrics_aggregation_fn
    # ----------------------------------------------------------

    def record_evaluation(
        self,
        client_name:  str,
        accuracy:     float,
        loss:         float,
        f1:           float = 0.0,
        precision:    float = 0.0,
        recall:       float = 0.0,
        num_examples: int   = 0,
    ) -> None:
        """Record evaluation metrics for a client (from evaluate() return)."""
        self._init_client(client_name)
        s = self.client_state[client_name]
        s["current_acc"]  = float(accuracy)
        s["current_loss"] = float(loss)
        s["current_f1"]   = float(f1)

    # ----------------------------------------------------------
    # Main: compute and finalise trust for all clients this round
    # ----------------------------------------------------------

    def finalize_round(self, server_round: int) -> Dict[str, Dict]:
        """
        Compute trust scores for all clients that participated in this round.
        Call this at the END of evaluate_metrics_aggregation_fn.
        Returns {client_name: trust_result_dict}.
        """
        results: Dict[str, Dict] = {}

        # --- Handle pure dropouts (failed fit, no update submitted) ---
        pure_dropouts = self._round_dropouts - self._round_participants
        for name in pure_dropouts:
            self._init_client(name)
            self.client_state[name]["participation"]["total"] += 1

        # --- Compute trust for clients that submitted updates ---
        for name in sorted(self._round_participants):
            self._init_client(name)
            data = self.client_state[name]

            # Component scores
            update_score   = self._calc_update_score(name)
            training_score = self._calc_training_score(name)
            hist_score     = data["historical_trust"]
            part_score     = self._calc_participation_score_preview(name)

            # Weighted combination
            w = TRUST_CONFIG["weights"]
            trust_score = (
                w["update_behaviour"]          * update_score   +
                w["training_behaviour"]        * training_score +
                w["historical_reputation"]     * hist_score     +
                w["participation_reliability"] * part_score
            )
            trust_score = float(np.clip(trust_score, 0.0, 100.0))

            # --- Update histories (AFTER using them for scoring) ---
            current_norm = data.pop("current_norm", None)
            current_acc  = data.pop("current_acc",  None)
            current_loss = data.pop("current_loss", None)
            data.pop("current_f1",           None)
            data.pop("current_fit_metrics",  None)

            if current_norm is not None and current_norm > 0.0:
                data["norm_history"].append(float(current_norm))
            if current_acc  is not None:
                data["acc_history"].append(float(current_acc))
            if current_loss is not None:
                data["loss_history"].append(float(current_loss))

            # --- Update participation ---
            data["participation"]["total"]   += 1
            data["participation"]["success"] += 1
            final_part_score = self._calc_participation_score(name)

            # --- Update historical trust (EMA) ---
            alpha = TRUST_CONFIG["historical_ema_alpha"]
            data["historical_trust"] = (
                (1.0 - alpha) * data["historical_trust"] +
                alpha         * trust_score
            )

            # Assign tag
            tag = self._assign_tag(trust_score)

            # Build result dict
            result = {
                "client_id":         name,
                "round":             server_round,
                "update_score":      round(update_score,     1),
                "training_score":    round(training_score,   1),
                "historical_score":  round(hist_score,       1),
                "reliability_score": round(final_part_score, 1),
                "trust_score":       round(trust_score,      1),
                "tag":               tag,
            }

            # Append to per-client round history
            data["trust_rounds"].append(
                (server_round, round(trust_score, 1), tag)
            )
            result["history"] = list(data["trust_rounds"])

            # Store as latest_trust (for next round's configure_fit injection)
            data["latest_trust"] = result
            results[name] = result

        # Reset round tracking sets
        self._round_participants = set()
        self._round_dropouts     = set()

        # Publish results
        if results:
            self._log_to_csv(results, server_round)
            self._print_trust_table(results, server_round)
            self._update_dashboard_state(results)

        self.last_round_results = results
        return results

    # ----------------------------------------------------------
    # Configure_fit helper
    # ----------------------------------------------------------

    def get_trust_for_cid(self, cid: str) -> Optional[Dict]:
        """
        Return last round's trust result for the client with this CID.
        Returns None if unknown (e.g., first round).
        """
        name = self.cid_to_name.get(cid)
        if not name:
            return None
        return self.client_state.get(name, {}).get("latest_trust")

    # ----------------------------------------------------------
    # Score calculations (all Non-IID safe: client vs. own history)
    # ----------------------------------------------------------

    def _compute_update_norm(self, client_ndarrays: List[np.ndarray]) -> float:
        """
        Compute L2 norm of the parameter update (client_params - global_params).
        If global_params is None (first round), uses raw parameter norm.
        """
        if (self._global_params is None or
                len(self._global_params) != len(client_ndarrays)):
            # First round: use raw parameter norm as baseline
            return float(sum(
                np.linalg.norm(p.astype(np.float64).flatten())
                for p in client_ndarrays
            ))

        total_sq = 0.0
        for c, g in zip(client_ndarrays, self._global_params):
            diff      = c.astype(np.float64) - g
            total_sq += float(np.sum(diff ** 2))
        return float(np.sqrt(total_sq))

    def _calc_update_score(self, name: str) -> float:
        """
        Score how normal this client's update norm is relative to its OWN history.
        Uses z-score anomaly detection — purely client-specific, Non-IID safe.
        """
        data     = self.client_state[name]
        history  = data["norm_history"]            # previous rounds' norms
        current  = data.get("current_norm")

        if current is None or current == 0.0:
            return TRUST_CONFIG["neutral_score"]

        min_hist = TRUST_CONFIG["min_history_for_anomaly"]
        if len(history) < min_hist:
            # Not enough data to judge — give neutral, not a penalty
            return TRUST_CONFIG["neutral_score"]

        mean_n = float(np.mean(history))
        std_n  = float(np.std(history))

        if std_n < 1e-8:
            # Client is extremely stable — check absolute ratio
            ratio = current / (mean_n + 1e-8)
            if ratio < 3.0:   return 90.0
            elif ratio < 5.0: return 55.0
            else:             return 20.0

        z = (current - mean_n) / std_n

        # Map z-score to score (z > 0 means larger-than-typical update)
        if z <= 0:
            score = 90.0                          # below-average norm: healthy
        elif z <= 1:
            score = 90.0 - 10.0 * z              # 80–90
        elif z <= 2:
            score = 80.0 - 15.0 * (z - 1.0)     # 65–80
        elif z <= 3:
            score = 65.0 - 25.0 * (z - 2.0)     # 40–65
        else:
            score = max(10.0, 40.0 - 10.0 * (z - 3.0))

        return float(np.clip(score, 0.0, 100.0))

    def _calc_training_score(self, name: str) -> float:
        """
        Score based on accuracy/loss trend relative to THIS CLIENT'S own baseline.
        A hospital with naturally lower accuracy is NOT penalised — only sudden
        drops or abnormal spikes relative to its own trajectory are flagged.
        """
        data      = self.client_state[name]
        acc_hist  = data["acc_history"]
        loss_hist = data["loss_history"]
        curr_acc  = data.get("current_acc")
        curr_loss = data.get("current_loss")

        if curr_acc is None:
            return TRUST_CONFIG["neutral_score"]

        score = 85.0  # generous base

        if acc_hist:  # at least one previous round available
            prev_acc  = acc_hist[-1]
            prev_loss = loss_hist[-1] if loss_hist else None

            acc_delta  = curr_acc - prev_acc
            loss_delta = (
                (curr_loss - prev_loss)
                if (prev_loss is not None and curr_loss is not None)
                else 0.0
            )

            # Penalise sudden accuracy drops
            if   acc_delta < -0.20: score -= 30
            elif acc_delta < -0.10: score -= 15
            elif acc_delta < -0.05: score -=  5

            # Penalise sudden loss spikes
            if   loss_delta > 1.0:  score -= 20
            elif loss_delta > 0.5:  score -= 10
            elif loss_delta > 0.2:  score -=  5

            # Reward genuine improvement
            if   acc_delta > 0.05:  score += 10
            elif acc_delta > 0.0:   score +=  5

        # Absolute sanity check:
        # Binary classification should beat 30% even with imbalanced Non-IID data.
        if curr_acc < 0.30:
            score -= 20

        return float(np.clip(score, 0.0, 100.0))

    def _calc_participation_score_preview(self, name: str) -> float:
        """
        Score BEFORE updating participation counter (used during score calculation).
        """
        p = self.client_state[name]["participation"]
        total   = p["total"]
        success = p["success"]
        if total == 0:
            return TRUST_CONFIG["neutral_score"]
        return float((success / total) * 100.0)

    def _calc_participation_score(self, name: str) -> float:
        """Score AFTER updating participation counter."""
        p = self.client_state[name]["participation"]
        total   = p["total"]
        success = p["success"]
        if total == 0:
            return TRUST_CONFIG["neutral_score"]
        return float((success / total) * 100.0)

    def _assign_tag(self, score: float) -> str:
        """Assign a trust tag based on configurable thresholds."""
        t = TRUST_CONFIG["thresholds"]
        if   score >= t["TRUSTED"]:    return "TRUSTED"
        elif score >= t["SUSPICIOUS"]: return "SUSPICIOUS"
        else:                          return "UNTRUSTED"

    # ----------------------------------------------------------
    # Terminal output
    # ----------------------------------------------------------

    def _print_trust_table(self, results: Dict[str, Dict], server_round: int) -> None:
        W = 70
        print()
        print("=" * W)
        title = f"CLIENT TRUST STATUS — Round {server_round}"
        print(f"{title:^{W}}")
        print("=" * W)
        hdr = (
            f"{'Client':<18} {'Update':>7} {'Training':>9} "
            f"{'History':>8} {'Reliab':>7} {'SCORE':>7}  {'TAG'}"
        )
        print(hdr)
        print("-" * W)

        for name, r in sorted(results.items()):
            emoji = TAG_EMOJI.get(r["tag"], r["tag"])
            print(
                f"{name:<18} {r['update_score']:>7.1f} {r['training_score']:>9.1f} "
                f"{r['historical_score']:>8.1f} {r['reliability_score']:>7.1f} "
                f"{r['trust_score']:>7.1f}  {emoji}"
            )

        print("=" * W)

        # Detailed per-client breakdown
        for name, r in sorted(results.items()):
            print()
            print(f"  {name}")
            print(f"  {'Update Behaviour':<26}: {r['update_score']:.1f} / 100")
            print(f"  {'Training Behaviour':<26}: {r['training_score']:.1f} / 100")
            print(f"  {'Historical Reputation':<26}: {r['historical_score']:.1f} / 100")
            print(f"  {'Participation Reliability':<26}: {r['reliability_score']:.1f} / 100")
            print(f"  {'-' * 38}")
            print(f"  {'Final Trust Score':<26}: {r['trust_score']:.1f} / 100")
            print(f"  {'Tag':<26}: {TAG_EMOJI.get(r['tag'], r['tag'])}")

        print()

    # ----------------------------------------------------------
    # CSV logging (separate from existing CSVs)
    # ----------------------------------------------------------

    def _get_csv_path(self) -> str:
        if self._csv_path is None:
            results_dir, suffix = _get_results_dir()
            self._csv_path = os.path.join(results_dir, f"trust_{suffix}.csv")
        return self._csv_path

    def _log_to_csv(self, results: Dict[str, Dict], server_round: int) -> None:
        try:
            path         = self._get_csv_path()
            write_header = not os.path.exists(path)
            with open(path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow([
                        "Round", "Client",
                        "UpdateScore", "TrainingScore",
                        "HistoricalScore", "ReliabilityScore",
                        "TrustScore", "Tag",
                    ])
                for name, r in sorted(results.items()):
                    writer.writerow([
                        server_round,        name,
                        r["update_score"],   r["training_score"],
                        r["historical_score"], r["reliability_score"],
                        r["trust_score"],    r["tag"],
                    ])
        except Exception as e:
            print(f"[TrustManager] CSV write warning: {e}")

    # ----------------------------------------------------------
    # Dashboard state update (optional — dashboard_state may not be loaded)
    # ----------------------------------------------------------

    def _update_dashboard_state(self, results: Dict[str, Dict]) -> None:
        try:
            import dashboard_state as _ds
            with _ds._lock:
                if "trust_data" not in _ds.LIVE_STATE:
                    _ds.LIVE_STATE["trust_data"] = {}
                for name, r in results.items():
                    _ds.LIVE_STATE["trust_data"][name] = r
                _ds._bump_version()
                _ds.write_state_file()
        except Exception:
            pass  # Dashboard is optional — training must never fail because of this


# =============================================================
# Module-level singleton
# =============================================================

trust_manager = TrustManager()
