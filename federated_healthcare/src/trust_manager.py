"""
trust_manager.py
================
Phase-1 Client Trust / Tagging Module for Federated Healthcare AI.

DESIGN PRINCIPLES
-----------------
1. NON-IID SAFE:
   Each client is compared ONLY to its OWN history. Hospital B with
   naturally lower accuracy is NOT penalised relative to Hospital A.

2. GRADUAL TRUST CHANGES (EMA):
   One bad round shifts trust by at most historical_ema_alpha * gap.
   A single anomalous round cannot make a TRUSTED client UNTRUSTED.

3. DISTRIBUTION SHIFT IS CONTEXT, NOT EVIDENCE:
   Per-client class-distribution is tracked each round. A large shift
   in NORMAL/PNEUMONIA ratio is used to REDUCE training anomaly penalties,
   not to trigger quarantine. Distribution shift ≠ malicious behaviour.

4. TWO-STAGE TRUST ARCHITECTURE:
   Stage 1 (Pre-Aggregation): Anomaly Score + PRE_AGGREGATION_DECISION
       — Fast, based on update norm, cosine similarity, training context
       — Controls whether client enters FedAvg (ACCEPT/QUARANTINE/REJECT)
   Stage 2 (Post-Evaluation): Final Trust Score + Tag
       — Computed after evaluation metrics are available
       — EMA-smoothed long-term reputation

5. TRUST SCORE ≠ FEDAVG WEIGHT:
   FedAvg continues to use sample-count-based weighting. Trust controls
   ACCEPT/QUARANTINE/REJECT — it does NOT replace sample weighting.

6. SEPARATE TRUST SCORE AND ANOMALY SCORE:
   Trust Score:  Long-term behavioural reputation (0–100, EMA-smoothed).
   Anomaly Score: Current-round abnormality indicator (0–100, per round).
   Both are logged and exposed to the dashboard.

Trust Score (4 components, unchanged from Phase 1):
    40%  Update Behaviour Score    (L2 norm z-score vs own history)
    30%  Training Behaviour Score  (acc/loss trend vs own baseline, dist-modulated)
    20%  Historical Reputation     (EMA of past trust scores)
    10%  Participation Reliability (rounds succeeded / rounds total)

Tagging thresholds (configurable in TRUST_CONFIG):
    80–100 → TRUSTED
    50–79  → SUSPICIOUS
    0–49   → UNTRUSTED

Aggregation Decision (from Trust Tag post-evaluation):
    TRUSTED    → ACCEPT
    SUSPICIOUS → QUARANTINE
    UNTRUSTED  → REJECT

New in this version:
    - Per-client class-distribution tracking (normal/pneumonia ratio)
    - Cosine similarity of consecutive update layer-norm fingerprints
    - Current-round Anomaly Score (separate from Trust Score)
    - Pre-Aggregation Decision (ACCEPT / QUARANTINE / REJECT)
    - Extended CSV output (backward compatible: new columns appended)
    - Extended dashboard state (backward compatible: new fields added)
    - Extended client feedback via configure_fit config dict

Usage in server.py:
    from trust_manager import trust_manager as tm
    # configure_fit    : tm.set_global_params(...)
    # aggregate_fit    : tm.record_update(...),  tm.record_dropout(...)
    #                  : tm.get_pre_agg_decisions() → gating dict
    # evaluate_agg_fn  : tm.record_evaluation(...), tm.finalize_round(round)
"""

import csv
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

# Enable ANSI colors & UTF-8 output on Windows terminals
try:
    import colorama
    colorama.init(autoreset=False)
except Exception:
    pass

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Pre-aggregation gate (anomaly scoring and decision)
try:
    from pre_agg_gate import (
        PRE_AGG_CONFIG,
        compute_update_layer_norms,
        compute_cosine_similarity,
        compute_distribution_shift,
        compute_anomaly_score,
        make_pre_agg_decision,
        tag_to_aggregation_decision,
    )
    _PRE_AGG_OK = True
except Exception as _pae:
    _PRE_AGG_OK = False
    print(f"[TrustManager] pre_agg_gate import warning: {_pae}")


# =============================================================
# CONFIGURABLE CONSTANTS
# Change these values to tune the trust system behaviour.
# =============================================================

TRUST_CONFIG: Dict = {
    # ── Weighted combination of the four component scores ────────
    "weights": {
        "update_behaviour":          0.40,
        "training_behaviour":        0.30,
        "historical_reputation":     0.20,
        "participation_reliability": 0.10,
    },

    # ── Tag boundaries (0–100 scale) ─────────────────────────────
    # score >= TRUSTED    → TRUSTED
    # score >= SUSPICIOUS → SUSPICIOUS
    # below SUSPICIOUS    → UNTRUSTED
    "thresholds": {
        "TRUSTED":    80.0,
        "SUSPICIOUS": 50.0,
    },

    # ── EMA decay ────────────────────────────────────────────────
    # New_hist = (1-alpha)*Old_hist + alpha*CurrentScore
    # alpha=0.30 → one bad round shifts trust by at most 30% of the gap
    "historical_ema_alpha": 0.30,

    # ── Minimum rounds of norm history before anomaly starts ─────
    # Before this, the update score returns neutral_score (no penalty).
    "min_history_for_anomaly": 3,

    # ── Default score used when insufficient history / missing data
    # Set to 82 so new clients start as TRUSTED (threshold=80) rather than
    # SUSPICIOUS. Clients that misbehave will fall below 80 naturally via EMA.
    "neutral_score": 82.0,

    # ── Initial historical trust for a brand-new client ──────────
    # 82 > TRUSTED threshold (80) → round-1 clients are TRUSTED, not SUSPICIOUS.
    "initial_historical_trust": 82.0,

    # ── Distribution shift: training score moderation ────────────
    # When distribution_shift exceeds this, training score penalties
    # are reduced because a Non-IID shift can legitimately explain
    # metric changes. This is a contextual adjustment, NOT a way to
    # excuse a genuinely suspicious update.
    "training_dist_moderation_threshold": 0.20,   # above this → partial reduction
    "training_dist_moderation_max":       0.50,   # max reduction fraction (50%)
}

# Terminal ANSI Color Codes
_COLOR_GREEN  = "\033[1;92m"   # Bold Bright Green
_COLOR_YELLOW = "\033[1;93m"   # Bold Bright Yellow/Amber
_COLOR_RED    = "\033[1;91m"   # Bold Bright Red
_COLOR_RESET  = "\033[0m"

TAG_COLORS = {
    "TRUSTED":    _COLOR_GREEN,
    "SUSPICIOUS": _COLOR_YELLOW,
    "UNTRUSTED":  _COLOR_RED,
}

DECISION_COLORS = {
    "ACCEPT":     _COLOR_GREEN,
    "QUARANTINE": _COLOR_YELLOW,
    "REJECT":     _COLOR_RED,
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

DECISION_EMOJI = {
    "ACCEPT":     "✅ ACCEPT",
    "QUARANTINE": "🔶 QUARANTINE",
    "REJECT":     "❌ REJECT",
}

def colorize_tag(tag: str, width: Optional[int] = None) -> str:
    """Format trust tag with circular badge and terminal color."""
    emoji = {"TRUSTED": "🟢", "SUSPICIOUS": "🟡", "UNTRUSTED": "🔴"}.get(tag, "⚪")
    color = TAG_COLORS.get(tag, "")
    text  = f"{emoji} {tag}"
    if width is not None:
        text = f"{text:<{width}}"
    return f"{color}{text}{_COLOR_RESET}"

def colorize_decision(decision: str, width: Optional[int] = None) -> str:
    """Format gating decision with badge and terminal color."""
    emoji = {"ACCEPT": "✅", "QUARANTINE": "🔶", "REJECT": "❌"}.get(decision, "•")
    color = DECISION_COLORS.get(decision, "")
    text  = f"{emoji} {decision}"
    if width is not None:
        text = f"{text:<{width}}"
    return f"{color}{text}{_COLOR_RESET}"

# Reputation penalty added to effective anomaly score in get_pre_agg_decisions().
# The previous round's Trust Tag makes the pre-aggregation gate stricter for
# clients with declining reputations, closing the gap between tag and gate.
#
# Combined with new thresholds (quarantine=40, reject=70):
#   TRUSTED    +0   → gate unchanged
#   SUSPICIOUS +20  → even a moderate anomaly (20+) triggers quarantine
#   UNTRUSTED  +40  → near-auto quarantine; extreme anomaly needed to reject
_REPUTATION_PENALTY: Dict[str, float] = {
    "TRUSTED":    0.0,
    "SUSPICIOUS": 20.0,
    "UNTRUSTED":  40.0,
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
        configure_fit → aggregate_fit → configure_evaluate → aggregate_evaluate
    so we do not need internal locks.

    State per client:
        norm_history     : L2 update norms from previous rounds
        acc_history      : Evaluation accuracy from previous rounds
        loss_history     : Evaluation loss from previous rounds
        historical_trust : EMA-smoothed trust score
        participation    : {success, total}
        trust_rounds     : [(round, score, tag), ...]
        latest_trust     : Last completed round's result dict
        dist_history     : Class-distribution ratio history
                           [{normal_ratio, pneumonia_ratio}, ...]
        prev_layer_norms : Layer-norm fingerprint from previous update
                           (used for cosine similarity computation)
        rounds_of_data   : Number of rounds this client has submitted data

    New per-round transient fields (cleared after finalize_round):
        current_norm            : L2 norm of this round's update
        current_acc             : Evaluation accuracy this round
        current_loss            : Evaluation loss this round
        current_f1              : F1 score this round
        current_layer_norms     : Layer-norm fingerprint this round
        current_cosine_sim      : Cosine similarity to previous update
        current_dist_shift      : Distribution shift indicator
        current_anomaly_score   : Pre-aggregation anomaly score
        current_pre_agg_decision: ACCEPT / QUARANTINE / REJECT
        current_normal_count    : Normal samples reported this round
        current_pneumonia_count : Pneumonia samples reported this round
    """

    def __init__(self):
        # Per-client persistent state
        self.client_state: Dict[str, Dict] = {}

        # Previous round's global parameters (list of float64 ndarrays)
        self._global_params: Optional[List[np.ndarray]] = None

        # CID to client name mapping (from fit_res.metrics["client_name"])
        self.cid_to_name: Dict[str, str] = {}

        # Sets tracking participation in the CURRENT round
        self._round_participants: set = set()   # submitted update successfully
        self._round_dropouts:     set = set()   # failed / timed out

        # Last completed round results (used by configure_fit for feedback)
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

                # ── New: Distribution tracking ────────────────
                # List of {normal_ratio, pneumonia_ratio} per round
                "dist_history": [],

                # ── New: Cosine similarity baseline ──────────
                # Layer-norm fingerprint of the PREVIOUS round's update.
                # None = first round (no baseline available).
                "prev_layer_norms": None,

                # Number of rounds for which we have submitted data
                "rounds_of_data": 0,
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
        """
        Record a client's model update for trust computation.

        Called AFTER dequantization so client_ndarrays are always FP32,
        which makes L2 norm and cosine computation comparable across rounds
        regardless of whether quantization is enabled.

        STAGE 1 (Pre-Aggregation) is computed here:
        - L2 update norm
        - Layer-norm fingerprint for cosine similarity
        - Distribution shift (from fit_metrics)
        - Anomaly Score
        - Pre-Aggregation Decision (ACCEPT / QUARANTINE / REJECT)

        The result of get_pre_agg_decisions() is used by server.py to
        gate which updates enter the FedAvg call.
        """
        self._init_client(client_name)
        self.cid_to_name[cid] = client_name
        self._round_participants.add(client_name)

        data = self.client_state[client_name]

        # ── Compute L2 norm of the update ────────────────────────
        norm = self._compute_update_norm(client_ndarrays)
        data["current_norm"] = norm

        # ── Compute layer-norm fingerprint ───────────────────────
        if _PRE_AGG_OK:
            try:
                current_layer_norms = compute_update_layer_norms(
                    client_ndarrays, self._global_params
                )
                data["current_layer_norms"] = current_layer_norms

                # ── Cosine similarity to previous update ─────────
                cosine_sim = compute_cosine_similarity(
                    current_layer_norms,
                    data.get("prev_layer_norms"),
                )
                data["current_cosine_sim"] = cosine_sim

            except Exception as e:
                print(f"[TrustManager] Layer-norm/cosine warning for {client_name}: {e}")
                data["current_layer_norms"] = None
                data["current_cosine_sim"]  = 1.0   # neutral fallback
        else:
            data["current_layer_norms"] = None
            data["current_cosine_sim"]  = 1.0

        # ── Extract class distribution from fit_metrics ──────────
        normal_count    = 0
        pneumonia_count = 0
        if fit_metrics:
            normal_count    = int(fit_metrics.get("normal_count",    0))
            pneumonia_count = int(fit_metrics.get("pneumonia_count", 0))

        data["current_normal_count"]    = normal_count
        data["current_pneumonia_count"] = pneumonia_count

        total_samples = normal_count + pneumonia_count
        if total_samples > 0:
            current_pneumonia_ratio = pneumonia_count / total_samples
            current_normal_ratio    = normal_count    / total_samples
        else:
            current_pneumonia_ratio = None
            current_normal_ratio    = None

        data["current_pneumonia_ratio"] = current_pneumonia_ratio
        data["current_normal_ratio"]    = current_normal_ratio

        # ── Distribution shift ───────────────────────────────────
        prev_dist = data["dist_history"][-1] if data["dist_history"] else None
        prev_pneumonia_ratio = prev_dist["pneumonia_ratio"] if prev_dist else None

        dist_shift = compute_distribution_shift(
            current_pneumonia_ratio, prev_pneumonia_ratio
        ) if _PRE_AGG_OK else 0.0

        data["current_dist_shift"] = dist_shift

        # ── Anomaly Score (Stage 1 — PRE-AGGREGATION GATE) ──────────
        #
        # The gate uses ONLY signals available at fit time:
        #   - L2 update norm vs client's own history  (0–50 contribution)
        #   - Cosine similarity to client's own previous update  (0–30)
        #
        # Evaluation metrics (accuracy, loss) are NOT available here.
        # evaluate() runs AFTER FedAvg, so using them at gate time is
        # architecturally incorrect. The training contribution (0–20) is
        # added AFTER evaluation in record_evaluation() for the refined
        # anomaly score that feeds back to clients next round.
        #
        # Distribution shift is passed through as modulation context for
        # the training contribution — it has no effect here since
        # current_acc=None → training contribution = 0 at this stage.
        #
        if _PRE_AGG_OK:
            try:
                cosine_sim   = data.get("current_cosine_sim", 1.0)
                rounds_avail = data["rounds_of_data"]

                anomaly_score, anomaly_breakdown = compute_anomaly_score(
                    current_norm       = norm,
                    norm_history       = list(data["norm_history"]),
                    cosine_similarity  = cosine_sim,
                    rounds_of_history  = rounds_avail,
                    current_acc        = None,   # evaluation not available at gate time
                    prev_acc           = None,
                    current_loss       = None,
                    prev_loss          = None,
                    distribution_shift = dist_shift,
                )

                # pre_agg_decision stored here is based on anomaly score alone.
                # get_pre_agg_decisions() adds reputation penalty before the
                # final gate decision is returned to server.py.
                pre_agg_decision = make_pre_agg_decision(anomaly_score)

            except Exception as e:
                print(f"[TrustManager] Anomaly score warning for {client_name}: {e}")
                anomaly_score     = 0.0
                anomaly_breakdown = {}
                pre_agg_decision  = "ACCEPT"
        else:
            anomaly_score     = 0.0
            anomaly_breakdown = {}
            pre_agg_decision  = "ACCEPT"

        data["current_anomaly_score"]    = anomaly_score
        data["current_anomaly_breakdown"] = anomaly_breakdown
        data["current_pre_agg_decision"] = pre_agg_decision

        # ── Log pre-agg decision ─────────────────────────────────
        emoji = DECISION_EMOJI.get(pre_agg_decision, pre_agg_decision)
        try:
            print(
                f"[TrustManager] {client_name:<18s}  "
                f"L2={norm:.2f}  "
                f"Cosine={data.get('current_cosine_sim', 1.0):.3f}  "
                f"DistShift={dist_shift:.3f}  "
                f"AnomalyScore={anomaly_score:.1f}  "
                f"-> {emoji}"
            )
        except UnicodeEncodeError:
            # Fallback for terminals that cannot render emoji (e.g. Windows cp1252)
            print(
                f"[TrustManager] {client_name:<18s}  "
                f"L2={norm:.2f}  "
                f"Cosine={data.get('current_cosine_sim', 1.0):.3f}  "
                f"DistShift={dist_shift:.3f}  "
                f"AnomalyScore={anomaly_score:.1f}  "
                f"-> {pre_agg_decision}"
            )


    # ----------------------------------------------------------
    # Called from aggregate_fit after record_update — provides
    # pre-aggregation gating decisions to server.py
    # ----------------------------------------------------------

    def get_pre_agg_decisions(self) -> Dict[str, str]:
        """
        Return pre-aggregation gating decisions for all clients that
        submitted updates this round.

        Combines two signals:
          1. current_anomaly_score  (L2 + Cosine, from record_update)
          2. Reputation penalty from previous round's Trust Tag
             TRUSTED    → +0   (no added pressure)
             SUSPICIOUS → +20  (stricter gate — SUSPICIOUS clients need less
                                 current anomaly to be quarantined)
             UNTRUSTED  → +40  (near-auto quarantine — even a small current
                                 anomaly crosses the gate threshold)

        Effective anomaly = current_anomaly + reputation_penalty
        make_pre_agg_decision(effective_anomaly) gives the final decision.

        Fail-safe: if _PRE_AGG_OK is False, returns ACCEPT for all clients
        so that trust failures never block FL training.

        Returns:
            {client_name: "ACCEPT" | "QUARANTINE" | "REJECT"}
        """
        decisions = {}
        for name in self._round_participants:
            data         = self.client_state.get(name, {})
            current_anom = float(data.get("current_anomaly_score", 0.0))

            # -- Reputation feedback: previous round's Trust Tag --------
            # A client with SUSPICIOUS/UNTRUSTED tag from round N faces a
            # stricter effective threshold in round N+1's gate.
            reputation_penalty = 0.0
            prev_trust = data.get("latest_trust")
            if prev_trust:
                prev_tag           = prev_trust.get("tag", "TRUSTED")
                reputation_penalty = _REPUTATION_PENALTY.get(prev_tag, 0.0)

            effective_anomaly = float(np.clip(current_anom + reputation_penalty, 0.0, 100.0))

            # Log when reputation adds pressure
            if reputation_penalty > 0:
                try:
                    print(
                        f"[TrustGate]    {name:<18s}  "
                        f"Anomaly={current_anom:.1f}  "
                        f"RepPenalty=+{reputation_penalty:.0f}  "
                        f"Effective={effective_anomaly:.1f}"
                    )
                except UnicodeEncodeError:
                    pass

            decision = make_pre_agg_decision(effective_anomaly) if _PRE_AGG_OK else "ACCEPT"
            decisions[name] = decision

        return decisions

    # ----------------------------------------------------------
    # Called from aggregate_fit for each failed client
    # ----------------------------------------------------------

    def record_dropout(self, cid_or_name: str) -> None:
        """Record a client dropout/failure."""
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
        """
        Record evaluation metrics for a client (from evaluate() return).

        Called from evaluate_metrics_aggregation_fn AFTER all clients
        have evaluated and AFTER FedAvg has already run.

        IMPORTANT: The pre-aggregation gate (get_pre_agg_decisions) was
        already applied before FedAvg. This method does NOT change which
        clients entered FedAvg this round. Its purposes are:
          1. Store evaluation acc/loss for Stage 2 (finalize_round Trust Score).
          2. Refine the stored anomaly score with the training contribution
             (now that evaluation metrics are available).
          3. The refined anomaly score is logged to CSV/dashboard and fed
             back to clients via configure_fit in the NEXT round.
        """
        self._init_client(client_name)
        s = self.client_state[client_name]
        s["current_acc"]  = float(accuracy)
        s["current_loss"] = float(loss)
        s["current_f1"]   = float(f1)

        # ── Refine anomaly score with evaluation metrics ──────────
        # If we have evaluation acc/loss now, and the anomaly score
        # was computed without them (fit metrics not available),
        # recompute with better data.
        if _PRE_AGG_OK and "current_pre_agg_decision" in s:
            try:
                prev_acc  = s["acc_history"][-1]  if s["acc_history"]  else None
                prev_loss = s["loss_history"][-1] if s["loss_history"] else None

                if prev_acc is not None:
                    # Recompute only the training contribution and update totals
                    from pre_agg_gate import (
                        _training_anomaly_contribution,
                        PRE_AGG_CONFIG,
                    )
                    old_breakdown   = s.get("current_anomaly_breakdown", {})
                    dist_shift      = s.get("current_dist_shift", 0.0)

                    new_training = _training_anomaly_contribution(
                        float(accuracy), prev_acc,
                        float(loss),     prev_loss,
                        dist_shift,      PRE_AGG_CONFIG,
                    )

                    old_total      = s.get("current_anomaly_score", 0.0)
                    old_training   = old_breakdown.get("training_contribution", 0.0)
                    refined_total  = float(
                        np.clip(old_total - old_training + new_training, 0.0, 100.0)
                    )

                    s["current_anomaly_score"] = refined_total
                    if old_breakdown:
                        old_breakdown["training_contribution"] = round(new_training, 2)

                    # Update decision based on refined score
                    s["current_pre_agg_decision"] = make_pre_agg_decision(refined_total)

            except Exception as e:
                print(f"[TrustManager] Anomaly refinement warning for {client_name}: {e}")

    # ----------------------------------------------------------
    # Main: compute and finalise trust for all clients this round
    # ----------------------------------------------------------

    def finalize_round(self, server_round: int) -> Dict[str, Dict]:
        """
        Stage 2: Compute final Trust Scores for all clients that
        participated in this round.

        Call this at the END of evaluate_metrics_aggregation_fn, AFTER
        all record_evaluation() calls have been made.

        Uses:
        - Evaluation acc/loss (now available post-evaluation)
        - Distribution shift (computed in record_update)
        - Update L2 norm history
        - Historical trust (EMA)
        - Participation tracking

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

            # ── Component scores ──────────────────────────────────
            update_score   = self._calc_update_score(name)
            training_score = self._calc_training_score(name)
            hist_score     = data["historical_trust"]
            part_score     = self._calc_participation_score_preview(name)

            # ── Weighted combination (unchanged formula) ──────────
            w = TRUST_CONFIG["weights"]
            trust_score = (
                w["update_behaviour"]          * update_score   +
                w["training_behaviour"]        * training_score +
                w["historical_reputation"]     * hist_score     +
                w["participation_reliability"] * part_score
            )
            trust_score = float(np.clip(trust_score, 0.0, 100.0))

            # ── Assign tag ────────────────────────────────────────
            tag = self._assign_tag(trust_score)

            # ── Aggregation Decision (from final Trust Tag) ───────
            agg_decision = tag_to_aggregation_decision(tag) \
                           if _PRE_AGG_OK else "ACCEPT"

            # ── Snapshot transient values before clearing ─────────
            current_norm           = data.pop("current_norm",            None)
            current_acc            = data.pop("current_acc",             None)
            current_loss           = data.pop("current_loss",            None)
            data.pop("current_f1",                None)
            data.pop("current_fit_metrics",       None)
            current_layer_norms    = data.pop("current_layer_norms",     None)
            cosine_sim             = data.pop("current_cosine_sim",      1.0)
            dist_shift             = data.pop("current_dist_shift",      0.0)
            anomaly_score          = data.pop("current_anomaly_score",   0.0)
            data.pop("current_anomaly_breakdown",  None)
            pre_agg_decision       = data.pop("current_pre_agg_decision", "ACCEPT")
            normal_count           = data.pop("current_normal_count",    0)
            pneumonia_count        = data.pop("current_pneumonia_count", 0)
            current_pneu_ratio     = data.pop("current_pneumonia_ratio", None)
            current_norm_ratio     = data.pop("current_normal_ratio",    None)

            # ── Update histories (AFTER using them for scoring) ───
            if current_norm is not None and current_norm > 0.0:
                data["norm_history"].append(float(current_norm))
            if current_acc  is not None:
                data["acc_history"].append(float(current_acc))
            if current_loss is not None:
                data["loss_history"].append(float(current_loss))

            # ── Update class-distribution history ─────────────────
            if current_pneu_ratio is not None:
                data["dist_history"].append({
                    "normal_ratio":    float(current_norm_ratio)  if current_norm_ratio  is not None else 0.0,
                    "pneumonia_ratio": float(current_pneu_ratio)  if current_pneu_ratio  is not None else 0.0,
                    "normal_count":    normal_count,
                    "pneumonia_count": pneumonia_count,
                })

            # ── Update layer-norm fingerprint (for next round's cosine)
            if current_layer_norms is not None:
                data["prev_layer_norms"] = current_layer_norms

            data["rounds_of_data"] += 1

            # ── Update participation ───────────────────────────────
            data["participation"]["total"]   += 1
            data["participation"]["success"] += 1
            final_part_score = self._calc_participation_score(name)

            # ── Update historical trust (EMA) ─────────────────────
            alpha = TRUST_CONFIG["historical_ema_alpha"]
            data["historical_trust"] = (
                (1.0 - alpha) * data["historical_trust"] +
                alpha         * trust_score
            )

            # ── Build result dict ─────────────────────────────────
            result = {
                # ── Original Phase-1 fields (preserved exactly) ───
                "client_id":         name,
                "round":             server_round,
                "update_score":      round(update_score,     1),
                "training_score":    round(training_score,   1),
                "historical_score":  round(hist_score,       1),
                "reliability_score": round(final_part_score, 1),
                "trust_score":       round(trust_score,      1),
                "tag":               tag,

                # ── New: Anomaly & Distribution fields ─────────────
                "anomaly_score":         round(float(anomaly_score), 1),
                "cosine_similarity":     round(float(cosine_sim),    4),
                "distribution_shift":    round(float(dist_shift),    4),
                "l2_norm":               round(float(current_norm) if current_norm else 0.0, 4),

                # ── New: Decision fields ───────────────────────────
                "pre_agg_decision":   pre_agg_decision,    # ACCEPT|QUARANTINE|REJECT
                "aggregation_decision": agg_decision,       # ACCEPT|QUARANTINE|REJECT

                # ── Distribution snapshot ──────────────────────────
                "normal_count":     normal_count,
                "pneumonia_count":  pneumonia_count,
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
        Used by configure_fit to inject trust feedback into client config.
        """
        name = self.cid_to_name.get(cid)
        if not name:
            return None
        return self.client_state.get(name, {}).get("latest_trust")

    def get_suspicious_clients(self) -> set:
        """
        Return the set of client names whose most recent Trust Tag is SUSPICIOUS.

        Used by server.py's _check_promote_to_excluded() as a clean public
        accessor — avoids callers reaching into client_state internals.

        Returns an empty set if no clients have been evaluated yet.
        """
        return {
            name
            for name, data in self.client_state.items()
            if data.get("latest_trust", {}).get("tag") == "SUSPICIOUS"
        }

    def get_untrusted_clients(self) -> set:
        """
        Return the set of client names whose most recent Trust Tag is UNTRUSTED.

        Used by server.py's _check_promote_to_excluded() as a clean public
        accessor — avoids callers reaching into client_state internals.

        Returns an empty set if no clients have been evaluated yet.
        """
        return {
            name
            for name, data in self.client_state.items()
            if data.get("latest_trust", {}).get("tag") == "UNTRUSTED"
        }


    # ----------------------------------------------------------
    # Score calculations (all Non-IID safe: client vs. own history)
    # ----------------------------------------------------------

    def _compute_update_norm(self, client_ndarrays: List[np.ndarray]) -> float:
        """
        Compute L2 norm of the parameter update (client_params - global_params).
        If global_params is None (first round), uses raw parameter norm.

        Always called on dequantized (FP32) parameters so results are
        comparable across quantized and non-quantized rounds.
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

        This is the Update Behaviour component (40% weight in Trust Score).
        The Anomaly Score also includes an L2 contribution, but the two are
        computed independently: Trust Score uses this for long-term reputation
        smoothed via EMA, while Anomaly Score uses it for immediate detection.
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

        IMPROVED in this version:
        A hospital with naturally lower accuracy is NOT penalised — only sudden
        drops or abnormal spikes relative to its own trajectory are flagged.

        DISTRIBUTION SHIFT MODERATION:
        If this client reported a large class-distribution shift this round,
        accuracy/loss changes are expected (legitimate Non-IID behaviour).
        Penalties are reduced proportionally to the distribution shift.
        This makes the system Non-IID safe for healthcare scenarios where
        a hospital may process a different patient cohort each round.

        Example:
        - Round 5: Hospital B  Normal=700,  Pneumonia=300  → accuracy=0.82
        - Round 6: Hospital B  Normal=300,  Pneumonia=700  → accuracy=0.74
        Distribution shift = |0.70 - 0.30| = 0.40 (high)
        → Training score penalty is reduced by up to 50%
        → Client is not flagged as suspicious based on metrics alone

        The Update Behaviour score (L2 norm z-score) remains unaffected by
        distribution shift, providing an independent anomaly signal.
        """
        data      = self.client_state[name]
        acc_hist  = data["acc_history"]
        loss_hist = data["loss_history"]
        curr_acc  = data.get("current_acc")
        curr_loss = data.get("current_loss")
        dist_shift = data.get("current_dist_shift", 0.0)

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

            # Base penalties (same as Phase-1 — preserved exactly)
            penalty = 0.0

            if   acc_delta < -0.20: penalty += 30
            elif acc_delta < -0.10: penalty += 15
            elif acc_delta < -0.05: penalty +=  5

            if   loss_delta > 1.0:  penalty += 20
            elif loss_delta > 0.5:  penalty += 10
            elif loss_delta > 0.2:  penalty +=  5

            # ── Apply distribution shift moderation ───────────────
            # High shift → legitimate Non-IID change → reduce penalty
            # Low shift  → no moderation → full penalty applies
            dist_thresh = TRUST_CONFIG["training_dist_moderation_threshold"]
            max_mod     = TRUST_CONFIG["training_dist_moderation_max"]

            if dist_shift > dist_thresh and penalty > 0:
                # Scale moderation from 0 at threshold to max_mod at 0.50
                t          = min((dist_shift - dist_thresh) / 0.30, 1.0)
                moderation = max_mod * t
                penalty   *= (1.0 - moderation)

            score -= penalty

            # Reward genuine improvement (unchanged from Phase-1)
            if   acc_delta > 0.05:  score += 10
            elif acc_delta > 0.0:   score +=  5

        # Absolute sanity check (unchanged from Phase-1):
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
        # Guard: use ASCII-only output on terminals that cannot render Unicode (e.g. Windows cp1252)
        def _safe_print(text: str) -> None:
            try:
                print(text)
            except UnicodeEncodeError:
                print(text.encode("ascii", errors="replace").decode("ascii"))

        W = 80
        _safe_print("")
        _safe_print("=" * W)
        title = f"CLIENT TRUST STATUS -- Round {server_round}"
        _safe_print(f"{title:^{W}}")
        _safe_print("=" * W)

        # Header
        hdr = (
            f"{'Client':<18} {'Update':>7} {'Training':>9} "
            f"{'History':>8} {'Reliab':>7} {'SCORE':>7}  {'TAG':<12} {'ANOMALY':>8}  PRE-AGG"
        )
        _safe_print(hdr)
        _safe_print("-" * W)

        for name, r in sorted(results.items()):
            tag       = r["tag"]
            pre       = r["pre_agg_decision"]
            _safe_print(
                f"{name:<18} {r['update_score']:>7.1f} {r['training_score']:>9.1f} "
                f"{r['historical_score']:>8.1f} {r['reliability_score']:>7.1f} "
                f"{r['trust_score']:>7.1f}  {tag:<12} {r['anomaly_score']:>8.1f}  "
                f"{pre}"
            )

        _safe_print("=" * W)

        # Detailed per-client breakdown
        for name, r in sorted(results.items()):
            _safe_print("")
            _safe_print(f"  {name}")
            _safe_print(f"  {'-' * 50}")
            _safe_print(f"  {'Update Behaviour':<28}: {r['update_score']:.1f} / 100")
            _safe_print(f"  {'Training Behaviour':<28}: {r['training_score']:.1f} / 100")
            _safe_print(f"  {'Historical Reputation':<28}: {r['historical_score']:.1f} / 100")
            _safe_print(f"  {'Participation Reliability':<28}: {r['reliability_score']:.1f} / 100")
            _safe_print(f"  {'-' * 38}")
            _safe_print(f"  {'Final Trust Score':<28}: {r['trust_score']:.1f} / 100")
            _safe_print(f"  {'Trust Tag':<28}: {r['tag']}")
            _safe_print(f"  {'-' * 38}")
            _safe_print(f"  {'Anomaly Score':<28}: {r['anomaly_score']:.1f} / 100")
            _safe_print(f"  {'L2 Update Norm':<28}: {r['l2_norm']:.4f}")
            _safe_print(f"  {'Cosine Similarity':<28}: {r['cosine_similarity']:.4f}")
            _safe_print(f"  {'Distribution Shift':<28}: {r['distribution_shift']:.4f}")
            _safe_print(f"  {'Pre-Aggregation Decision':<28}: {r['pre_agg_decision']}")
            _safe_print(f"  {'Aggregation Decision':<28}: {r['aggregation_decision']}")
            if r.get("normal_count") or r.get("pneumonia_count"):
                _safe_print(f"  {'Normal Count':<28}: {r['normal_count']}")
                _safe_print(f"  {'Pneumonia Count':<28}: {r['pneumonia_count']}")

        _safe_print("")


    # ----------------------------------------------------------
    # CSV logging (backward compatible — new columns appended)
    # ----------------------------------------------------------

    def _get_csv_path(self) -> str:
        if self._csv_path is None:
            results_dir, suffix = _get_results_dir()
            self._csv_path = os.path.join(results_dir, f"trust_{suffix}.csv")
        return self._csv_path

    def _log_to_csv(self, results: Dict[str, Dict], server_round: int) -> None:
        """
        Write trust results to CSV.

        BACKWARD COMPATIBLE: The file is only created fresh at the start
        of a new run (when it doesn't exist). Existing runs that started
        with Phase-1 columns will continue appending to their own file.
        New runs get the extended column set.
        """
        try:
            path         = self._get_csv_path()
            write_header = not os.path.exists(path)
            with open(path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if write_header:
                    # Extended header — backward compatible with Phase-1
                    # (Phase-1 columns first, new columns appended at end)
                    writer.writerow([
                        # Phase-1 columns (preserved exactly)
                        "Round", "Client",
                        "UpdateScore", "TrainingScore",
                        "HistoricalScore", "ReliabilityScore",
                        "TrustScore", "Tag",
                        # New columns
                        "L2Norm", "CosineSimilarity",
                        "DistributionShift", "AnomalyScore",
                        "PreAggregationDecision", "AggregationDecision",
                        "NormalCount", "PneumoniaCount",
                    ])
                for name, r in sorted(results.items()):
                    writer.writerow([
                        # Phase-1 values (preserved exactly)
                        server_round,           name,
                        r["update_score"],      r["training_score"],
                        r["historical_score"],  r["reliability_score"],
                        r["trust_score"],       r["tag"],
                        # New values
                        r["l2_norm"],
                        r["cosine_similarity"],
                        r["distribution_shift"],
                        r["anomaly_score"],
                        r["pre_agg_decision"],
                        r["aggregation_decision"],
                        r.get("normal_count",    0),
                        r.get("pneumonia_count", 0),
                    ])
        except Exception as e:
            print(f"[TrustManager] CSV write warning: {e}")

    # ----------------------------------------------------------
    # Dashboard state update (backward compatible — new fields added)
    # ----------------------------------------------------------

    def _update_dashboard_state(self, results: Dict[str, Dict]) -> None:
        """
        Update dashboard state with trust results.

        BACKWARD COMPATIBLE: Existing `trust_data` structure is extended
        with new fields. The dashboard will display new fields if it reads
        them, and ignore them if it doesn't.
        """
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
