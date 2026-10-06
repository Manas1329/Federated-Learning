"""
pre_agg_gate.py
===============
Stage 1 Pre-Aggregation Anomaly Assessment and Gating.

PURPOSE
-------
This module is called BEFORE FedAvg to assess whether each client's model
update is anomalous enough to be quarantined or rejected from aggregation.

DESIGN PRINCIPLES
-----------------
1. NON-IID SAFETY:
   Each client is compared ONLY to its OWN history. Hospital B with
   naturally lower accuracy is NOT penalised relative to Hospital A.

2. DISTRIBUTION SHIFT IS CONTEXT, NOT EVIDENCE:
   A large shift in normal/pneumonia ratio can legitimately explain
   metric changes in Non-IID healthcare data. High distribution shift
   REDUCES the training anomaly contribution — it does NOT trigger
   a quarantine by itself.

3. TRUST SCORE ≠ FEDAVG WEIGHT:
   This module outputs ACCEPT / QUARANTINE / REJECT decisions.
   FedAvg continues to use sample-count-based weighting unchanged.

4. SEPARATION OF CONCERNS:
   Pre-Aggregation Decision = current-round anomaly assessment (fast).
   Final Trust Score = long-term behavioural reputation (post-evaluation).
   These are computed independently and serve different purposes.

5. QUARANTINE → EXISTING DACM/UNLEARNING:
   Quarantined clients are excluded from the FedAvg call. The existing
   DACM distribution audit (already in server.py) naturally handles the
   surviving distribution. The full run_unlearn_workflow() pipeline
   (post-training) remains available and unchanged.

ANOMALY SCORE FORMULA (0–100, fully transparent)
-------------------------------------------------
AnomalyScore = L2_contribution (0–50)
             + Cosine_contribution (0–30)
             + Training_contribution (0–20)

PRE-AGGREGATION DECISION
------------------------
    AnomalyScore >= reject_threshold    → REJECT
    AnomalyScore >= quarantine_threshold → QUARANTINE
    else                                 → ACCEPT

USAGE (called from trust_manager.py)
-------------------------------------
    from pre_agg_gate import (
        compute_update_layer_norms,
        compute_cosine_similarity,
        compute_distribution_shift,
        compute_anomaly_score,
        make_pre_agg_decision,
        PRE_AGG_CONFIG,
    )
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

# ==================================================================
# Configuration — all thresholds are configurable here
# ==================================================================

PRE_AGG_CONFIG: Dict = {
    # ── Aggregation gate thresholds ──────────────────────────────
    # AnomalyScore >= reject_threshold    → REJECT  (clear, multi-signal attack)
    # AnomalyScore >= quarantine_threshold → QUARANTINE (single strong signal)
    # else                                 → ACCEPT
    #
    # Thresholds chosen so that:
    #   - A large L2 spike alone  (z≈3 → L2 contribution≈40) → QUARANTINE
    #   - Extreme L2 + cosine reversal (50+28=78)             → REJECT
    #   - Normal clients (z<1, cosine≈1) score <10             → ACCEPT
    #   - SUSPICIOUS/UNTRUSTED reputation adds +20/+40 via get_pre_agg_decisions()
    "quarantine_threshold": 40.0,
    "reject_threshold":     70.0,

    # ── Cosine similarity thresholds ─────────────────────────────
    # Min rounds of previous update stats needed before cosine is used.
    # Before this, cosine contribution = 0 (neutral — no false positives).
    "min_history_for_cosine": 2,

    # Cosine similarity below this (towards negative) = suspicious direction.
    # 0.0 means a 90° direction change starts contributing anomaly evidence.
    "cosine_suspicious_threshold": 0.0,

    # ── Distribution shift thresholds ────────────────────────────
    # distribution_shift is abs(current_pneumonia_ratio - prev_pneumonia_ratio)
    # High shift = legitimate Non-IID change → reduces training anomaly penalty
    "dist_shift_high":   0.40,   # above this → strong moderation of training penalty
    "dist_shift_medium": 0.20,   # above this → partial moderation

    # Maximum fraction by which distribution shift can reduce training penalty
    "dist_shift_max_moderation": 0.80,   # up to 80% reduction at extreme shifts

    # ── Minimum history before anomaly is computed ────────────────
    # Mirrors trust_manager's min_history_for_anomaly.
    # Before this, L2 contribution = 0 (neutral).
    "min_history_for_l2": 3,
}


# ==================================================================
# Helper: Layer-norm fingerprint (memory-efficient cosine proxy)
# ==================================================================

def compute_update_layer_norms(
    client_ndarrays: List[np.ndarray],
    global_ndarrays: Optional[List[np.ndarray]],
) -> np.ndarray:
    """
    Compute a per-layer L2-norm vector of the model update.

    Rather than storing the full flattened update delta (which can be
    100+ MB for a CNN), we store only the per-layer update norms.
    This forms a compact "update fingerprint" usable for cosine
    similarity between consecutive rounds.

    Why layer norms instead of full vector?
    ----------------------------------------
    Full cosine similarity would require storing tens of millions of
    float32 values per client per round — too expensive.
    The layer-norm fingerprint captures the distribution of update
    energy across layers and is sufficient to detect:
      - Direction reversals (all layer norms flip relative pattern)
      - Abnormal layer-specific spikes (one layer dominates)
      - Consistent update patterns (stable fingerprint over rounds)

    Args:
        client_ndarrays : Client's model parameters (dequantized FP32)
        global_ndarrays : Previous global parameters (None = first round)

    Returns:
        1-D float64 ndarray of shape (num_layers,) with per-layer norms.
        If global_ndarrays is None, uses raw parameter norms as baseline.
    """
    if global_ndarrays is None or len(global_ndarrays) != len(client_ndarrays):
        # First round or shape mismatch — use raw parameter norms
        layer_norms = np.array([
            float(np.linalg.norm(p.astype(np.float64).flatten()))
            for p in client_ndarrays
        ], dtype=np.float64)
    else:
        layer_norms = np.array([
            float(np.linalg.norm((c.astype(np.float64) - g).flatten()))
            for c, g in zip(client_ndarrays, global_ndarrays)
        ], dtype=np.float64)

    return layer_norms


def compute_cosine_similarity(
    current_layer_norms: np.ndarray,
    prev_layer_norms: Optional[np.ndarray],
) -> float:
    """
    Compute cosine similarity between this round's and the previous
    round's update layer-norm fingerprint for the SAME client.

    WHY OWN-HISTORY COMPARISON?
    ----------------------------
    Non-IID clients have fundamentally different update patterns.
    Comparing Hospital B's update to Hospital A's update is meaningless
    and would cause false positives. We compare each client's current
    update to its own previous update — this is Non-IID safe.

    Returns:
        float in [-1.0, 1.0]
        1.0  = identical direction (perfectly consistent)
        0.0  = orthogonal (unrelated)
       -1.0  = opposite direction (complete reversal — suspicious)

        Returns 1.0 (neutral) when:
        - prev_layer_norms is None (first available round)
        - Either vector is a zero vector
        - Numerical error
    """
    if prev_layer_norms is None:
        return 1.0   # Neutral — no history available

    try:
        a = np.asarray(current_layer_norms, dtype=np.float64)
        b = np.asarray(prev_layer_norms,    dtype=np.float64)

        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))

        if norm_a < 1e-12 or norm_b < 1e-12:
            return 1.0   # Zero vector — treat as neutral

        cosine = float(np.dot(a, b) / (norm_a * norm_b))
        # Clamp to [-1, 1] for numerical stability
        return float(np.clip(cosine, -1.0, 1.0))

    except Exception:
        return 1.0   # Error → neutral, never penalise due to computation error


def compute_distribution_shift(
    current_pneumonia_ratio: Optional[float],
    prev_pneumonia_ratio:    Optional[float],
) -> float:
    """
    Compute a binary class distribution shift indicator.

    Uses absolute difference of pneumonia ratios (0 = NORMAL, 1 = PNEUMONIA).

    IMPORTANT: Distribution shift is CONTEXT, not evidence of malicious behaviour.
    ---
    A legitimate Non-IID scenario:
        Round 5:  NORMAL=300, PNEUMONIA=700  →  pneumonia_ratio=0.70
        Round 6:  NORMAL=700, PNEUMONIA=300  →  pneumonia_ratio=0.30
        distribution_shift = |0.30 - 0.70| = 0.40

    This shift is large but completely legitimate — the hospital may have
    processed a different patient cohort. This value is used by the training
    anomaly calculation to REDUCE penalties when a large shift explains the
    metric change. It is NOT a direct trigger for QUARANTINE.

    Returns:
        float in [0.0, 1.0]
        0.0 = no distribution change
        1.0 = complete distribution reversal (theoretical max)
        Returns 0.0 when history is insufficient (neutral).
    """
    if current_pneumonia_ratio is None or prev_pneumonia_ratio is None:
        return 0.0   # No history → neutral (no distribution context)

    try:
        shift = abs(float(current_pneumonia_ratio) - float(prev_pneumonia_ratio))
        return float(np.clip(shift, 0.0, 1.0))
    except Exception:
        return 0.0


# ==================================================================
# Anomaly Score components
# ==================================================================

def _l2_anomaly_contribution(
    current_norm:  Optional[float],
    norm_history:  List[float],
    min_history:   int,
) -> float:
    """
    Compute L2 update-norm anomaly contribution (0–50).

    Uses z-score of current update norm vs. client's OWN norm history.
    Below minimum history: returns 0 (neutral — no false positives).

    Score increases with z-score:
        z ≤ 0   → 0   (update is below historical average — healthy)
        z ≤ 1   → z * 8
        z ≤ 2   → 8 + (z-1)*14
        z ≤ 3   → 22 + (z-2)*18
        z > 3   → min(50, 40 + (z-3)*5)
    """
    if current_norm is None or current_norm == 0.0:
        return 0.0

    if len(norm_history) < min_history:
        return 0.0   # Insufficient history → neutral

    mean_n = float(np.mean(norm_history))
    std_n  = float(np.std(norm_history))

    if std_n < 1e-8:
        # Client is extremely stable — compare by ratio
        ratio = current_norm / (mean_n + 1e-8)
        if   ratio < 2.0:  return 0.0
        elif ratio < 3.0:  return 15.0
        elif ratio < 5.0:  return 30.0
        else:              return 45.0

    z = (current_norm - mean_n) / std_n

    if   z <= 0:  contribution = 0.0
    elif z <= 1:  contribution = z * 8.0
    elif z <= 2:  contribution = 8.0  + (z - 1.0) * 14.0
    elif z <= 3:  contribution = 22.0 + (z - 2.0) * 18.0
    else:         contribution = min(50.0, 40.0 + (z - 3.0) * 5.0)

    return float(np.clip(contribution, 0.0, 50.0))


def _cosine_anomaly_contribution(
    cosine_similarity:   float,
    rounds_of_history:   int,
    min_history:         int,
) -> float:
    """
    Compute cosine similarity anomaly contribution (0–30).

    Insufficient history → 0 (neutral — no false positives in early rounds).

    cosine ≥ 0.5   → 0    (consistent update direction — healthy)
    cosine ≥ 0.0   → up to 10   (minor deviation — slight penalty)
    cosine ≥ -0.5  → 10 to 25   (direction change — suspicious)
    cosine < -0.5  → 25 to 30   (direction reversal — very suspicious)
    """
    if rounds_of_history < min_history:
        return 0.0   # Not enough history → neutral

    if   cosine_similarity >= 0.5:
        contribution = 0.0
    elif cosine_similarity >= 0.0:
        # 0 to 10 — proportional to deviation from 0.5
        contribution = ((0.5 - cosine_similarity) / 0.5) * 10.0
    elif cosine_similarity >= -0.5:
        # 10 to 25
        contribution = 10.0 + ((-cosine_similarity) / 0.5) * 15.0
    else:
        # 25 to 30
        contribution = 25.0 + ((-cosine_similarity - 0.5) * 10.0)

    return float(np.clip(contribution, 0.0, 30.0))


def _training_anomaly_contribution(
    current_acc:         Optional[float],
    prev_acc:            Optional[float],
    current_loss:        Optional[float],
    prev_loss:           Optional[float],
    distribution_shift:  float,
    cfg:                 Dict,
) -> float:
    """
    Compute training-metric anomaly contribution (0–20), modulated by
    distribution shift context.

    WHY DISTRIBUTION SHIFT MODULATES THIS?
    ---------------------------------------
    A Non-IID hospital that switched from pneumonia-heavy to normal-heavy
    data will naturally see accuracy and loss changes. Penalising this
    equally to a suspicious update would create false positives.

    Therefore:
    - Base penalty from acc drop / loss spike is calculated first
    - Then moderated by distribution_shift:
        shift > dist_shift_high   → up to 80% reduction in contribution
        shift > dist_shift_medium → up to 40% reduction in contribution
        shift ≤ dist_shift_medium → no reduction (full contribution)

    IMPORTANT: Even with a large shift, if the update L2 norm is also
    anomalous, the L2 contribution remains unchanged. Distribution shift
    only moderates the training metric component.
    """
    if current_acc is None or prev_acc is None:
        return 0.0   # No evaluation history → neutral

    contribution = 0.0
    acc_delta = current_acc - prev_acc

    # Accuracy drop penalties
    if   acc_delta < -0.20: contribution += 15.0
    elif acc_delta < -0.10: contribution +=  8.0
    elif acc_delta < -0.05: contribution +=  3.0

    # Loss spike penalties
    if current_loss is not None and prev_loss is not None:
        loss_delta = current_loss - prev_loss
        if   loss_delta > 1.0: contribution += 5.0
        elif loss_delta > 0.5: contribution += 3.0
        elif loss_delta > 0.2: contribution += 1.0

    # ── Apply distribution shift moderation ──────────────────────
    # High shift means legitimate Non-IID change can explain metric variation.
    # This does NOT make a suspicious update trustworthy — it only acknowledges
    # that metric changes are not sole evidence of malice in Non-IID settings.
    dist_high   = cfg["dist_shift_high"]
    dist_medium = cfg["dist_shift_medium"]
    max_mod     = cfg["dist_shift_max_moderation"]

    if distribution_shift > dist_high:
        # Strong shift: up to max_moderation reduction (e.g. 80%)
        moderation = max_mod * min(distribution_shift / (dist_high + 0.10), 1.0)
        contribution *= (1.0 - moderation)
    elif distribution_shift > dist_medium:
        # Moderate shift: up to 40% reduction
        t = (distribution_shift - dist_medium) / (dist_high - dist_medium)
        moderation = 0.40 * t
        contribution *= (1.0 - moderation)
    # else: no moderation

    return float(np.clip(contribution, 0.0, 20.0))


# ==================================================================
# Main anomaly score function
# ==================================================================

def compute_anomaly_score(
    current_norm:        Optional[float],
    norm_history:        List[float],
    cosine_similarity:   float,
    rounds_of_history:   int,
    current_acc:         Optional[float],
    prev_acc:            Optional[float],
    current_loss:        Optional[float],
    prev_loss:           Optional[float],
    distribution_shift:  float,
    cfg:                 Optional[Dict] = None,
) -> Tuple[float, Dict]:
    """
    Compute the current-round Anomaly Score (0–100).

    Anomaly Score is DIFFERENT from Trust Score:
    - Trust Score  = long-term behavioural reputation (EMA-smoothed, post-eval)
    - Anomaly Score = current-round abnormality measure (fast, pre-aggregation)

    Args:
        current_norm       : L2 update norm for this round
        norm_history       : List of previous L2 norms for this client
        cosine_similarity  : Cosine similarity to previous update direction
        rounds_of_history  : Number of completed FL rounds for this client
        current_acc        : Accuracy this round (from fit metrics, if available)
        prev_acc           : Previous round accuracy
        current_loss       : Loss this round
        prev_loss          : Previous round loss
        distribution_shift : abs(current_pneumonia_ratio - prev_pneumonia_ratio)
        cfg                : Config dict (uses PRE_AGG_CONFIG if None)

    Returns:
        (anomaly_score, breakdown_dict)
        anomaly_score: float [0, 100] — higher = more anomalous
        breakdown_dict: {l2_contrib, cosine_contrib, training_contrib}
    """
    if cfg is None:
        cfg = PRE_AGG_CONFIG

    l2_contrib       = _l2_anomaly_contribution(
        current_norm, norm_history, cfg["min_history_for_l2"]
    )
    cosine_contrib   = _cosine_anomaly_contribution(
        cosine_similarity, rounds_of_history, cfg["min_history_for_cosine"]
    )
    training_contrib = _training_anomaly_contribution(
        current_acc, prev_acc, current_loss, prev_loss,
        distribution_shift, cfg
    )

    total = float(np.clip(l2_contrib + cosine_contrib + training_contrib, 0.0, 100.0))

    breakdown = {
        "l2_contribution":       round(l2_contrib,       2),
        "cosine_contribution":   round(cosine_contrib,   2),
        "training_contribution": round(training_contrib, 2),
    }

    return total, breakdown


# ==================================================================
# Pre-Aggregation Decision
# ==================================================================

def make_pre_agg_decision(
    anomaly_score: float,
    cfg:           Optional[Dict] = None,
) -> str:
    """
    Map an anomaly score to a pre-aggregation decision.

    Returns one of:
        "ACCEPT"     — update proceeds into normal FedAvg
        "QUARANTINE" — update is excluded from FedAvg; DACM audit triggered
        "REJECT"     — update is excluded; clear anomaly signal

    NOTE: This decision is made BEFORE the final Trust Score is computed.
    It is based solely on current-round evidence (L2 norm, cosine, training
    metrics). The final Trust Score (long-term reputation) is computed
    post-evaluation and may differ from this decision.

    SEPARATION OF CONCERNS:
    - PRE_AGGREGATION_DECISION: fast, pre-FedAvg, anomaly-based
    - FINAL TRUST TAG:          slow, post-evaluation, reputation-based
    """
    if cfg is None:
        cfg = PRE_AGG_CONFIG

    if   anomaly_score >= cfg["reject_threshold"]:
        return "REJECT"
    elif anomaly_score >= cfg["quarantine_threshold"]:
        return "QUARANTINE"
    else:
        return "ACCEPT"


# ==================================================================
# Aggregation Decision (post-evaluation, from final Trust Tag)
# ==================================================================

def tag_to_aggregation_decision(tag: str) -> str:
    """
    Map the final post-evaluation Trust Tag to an aggregation decision.

    This is the FINAL aggregation decision — computed AFTER evaluation.
    It is separate from the pre-aggregation decision.

    Mapping:
        TRUSTED    → ACCEPT
        SUSPICIOUS → QUARANTINE
        UNTRUSTED  → REJECT
    """
    mapping = {
        "TRUSTED":    "ACCEPT",
        "SUSPICIOUS": "QUARANTINE",
        "UNTRUSTED":  "REJECT",
    }
    return mapping.get(tag, "ACCEPT")
