# 🛡️ Client Trust Scoring & Tagging System

> **File:** `src/trust_manager.py` · **Used by:** `src/server.py`
>
> A Phase-1 non-IID-safe trust evaluation system that runs **on the server side**, scores every federated client after each round, and assigns a human-readable trust tag without touching aggregation, training, or differential privacy.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture & Data Flow](#2-architecture--data-flow)
3. [The Four Score Components](#3-the-four-score-components)
4. [Final Trust Score Formula](#4-final-trust-score-formula)
5. [Tagging Thresholds](#5-tagging-thresholds)
6. [Round-by-Round Lifecycle](#6-round-by-round-lifecycle)
7. [Non-IID Safety Guarantees](#7-non-iid-safety-guarantees)
8. [Configuration Reference](#8-configuration-reference)
9. [Output & Logging](#9-output--logging)
10. [Worked Example (Real Run)](#10-worked-example-real-run)
11. [What the Tags Mean](#11-what-the-tags-mean)
12. [Key Design Decisions](#12-key-design-decisions)

---

## 1. Overview

Every Federated Learning round, three hospitals submit model updates. The Trust Manager answers:

> *"Can we trust that this hospital's update is behaviorally consistent and not anomalous?"*

It does this by computing **four independent scores** and blending them into a single **0–100 Trust Score**, which is then bucketed into one of three tags:

| Tag | Emoji | Score Range |
|-----|-------|-------------|
| TRUSTED | 🟢 | 80 – 100 |
| SUSPICIOUS | 🟡 | 50 – 79 |
| UNTRUSTED | 🔴 | 0 – 49 |

> **Critical Design Choice:** All scoring is relative to each client's **own history**. Hospital B with naturally lower accuracy is NOT penalised just because Hospital A has higher accuracy. This makes the system Non-IID safe.

---

## 2. Architecture & Data Flow

```
                          ┌─────────────────────────────────────────────┐
                          │                 SERVER.PY                   │
                          │                                             │
  Round N starts          │  configure_fit()                            │
  ─────────────────►      │    └─► tm.set_global_params(global_weights) │
                          │         (saves current global model weights) │
                          │                                             │
  Clients train &         │  aggregate_fit()                            │
  send updates  ─────────►│    └─► tm.record_update(client_name,        │
                          │              client_weights, cid)           │
                          │         (computes L2 norm of update)        │
                          │                                             │
  Clients evaluate &      │  evaluate_metrics_aggregation_fn()          │
  send metrics  ─────────►│    └─► tm.record_evaluation(client_name,    │
                          │              accuracy, loss, f1, ...)       │
                          │         (stores this round's metrics)       │
                          │                                             │
                          │    └─► tm.finalize_round(round_number)      │
                          │         (computes all 4 scores,             │
                          │          assigns tag, logs CSV,             │
                          │          prints table, updates dashboard)   │
                          └─────────────────────────────────────────────┘
                                            │
                                            ▼
                          ┌─────────────────────────────────────────────┐
                          │         Next Round: configure_fit()         │
                          │    └─► Trust score injected back into       │
                          │        each client's FitIns config          │
                          │        (client receives its own tag)        │
                          └─────────────────────────────────────────────┘
```

**The `TrustManager` class is a singleton** — one instance lives for the entire server process. It maintains per-client state dictionaries that grow each round.

---

## 3. The Four Score Components

### 3.1 Update Behaviour Score (Weight: 40%)

**What it measures:** Whether the size of a client's model update is normal for *that specific client*.

**How it is computed:**

**Step 1 — Compute L2 Update Norm:**
```
update_norm = sqrt( sum( (client_weight_i - global_weight_i)^2 ) )
              over all layers i
```
This is the Euclidean distance between what the client sent back and what the server sent out. A very large norm = very aggressive local update. A very small norm = almost no learning happened.

**Step 2 — Compare against client's OWN norm history:**

The system keeps a list of all previous norms for this client. Once there are at least **3 rounds** of history, it computes a **z-score**:

```
z = (current_norm - mean(past_norms)) / std(past_norms)
```

**Step 3 — Map z-score to a score (0–100):**

| z-score | Meaning | Score |
|---------|---------|-------|
| z ≤ 0 | Below average norm (healthy, smaller than usual update) | 90.0 |
| 0 < z ≤ 1 | Slightly above average (1 std dev) | 80–90 |
| 1 < z ≤ 2 | Notably above average (2 std devs) | 65–80 |
| 2 < z ≤ 3 | Suspicious spike (3 std devs) | 40–65 |
| z > 3 | Extreme anomaly | 10–40 (min 10) |

**Edge cases:**
- **< 3 rounds of history:** Returns neutral score (75.0). No penalty during warmup.
- **std ≈ 0 (extremely stable client):** Uses absolute ratio current / mean:
  - ratio < 3x → 90.0
  - ratio 3–5x → 55.0
  - ratio > 5x → 20.0

---

### 3.2 Training Behaviour Score (Weight: 30%)

**What it measures:** Whether a client's local accuracy and loss are improving, stable, or suddenly degrading compared to *its own previous round*.

**Starting point:** 85.0 (generous base — most honest clients stay here).

**Penalties applied based on accuracy delta (curr_acc - prev_acc):**

| Accuracy Change | Meaning | Penalty |
|----------------|---------|---------|
| drop > 20% | Catastrophic degradation | -30 |
| drop 10–20% | Severe drop | -15 |
| drop 5–10% | Notable drop | -5 |
| stable or improving | Normal | 0 |

**Penalties applied based on loss delta (curr_loss - prev_loss):**

| Loss Change | Meaning | Penalty |
|------------|---------|---------|
| increase > 1.0 | Diverging model | -20 |
| increase 0.5–1.0 | Large spike | -10 |
| increase 0.2–0.5 | Moderate spike | -5 |
| stable or decreasing | Normal | 0 |

**Rewards for improvement:**

| Accuracy Change | Reward |
|----------------|--------|
| improvement > 5% | +10 |
| any improvement > 0% | +5 |

**Absolute sanity check:** If current_accuracy < 30%, deduct an additional -20.
(Even imbalanced non-IID binary classification should beat 30%.)

**First-round behaviour:** If there is no previous round data, only the sanity check applies. No delta-based penalties.

---

### 3.3 Historical Reputation Score (Weight: 20%)

**What it measures:** A "memory" of how this client has behaved across all previous rounds. It is never reset.

**Mechanism — Exponential Moving Average (EMA):**

After each round's final trust score is computed, the historical trust is updated:

```
historical_trust_new = (1 - alpha) x historical_trust_old  +  alpha x this_round_trust_score
```

Where **alpha = 0.30** (configurable).

**Intuition:**
- One bad round shifts historical trust by at most 30% of the gap.
- A client trusted for 9 rounds cannot become untrusted after 1 bad round.
- A consistently bad client will slowly see its historical trust decline.

**Initialization:** Every new client starts at **75.0** (neutral, not penalised for being new).

**Example EMA trace (alpha = 0.30):**
```
Round 1: trust=82 → hist = 0.7x75 + 0.3x82 = 77.1
Round 2: trust=88 → hist = 0.7x77.1 + 0.3x88 = 80.0
Round 3: trust=30 → hist = 0.7x80.0 + 0.3x30 = 65.0   ← one bad round
Round 4: trust=85 → hist = 0.7x65.0 + 0.3x85 = 71.0   ← recovering
```

---

### 3.4 Participation Reliability Score (Weight: 10%)

**What it measures:** The fraction of rounds this client successfully completed out of all rounds it was selected for.

**Formula:**
```
participation_score = (successful_rounds / total_selected_rounds) x 100
```

| Scenario | Score |
|----------|-------|
| Never dropped out | 100.0 |
| Dropped out 1 of 10 rounds | 90.0 |
| Dropped out 5 of 10 rounds | 50.0 |
| New client (0 rounds so far) | 75.0 (neutral) |

**How dropouts are detected:** In `aggregate_fit()`, the server receives a `failures` list. Each failure is recorded via `tm.record_dropout(cid)`. Clients in `failures` but not in the success set have their `total` counter incremented but NOT their `success` counter.

---

## 4. Final Trust Score Formula

```
Trust Score =
    0.40 x Update Behaviour Score        (L2 norm anomaly detection)
  + 0.30 x Training Behaviour Score      (acc/loss trend per client baseline)
  + 0.20 x Historical Reputation Score   (EMA of all past trust scores)
  + 0.10 x Participation Reliability     (rounds succeeded / rounds selected)
```

The result is **clipped to [0.0, 100.0]**.

**Why these weights?**

| Component | Weight | Rationale |
|-----------|--------|-----------|
| Update Behaviour | 40% | Model poisoning shows up here first — a compromised client sends anomalous weight updates |
| Training Behaviour | 30% | Honest clients generally improve; sudden drops suggest data corruption or poisoning |
| Historical Reputation | 20% | Long-term trust is a meaningful signal; one bad round should not destroy trust |
| Participation Reliability | 10% | Low weight because dropouts may be due to hardware/network, not malicious intent |

---

## 5. Tagging Thresholds

```python
TRUST_CONFIG["thresholds"] = {
    "TRUSTED":    80.0,   # score >= 80  -> TRUSTED
    "SUSPICIOUS": 50.0,   # score >= 50  -> SUSPICIOUS
}
                           # score <  50  -> UNTRUSTED
```

| Score Range | Tag | Emoji | Meaning |
|-------------|-----|-------|---------|
| 80 – 100 | TRUSTED | 🟢 | Consistent, well-behaved client |
| 50 – 79 | SUSPICIOUS | 🟡 | Anomalous but not definitively malicious |
| 0 – 49 | UNTRUSTED | 🔴 | Strongly anomalous — possible poisoning or data corruption |

> **Important:** Tags are **informational only in Phase 1**. They do NOT affect FedAvg aggregation weights. The trust system is fully decoupled from training.

---

## 6. Round-by-Round Lifecycle

```
ROUND START
    │
    ├── configure_fit()
    │       └── tm.set_global_params(current_global_weights)
    │           (snapshot the model before clients train)
    │
    ├── [Clients train locally and return updates]
    │
    ├── aggregate_fit()
    │       ├── Dequantize INT8 -> FP32 (if quantization enabled)
    │       ├── tm.record_update(client_name, fp32_weights, cid)
    │       │       └── Computes L2 norm: ||client_weights - global_weights||_2
    │       │           Stored as client_state[name]["current_norm"]
    │       ├── tm.record_dropout(cid)  <- for each failure
    │       └── FedAvg aggregation -> new global model
    │
    ├── [Server sends global model to eval clients]
    │
    ├── evaluate_metrics_aggregation_fn()
    │       ├── tm.record_evaluation(client_name, accuracy, loss, f1, ...)
    │       │       └── Stored as current_acc, current_loss in client_state
    │       └── tm.finalize_round(round_number)
    │               ├── For each participant:
    │               │       ├── update_score   = _calc_update_score()
    │               │       ├── training_score = _calc_training_score()
    │               │       ├── hist_score     = historical_trust (EMA)
    │               │       ├── part_score     = success/total x 100
    │               │       ├── trust_score    = weighted sum (clipped 0-100)
    │               │       ├── tag            = TRUSTED / SUSPICIOUS / UNTRUSTED
    │               │       ├── Append current_norm  -> norm_history
    │               │       ├── Append current_acc   -> acc_history
    │               │       ├── Append current_loss  -> loss_history
    │               │       └── Update historical_trust via EMA
    │               ├── Log to CSV (trust_b_quantized.csv)
    │               ├── Print trust table to terminal
    │               └── Push to dashboard state
    │
ROUND END -> Next round: trust score injected into each client's config
```

---

## 7. Non-IID Safety Guarantees

In healthcare federated learning, different hospitals have **fundamentally different data distributions** (Non-IID). A naive comparison would unfairly penalise hospitals with harder data.

This system avoids that through three rules:

| Mechanism | How It Prevents Non-IID Bias |
|-----------|------------------------------|
| **Own-history comparison** | Update norm is compared against the same client's past norms, not against other clients |
| **Own-baseline training score** | Accuracy drop is compared against the client's own previous round, not against global average |
| **Generous neutral score (75.0)** | New clients and clients with insufficient history get 75/100, not 0 |

> A hospital that consistently achieves 60% accuracy and suddenly drops to 38% IS flagged.
> A hospital that consistently achieves 60% while another consistently achieves 95% is **not** penalised.

---

## 8. Configuration Reference

All tunable values are in `TRUST_CONFIG` at the top of `trust_manager.py`:

```python
TRUST_CONFIG = {
    "weights": {
        "update_behaviour":          0.40,  # Weight for update norm score
        "training_behaviour":        0.30,  # Weight for acc/loss trend score
        "historical_reputation":     0.20,  # Weight for EMA history score
        "participation_reliability": 0.10,  # Weight for dropout rate score
    },
    "thresholds": {
        "TRUSTED":    80.0,   # Minimum score for TRUSTED tag
        "SUSPICIOUS": 50.0,   # Minimum score for SUSPICIOUS tag
                              # Below 50 -> UNTRUSTED
    },
    "historical_ema_alpha":     0.30,   # EMA smoothing factor (0=no update, 1=overwrite)
    "min_history_for_anomaly":  3,      # Rounds needed before norm anomaly detection
    "neutral_score":            75.0,   # Score used when no history available
    "initial_historical_trust": 75.0,   # Starting historical trust for new clients
}
```

---

## 9. Output & Logging

### Terminal Output (per round)

```
======================================================================
              CLIENT TRUST STATUS — Round 10
======================================================================
Client             Update  Training  History  Reliab   SCORE  TAG
----------------------------------------------------------------------
Hospital_A           90.0      95.0     85.0   100.0    90.5  🟢 TRUSTED
Hospital_B           81.9      85.0     85.8   100.0    85.4  🟢 TRUSTED
Hospital_C           88.0      90.0     82.0   100.0    87.6  🟢 TRUSTED
======================================================================

  Hospital_B
  Update Behaviour          : 81.9 / 100
  Training Behaviour        : 85.0 / 100
  Historical Reputation     : 85.8 / 100
  Participation             : 100.0 / 100
  --------------------------------------
  Final Trust Score         : 85.4 / 100
  Tag                       : 🟢 TRUSTED
```

### CSV File

Written to: `dashboard/results/<suffix>/trust_<suffix>.csv`

| Column | Description |
|--------|-------------|
| Round | FL round number |
| Client | Hospital name |
| UpdateScore | Component 1 score (0–100) |
| TrainingScore | Component 2 score (0–100) |
| HistoricalScore | Component 3 EMA score (0–100) |
| ReliabilityScore | Component 4 score (0–100) |
| TrustScore | Final blended score (0–100) |
| Tag | TRUSTED / SUSPICIOUS / UNTRUSTED |

### Dashboard State

Trust data is pushed to `dashboard_state.py`'s `LIVE_STATE["trust_data"]` dictionary after each round, making it available to the live dashboard API.

### Trust Score Fed Back to Clients

At the start of each round, the server injects the **previous round's trust score** into each client's `FitIns` config:

```python
config["trust_score"]       = 85.4
config["trust_tag"]         = "TRUSTED"
config["trust_update"]      = 81.9
config["trust_training"]    = 85.0
config["trust_history"]     = 85.8
config["trust_reliability"] = 100.0
```

The client prints this at the start of training so each hospital can see its own standing.

---

## 10. Worked Example (Real Run)

From your actual run output (Round 10, Hospital_B):

```
Hospital_B
  Update Behaviour      : 81.9 / 100
  Training Behaviour    : 85.0 / 100
  Historical Trust      : 85.8 / 100
  Participation         : 100.0 / 100
  Final Trust Score     : 85.4 / 100
  Client Tag            : 🟢 TRUSTED
```

**Step-by-step verification:**

```
Trust Score = 0.40 x 81.9
            + 0.30 x 85.0
            + 0.20 x 85.8
            + 0.10 x 100.0

           = 32.76
           + 25.50
           + 17.16
           + 10.00

           = 85.42  ≈  85.4  ✓
```

**What each score tells us:**
- **Update score 81.9:** Hospital B's update norm was slightly above its own historical mean (z ≈ 0.8 → score ≈ 82), meaning it learned a bit more than usual — which is fine, not anomalous.
- **Training score 85.0:** Accuracy and loss were stable or gently improving from the previous round. Started at base 85, slight reward for improvement applied.
- **Historical score 85.8:** Built up over 10 rounds of consistently good behavior. The EMA has converged above 85 indicating a long track record.
- **Participation 100.0:** Hospital B connected and submitted updates in every single round.
- **Final: 85.4 → 🟢 TRUSTED** (above 80 threshold).

---

## 11. What the Tags Mean

### 🟢 TRUSTED (80–100)

The client is behaving consistently and predictably:
- Its model updates are the normal size for this client
- Its accuracy/loss trajectory is stable or improving
- It has a good track record across rounds
- It shows up reliably every round

**What this means for the project:** Include its updates in FedAvg aggregation with full confidence. This client is contributing valid, honest model improvements.

---

### 🟡 SUSPICIOUS (50–79)

Something is slightly off. Possible causes:
- A one-off anomalous update (bad batch of data, hardware instability)
- A small but persistent downward trend in accuracy
- Occasional missed rounds
- Model updates noticeably larger/smaller than usual

**What this means for the project:** Monitor this client closely. In Phase 2, you might reduce its aggregation weight proportionally to its trust score.

---

### 🔴 UNTRUSTED (0–49)

Strongly anomalous behavior detected. Possible causes:
- Update norm is 3–5 standard deviations above this client's own norm
- Massive sudden accuracy drop (>20%) suggesting poisoned local data
- Very high dropout rate
- Multiple negative signals combining in the same round

**What this means for the project:** In Phase 1, this is a flag for investigation. In Phase 2, this client could be excluded from aggregation entirely to protect the global model.

---

## 12. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Singleton TrustManager** | State must persist across Flower callbacks within a round and across rounds |
| **Try/except around all trust calls** | Trust is a monitoring overlay — it must NEVER crash FL training |
| **EMA for historical trust** | Gradual change prevents flip-flopping; one bad round does not equal untrusted |
| **z-score for update anomaly** | Scale-invariant; works regardless of model size or learning rate |
| **Minimum 3 rounds before anomaly detection** | Prevents false positives during cold start warmup period |
| **75.0 neutral score for new clients** | Innocent until proven suspicious; not penalised for being new |
| **No cross-client comparison** | Non-IID hospitals have different natural norms — comparing across hospitals would be unfair |
| **Tags don't affect aggregation** | Phase 1 is observational only; aggregation modification is Phase 2 |
| **Trust injected back to clients** | Hospitals can see their own standing; provides transparency and accountability |

---

*Source files: `src/trust_manager.py` and `src/server.py`*
