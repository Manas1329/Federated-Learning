# Implementation Plan: Fix Adaptive Multi-Drop Quorum Handling

## Proposed Changes

### 1. Update `AdaptiveServer.fit_round` in `dropout_handler.py`
The issue occurs when `self.engine.evaluate_missing_clients()` returns multiple `DROP` decisions, and the server blindly cancels all of them without checking quorum. I will modify the cancellation loop:

```python
# Evaluate decisions
decisions = self.engine.evaluate_missing_clients()

futures_to_cancel = []
for future, (client_proxy, _) in future_to_client.items():
    cid = str(client_proxy.cid)
    decision = decisions.get(cid)
    
    if decision and not decision.should_wait:
        futures_to_cancel.append((future, client_proxy, decision.reason))

# Quorum protection logic
max_drops_allowed = max(0, len(results) + len(future_to_client) - self.min_clients)

# If we have more drop candidates than allowed drops, we must retain some
while len(futures_to_cancel) > max_drops_allowed:
    # Pop one candidate from the cancel list to retain it (keep it running)
    retained_future, cp, reason = futures_to_cancel.pop()
    print(f"[AdaptiveServer] Quorum protection prevented dropping client {cp.cid} despite decision: {reason}")
    # Tell the engine we decided to wait anyway
    # Note: engine state for this client will remain STRAGGLER, so it's safe.

# Apply the allowed drops
for future, client_proxy, reason in futures_to_cancel:
    future.cancel()
    future_to_client.pop(future)
    print(f"[AdaptiveServer] Engine decision: DROP client {client_proxy.cid} ({reason})")
    self.engine.record_straggler_drop(str(client_proxy.cid))
    failures.append(Exception("Dropped by Adaptive Dropout Engine"))
    self._record_participation(str(client_proxy.cid), success=False)
```

### 2. Add Regression Tests in `tests/test_dropout_handler.py`
I will add focused tests to verify that `AdaptiveServer` preserves quorum when faced with multiple `DROP` candidates:
- **Test 1**: Two DROP candidates, one completed client, quorum=2 -> At most 1 drop allowed.
- **Test 2**: Two completed clients, one DROP candidate, quorum=2 -> 1 drop allowed.
- **Test 3**: No completed clients, two DROP candidates, quorum=2 -> 0 drops allowed.

### 3. Verification & Experiment Execution
- Run `pytest tests/test_dropout_handler.py` to ensure all tests (including new ones) pass.
- Rerun Experiment 2 (`python run_experiments.py --exp exp2_one_straggler --runs 1`).
- Analyze the logs to confirm the anomaly (B and C being simultaneously dropped leading to a quorum failure) is resolved.
- Provide the final requested report.

## Verification Plan
1. Tests must pass locally.
2. The `exp2_one_straggler/run_03` execution logs must show `[AdaptiveServer] Quorum protection prevented dropping client` and show 100% successful aggregations (no aborted rounds due to premature multiple drops).
