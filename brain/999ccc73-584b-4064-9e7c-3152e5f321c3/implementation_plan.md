# Implementation Plan: Rerun Exp 2 & Implement Exp 8

## Proposed Changes

### 1. Update `run_experiments.py`
- Add `fixed_deadline_control: bool = False` to `run_experiment()` signature.
- Include `"FIXED_DEADLINE_CONTROL": int(fixed_deadline_control)` in the `config.json` serialization.
- Pass `FIXED_DEADLINE_CONTROL` to the server process via environment variables.
- Add `"exp8_fixed_deadline"` to the list of experiments.
- When `exp == "exp8_fixed_deadline"`, call `run_experiment` with `adaptive_dropout_enabled=False`, `fixed_deadline_control=True`, and `artificial_delays={"Hospital_B": 70}`.

### 2. Update `demo_server.py`
- Read `FIXED_DEADLINE_CONTROL` from environment variables, defaulting to `"0"`.
- Pass `fixed_deadline_control` to the `AdaptiveServer` constructor.

### 3. Update `dropout_handler.py` (`AdaptiveServer`)
- Add `fixed_deadline_control: bool = False` to `__init__`.
- Inside `fit_round`'s `while future_to_client:` loop, add a fixed deadline policy block:
```python
    if future_to_client and self.adaptive_dropout_enabled:
        # Existing adaptive dropout logic
        ...
    elif future_to_client and self.fixed_deadline_control:
        # Fixed Hard-Deadline Control
        if self.engine.get_elapsed_time() >= self.engine.hard_deadline:
            # Exclude incomplete clients ONLY if minimum quorum is preserved
            if len(results) >= self.min_clients:
                futures_to_cancel = list(future_to_client.keys())
                for future in futures_to_cancel:
                    client_proxy, _ = future_to_client.pop(future)
                    future.cancel()
                    self.engine.record_straggler_drop(str(client_proxy.cid))
                    failures.append(Exception("Dropped by Fixed Deadline Control"))
                    self._record_participation(str(client_proxy.cid), success=False)
```
- This ensures that if the deadline passes, incomplete clients are dropped *only* if `len(results) >= min_clients`. If quorum is not met, the server continues waiting until another client finishes, at which point the check evaluates again and drops any remaining clients.

## Verification Plan
1. **Tests**: Run existing tests (`pytest tests/test_dropout_handler.py`) to ensure no regressions.
2. **Exp 2 Rerun**: Execute `python run_experiments.py --exp exp2_one_straggler --runs 1` and analyze the logs for `run_02`.
3. **Exp 8 Rerun**: Execute `python run_experiments.py --exp exp8_fixed_deadline --runs 1` and analyze the logs for `run_01`.
4. Provide the final report as requested.
