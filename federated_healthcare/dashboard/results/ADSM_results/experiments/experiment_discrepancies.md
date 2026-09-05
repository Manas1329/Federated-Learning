# Experiment Discrepancies Report

This document records the discrepancies between the previously existing repository artifacts (`experiment_summary.md`) and the verified authoritative audit of the completed experiments.

## Discrepancy 1: Stale Run Tracking and Resolution of GRPC Errors
- **What the artifact (`experiment_summary.md`) says:** 
  Experiments 1, 2, 3, 4, 5, and 7 exhibit `GRPC_BRIDGE_ERROR` anomalies. It claims `exp3_two_stragglers` and `exp7_nonstationary` failed completely with 0 successful clients in `run_01`.
- **What the audit says:** 
  The GRPC bridge proxy-corruption bug was definitively fixed by introducing the `busy_clients` safeguard to the `AdaptiveServer`. In the authoritative runs (`run_01` for Exps 1, 2, 4, 5, 6; `run_05` for Exp 3; `run_02` for Exp 7), there are exactly **zero** `ValueError: This should not happen` or GRPC bridge errors.
- **Can it be resolved:** Yes.
- **Authoritative source:** The raw `execution.log` of the authoritative runs, which prove the errors no longer exist.

## Discrepancy 2: Successful/Failed Client Counts
- **What the artifact (`experiment_summary.md`) says:**
  - Exp 2: 6 successful, 3 failed
  - Exp 4: 18 successful, 1 failed
  - Exp 5: 18 successful, 9 failed
- **What the audit says:**
  - Exp 2 (run_01): 13 successful, 17 failed
  - Exp 4 (run_01): 19 successful, 3 failed
  - Exp 5 (run_01): 19 successful, 11 failed
- **Can it be resolved:** Yes.
- **Authoritative source:** The raw `execution.log` of `run_01` for each respective experiment. The artifact summarizes early aborted/stale runs prior to the final verified executions.

## Discrepancy 3: Missing Limitations Acknowledgment
- **What the artifact (`experiment_summary.md`) says:**
  Makes no mention of the physical hardware limitations or CPU thread oversubscription.
- **What the audit says:**
  Exp 3 and Exp 7 suffered aborted aggregations due to local CPU contention inflating Hospital A's execution time past the 60.0s hard deadline.
- **Can it be resolved:** Yes.
- **Authoritative source:** The verified experimental observations in `final_experiment_results.md` must be treated as the authoritative explanation for aborted rounds in Exps 3 and 7.

## Conclusion
The existing `experiment_summary.md` and `experiment_summary.csv` contain stale metrics from before the `busy_clients` proxy fix and the CPU thread limiting mitigations. They should be considered deprecated. The new `final_experiment_results.csv` and `final_experiment_results.md` are the single source of truth based exclusively on the verified raw execution logs.
