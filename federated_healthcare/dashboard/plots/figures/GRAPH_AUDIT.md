# GRAPH AUDIT

## Figure 1: Final Accuracy Comparison across Experiments
- **Filename**: `fig1_final_accuracy.png` / `.pdf`
- **Purpose**: Compare final model accuracy across the seven main federated learning scenarios to establish performance viability.
- **Source data**: `final_experiment_results.csv`
- **Exact experiments/runs used**: `exp1_baseline` (run_01), `exp2_one_straggler` (run_04), `exp3_two_stragglers` (run_05), `exp4_network_dropout` (run_01), `exp5_recovery` (run_01), `exp6_adaptive_off` (run_01), `exp7_nonstationary` (run_02)
- **What each series represents**: The final accuracy percentage achieved by the global model at the end of the experiment.
- **Important limitations**: Exp 2 only completed 4 out of 10 rounds before reaching the 600s global timeout, which inherently caps its accuracy compared to 10-round completions. Exp 3 and 7 suffered from CPU contention slowing down the safe clients.
- **Suitable for IEEE paper**: Yes. The layout and typography adhere to standard IEEE aesthetic limits.
- **Should it appear in the paper**: Yes, as the primary global performance metric visualization.

## Figure 2: Operational Outcomes per Experiment
- **Filename**: `fig2_client_round_outcomes.png` / `.pdf`
- **Purpose**: Quantify the raw volume of successful client participations versus failed/dropped participations and aborted aggregations.
- **Source data**: `final_experiment_results.csv`
- **Exact experiments/runs used**: Same as Figure 1.
- **What each series represents**: 
  - Green bars: Count of successful client-rounds safely aggregated.
  - Red bars: Count of client-rounds that failed, dropped, or timed out.
  - Grey bars: Count of entirely aborted aggregations (where minimum quorum failed).
- **Important limitations**: The definition of "failed" encompasses network disconnects, predictive straggler drops, and end-of-round cancellation timeouts. 
- **Suitable for IEEE paper**: Yes. Clearly differentiates system efficiency without confounding different failure types.
- **Should it appear in the paper**: Yes, it visually explains the difference between the robust fault tolerance in Exp4 and the highly constrained nature of Exp3/7.

## Figure 3: Adaptive Decision Mechanism
- **Filename**: `fig3_adaptive_decision_response.png` / `.pdf`
- **Purpose**: Detail the chronological observation, prediction, and preemption loop for a dynamic straggler (Hospital C).
- **Source data**: `execution.log` (Exp 7) 
- **Exact experiments/runs used**: `exp7_nonstationary`
- **What each series represents**:
  - Blue circles: Ground truth observed completion time of Hospital C.
  - Orange squares: The engine's subsequent predicted safe completion time based on the EMA update.
  - Red dashed line: The 60-second absolute deadline limit.
  - Red X marks: Preemptive dropping decisions executed precisely when the prediction exceeded the deadline.
- **Important limitations**: Relies heavily on Exp 7's timing, which was partially affected by CPU contention (causing the spikes to >100s).
- **Suitable for IEEE paper**: Yes.
- **Should it appear in the paper**: Yes, it is the most critical figure to explain the novelty of the algorithm visually.

## Figure 4: Exp 7 Adaptive Prediction of Nonstationary Delay
- **Filename**: `fig4_nonstationary_hospital_c.png` / `.pdf`
- **Purpose**: Demonstrates the exponential moving average (EMA) trailing the actual completion times.
- **Source data**: `execution.log` (Exp 7)
- **Exact experiments/runs used**: `exp7_nonstationary`
- **What each series represents**:
  - Blue line: Observed completion time.
  - Orange line: Predicted completion time (EMA + Dev).
- **Important limitations**: Redundant with Figure 3; essentially shows the same Hospital C nonstationary behavior in a slightly different format.
- **Suitable for IEEE paper**: Yes, but redundant.
- **Should it appear in the paper**: No, Figure 3 is superior as it explicitly calls out the mechanism steps. Figure 4 can be omitted to save space, or kept as supplementary material.

## Figure 6: Global Evaluation Accuracy Trajectory Across Rounds
- **Filename**: `fig6_round_accuracy.png` / `.pdf`
- **Purpose**: Illustrate convergence progression of the global model across federated rounds for primary experiments.
- **Source data**: `metrics.csv` (`Round`, `Accuracy`) across primary experiment runs (`exp1_baseline/run_01`, `exp2_one_straggler/run_04`, `exp3_two_stragglers/run_05`, `exp4_network_dropout/run_01`, `exp5_recovery/run_01`, `exp6_adaptive_off/run_01`, `exp7_nonstationary/run_02`).
- **What each series represents**: Global test accuracy evaluated at the conclusion of each aggregated federated round.
- **Important limitations**: Incomplete runs (Exp 2 halted at round 4 by 600s timeout; Exp 3 and 7 suffered CPU contention causing aborted aggregations) plot strictly measured rounds without synthetic interpolation.
- **Suitable for IEEE paper**: Yes.

## Figure 7: Global Evaluation Loss Trajectory Across Rounds
- **Filename**: `fig7_round_loss.png` / `.pdf`
- **Purpose**: Companion loss curve illustrating convergence stability and penalty under adverse client conditions.
- **Source data**: `metrics.csv` (`Round`, `Loss`) across primary experiment runs.
- **What each series represents**: Global test cross-entropy loss evaluated at the end of each aggregated round.
- **Important limitations**: Plotted strictly for confirmed aggregated rounds matching Figure 6.
- **Suitable for IEEE paper**: Yes.

## Figure 8: Client Training Completion Time vs. Hard Deadline (Exp 7)
- **Filename**: `fig8_client_completion_time.png` / `.pdf`
- **Purpose**: Contrast observed client training completion times across federated rounds against the configured hard deadline ($T_{\text{hard}} = 60\text{s}$) in the nonstationary experiment.
- **Source data**: `Hospital_A.csv`, `Hospital_B.csv`, `Hospital_C.csv`, and `config.json` in `exp7_nonstationary/run_02`.
- **What each series represents**: Measured training duration for Hospital A, Hospital B, and Hospital C per round.
- **Important limitations**: Shows actual measured training completion times; rounds where Hospital C was preemptively dropped are annotated with markers.
- **Suitable for IEEE paper**: Yes.

## Figure 9: Client Participation Heatmap (Status Audit)
- **Status**: Skipped intentionally per experimental integrity guidelines.
- **Reason**: Flower server allocates ephemeral UUID CIDs without persistent client identification in server-side logs, and dropped/timed-out clients do not generate local CSV logs. Constructing a granular status heatmap would require unverifiable synthetic inference.

## Figure 10: Communication Payload Comparison: FP32 vs. INT8 Quantization
- **Filename**: `fig10_quantization_payload.png` / `.pdf`
- **Purpose**: Empirically compare model weight transmission payloads between standard FP32 and INT8 quantization.
- **Source data**: `Hospital_A.csv`, `Hospital_B.csv`, `Hospital_C.csv` (`payload_size_mb`, `quantized_payload_size_mb`, `compression_ratio`).
- **What each series represents**: Unquantized FP32 model payload (32.07 MB) vs. INT8 quantized payload (8.02 MB) per client.
- **Important limitations**: Demonstrates exact 4.00x weight compression (75.0% reduction) measured in every round; latency reduction is dependent on link bandwidth.
- **Suitable for IEEE paper**: Yes.

## Figure 11: Client Disconnection and Recovery Timeline
- **Filename**: `fig11_recovery_timeline.png` / `.pdf`
- **Purpose**: Analyze client drop-off and recovery under abrupt network disconnection (Exp 4 permanent vs. Exp 5 recovery).
- **Source data**: `round_metrics.csv` (`Successful_Clients`) and `metrics.csv` (`Accuracy`) in `exp4_network_dropout/run_01` and `exp5_recovery/run_01`.
- **What each series represents**:
  - Top Panel: Active client participation count over rounds 1–10.
  - Bottom Panel: Global accuracy trajectory maintaining convergence post-dropout and post-reconnection.
- **Important limitations**: Clarifies that transport-level reconnection assigned a fresh Flower CID rather than restoring prior client session state.
- **Suitable for IEEE paper**: Yes.

## Figure 12: Expanded Operational Reliability Comparison
- **Filename**: `fig12_operational_reliability.png` / `.pdf`
- **Purpose**: Multi-level operational reliability breakdown separating client-level participation outcomes from server-level round outcomes.
- **Source data**: `final_experiment_results.csv`.
- **What each series represents**:
  - Top Panel: Client-Rounds count (Successful vs. Failed/Dropped client-rounds, max 30).
  - Bottom Panel: Federated Rounds count (Completed aggregations vs. Aborted aggregations, max 10).
- **Important limitations**: Resolves the dimensional mixing present in earlier visualizations; annotates wall-clock timeout and thread contention constraints.
- **Suitable for IEEE paper**: Yes.
