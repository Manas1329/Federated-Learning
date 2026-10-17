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
