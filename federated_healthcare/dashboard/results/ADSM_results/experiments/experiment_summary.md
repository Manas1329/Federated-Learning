# Federated Learning Experiments Summary

## Overview
Total experiments executed: 7

## Aggregate Results
| Experiment           | Run    |   Num_Rounds |   Adaptive_Dropout |   Final_Accuracy |   Final_Loss |   Total_Server_Time_sec |   Total_Successful_Client_Rounds |   Total_Failed_Client_Rounds | Anomalies         |
|:---------------------|:-------|-------------:|-------------------:|-----------------:|-------------:|------------------------:|---------------------------------:|-----------------------------:|:------------------|
| exp1_baseline        | run_01 |           10 |                  1 |         0.9729   |    0.0741504 |                 535.528 |                               20 |                           10 | GRPC_BRIDGE_ERROR |
| exp2_one_straggler   | run_01 |           10 |                  1 |         0.764706 |    0.47722   |                 244.989 |                                6 |                            3 | GRPC_BRIDGE_ERROR |
| exp3_two_stragglers  | run_01 |           10 |                  1 |         0        |    0         |                   0     |                                0 |                            0 | GRPC_BRIDGE_ERROR |
| exp4_network_dropout | run_01 |           10 |                  1 |         0.97     |    0.0957915 |                 584.959 |                               18 |                            1 | GRPC_BRIDGE_ERROR |
| exp5_recovery        | run_01 |           10 |                  1 |         0.948    |    0.162187  |                 490.486 |                               18 |                            9 | GRPC_BRIDGE_ERROR |
| exp6_adaptive_off    | run_01 |           10 |                  0 |         0.974    |    0.093546  |                1394.71  |                               30 |                            0 |                   |
| exp7_nonstationary   | run_01 |           10 |                  1 |         0.818428 |    0.650221  |                   0     |                                0 |                            0 | GRPC_BRIDGE_ERROR |

## Findings & Anomalies
- **Experiment 5 (Recovery) Finding:** As documented during validation, a reconnected Flower client is assigned a new `cid` by the internal connection handler. The `AdaptiveDropoutDecisionEngine` natively relies on this `cid`, meaning the client's previous adaptive history is not automatically restored. This behavior reflects a limitation of the current mechanism's tight coupling with Flower's lifecycle management.
- **gRPC Bridge Errors:** Some runs experienced 'This should not happen' errors in Flower's grpc_bridge, indicating race conditions or unsupported connection states in Flower's proxy management.

*(Report generated automatically based on raw results.)*
