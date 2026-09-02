import os
import glob
import pandas as pd
import json
from pathlib import Path

RESULTS_BASE = Path("results/experiments")
SUMMARY_CSV = Path("results/experiments/experiment_summary.csv")
SUMMARY_MD = Path("results/experiments/experiment_summary.md")

def aggregate_results():
    all_runs = []
    
    # Iterate over all experiment directories
    if not RESULTS_BASE.exists():
        print(f"Results base {RESULTS_BASE} does not exist.")
        return
        
    for exp_dir in RESULTS_BASE.iterdir():
        if not exp_dir.is_dir(): continue
        exp_name = exp_dir.name
        
        for run_dir in exp_dir.iterdir():
            if not run_dir.is_dir() or not run_dir.name.startswith("run_"): continue
            run_num = run_dir.name
            
            # Read Config
            config_file = run_dir / "config.json"
            config = {}
            if config_file.exists():
                with open(config_file, "r") as f:
                    config = json.load(f)
                    
            # Read Server Metrics (Global Accuracy/Loss)
            metrics_file = run_dir / "metrics.csv"
            final_accuracy = 0.0
            final_loss = 0.0
            if metrics_file.exists():
                try:
                    metrics_df = pd.read_csv(metrics_file)
                    if not metrics_df.empty:
                        # Get the last round's metrics
                        last_row = metrics_df.iloc[-1]
                        final_accuracy = last_row.get("Accuracy", 0.0)
                        final_loss = last_row.get("Loss", 0.0)
                except Exception as e:
                    print(f"Error reading {metrics_file}: {e}")

            # Read Server Round Metrics (Timing)
            round_metrics_file = run_dir / "round_metrics.csv"
            total_server_time = 0.0
            total_successful_clients = 0
            total_failed_clients = 0
            if round_metrics_file.exists():
                try:
                    rm_df = pd.read_csv(round_metrics_file)
                    if not rm_df.empty:
                        total_server_time = rm_df["Total_Round_Time_sec"].sum()
                        total_successful_clients = rm_df["Successful_Clients"].sum()
                        total_failed_clients = rm_df["Failed_Clients"].sum()
                except Exception as e:
                    print(f"Error reading {round_metrics_file}: {e}")

            # Check execution log for anomalies
            log_file = run_dir / "execution.log"
            anomalies = []
            if log_file.exists():
                with open(log_file, "r") as f:
                    log_text = f.read()
                    if "TIMEOUT" in log_text:
                        anomalies.append("TIMEOUT")
                    if "This should not happen" in log_text:
                        anomalies.append("GRPC_BRIDGE_ERROR")
            
            run_record = {
                "Experiment": exp_name,
                "Run": run_num,
                "Num_Rounds": config.get("num_rounds", 0),
                "Adaptive_Dropout": config.get("ADAPTIVE_DROPOUT_ENABLED", 1),
                "Final_Accuracy": final_accuracy,
                "Final_Loss": final_loss,
                "Total_Server_Time_sec": total_server_time,
                "Total_Successful_Client_Rounds": total_successful_clients,
                "Total_Failed_Client_Rounds": total_failed_clients,
                "Anomalies": ", ".join(anomalies)
            }
            all_runs.append(run_record)
            
    if not all_runs:
        print("No run data found.")
        return
        
    summary_df = pd.DataFrame(all_runs)
    # Sort by Experiment and Run
    summary_df = summary_df.sort_values(by=["Experiment", "Run"])
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"Aggregated {len(summary_df)} runs into {SUMMARY_CSV}")
    
    # Generate Markdown Report
    with open(SUMMARY_MD, "w") as f:
        f.write("# Federated Learning Experiments Summary\n\n")
        f.write("## Overview\n")
        f.write(f"Total experiments executed: {len(summary_df)}\n\n")
        
        f.write("## Aggregate Results\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## Findings & Anomalies\n")
        f.write("- **Experiment 5 (Recovery) Finding:** As documented during validation, a reconnected Flower client is assigned a new `cid` by the internal connection handler. The `AdaptiveDropoutDecisionEngine` natively relies on this `cid`, meaning the client's previous adaptive history is not automatically restored. This behavior reflects a limitation of the current mechanism's tight coupling with Flower's lifecycle management.\n")
        
        # Check for specific anomalies
        has_grpc_errors = summary_df["Anomalies"].str.contains("GRPC_BRIDGE_ERROR").any()
        if has_grpc_errors:
            f.write("- **gRPC Bridge Errors:** Some runs experienced 'This should not happen' errors in Flower's grpc_bridge, indicating race conditions or unsupported connection states in Flower's proxy management.\n")
            
        has_timeouts = summary_df["Anomalies"].str.contains("TIMEOUT").any()
        if has_timeouts:
            f.write("- **Timeouts:** Some runs hit the hard orchestration timeout.\n")
            
        f.write("\n*(Report generated automatically based on raw results.)*\n")
        
    print(f"Generated markdown summary at {SUMMARY_MD}")

if __name__ == "__main__":
    aggregate_results()
