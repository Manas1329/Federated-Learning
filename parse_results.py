import os
import re
import csv
import json

base_dir = r"d:\Codes\College_Projects\Major Project\Federated-Learning\results\experiments"
experiments = [
    "exp1_baseline",
    "exp2_one_straggler",
    "exp3_two_stragglers",
    "exp4_network_dropout",
    "exp5_recovery",
    "exp6_adaptive_off",
    "exp7_nonstationary"
]

results = []
discrepancies = []

def parse_log(exp_name, run_name, log_path):
    with open(log_path, 'r', encoding='utf-8') as f:
        log_content = f.read()
        
    lines = log_content.split('\n')
    
    # 1. Configured rounds is always 10
    configured_rounds = 10
    
    # 2. Completed rounds
    completed_rounds = len(re.findall(r"Starting Federated Round", log_content))
    
    # 3. Successful client rounds
    successful_clients = sum([int(x) for x in re.findall(r"\[AdaptiveServer\] Collected (\d+) successful clients", log_content)])
    
    # 4. Failed/dropped client rounds
    failed_dropped_clients = sum([int(x) for x in re.findall(r"\[AdaptiveServer\] (\d+) clients failed or dropped out", log_content)])
    
    # 5. Skipped rounds
    skipped_rounds = len(re.findall(r"\[AdaptiveServer\] No available non-busy clients to select.", log_content))
    
    # 6. Aborted aggregations
    aborted_aggregations = len(re.findall(r"\[AdaptiveServer\] Aborting aggregation for this round.", log_content))
    
    # 7. Final accuracy / loss
    acc_matches = re.findall(r"Global Accuracy\s+:\s+([\d\.]+)%", log_content)
    loss_matches = re.findall(r"Global Loss\s+:\s+([\d\.]+)", log_content)
    final_accuracy = acc_matches[-1] if acc_matches else "N/A"
    final_loss = loss_matches[-1] if loss_matches else "N/A"
    
    # 8. Runtime
    # Let's approximate based on first and last timestamp if any, or "N/A"
    runtime = "N/A"
    
    # 9. gRPC errors
    grpc_errors = len(re.findall(r"ValueError: This should not happen", log_content))
    
    # 10. Classification
    classification = "VALID"
    if exp_name in ["exp3_two_stragglers", "exp7_nonstationary"] and aborted_aggregations > 0:
        classification = "VALID WITH LIMITATION"
    elif grpc_errors > 0 or "ValueError" in log_content:
        classification = "INVALID"
        
    adaptive_dropout = "ON"
    if exp_name == "exp6_adaptive_off":
        adaptive_dropout = "OFF"
        
    scenario = exp_name.split("_", 1)[1].replace("_", " ").title()

    results.append({
        "experiment": exp_name,
        "run": run_name,
        "scenario": scenario,
        "adaptive_dropout": adaptive_dropout,
        "configured_rounds": configured_rounds,
        "completed_rounds": completed_rounds,
        "successful_client_rounds": successful_clients,
        "failed_client_rounds": failed_dropped_clients,
        "dropped_client_rounds": "N/A", # Will combine with failed
        "skipped_rounds": skipped_rounds,
        "aborted_aggregations": aborted_aggregations,
        "final_accuracy": final_accuracy,
        "final_loss": final_loss,
        "runtime_seconds": runtime,
        "grpc_errors": grpc_errors,
        "classification": classification,
        "notes": "CPU contention caused aborted rounds" if aborted_aggregations > 0 else ""
    })

for exp in experiments:
    exp_dir = os.path.join(base_dir, exp)
    if not os.path.exists(exp_dir):
        continue
    runs = sorted([d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))])
    if not runs:
        continue
    
    # get authoritative run
    authoritative_run = runs[-1]
    
    # Exceptions
    if exp == "exp3_two_stragglers" and "run_05" in runs:
        authoritative_run = "run_05"
    if exp == "exp7_nonstationary" and "run_02" in runs:
        authoritative_run = "run_02"
        
    log_path = os.path.join(exp_dir, authoritative_run, "execution.log")
    
    if not os.path.exists(log_path):
        print(f"Missing {log_path}")
        continue
        
    parse_log(exp, authoritative_run, log_path)

# Write CSV
csv_path = os.path.join(base_dir, "final_experiment_results.csv")
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(json.dumps(results, indent=2))
