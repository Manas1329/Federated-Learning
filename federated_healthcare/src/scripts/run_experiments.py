import os
import subprocess
import time
import json
import shutil
import platform
from pathlib import Path

# Paths
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from federated_healthcare.src.paths import EXPERIMENTS_RESULTS_DIR, SRC_DIR

RESULTS_BASE = EXPERIMENTS_RESULTS_DIR

def run_experiment(
    experiment_name: str,
    run_number: int,
    num_rounds: int = 10,
    adaptive_dropout_enabled: bool = True,
    fixed_deadline_control: bool = False,
    dropout_hard_deadline: float = 60.0,
    round_timeout: float = 300.0,
    artificial_delays: dict = None,
    network_dropout_configuration: dict = None,
    nonstationary_delays: dict = None,
    reconnect_configuration: dict = None
):
    run_dir = RESULTS_BASE / experiment_name / f"run_{run_number:02d}"
    
    while run_dir.exists():
        run_number += 1
        run_dir = RESULTS_BASE / experiment_name / f"run_{run_number:02d}"
        
    run_dir.mkdir(parents=True)
    
    # Write config.json
    config = {
        "experiment_name": experiment_name,
        "run_number": run_number,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "clients": ["Hospital_A", "Hospital_B", "Hospital_C"],
        "num_rounds": num_rounds,
        "local_epochs": 2, # Hardcoded in client demo
        "learning_rate": 0.001, # Hardcoded in client
        "USE_DP": 0,
        "USE_QUANTIZATION": 1,
        "ADAPTIVE_DROPOUT_ENABLED": int(adaptive_dropout_enabled),
        "FIXED_DEADLINE_CONTROL": int(fixed_deadline_control),
        "DROPOUT_HARD_DEADLINE": dropout_hard_deadline,
        "ROUND_TIMEOUT": round_timeout,
        "MIN_CLIENTS": 2,
        "artificial_delays": artificial_delays or {},
        "network_dropout_configuration": network_dropout_configuration or {},
        "nonstationary_delays": nonstationary_delays or {},
        "reconnect_configuration": reconnect_configuration or {},
        "random_seed": 42,
        "device": "cpu",
        "operating_system": platform.system()
    }
    
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)
        
    env = os.environ.copy()
    env["USE_DP"] = str(config["USE_DP"])
    env["USE_QUANTIZATION"] = str(config["USE_QUANTIZATION"])
    env["ADAPTIVE_DROPOUT_ENABLED"] = str(config["ADAPTIVE_DROPOUT_ENABLED"])
    env["FIXED_DEADLINE_CONTROL"] = str(config["FIXED_DEADLINE_CONTROL"])
    env["DROPOUT_HARD_DEADLINE"] = str(config["DROPOUT_HARD_DEADLINE"])
    env["ROUND_TIMEOUT"] = str(config["ROUND_TIMEOUT"])
    
    # Thread limit environment variables to prevent CPU oversubscription
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    env["OPENBLAS_NUM_THREADS"] = "2"
    env["VECLIB_MAXIMUM_THREADS"] = "2"
    env["NUMEXPR_NUM_THREADS"] = "2"
    env["TARGET_CLIENTS"] = "3"
    env["NUM_ROUNDS"] = str(num_rounds)
    env["EXPERIMENT_RESULTS_DIR"] = str(run_dir)
    
    try:
        with open(run_dir / "execution.log", "w") as log_f:
            log_f.write(f"=== Starting Experiment: {experiment_name} Run {run_number} ===\n\n")
            
            # Start Server
            log_f.write("Starting Server...\n")
            server_proc = subprocess.Popen(
                ["python", "demo_server.py", "--num_rounds", str(num_rounds)],
                cwd=str(SRC_DIR),
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True
            )
            time.sleep(3) # Wait for server to bind

            clients = {}
            for cid in config["clients"]:
                c_env = env.copy()
                if artificial_delays and cid in artificial_delays:
                    c_env["ARTIFICIAL_DELAY_SEC"] = str(artificial_delays[cid])
                if nonstationary_delays and cid in nonstationary_delays:
                    # Expect dict like {1: 10, 2: 50}
                    ns_str = ",".join([f"{k}:{v}" for k,v in nonstationary_delays[cid].items()])
                    c_env["NONSTATIONARY_DELAYS"] = ns_str
                if network_dropout_configuration and cid in network_dropout_configuration:
                    c_env["NETWORK_DROPOUT_ROUND"] = str(network_dropout_configuration[cid])
                    
                c_proc = subprocess.Popen(
                    ["python", "demo_client.py", "--server_ip", "127.0.0.1", "--client_id", cid],
                    cwd=str(SRC_DIR),
                    env=c_env,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                clients[cid] = c_proc

            start_time = time.time()
            
            # Fetch timeout from environment or use default 600
            try:
                timeout = int(os.environ.get("EXPERIMENT_TIMEOUT_SEC", 600))
            except ValueError:
                timeout = 600
            
            # Local copy to safely mutate during iteration
            active_network_dropouts = network_dropout_configuration.copy() if network_dropout_configuration else {}

            while server_proc.poll() is None:
                if time.time() - start_time > timeout:
                    log_f.write("\n[TIMEOUT] Experiment timed out. Killing all processes.\n")
                    break
                    
                # For network dropout simulations, detect process death
                for cid in list(active_network_dropouts.keys()):
                    if clients[cid].poll() is not None:
                        log_f.write(f"\n[NETWORK DROPOUT SIMULATION] Detected termination of {cid} at {time.time() - start_time:.1f}s\n")
                        log_f.flush()
                        # Simulate reconnect?
                        if reconnect_configuration and cid in reconnect_configuration:
                            reconnect_delay = reconnect_configuration[cid]
                            time.sleep(reconnect_delay)
                            log_f.write(f"\n[RECOVERY SIMULATION] Restarting {cid}\n")
                            log_f.flush()
                            c_env = env.copy()
                            # Do not set NETWORK_DROPOUT_ROUND again so it doesn't die again
                            if artificial_delays and cid in artificial_delays:
                                c_env["ARTIFICIAL_DELAY_SEC"] = str(artificial_delays[cid])
                            if nonstationary_delays and cid in nonstationary_delays:
                                ns_str = ",".join([f"{k}:{v}" for k,v in nonstationary_delays[cid].items()])
                                c_env["NONSTATIONARY_DELAYS"] = ns_str
                                
                            clients[cid] = subprocess.Popen(
                                ["python", "demo_client.py", "--server_ip", "127.0.0.1", "--client_id", cid],
                                cwd=str(SRC_DIR),
                                env=c_env,
                                stdout=log_f,
                                stderr=subprocess.STDOUT,
                                text=True
                            )
                        # Remove from tracking so we don't process it again
                        del active_network_dropouts[cid]
                
                time.sleep(1)
                
            log_f.write("\n=== Experiment Completed ===\n")
    finally:
        # Cleanup
        if 'server_proc' in locals() and server_proc.poll() is None:
            server_proc.kill()
        if 'clients' in locals():
            for c in clients.values():
                if c.poll() is None:
                    c.kill()

    suffix_dir = run_dir / "b_quantized"
    if suffix_dir.exists():
        for f in suffix_dir.glob("*.csv"):
            new_name = f.name.replace("_b_quantized", "").replace("_a_pure", "")
            shutil.move(str(f), str(run_dir / new_name))
        suffix_dir.rmdir()
        
    print(f"Finished {experiment_name} run {run_number}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, help="Experiment to run (e.g. exp1_baseline, all)")
    parser.add_argument("--runs", type=int, default=1, help="Number of repetitions")
    args = parser.parse_args()

    experiments_to_run = [args.exp] if args.exp != "all" else [
        "exp1_baseline", "exp2_one_straggler", "exp3_two_stragglers", "exp4_network_dropout",
        "exp5_recovery", "exp6_adaptive_off", "exp7_nonstationary", "exp8_fixed_deadline"
    ]

    for exp in experiments_to_run:
        for i in range(1, args.runs + 1):
            print(f"Running {exp} Run {i}")
            if exp == "exp1_baseline":
                run_experiment("exp1_baseline", i, num_rounds=10)
            elif exp == "exp2_one_straggler":
                run_experiment("exp2_one_straggler", i, num_rounds=10, artificial_delays={"Hospital_B": 70}) # > 60s deadline
            elif exp == "exp3_two_stragglers":
                run_experiment("exp3_two_stragglers", i, num_rounds=10, artificial_delays={"Hospital_B": 70, "Hospital_C": 75})
            elif exp == "exp4_val":
                run_experiment("exp4_val", i, num_rounds=3, network_dropout_configuration={"Hospital_C": 2})
            elif exp == "exp5_val":
                run_experiment("exp5_val", i, num_rounds=4, network_dropout_configuration={"Hospital_C": 2}, reconnect_configuration={"Hospital_C": 5})
            elif exp == "exp4_network_dropout":
                run_experiment("exp4_network_dropout", i, num_rounds=10, network_dropout_configuration={"Hospital_C": 2})
            elif exp == "exp5_recovery":
                run_experiment("exp5_recovery", i, num_rounds=10, network_dropout_configuration={"Hospital_C": 2}, reconnect_configuration={"Hospital_C": 5})
            elif exp == "exp6_adaptive_off":
                run_experiment("exp6_adaptive_off", i, num_rounds=10, adaptive_dropout_enabled=False, artificial_delays={"Hospital_B": 70})
            elif exp == "exp7_nonstationary":
                run_experiment("exp7_nonstationary", i, num_rounds=10, nonstationary_delays={"Hospital_C": {3: 70, 7: 70}})
            elif exp == "exp8_fixed_deadline":
                run_experiment("exp8_fixed_deadline", i, num_rounds=10, adaptive_dropout_enabled=False, fixed_deadline_control=True, artificial_delays={"Hospital_B": 70})


