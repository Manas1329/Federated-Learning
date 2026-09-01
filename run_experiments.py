import os
import subprocess
import time
import json
import shutil
import platform
from pathlib import Path

# Paths
PROJECT_ROOT = Path("d:/Codes/College_Projects/Major Project/Federated-Learning")
SRC_DIR = PROJECT_ROOT / "federated_healthcare" / "src"
RESULTS_BASE = PROJECT_ROOT / "results" / "experiments"

def run_experiment(
    experiment_name: str,
    run_number: int,
    num_rounds: int = 10,
    adaptive_dropout_enabled: bool = True,
    dropout_hard_deadline: float = 60.0,
    round_timeout: float = 300.0,
    artificial_delays: dict = None,
    network_dropout_configuration: dict = None,
    nonstationary_delays: dict = None
):
    run_dir = RESULTS_BASE / experiment_name / f"run_{run_number:02d}"
    
    if run_dir.exists():
        raise FileExistsError(f"Run directory {run_dir} already exists. Never overwrite!")
        
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
        "DROPOUT_HARD_DEADLINE": dropout_hard_deadline,
        "ROUND_TIMEOUT": round_timeout,
        "MIN_CLIENTS": 2,
        "artificial_delays": artificial_delays or {},
        "network_dropout_configuration": network_dropout_configuration or {},
        "nonstationary_delays": nonstationary_delays or {},
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
    env["DROPOUT_HARD_DEADLINE"] = str(config["DROPOUT_HARD_DEADLINE"])
    env["ROUND_TIMEOUT"] = str(config["ROUND_TIMEOUT"])
    env["TARGET_CLIENTS"] = "3"
    env["NUM_ROUNDS"] = str(num_rounds)
    env["EXPERIMENT_RESULTS_DIR"] = str(run_dir)
    
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
        timeout = 3600
        
        while server_proc.poll() is None:
            if time.time() - start_time > timeout:
                log_f.write("\n[TIMEOUT] Experiment timed out. Killing all processes.\n")
                server_proc.kill()
                for c in clients.values(): c.kill()
                break
                
            # For network dropout simulations, check config
            if network_dropout_configuration:
                for cid, drop_round in network_dropout_configuration.items():
                    # For simplicity without parsing live logs, we drop the client after N seconds
                    # Round time is roughly 60 seconds
                    drop_time = drop_round * 60
                    if (time.time() - start_time) > drop_time and clients[cid].poll() is None:
                        log_f.write(f"\n[NETWORK DROPOUT SIMULATION] Terminating {cid} at {time.time() - start_time:.1f}s\n")
                        clients[cid].kill()
                        # Simulate reconnect?
                        if config.get("reconnect_configuration") and cid in config["reconnect_configuration"]:
                            reconnect_delay = config["reconnect_configuration"][cid]
                            time.sleep(reconnect_delay)
                            log_f.write(f"\n[RECOVERY SIMULATION] Restarting {cid}\n")
                            c_env = env.copy()
                            clients[cid] = subprocess.Popen(
                                ["python", "demo_client.py", "--server_ip", "127.0.0.1", "--client_id", cid],
                                cwd=str(SRC_DIR),
                                env=c_env,
                                stdout=log_f,
                                stderr=subprocess.STDOUT,
                                text=True
                            )
                        # Ensure we don't drop again
                        network_dropout_configuration[cid] = 9999
            
            time.sleep(1)
            
        for c in clients.values():
            if c.poll() is None:
                c.kill()
                
        log_f.write("\n=== Experiment Completed ===\n")

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
        "exp2_one_straggler", "exp3_two_stragglers", "exp4_network_dropout",
        "exp5_recovery", "exp6_adaptive_off", "exp7_nonstationary"
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
            elif exp == "exp4_network_dropout":
                run_experiment("exp4_network_dropout", i, num_rounds=10, network_dropout_configuration={"Hospital_B": 4})
            elif exp == "exp5_recovery":
                run_experiment("exp5_recovery", i, num_rounds=10, network_dropout_configuration={"Hospital_B": 4}) # wait I need to pass reconnect config
            elif exp == "exp6_adaptive_off":
                run_experiment("exp6_adaptive_off", i, num_rounds=10, adaptive_dropout_enabled=False, artificial_delays={"Hospital_B": 70})
            elif exp == "exp7_nonstationary":
                run_experiment("exp7_nonstationary", i, num_rounds=10, nonstationary_delays={"Hospital_C": {3: 70, 7: 70}})


