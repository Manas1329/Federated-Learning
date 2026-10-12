import os, sys, subprocess, time, threading
from pathlib import Path
import torch

PROJECT_ROOT = Path(r"c:\Users\Public\Documents\Major Project\Federated-Learning")
SRC_DIR = PROJECT_ROOT / "federated_healthcare" / "src"
MODELS_DIR = PROJECT_ROOT / "federated_healthcare" / "models"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))


def banner(msg):
    print("\n" + "=" * 70)
    print("  " + msg)
    print("=" * 70, flush=True)


stop_event = threading.Event()
_procs = {}


def _stream(pipe, prefix):
    try:
        for raw in iter(pipe.readline, ""):
            if stop_event.is_set():
                break
            line = raw.rstrip()
            if line:
                print("[" + prefix + "] " + line, flush=True)
    except Exception:
        pass
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def launch(label, cmd, env, cwd):
    p = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env, cwd=str(cwd), text=True, bufsize=1,
    )
    _procs[label] = p
    threading.Thread(target=_stream, args=(p.stdout, label), daemon=True).start()
    threading.Thread(target=_stream, args=(p.stderr, label), daemon=True).start()
    return p


def kill_all():
    stop_event.set()
    for _, p in list(_procs.items()):
        if p.poll() is None:
            try:
                p.terminate()
            except Exception:
                pass
    time.sleep(2)
    for _, p in list(_procs.items()):
        if p.poll() is None:
            try:
                p.kill()
            except Exception:
                pass


def base_env(extra=None):
    env = os.environ.copy()
    pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + str(SRC_DIR) + (os.pathsep + pp if pp else "")
    env["USE_QUANTIZATION"] = "1"
    env["USE_DP"] = "0"
    env["FORCE_CPU"] = "0"
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    if extra:
        env.update({str(k): str(v) for k, v in extra.items()})
    return env


def run_fl(phase, exp_name, rounds, target, hospitals, port, init_model=None, wait=6.0):
    banner(phase + ": exp=" + exp_name + ", rounds=" + str(rounds) + ", clients=" + str(hospitals))
    extra = {
        "EXPERIMENT_NAME": exp_name,
        "NUM_ROUNDS": str(rounds),
        "TARGET_CLIENTS": str(target),
        "FL_PORT": str(port),
        "FL_SERVER_BIND": "0.0.0.0",
        "KNOWN_CLIENTS": ",".join(hospitals),
    }
    if init_model:
        extra["INIT_MODEL_PATH"] = str(init_model)
        print("  Warm-start from: " + str(init_model))

    srv_env = base_env(dict(list(extra.items()) + [("SERVER_ADDRESS", "0.0.0.0:" + str(port))]))
    srv = launch("SERVER", [sys.executable, str(SRC_DIR / "server.py")], srv_env, SRC_DIR)
    print("  Server PID " + str(srv.pid) + ", waiting " + str(wait) + "s...", flush=True)
    time.sleep(wait)
    if srv.poll() is not None:
        print("  ERROR: Server exited (code " + str(srv.returncode) + ")")
        kill_all()
        return False

    client_procs = {}
    for code in hospitals:
        data_path = PROJECT_ROOT / "federated_healthcare" / "data" / code.lower()
        c_env = base_env(dict(list(extra.items()) + [
            ("CLIENT_NAME", code),
            ("DATA_PATH", str(data_path)),
            ("SERVER_ADDRESS", "127.0.0.1:" + str(port)),
        ]))
        p = launch(code, [sys.executable, str(SRC_DIR / "client.py")], c_env, SRC_DIR)
        client_procs[code] = p
        print("  " + code + " PID " + str(p.pid), flush=True)

    try:
        while True:
            time.sleep(2)
            if srv.poll() is not None:
                rc = srv.returncode
                print("  Server done (code " + str(rc) + ").")
                break
            alive = [c for c, p in client_procs.items() if p.poll() is None]
            if not alive:
                print("  All clients done. Finalizing...")
                try:
                    srv.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    print("  Server timeout — terminating.")
                break
    except KeyboardInterrupt:
        print("  Interrupted.")
        kill_all()
        return False
    finally:
        kill_all()
        stop_event.clear()
        _procs.clear()

    out_path = MODELS_DIR / ("global_model_" + exp_name + ".pth")
    if out_path.exists():
        print("  Model saved: " + str(out_path))
        return True
    print("  ERROR: Model not found at " + str(out_path))
    return False


# =========================================================
# PHASE 1: Train A+B+C for 10 rounds
# =========================================================
ok = run_fl(
    "PHASE 1", "abc_base", 10, 3,
    ["Hospital_A", "Hospital_B", "Hospital_C"], 8082,
    wait=20.0,
)
if not ok:
    print("PHASE 1 FAILED — aborting pipeline.")
    sys.exit(1)

ABC_MODEL = MODELS_DIR / "global_model_abc_base.pth"
banner("Phase 1 complete. Model: " + str(ABC_MODEL))

# =========================================================
# PHASE 2: Unlearn Hospital B via gradient ascent
# =========================================================
banner("PHASE 2 - Gradient-Ascent Unlearning of Hospital B")

from unlearn_baseline import execute_baseline_unlearning
from client import load_partitions
import numpy as np
from sklearn.metrics import accuracy_score

print("Loading Hospital B training data...")
trainloader_b, testloader_b, _ = load_partitions(client_id="Hospital_B")

print("Running gradient-ascent unlearning (lr=1e-4, epochs=2)...")
unlearned_model = execute_baseline_unlearning(
    global_model_path=str(ABC_MODEL),
    target_loader=trainloader_b,
    target_client="Hospital_B",
    surviving_loaders=[],
    lr=1e-4,
    unlearn_epochs=2,
    finetune_epochs=0,
)

UNLEARNED_PATH = MODELS_DIR / "unlearned_Hospital_B_abc_base.pth"
torch.save(unlearned_model.state_dict(), str(UNLEARNED_PATH))
print("Unlearned model saved: " + str(UNLEARNED_PATH))

# Quick accuracy check on Hospital B (expect degraded)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unlearned_model.eval().to(device)
preds, labs = [], []
with torch.no_grad():
    for imgs, lbls in testloader_b:
        out = unlearned_model(imgs.to(device))
        _, p = torch.max(out, 1)
        preds.extend(p.cpu().numpy())
        labs.extend(lbls.numpy())
acc_b = accuracy_score(labs, preds) * 100
print("Unlearned model acc on Hospital B test set: " + str(round(acc_b, 2)) + "% (expect degraded)")

# =========================================================
# PHASE 3: Retrain A+C for 10 rounds warm-started from unlearned model
# =========================================================
ok = run_fl(
    "PHASE 3", "post_unlearn_ac", 10, 2,
    ["Hospital_A", "Hospital_C"], 8083,
    init_model=UNLEARNED_PATH,
    wait=15.0,
)
if not ok:
    print("PHASE 3 FAILED.")
    sys.exit(1)

banner("PIPELINE COMPLETE")
POST_MODEL = MODELS_DIR / "global_model_post_unlearn_ac.pth"
print("  global_model_abc_base.pth          -> Phase 1: A+B+C base model (10 rounds)")
print("  unlearned_Hospital_B_abc_base.pth  -> Phase 2: After unlearning Hospital B")
print("  global_model_post_unlearn_ac.pth   -> Phase 3: Post-unlearn A+C retrain (10 rounds)")
print("=" * 70)
