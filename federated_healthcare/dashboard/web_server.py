import os
import sys
import time
import subprocess
import threading
import queue
from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Ensure project root and src are in path
DASHBOARD_DIR = Path(__file__).resolve().parent
SRC_DIR = DASHBOARD_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from model import ChestCNN
from paths import MODELS_DIR, PLOTS_DIR, RESULTS_DIR

app = Flask(__name__, template_folder=str(DASHBOARD_DIR / "templates"), static_folder=str(DASHBOARD_DIR))

# Global process tracking
subprocesses = {}
log_history = {"Hospital_A": [], "Hospital_B": [], "Hospital_C": [], "System": []}
log_queues = {"Hospital_A": queue.Queue(), "Hospital_B": queue.Queue(), "Hospital_C": queue.Queue()}

def log_message(client_id, text):
    clean_text = text.strip()
    if clean_text:
        log_history[client_id].append(clean_text)
        if client_id in log_queues:
            log_queues[client_id].put(clean_text)

def read_output(proc, client_id):
    for line in iter(proc.stdout.readline, ""):
        log_message(client_id, line)
    proc.stdout.close()
    proc.wait()
    log_message(client_id, f"--- {client_id} Execution Finished with Exit Code {proc.returncode} ---")
    if client_id in subprocesses:
        del subprocesses[client_id]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/start_training", methods=["POST"])
def start_training():
    data = request.json or {}
    client_id = data.get("client_id")
    server_ip = data.get("server_ip", "127.0.0.1")
    port = data.get("port", "8080")

    if not client_id or client_id not in log_history:
        return jsonify({"status": "error", "message": "Invalid client ID"}), 400

    if client_id in subprocesses:
        return jsonify({"status": "error", "message": f"{client_id} is already running training."}), 400

    env = os.environ.copy()
    # Resolve custom data paths if needed or use defaults
    base_dir = DASHBOARD_DIR.parent.parent
    folder_name = client_id.lower().replace("-", "_")
    data_path = os.path.abspath(os.path.join(base_dir, "data", folder_name))

    env["SERVER_ADDRESS"] = f"{server_ip}:{port}"
    env["CLIENT_NAME"] = client_id
    env["DATA_PATH"] = data_path
    env["USE_QUANTIZATION"] = "1"
    env["USE_DP"] = "0"

    cmd = [sys.executable, "client.py", "--server_ip", server_ip, "--port", str(port), "--client_id", client_id]

    try:
        log_history[client_id] = [] # Clear old history
        log_message(client_id, f"--- Launching {client_id} training subprocess... ---")
        log_message(client_id, f"Command: {' '.join(cmd)}")

        proc = subprocess.Popen(
            cmd,
            cwd=str(SRC_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        subprocesses[client_id] = proc

        t = threading.Thread(target=read_output, args=(proc, client_id), daemon=True)
        t.start()

        return jsonify({"status": "success", "message": f"Started training for {client_id}"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/stop_training", methods=["POST"])
def stop_training():
    data = request.json or {}
    client_id = data.get("client_id")

    if client_id in subprocesses:
        proc = subprocesses[client_id]
        proc.terminate()
        log_message(client_id, f"--- Training manually terminated by user ---")
        return jsonify({"status": "success", "message": f"Terminated {client_id} training."})
    return jsonify({"status": "error", "message": "No active training process found for this client."}), 400

@app.route("/api/logs/<client_id>")
def get_logs(client_id):
    if client_id in log_history:
        return jsonify({"status": "success", "logs": log_history[client_id]})
    return jsonify({"status": "error", "message": "Client not found"}), 404

@app.route("/api/stream_logs/<client_id>")
def stream_logs(client_id):
    if client_id not in log_queues:
        return "Client invalid", 404

    def generate():
        # First send existing logs history
        for line in log_history[client_id]:
            yield f"data: {line}\n\n"

        while True:
            try:
                line = log_queues[client_id].get(timeout=5)
                yield f"data: {line}\n\n"
            except queue.Empty:
                # Heartbeat to keep connection alive
                yield "data: :heartbeat\n\n"
            except Exception:
                break

    return Response(generate(), mimetype="text/event-stream")

@app.route("/api/evaluate", methods=["POST"])
def execute_evaluate():
    try:
        log_message("System", "Running comparison evaluation script...")
        cmd_comp = [sys.executable, "comparison.py"]
        res_comp = subprocess.run(cmd_comp, cwd=str(SRC_DIR), capture_output=True, text=True)

        cmd_eval = [sys.executable, "evaluate.py"]
        env = os.environ.copy()
        env["CLIENT_NAME"] = "Hospital_A"
        res_eval = subprocess.run(cmd_eval, cwd=str(SRC_DIR), env=env, capture_output=True, text=True)

        output = f"--- Comparison Results ---\n{res_comp.stdout}\n{res_comp.stderr}\n\n--- Evaluation Results ---\n{res_eval.stdout}\n{res_eval.stderr}"
        return jsonify({"status": "success", "output": output})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/generate_graphs", methods=["POST"])
def execute_graphs():
    try:
        cmd = [sys.executable, "graph.py"]
        res = subprocess.run(cmd, cwd=str(SRC_DIR), capture_output=True, text=True)
        return jsonify({"status": "success", "output": res.stdout + "\n" + res.stderr})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"status": "error", "message": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "Empty filename"}), 400

    try:
        # Load global model dynamically based on availability
        model_path = None
        for suffix in ["b_quantized", "a_pure", "c_dp"]:
            p = MODELS_DIR / f"global_model_{suffix}.pth"
            if p.exists():
                model_path = p
                break

        if not model_path:
            # Fallback to check if any model file is in models directory
            models = list(MODELS_DIR.glob("*.pth"))
            if models:
                model_path = models[0]

        if not model_path or not model_path.exists():
            return jsonify({"status": "error", "message": "Trained global model file (.pth) not found. Please complete federated learning rounds first."}), 400

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ChestCNN().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # Image pre-processing (Grayscale, 128x128 resize)
        image = Image.open(file.stream)
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]
            prediction_idx = torch.argmax(outputs, dim=1).item()

        classes = ["NORMAL", "PNEUMONIA"]
        result = {
            "prediction": classes[prediction_idx],
            "confidence": float(probabilities[prediction_idx]),
            "probabilities": {
                "NORMAL": float(probabilities[0]),
                "PNEUMONIA": float(probabilities[1])
            },
            "model_used": model_path.name
        }
        return jsonify({"status": "success", "result": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/plots/<path:filename>")
def get_plot_image(filename):
    # Safe serving of plot images
    from flask import send_from_directory
    for folder in [PLOTS_DIR / "b_quantized", PLOTS_DIR / "a_pure", PLOTS_DIR / "comparisons" / "LvF", PLOTS_DIR]:
        if (folder / filename).exists():
            return send_from_directory(str(folder), filename)
    return "Plot not found", 404

if __name__ == "__main__":
    print(f"Starting Hospital Federated Learning Adapter Web Server on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
