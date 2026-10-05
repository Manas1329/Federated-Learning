# Federated Learning — Multi-Device Deployment Guide

This project implements **real distributed federated learning** across 4 physical devices.
Each hospital trains its model locally on its own data and communicates only model
parameters (never raw patient data) to the central server over the network.

---

## Primary Architecture

```
DEVICE 1 (Hospital A)          DEVICE 4 (Central FL Server)
  client.py                            server.py
  local data: hospital_A/   ─gRPC──►  0.0.0.0:8080
                             ◄──────  global model params

DEVICE 2 (Hospital B)
  client.py
  local data: hospital_B/   ─gRPC──►  (same server)
                             ◄──────  global model params

DEVICE 3 (Hospital C)
  client.py
  local data: hospital_C/   ─gRPC──►  (same server)
                             ◄──────  global model params
```

**What is exchanged over the network:**
- Global model parameters → from server to each hospital client (start of round)
- Local model updates (quantized INT8 if enabled) → from each hospital to server
- Evaluation metrics → from each hospital to server

**What stays on each hospital device:**
- Raw chest X-ray images
- Training data paths
- Local optimizer state
- DP privacy budget (if enabled)

---

## Network Requirements

| Requirement | Detail |
|---|---|
| **Network** | All 4 devices must be on the same LAN/Wi-Fi, or reachable by IP |
| **Firewall** | Port 8080 (TCP) must be open **inbound** on the server device |
| **Protocol** | Flower uses gRPC over HTTP/2 — no special VPN/tunnel needed on LAN |
| **Bandwidth** | ~2–8 MB per client per round (quantized); ~8–32 MB unquantized |

---

## Step 1 — Find the Server Device's LAN IP

On **Device 4 (Server)**, run:

**Windows:**
```
ipconfig
```
Look for `IPv4 Address` under your Wi-Fi or Ethernet adapter, e.g. `192.168.1.10`.

**Linux/macOS:**
```bash
ip addr show   # or: ifconfig
```

Note this IP — all hospital devices will need it.

---

## Step 2 — Open Firewall Port on the Server Device

**Windows (run as Administrator):**
```
netsh advfirewall firewall add rule name="FL Server Port 8080" dir=in action=allow protocol=TCP localport=8080
```

**Linux (ufw):**
```bash
sudo ufw allow 8080/tcp
```

**Linux (firewalld):**
```bash
sudo firewall-cmd --add-port=8080/tcp --permanent
sudo firewall-cmd --reload
```

**Test connectivity** (from any hospital device after server is started):

Windows:
```
Test-NetConnection -ComputerName 192.168.1.10 -Port 8080
```
Expected: `TcpTestSucceeded : True`

Linux/macOS:
```bash
nc -zv 192.168.1.10 8080
```

---

## Step 3 — Install Dependencies on Each Device

Each device needs the project environment:

```bash
# Clone (or copy) the repository on each device
git clone https://github.com/Manas1329/Federated-Learning
cd Federated-Learning

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/macOS)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Step 4 — Copy Hospital Data to Each Client Device

Only the relevant hospital folder is needed on each device:

| Device | Copy this data folder |
|---|---|
| Device 1 (Hospital A) | `data/hospital_A/` |
| Device 2 (Hospital B) | `data/hospital_B/` |
| Device 3 (Hospital C) | `data/hospital_C/` |
| Device 4 (Server) | **No data needed** |

The server **never** accesses patient data.

---

## Step 5 — Configure Each Device

### Device 4 — Central FL Server

Create or edit `.env`:
```ini
# Server binds to all interfaces so remote clients can reach it
FL_SERVER_BIND=0.0.0.0
FL_PORT=8080

# FL parameters
NUM_ROUNDS=10
TARGET_CLIENTS=3
USE_QUANTIZATION=1
USE_DP=0
```

### Device 1 — Hospital A

Create or edit `.env` (or set environment variables directly):
```ini
# Point to the central server's LAN IP
SERVER_ADDRESS=192.168.1.10:8080

# This hospital's identity and local data
CLIENT_NAME=Hospital_A
DATA_PATH=/path/to/data/hospital_A

# Must match server configuration
USE_QUANTIZATION=1
USE_DP=0
```

### Device 2 — Hospital B

```ini
SERVER_ADDRESS=192.168.1.10:8080
CLIENT_NAME=Hospital_B
DATA_PATH=/path/to/data/hospital_B
USE_QUANTIZATION=1
USE_DP=0
```

### Device 3 — Hospital C

```ini
SERVER_ADDRESS=192.168.1.10:8080
CLIENT_NAME=Hospital_C
DATA_PATH=/path/to/data/hospital_C
USE_QUANTIZATION=1
USE_DP=0
```

> Replace `192.168.1.10` with the actual LAN IP of your server device.

---

## Step 6 — Start the System

### Start server FIRST (Device 4)

```bash
# Windows
cd "path\to\Federated-Learning"
venv\Scripts\activate
python federated_healthcare\src\server.py
```

```bash
# Linux/macOS
cd /path/to/Federated-Learning
source venv/bin/activate
python federated_healthcare/src/server.py
```

Expected output:
```
Starting Adaptive Flower Server with Dropout Handling...
  Binding on          : 0.0.0.0:8080
  Target clients      : 3
  Minimum clients     : 2
  FL rounds           : 10
  Round timeout       : 300.0 sec
  ...
  Remote hospital clients should connect with:
    SERVER_ADDRESS=<this-machine-LAN-IP>:8080
    python federated_healthcare/src/client.py
```

The server blocks and waits for clients to connect.

---

### Start each hospital client (Devices 1, 2, 3) — independently and simultaneously

**Device 1 — Hospital A:**
```bash
cd "path\to\Federated-Learning"
venv\Scripts\activate
# Windows
set SERVER_ADDRESS=192.168.1.10:8080
set CLIENT_NAME=Hospital_A
set DATA_PATH=C:\path\to\data\hospital_A
python federated_healthcare\src\client.py
```

```bash
# Linux/macOS (or if configured via .env)
export SERVER_ADDRESS=192.168.1.10:8080
export CLIENT_NAME=Hospital_A
export DATA_PATH=/path/to/data/hospital_A
python federated_healthcare/src/client.py
```

**Device 2 — Hospital B:**
```bash
set SERVER_ADDRESS=192.168.1.10:8080
set CLIENT_NAME=Hospital_B
set DATA_PATH=C:\path\to\data\hospital_B
python federated_healthcare\src\client.py
```

**Device 3 — Hospital C:**
```bash
set SERVER_ADDRESS=192.168.1.10:8080
set CLIENT_NAME=Hospital_C
set DATA_PATH=C:\path\to\data\hospital_C
python federated_healthcare\src\client.py
```

> The three hospital clients start independently and connect to the central server concurrently.
> There is no required startup order between Hospital A, B, and C.
> The server will begin a round only when the required number of clients have connected.

---

## Alternative: Using demo_server.py / demo_client.py

The project also includes `demo_server.py` and `demo_client.py` which wrap the same
underlying logic with additional CLI argument parsing:

**Device 4 — Server:**
```bash
cd federated_healthcare/src
python demo_server.py --num_rounds 10 --target_clients 3
```

**Device 1 — Hospital A:**
```bash
cd federated_healthcare/src
python demo_client.py --server_ip 192.168.1.10 --client_id Hospital_A
```

**Device 2 — Hospital B:**
```bash
python demo_client.py --server_ip 192.168.1.10 --client_id Hospital_B
```

**Device 3 — Hospital C:**
```bash
python demo_client.py --server_ip 192.168.1.10 --client_id Hospital_C
```

---

## How Concurrent Client Communication Works

Flower uses gRPC over HTTP/2, which natively multiplexes multiple concurrent
connections over the same TCP stream. The server does not need any special
threading configuration to handle multiple simultaneous client connections.

**Round execution flow:**
```
1. Server sends FitIns (global model) to all selected clients simultaneously
                │
        ┌───────┼───────┐
        │       │       │
   Hospital A  B       C
   (train)  (train)  (train)
        │       │       │       ← Independent local training on each device
        └───────┼───────┘
                │
2. Server collects FitRes (model updates) as they arrive
3. Once quorum is reached (min_clients), FedAvg aggregation runs
4. New global model is broadcast to all clients for the next round
```

**Threading in this project:**
- `dropout_handler.py` uses `threading.Lock()` on the server side to protect the
  "busy clients" set — preventing the same client proxy from being submitted
  to two operations at once. This is server-side state management, not training.
- `webapp/core/process_manager.py` uses daemon threads to stream subprocess
  stdout/stderr to the dashboard without blocking the main process.
- PyTorch training on hospital devices is single-threaded per process.
- No Python threads are used around the neural network training loop.

---

## FL Round Log Reference

### Server output (per round):
```
============================================================
Starting Federated Round 1
============================================================

[AdaptiveServer] Round 1: Selected 3 clients.

[AdaptiveServer] Round 1 collection finished.
[AdaptiveServer] Collected 3 successful clients.
[AdaptiveServer] 0 clients failed or dropped out.

Round 1 completed
Successful Clients: 3
Failed Clients: 0
Aggregation Time: 0.0012 sec
Total Round Time: 45.22 sec

  GLOBAL ROUND 1 RESULTS
  Global Accuracy: 82.45%
  Global Loss: 0.4231
  F1 Score: 0.8012
```

### Hospital client output (per round):
```
[Hospital_A] Federated Round 1
[Hospital_A] Epoch 1/2: 12.34 sec
[Hospital_A] Epoch 2/2: 11.87 sec
[Hospital_A] Total Training Time: 24.21 sec
[Hospital_A] FP32 Payload Size: 8.2034 MB
[Hospital_A] INT8 Payload Size: 2.1008 MB
[Hospital_A] Compression Ratio: 3.90x
[Hospital_A] Payload Reduction: 74.39%
[Hospital_A] Round 1 Evaluation -> Loss: 0.4123, Accuracy: 0.8312, F1: 0.8011
```

---

## Client Dropout During a Round

If a hospital client disconnects mid-round, the AdaptiveServer handles it:

```
[AdaptiveServer] Round 3: Selected 3 clients.
[AdaptiveServer] Round 3 collection finished.
[AdaptiveServer] Collected 2 successful clients.  ← Hospital C dropped
[AdaptiveServer] 1 clients failed or dropped out.
```

Since `min_clients = ceil(0.6 × 3) = 2`, the server aggregates with 2 clients.
If fewer than `min_clients` respond, the server aborts aggregation for that round.

---

## DP-SGD Mode (Differential Privacy)

To enable DP, set the same variables on both server and all client devices:

```ini
USE_DP=1
DP_NOISE_MULTIPLIER=1.0
DP_MAX_GRAD_NORM=1.0
DP_DELTA=0.00001

# DP training is much slower — increase timeouts
DROPOUT_HARD_DEADLINE=180
ROUND_TIMEOUT=1800
```

DP-SGD training runs only on each hospital device. The server only receives
the resulting model update — it is unaware of the DP mechanism.

---

## Results & Output Files

Results are saved on the **server device** (or wherever `server.py` runs):

| File | Location | Content |
|---|---|---|
| Global model | `federated_healthcare/models/global_model_<suffix>.pth` | Final aggregated model |
| Round metrics | `federated_healthcare/dashboard/results/<suffix>/round_metrics_<suffix>.csv` | Per-round timing, client counts |
| Accuracy/Loss | `federated_healthcare/dashboard/results/<suffix>/metrics_<suffix>.csv` | Per-round accuracy, loss, F1 |

Per-client training metrics are saved on **each hospital device**:

| File | Content |
|---|---|
| `.../<suffix>/Hospital_A_<suffix>.csv` | Training time, payload size, wall-clock time |

---

## Troubleshooting

| Problem | Cause | Solution |
|---|---|---|
| Client cannot connect | Firewall blocking port | Open TCP 8080 inbound on server device (Step 2) |
| `Connection refused` | Server not started yet | Start `server.py` before starting any client |
| `localhost:8080` fails from client device | Wrong `SERVER_ADDRESS` | Set `SERVER_ADDRESS=<server-LAN-IP>:8080` |
| `Data path not found` | Wrong `DATA_PATH` | Set `DATA_PATH` to the local path on that device |
| Clients not joining | Wrong LAN/Wi-Fi | Verify all devices are on the same network |
| Timeout exceeded | DP training too slow | Set `DROPOUT_HARD_DEADLINE=180`, `ROUND_TIMEOUT=1800` |
| Only 1 client connects | Others failed | Check each client device separately |

---

## Local Development Simulation (Single Machine Only)

If you want to test the full system on **one machine only** (without 4 physical devices),
use the local simulation launcher:

```bash
python run_local.py
python run_local.py --force_cpu 1   # recommended to avoid GPU memory contention
```

> **This is NOT the primary deployment architecture.**
> It is a development convenience tool only.
> In production, each hospital device runs its own `client.py` independently.

---

## Quick Reference

| Device | Role | Command |
|---|---|---|
| **Device 4 (Server)** | FL Server | `python federated_healthcare/src/server.py` |
| **Device 1 (Hospital A)** | FL Client | `SERVER_ADDRESS=<ip>:8080 CLIENT_NAME=Hospital_A DATA_PATH=<local-path> python federated_healthcare/src/client.py` |
| **Device 2 (Hospital B)** | FL Client | `SERVER_ADDRESS=<ip>:8080 CLIENT_NAME=Hospital_B DATA_PATH=<local-path> python federated_healthcare/src/client.py` |
| **Device 3 (Hospital C)** | FL Client | `SERVER_ADDRESS=<ip>:8080 CLIENT_NAME=Hospital_C DATA_PATH=<local-path> python federated_healthcare/src/client.py` |
