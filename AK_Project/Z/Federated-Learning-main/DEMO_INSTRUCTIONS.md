# Federated Learning Live Multi-Laptop Demonstration
## Complete Setup & Run Guide

---

## Overview

| Laptop | Role | Command |
|--------|------|---------|
| **Laptop 1** | FL Server | `python demo_server.py` |
| **Laptop 2** | Hospital A | `python demo_client.py --server_ip <IP> --client_id Hospital_A` |
| **Laptop 3** | Hospital B | `python demo_client.py --server_ip <IP> --client_id Hospital_B` |
| **Laptop 4** | Hospital C | `python demo_client.py --server_ip <IP> --client_id Hospital_C` |

All laptops must be on the **same Wi-Fi / LAN network**.

---

## Pre-Requisites (One-time Setup on Each Laptop)

### 1. Install Python & Dependencies
Each laptop must have the project environment installed:
```
pip install flwr torch torchvision scikit-learn pandas opacus
```
Or activate the existing virtual environment in the `fl/` folder:
- Windows: `fl\Scripts\activate`

### 2. Copy Data to Client Laptops
Each client laptop needs the correct hospital data folder:

| Laptop | Data folder to copy |
|--------|---------------------|
| Hospital A (Laptop 2) | `data\hospital_A\` |
| Hospital B (Laptop 3) | `data\hospital_B\` |
| Hospital C (Laptop 4) | `data\hospital_c\` |

Copy the entire project folder to each laptop, or at minimum copy:
- The `src\` folder (all .py files)
- The `.env` file
- The relevant `data\hospital_X\` folder

---

## Step 1: Find Server Laptop IP Address

On **Laptop 1 (Server)**, open PowerShell or CMD and run:
```
ipconfig
```
Look for **IPv4 Address** under your Wi-Fi adapter. It will look like:
```
IPv4 Address. . . . . . . . . . . : 192.168.1.10
```
Note this IP — you will need it for all client laptops.

Alternatively, `demo_server.py` will print the IP automatically when started.

---

## Step 2: Allow Port Through Windows Firewall (Server Laptop Only)

Run this command **once** on the server laptop (as Administrator):
```
netsh advfirewall firewall add rule name="FL Server Port 8080" dir=in action=allow protocol=TCP localport=8080
```

Or via Windows Defender Firewall GUI:
1. Open **Windows Defender Firewall** → **Advanced Settings**
2. Click **Inbound Rules** → **New Rule**
3. Choose **Port** → **TCP** → **Specific local ports: 8080**
4. Allow the connection → Apply to all profiles → Name it "FL Server"

### Test Connectivity (from any client laptop)
```
Test-NetConnection -ComputerName 192.168.1.10 -Port 8080
```
You should see `TcpTestSucceeded : True` (run after server is started).

---

## Step 3: Start the FL Server (Laptop 1)

Open a terminal in the `src\` folder and run:
```
python demo_server.py
```

With custom options:
```
python demo_server.py --num_rounds 5 --target_clients 3
```

**Expected output:**
```
============================================================
   FEDERATED LEARNING SERVER  -  STARTING
============================================================
   Server IP          : 192.168.1.10
   Port               : 8080
   FL Mode            : INT8 Quantization
   Target Clients     : 3
   Minimum Clients    : 2
   FL Rounds          : 5
============================================================

   Clients should connect with:
   python demo_client.py --server_ip 192.168.1.10 --client_id Hospital_X

   Waiting for 3 client(s) to join...
============================================================
```

The server will block and wait for clients to join.

---

## Step 4: Connect Each Client Laptop

### Laptop 2 — Hospital A
```
cd src
python demo_client.py --server_ip 192.168.1.10 --client_id Hospital_A
```

### Laptop 3 — Hospital B
```
cd src
python demo_client.py --server_ip 192.168.1.10 --client_id Hospital_B
```

### Laptop 4 — Hospital C
```
cd src
python demo_client.py --server_ip 192.168.1.10 --client_id Hospital_C
```

If data is in a non-default location, add `--data_path`:
```
python demo_client.py --server_ip 192.168.1.10 --client_id Hospital_C --data_path C:\Projects\data\hospital_c
```

**Expected client output on connection:**
```
============================================================
   FL CLIENT  :  Hospital_A
============================================================
   This laptop IP     : 192.168.1.11
   Server address     : 192.168.1.10:8080
   FL Mode            : INT8 Quantization
   Data path          : C:\...\data\hospital_A
============================================================

   Connecting to FL Server at 192.168.1.10:8080 ...
============================================================

[Hospital_A] Data loaded. Connecting to server...
```

**Expected server output as clients join:**
```
  [SERVER] Client joined (cid=...) -- 1/3 connected
  [SERVER] Client joined (cid=...) -- 2/3 connected
  [SERVER] Client joined (cid=...) -- 3/3 connected

============================================================
  ALL REQUIRED CLIENTS CONNECTED
  Starting Federated Learning...
============================================================
```

---

## Step 5: Federated Learning Rounds

Once all clients connect, training begins automatically.

### Server Terminal (each round):
```
============================================================
Starting Federated Round 1
============================================================

[AdaptiveServer] Round 1: Selected 3 clients.
[AdaptiveServer] Target: 3, Minimum: 2

[AdaptiveServer] Round 1 collection finished.
[AdaptiveServer] Collected 3 successful clients.
[AdaptiveServer] 0 clients failed or dropped out.

============================================================
Round 1 completed
============================================================
Successful Clients: 3
Failed Clients: 0
Dequantization + Aggregation Time: 0.XXXX sec
Total Round Time: XX.XXXX sec

========================================================
  GLOBAL ROUND 1 RESULTS
========================================================
  Clients Evaluated   : 3
  Global Accuracy     : XX.XX%
  Global Loss         : X.XXXX
  F1 Score            : X.XXXX
  Precision           : X.XXXX
  Recall              : X.XXXX
========================================================
```

### Client Terminal (each round):
```
============================================================
[Hospital_A] Federated Round 1
============================================================
[Hospital_A] Epoch 1/2: XX.XX sec
[Hospital_A] Epoch 2/2: XX.XX sec
[Hospital_A] Total Training Time: XX.XX sec
[Hospital_A] FP32 Payload Size: X.XXXX MB
[Hospital_A] INT8 Payload Size: X.XXXX MB
[Hospital_A] Compression Ratio: X.XXx
[Hospital_A] Payload Reduction: XX.XX%

[Hospital_A] Round 1 Evaluation -> Loss: X.XXXX, Accuracy: X.XXXX, F1: X.XXXX, Precision: X.XXXX, Recall: X.XXXX
```

---

## Step 6: Demonstrating Client Dropout

During any training round, you can demonstrate dropout by:

**Option A — Close the terminal** on Hospital C's laptop.

**Option B — Press Ctrl+C** in Hospital C's terminal during training.

**Option C — Disconnect from Wi-Fi** on Hospital C's laptop.

### Expected Server Output (dropout detected):

```
[AdaptiveServer] Round 3: Selected 3 clients.
[AdaptiveServer] Target: 3, Minimum: 2

[AdaptiveServer] Round 3 collection finished.
[AdaptiveServer] Collected 2 successful clients.
[AdaptiveServer] 1 clients failed or dropped out.
```

Since `min_clients = 2` (= ceil(0.6 x 3) = 2), aggregation continues:
```
============================================================
Round 3 completed
============================================================
Successful Clients: 2
Failed Clients: 1
Aggregation + Dequantization Time: X.XXXX sec

========================================================
  GLOBAL ROUND 3 RESULTS
========================================================
  Clients Evaluated   : 2
  Global Accuracy     : XX.XX%
  ...
========================================================
```

If only 1 client responds (below minimum), the server will print:
```
[AdaptiveServer] ERROR: Minimum clients (2) not reached. Only 1 succeeded.
[AdaptiveServer] Aborting aggregation for this round.
```

---

## Command Reference

### Server Laptop (Laptop 1)
```
python demo_server.py [OPTIONS]

Options:
  --port PORT              Server port (default: 8080)
  --num_rounds N           Number of FL rounds (default: 5)
  --target_clients N       Target clients per round (default: 3)
  --min_clients N          Minimum required clients (default: ceil(0.6 x target))
```

### Client Laptops (Laptops 2, 3, 4)
```
python demo_client.py --server_ip IP [OPTIONS]

Required:
  --server_ip IP           LAN IP of the server laptop

Options:
  --port PORT              Server port (default: 8080)
  --client_id NAME         Hospital identity (default: Hospital_A)
  --data_path PATH         Path to hospital data folder (auto-detected if omitted)
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Client cannot connect | Check firewall (Step 2). Verify server IP and port. |
| "Data path not found" error | Use `--data_path` to specify the correct path |
| Client disconnects mid-round | This is the dropout scenario — server handles it |
| Server not detecting clients | Ensure all laptops are on the same Wi-Fi |
| Import errors | Ensure you are running from the `src/` directory |
| "TcpTestSucceeded: False" | Server firewall port not opened yet, or server not started |

---

## Results & Output Files

All existing CSV files and metrics are saved as normal:

| File | Content |
|------|---------|
| `dashboard/results/<suffix>/metrics_<suffix>.csv` | Per-round accuracy + loss |
| `dashboard/results/<suffix>/round_metrics_<suffix>.csv` | Round timing, client counts |
| `dashboard/results/<suffix>/Hospital_A_<suffix>.csv` | Per-client training records |
| `models/global_model_<suffix>.pth` | Saved global model (final round) |

Run `evaluate.py` after demo to generate confusion matrix, ROC curve, PR curve.

---

## Quick Start Cheat Sheet

**Server laptop:**
```
cd "path\to\federated_healthcare\src"
python demo_server.py --num_rounds 5
```

**Hospital A laptop:**
```
cd "path\to\federated_healthcare\src"
python demo_client.py --server_ip 192.168.1.10 --client_id Hospital_A
```

**Hospital B laptop:**
```
cd "path\to\federated_healthcare\src"
python demo_client.py --server_ip 192.168.1.10 --client_id Hospital_B
```

**Hospital C laptop:**
```
cd "path\to\federated_healthcare\src"
python demo_client.py --server_ip 192.168.1.10 --client_id Hospital_C
```
