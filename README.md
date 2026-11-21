# Clinical Federated Learning with Dynamic INT8 Quantization and DP-SGD

An end-to-end framework implementing medical image classification (Pneumonia detection in Chest X-rays) using Federated Learning (FL). This project utilizes the **Flower FL framework** and simulates three isolated hospital nodes under a non-IID split. 

Key Features:
* **Dynamic INT8 Quantization:** Reduces communication payload size by **75%** while retaining high performance.
* **Differential Privacy (DP-SGD):** Leverages PyTorch **Opacus** to enforce strict mathematical privacy bounds across federated clients without resetting privacy budgets across rounds.
* **Adaptive Dropout Handling:** Server dynamically manages timeouts and grace periods depending on whether computationally heavy DP-SGD is enabled.

---

## 📁 Consolidated Project Layout

The repository is structured to separate code, results, dashboard, and datasets cleanly:

```
Federated-Learning/
├── data/                                 # Federated dataset splits
│   ├── hospital_A/                       # Hospital A split (train/test directories)
│   ├── hospital_B/
│   └── hospital_C/
├── federated_healthcare/
│   ├── dashboard/                        # Web visualization & results
│   │   ├── app.py                        # Streamlit dashboard
│   │   ├── classification_reports/       # Evaluation metric report text files
│   │   │   ├── classification_report_no_quantization.txt
│   │   │   └── classification_report_quantized.txt
│   │   ├── results/                      # Raw log CSV files
│   │   │   ├── metrics_*.csv
│   │   │   ├── round_metrics_*.csv
│   │   │   └── Hospital_*_*.csv
│   │   └── plots/                        # Output plots
│   │       ├── comparison_summary.md     # Comparative markdown table
│   │       ├── payload_reduction.png
│   │       ├── quantized/                # INT8 run plots
│   │       └── no_quantization/          # FP32 run plots
│   ├── models/                           # PyTorch model checkpoints
│   ├── split_data.py                     # Non-IID dataset splitter utility
│   └── src/                              # Source code directory
│       ├── client.py                     # FL client implementation (DP-SGD + INT8)
│       ├── server.py                     # FL server implementation (Adaptive Timeouts)
│       ├── evaluate.py                   # Global evaluation script
│       ├── graph.py                      # Chart & metrics compiler script
│       ├── model.py                      # ChestCNN PyTorch architecture (GPU memory optimized)
│       ├── quantization.py               # INT8 quantization utilities
│       ├── dropout_handler.py            # Adaptive dropout handler logic
│       └── utils.py                      # Data loading & helper utilities
├── .env                                  # Active environment config (ignored by git)
├── .env_example                          # Template for environment configuration
├── requirements.txt                      # Python dependencies (strict pinned versions)
└── Dockerfile                            # Container configuration (optional)
```

---

## 🔧 Environment Configuration (`.env`)

We use a `.env` file to globally toggle training options without manually typing parameters in multiple client/server terminals.

Template: [**`.env_example`**]

```ini
# Enable (1) or disable (0) INT8 Quantization during communication
USE_QUANTIZATION=1

# Enable (1) or disable (0) Differential Privacy (DP-SGD)
USE_DP=0

# DP Hyperparameters (Only active if USE_DP=1)
DP_NOISE_MULTIPLIER=0.5
DP_MAX_GRAD_NORM=1.5
DP_DELTA=1e-5

# Force training on CPU (1) or let it auto-detect GPU (0)
FORCE_CPU=0
```

For the first training run, no global checkpoint exists yet, so start with:

```bat
set TRAINING_MODE=fresh&& set TARGET_CLIENTS=3&& set MIN_CLIENTS=2&& python federated_healthcare\src\server.py
```

Use `TRAINING_MODE=warm_start` only after an approved global checkpoint has
been created, and set `INIT_MODEL_PATH` to that `.pth` file when needed.

To initialize your settings:
```bash
cp .env_example .env
```

---

## 📊 Dataset Splitter (`split_data.py`)

The [**`split_data.py`**] script reads your raw Chest X-ray train dataset directory and distributes the files to simulate a **non-IID (Independent and Identically Distributed)** data environment representing realistic clinical distributions:

* **Hospital A**: 1,000 NORMAL, 250 PNEUMONIA
* **Hospital B**: 250 NORMAL, 1,000 PNEUMONIA
* **Hospital C**: 500 NORMAL, 500 PNEUMONIA

### Usage
1. Configure `RAW_DATA_DIR` in `split_data.py` to point to your raw dataset path.
2. Run the script:
   ```bash
   python federated_healthcare/split_data.py
   ```

---

## 🚀 Setup & Execution

### 1. Installation
Ensure you have Python 3.11 installed. Create a virtual environment and install the pinned dependencies:

```bash
python -m venv venv
venv\Scripts\activate      # On Windows
source venv/bin/activate    # On Linux/macOS
pip install -r requirements.txt
```
venv\Scripts\activate
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128


### 2. Running Federated Learning
Open 4 separate terminal windows:

* **Terminal 1: Server**
  ```bash
  venv\Scripts\Activate.bat
  set TARGET_CLIENTS=3&& set MIN_CLIENTS=2&& python federated_healthcare\src\server.py

  OR

  set TRAINING_MODE=fresh&& set TARGET_CLIENTS=3&& set MIN_CLIENTS=2&& python federated_healthcare\src\server.py
  ```
* **Terminal 2: Client A**
  ```bash
  venv\Scripts\Activate.bat
  set DATA_PATH=.\federated_healthcare\data\hospital_A&& set CLIENT_NAME=Hospital_A&& python federated_healthcare\src\client.py

  ```
* **Terminal 3: Client B**
  ```bash
  venv\Scripts\Activate.bat
  set DATA_PATH=.\federated_healthcare\data\hospital_B&& set CLIENT_NAME=Hospital_B&& python federated_healthcare\src\client.py

  ```
* **Terminal 4: Client C**
  ```bash
  venv\Scripts\Activate.bat
  set DATA_PATH=.\federated_healthcare\data\hospital_c&& set CLIENT_NAME=Hospital_C&& python federated_healthcare\src\client.py
  ```

## 🔄 Controlled Periodic Retraining

The system supports controlled, reproducible periodic retraining cycles with **zero modification to existing dataset directories**.

### 1. Dataset Directory Preservation & Logical Manifests
The existing directory structure is preserved completely untouched:
```
data/hospital_A/train/NORMAL/
data/hospital_A/train/PNEUMONIA/
data/hospital_B/train/NORMAL/
data/hospital_B/train/PNEUMONIA/
data/hospital_C/train/NORMAL/
data/hospital_C/train/PNEUMONIA/
```
No images are renamed, moved, or duplicated. No physical `validation/` or `test/` folders are created inside hospital data folders.

Instead, train/validation/test partitions are managed through immutable JSON manifests stored under `federated_healthcare/dataset_manifests/<hospital_id>/`:
- Manifests record relative image paths, SHA-256 hashes, class labels, and split assignments (`train: 0.70`, `validation: 0.15`, `test: 0.15`).
- Patient-level partition isolation is enforced where patient IDs are present in filenames.
- Historical test cohorts are preserved when new images arrive; new images are assigned deterministically without reshuffling the historical dataset.

### 2. Dataset Version Creation Command
Generate or update logical dataset manifests for hospitals:
```bash
# Generate version dataset_v001 for all hospitals:
python federated_healthcare/src/dataset_manifest.py --hospital all --version dataset_v001

# For a single hospital:
python federated_healthcare/src/dataset_manifest.py --hospital Hospital_A --version dataset_v001
```

When new clinically verified images are placed in the existing `NORMAL/` or `PNEUMONIA/` folders:
```bash
# Detect changes and generate dataset_v002 inheriting historical assignments:
python federated_healthcare/src/dataset_manifest.py --hospital all --version dataset_v002 --parent dataset_v001
```

### 3. Training Modes: Fresh vs Warm-Start

#### Fresh Initialisation (`TRAINING_MODE=fresh`)
- Initialises a new ChestCNN with deterministic random seed (`SPLIT_SEED=42`).
- Saves baseline checkpoint as `initial_model.pth`.

#### Warm-Start Retraining (`TRAINING_MODE=warm_start` — Default)
- Loads the approved global model from `INIT_MODEL_PATH` or `models/approved/global_model.pth`.
- Strictly verifies checkpoint existence, SHA-256 hash, architecture keys, and tensor shapes.
- With `ALLOW_RANDOM_FALLBACK=0`, the run stops immediately with a fatal error if the checkpoint is missing or invalid. Never silently falls back to random weights.

### 4. Periodic Retraining Workflow
1. **Initial Baseline Evaluation (Before Round 1)**: Evaluates the initial global model on every hospital's validation partition. Collects TP, TN, FP, FN, loss, accuracy, and F1.
2. **Round-Wise Validation**: In each federated round, the server aggregates client updates into a new candidate model. Clients evaluate this model on their validation partition in a single inference pass and return aggregate counts.
3. **Exact Checkpoint Selection**: The primary checkpoint criterion is pooled validation loss. When an improvement $\ge \text{MIN\_DELTA}$ (0.001) occurs, `best_model.pth` is saved atomically.
4. **Early Stopping**: If validation loss does not improve for `EARLY_STOPPING_PATIENCE` (5) consecutive rounds after `MIN_ROUNDS` (5), federated rounds stop cleanly.
5. **Final Independent Testing**: Evaluates `best_model.pth` on the reserved `test` partition across all hospitals.
6. **Model Approval & Rollback**: Evaluates candidate metrics against configurable experimental safeguards (`MODEL_MIN_ACCURACY=0.80`, `MODEL_MIN_PNEUMONIA_RECALL=0.85`). If approved, atomically promotes to `models/approved/global_model.pth` and archives the prior approved model to `models/approved/previous_global_model.pth`.

### 5. Launch Commands

#### Server Launch:
```bash
python federated_healthcare/src/server.py
```

#### Hospital Client Launches:
```bash
# Hospital A
set CLIENT_NAME=Hospital_A&& python federated_healthcare/src/client.py

# Hospital B
set CLIENT_NAME=Hospital_B&& python federated_healthcare/src/client.py

# Hospital C
set CLIENT_NAME=Hospital_C&& python federated_healthcare/src/client.py
```

#### Running Automated Tests:
```bash
python -m unittest discover -s federated_healthcare/src/tests
```

---

## 📈 Monitoring & Graphing

### 1. Real-Time Streamlit Dashboard
Launch the monitoring dashboard to observe training convergence in real-time:
```bash
streamlit run federated_healthcare/dashboard/app.py
```
### 2. Evaluating the Global Model
Evaluate the final global model checkpoint against the test set:
```bash
venv\Scripts\python.exe federated_healthcare\src\evaluate.py
```

### 3. Generating Analysis Graphs
Compile all raw CSV results into structured visualizations and a final comparison summary:
```bash
venv\Scripts\python.exe federated_healthcare\src\graph.py
```

---

## 🐳 Docker Support

You can package and containerize your FL node:

### 1. Build Docker Image
```bash
docker build -t federated-fl-node .
```

### 2. Run Container (e.g., Hospital A Node)
```bash
docker run -e CLIENT_NAME=Hospital_A -e DATA_PATH=/app/data/hospital_A -v $(pwd)/data:/app/data federated-fl-node
```

---

## 🔄 Git Workflow & Branching

Always perform development on a separate feature branch before pushing to the repository.

### 1. Clone the Repository
```bash
git clone <repository_url>
cd Federated-Learning
```

### 2. Create a Feature Branch
```bash
git checkout -b feature/restructure-and-quantization
```

### 3. Save Changes & Commit
```bash
git add .
git commit -m "feat: Restructure directory layout and enable consolidated results"
```

### 4. Push Branch to Remote
```bash
git push -u origin feature/restructure-and-quantization
```

### 5. Open a Pull Request
Go to your Git repository (e.g. GitHub/GitLab) and merge the feature branch into `main` after verification.



