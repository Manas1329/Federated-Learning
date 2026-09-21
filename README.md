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
│   │   ├── results/                      # Raw log CSV files (Organized by experiment)
│   │   │   ├── a_pure/                   # Baseline logs
│   │   │   ├── b_quantized/              # Quantized logs
│   │   │   └── c_dp/                     # Diff-Privacy logs
│   │   └── plots/                        # Output plots
│   │       ├── a_pure/                   
│   │       ├── b_quantized/              
│   │       ├── c_dp/                     
│   │       ├── comparisons/              # Cross-experiment comparative charts
│   │       └── figures/                  # Final presentation figures
│   ├── models/                           # PyTorch model checkpoints (.pth)
│   └── src/                              # Source code directory
│       ├── scripts/                      # Utility scripts (split_data, graphs, etc)
│       ├── tests/                        # Regression & Unit tests
│       ├── client.py                     # FL client implementation (DP-SGD + INT8)
│       ├── server.py                     # FL server implementation (Adaptive Timeouts)
│       ├── evaluate.py                   # Global evaluation script
│       ├── graph.py                      # Chart & metrics compiler script
│       ├── model.py                      # ChestCNN PyTorch architecture
│       ├── quantization.py               # INT8 quantization utilities
│       ├── dropout_handler.py            # Adaptive dropout handler logic
│       ├── dropout_engine.py             # Adaptive decision math engine
│       ├── trust_manager.py              # Trust scoring & outlier tagging
│       ├── paths.py                      # Centralized path handling
│       └── utils.py                      # Data loading & helper utilities
├── .env                                  # Active environment config (ignored by git)
├── .env_example                          # Template for environment configuration
├── requirements.txt                      # Python dependencies
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

### 2. Running Federated Learning
Open 4 separate terminal windows:

* **Terminal 1: Server**
  ```bash
  venv\Scripts\Activate.bat
  set TARGET_CLIENTS=3&& set MIN_CLIENTS=2&& python federated_healthcare\src\server.py

  ```
* **Terminal 2: Client A**
  ```bash
  venv\Scripts\Activate.bat
  set DATA_PATH=.\data\hospital_A&& set CLIENT_NAME=Hospital_A&& python federated_healthcare\src\client.py

  ```
* **Terminal 3: Client B**
  ```bash
  venv\Scripts\Activate.bat
  set DATA_PATH=.\data\hospital_B&& set CLIENT_NAME=Hospital_B&& python federated_healthcare\src\client.py

  ```
* **Terminal 4: Client C**
  ```bash
  venv\Scripts\Activate.bat
  set DATA_PATH=.\data\hospital_C&& set CLIENT_NAME=Hospital_C&& python federated_healthcare\src\client.py

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





