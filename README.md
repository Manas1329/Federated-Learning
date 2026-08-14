# Clinical Federated Learning with Dynamic INT8 Quantization

An end-to-end framework implementing medical image classification (Pneumonia detection in Chest X-rays) using Federated Learning (FL). This project utilizes the **Flower FL framework** and simulates three isolated hospital nodes under a non-IID split. It incorporates dynamic **INT8 Quantization** to reduce the communication payload size by **75%** while retaining high performance.

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
│       ├── client.py                     # FL client implementation
│       ├── server.py                     # FL server implementation
│       ├── evaluate.py                   # Global evaluation script
│       ├── graph.py                      # Chart & metrics compiler script
│       ├── model.py                      # ChestCNN PyTorch architecture
│       ├── quantization.py               # INT8 quantization utilities
│       └── utils.py                      # Data loading & helper utilities
├── .env                                  # Active environment config (ignored by git)
├── .env_example                          # Template for environment configuration
├── requirements.txt                      # Python dependencies
└── Dockerfile                            # Container configuration (optional)
```

---

## 🔧 Environment Configuration (`.env`)

We use a `.env` file to globally toggle training options without manually typing parameters in multiple client/server terminals.

Template: [**`.env_example`**](file:///d:/Codes/College%20Projects/Major%20Project/Federated-Learning/.env_example)

```ini
# Enable (1) or disable (0) INT8 Quantization during communication
USE_QUANTIZATION=1

# Force training on CPU (1) or let it auto-detect GPU (0)
FORCE_CPU=0
```

To initialize your settings:
```bash
cp .env_example .env
```

---

## 📊 Dataset Splitter (`split_data.py`)

The [**`split_data.py`**](file:///d:/Codes/College%20Projects/Major%20Project/Federated-Learning/federated_healthcare/split_data.py) script reads your raw Chest X-ray train dataset directory and distributes the files to simulate a **non-IID (Independent and Identically Distributed)** data environment representing realistic clinical distributions:

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
  venv\Scripts\python.exe federated_healthcare\src\server.py
  ```
* **Terminal 2: Client A**
  ```bash
  set CLIENT_NAME=Hospital_A
  set DATA_PATH=data\hospital_A
  venv\Scripts\python.exe federated_healthcare\src\client.py
  ```
* **Terminal 3: Client B**
  ```bash
  set CLIENT_NAME=Hospital_B
  set DATA_PATH=data\hospital_B
  venv\Scripts\python.exe federated_healthcare\src\client.py
  ```
* **Terminal 4: Client C**
  ```bash
  set CLIENT_NAME=Hospital_C
  set DATA_PATH=data\hospital_C
  venv\Scripts\python.exe federated_healthcare\src\client.py
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
