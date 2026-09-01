import os
from pathlib import Path

# Paths are resolved relative to THIS file (federated_healthcare/src/paths.py)
# src -> federated_healthcare -> Federated-Learning (Project Root)
SRC_DIR = Path(__file__).resolve().parent
FEDERATED_HEALTHCARE_DIR = SRC_DIR.parent
PROJECT_ROOT = FEDERATED_HEALTHCARE_DIR.parent

# Core Directories
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = FEDERATED_HEALTHCARE_DIR / "models"
DASHBOARD_DIR = FEDERATED_HEALTHCARE_DIR / "dashboard"
if "EXPERIMENT_RESULTS_DIR" in os.environ:
    RESULTS_DIR = Path(os.environ["EXPERIMENT_RESULTS_DIR"])
else:
    RESULTS_DIR = DASHBOARD_DIR / "results"
PLOTS_DIR = DASHBOARD_DIR / "plots"
REPORTS_DIR = DASHBOARD_DIR / "classification_reports"

# Ensure core directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def resolve_data_path(env_data_path: str = None, client_name: str = None) -> Path:
    """
    Resolve the data path for a client robustly.
    1. If env_data_path is an absolute path that exists, use it.
    2. If env_data_path is a relative path (e.g., './data/hospital_A' or 'data/hospital_A'), 
       resolve it relative to PROJECT_ROOT.
    3. If env_data_path is missing but client_name is provided (e.g., 'Hospital_A'), 
       derive it as PROJECT_ROOT / "data" / client_name.lower()
    4. Fallback to PROJECT_ROOT / "data" / "hospital_A"
    """
    if env_data_path:
        env_path = Path(env_data_path)
        if env_path.is_absolute():
            return env_path
        
        # If it starts with ./ or .\ strip it for clean path joining
        env_str = env_data_path.replace("./", "").replace(".\\", "")
        
        # If they just passed the folder name e.g. "hospital_A" instead of "data/hospital_A"
        if "data" not in env_str:
            return DATA_DIR / env_str
            
        return PROJECT_ROOT / env_str
        
    if client_name:
        return DATA_DIR / client_name.lower()
        
    return DATA_DIR / "hospital_A"
