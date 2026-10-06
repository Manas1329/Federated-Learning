import os
import sys
from pathlib import Path

# Paths are resolved relative to THIS file (federated_healthcare/src/paths.py)
# src -> federated_healthcare -> Federated-Learning (Project Root)
SRC_DIR = Path(__file__).resolve().parent
FEDERATED_HEALTHCARE_DIR = SRC_DIR.parent
PROJECT_ROOT = FEDERATED_HEALTHCARE_DIR.parent

# Inject project root robustly so that `import federated_healthcare` works anywhere
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Script & Test Directories
SCRIPTS_DIR = SRC_DIR / "scripts"
TESTS_DIR = SRC_DIR / "tests"

# Core Directories
DATA_DIR = FEDERATED_HEALTHCARE_DIR / "data" if (FEDERATED_HEALTHCARE_DIR / "data").exists() else PROJECT_ROOT / "data"
MODELS_DIR = FEDERATED_HEALTHCARE_DIR / "models"
DASHBOARD_DIR = FEDERATED_HEALTHCARE_DIR / "dashboard"

# Results & Plots Output Structure (Centralized)
RESULTS_DIR = DASHBOARD_DIR / "results"
ADSM_RESULTS_DIR = RESULTS_DIR / "ADSM_results"
EXPERIMENTS_RESULTS_DIR = ADSM_RESULTS_DIR / "experiments"

PLOTS_DIR = DASHBOARD_DIR / "plots"
FIGURES_DIR = PLOTS_DIR / "figures"
REPORTS_DIR = DASHBOARD_DIR / "classification_reports"

# Ensure core directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ADSM_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def resolve_data_path(env_data_path: str = None, client_name: str = None) -> Path:
    """
    Resolve the data path for a client robustly.
    Tries multiple candidate locations in order:
    1. If env_data_path is provided:
       - Direct absolute or relative to cwd
       - Relative to SRC_DIR (e.g. '../data/hospital_A')
       - Relative to FEDERATED_HEALTHCARE_DIR
       - Inside DATA_DIR by directory name
    2. If client_name is provided:
       - Case-insensitive search inside DATA_DIR
    3. Fallback to hospital_A inside DATA_DIR
    """
    if env_data_path:
        p = Path(env_data_path)
        candidates = [
            p.resolve() if p.is_absolute() else None,
            (Path.cwd() / p).resolve(),
            (SRC_DIR / p).resolve(),
            (FEDERATED_HEALTHCARE_DIR / p).resolve(),
            (DATA_DIR / p.name).resolve(),
        ]
        for candidate in candidates:
            if candidate and candidate.exists() and (candidate / "train").exists():
                return candidate

    target_name = (client_name or "").strip()
    if target_name:
        if (DATA_DIR / target_name).exists() and (DATA_DIR / target_name / "train").exists():
            return DATA_DIR / target_name
        if (DATA_DIR / target_name.lower()).exists() and (DATA_DIR / target_name.lower() / "train").exists():
            return DATA_DIR / target_name.lower()
        if DATA_DIR.exists():
            for d in DATA_DIR.iterdir():
                if d.is_dir() and d.name.lower() == target_name.lower() and (d / "train").exists():
                    return d

    for fallback in ["hospital_A", "hospital_a", "Hospital_A"]:
        cand = DATA_DIR / fallback
        if cand.exists() and (cand / "train").exists():
            return cand

    return DATA_DIR / (client_name or "hospital_A")
