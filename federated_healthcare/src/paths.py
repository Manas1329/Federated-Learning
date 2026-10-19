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

def _find_matching_folder(parent: Path, folder_name: str) -> Path:
    """Find a folder case-insensitively if exact match does not exist."""
    exact = parent / folder_name
    if exact.exists():
        return exact
    if parent.exists():
        target_lower = folder_name.lower()
        for child in parent.iterdir():
            if child.is_dir() and child.name.lower() == target_lower:
                return child
    return exact

def resolve_data_path(env_data_path: str = None, client_name: str = None) -> Path:
    """
    Resolve the data path for a client robustly.
    1. If env_data_path exists, use it.
    2. If env_data_path is given but doesn't exist, attempt resolution inside DATA_DIR.
    3. If client_name is provided, find matching folder in DATA_DIR case-insensitively.
    4. Fallback to DATA_DIR / 'hospital_A'.
    """
    if env_data_path:
        env_path = Path(env_data_path)
        if env_path.exists():
            return env_path
        
        # Strip common prefixes
        folder_name = env_path.name
        resolved = _find_matching_folder(DATA_DIR, folder_name)
        if resolved.exists():
            return resolved

        # Check relative to PROJECT_ROOT
        env_str = env_data_path.replace("./", "").replace(".\\", "")
        cand = PROJECT_ROOT / env_str
        if cand.exists():
            return cand
        cand_fh = FEDERATED_HEALTHCARE_DIR / env_str
        if cand_fh.exists():
            return cand_fh

    if client_name:
        resolved = _find_matching_folder(DATA_DIR, client_name)
        if resolved.exists():
            return resolved
        # Try hospital_<suffix>
        client_clean = client_name.replace("Hospital_", "hospital_").replace("hospital-", "hospital_")
        resolved = _find_matching_folder(DATA_DIR, client_clean)
        if resolved.exists():
            return resolved
        return DATA_DIR / client_name.lower()

    return _find_matching_folder(DATA_DIR, "hospital_A")
