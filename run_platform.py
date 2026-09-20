import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from federated_healthcare.webapp.main import run

if __name__ == "__main__":
    run()
