import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from webapp.database.init_db import init_db

if __name__ == "__main__":
    init_db()
