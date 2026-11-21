"""
run_manager.py
==============
Run lifecycle, provenance logging, directory management, and artifact tracking
for periodic retraining cycles in Federated Healthcare Learning.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import torch
import flwr
import opacus

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from paths import RUNS_MODELS_DIR, RUNS_RESULTS_DIR, PROJECT_ROOT


def get_git_commit() -> str:
    """Safely obtain current git commit SHA if available."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return "unknown_or_no_git"


class RetrainingRunManager:
    """Manages directories, provenance metadata, and metric persistence for a training run."""

    def __init__(self, run_id: Optional[str] = None):
        if not run_id:
            now_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            run_id = f"run_{now_str}"

        self.run_id = run_id
        self.models_dir = RUNS_MODELS_DIR / self.run_id
        self.results_dir = RUNS_RESULTS_DIR / self.run_id

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.initial_model_path = self.models_dir / "initial_model.pth"
        self.best_model_path = self.models_dir / "best_model.pth"
        self.final_model_path = self.models_dir / "final_model.pth"
        self.metadata_path = self.models_dir / "metadata.json"

        self.run_config_path = self.results_dir / "run_config.json"
        self.initial_metrics_path = self.results_dir / "initial_metrics.csv"
        self.client_metrics_path = self.results_dir / "client_metrics.csv"
        self.round_metrics_path = self.results_dir / "round_metrics.csv"
        self.final_test_metrics_path = self.results_dir / "final_test_metrics.csv"

        self.metadata: Dict[str, Any] = {
            "run_id": self.run_id,
            "training_start_time": datetime.utcnow().isoformat() + "Z",
            "training_end_time": None,
            "environment": {
                "python_version": sys.version,
                "pytorch_version": torch.__version__,
                "flower_version": flwr.__version__,
                "opacus_version": opacus.__version__,
                "git_commit": get_git_commit()
            },
            "best_round": None,
            "best_val_loss": None,
            "best_val_accuracy": None,
            "final_round": None,
            "early_stopped": False,
            "early_stopping_reason": None,
            "acceptance_result": None,
            "promotion_result": None
        }

    def init_run(self, config: Dict[str, Any]):
        """Record configuration and initialize run config JSON."""
        self.metadata["config"] = config
        with open(self.run_config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

    def update_best(self, round_num: int, val_loss: float, val_acc: float):
        self.metadata["best_round"] = int(round_num)
        self.metadata["best_val_loss"] = float(val_loss)
        self.metadata["best_val_accuracy"] = float(val_acc)
        self.save_metadata()

    def finalize(
        self,
        final_round: int,
        early_stopped: bool = False,
        early_stopping_reason: Optional[str] = None,
        final_test_metrics: Optional[Dict[str, Any]] = None
    ):
        self.metadata["training_end_time"] = datetime.utcnow().isoformat() + "Z"
        self.metadata["final_round"] = int(final_round)
        self.metadata["early_stopped"] = bool(early_stopped)
        self.metadata["early_stopping_reason"] = early_stopping_reason
        if final_test_metrics:
            self.metadata["final_test_metrics"] = final_test_metrics
        self.save_metadata()

    def save_metadata(self):
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)
