"""
run_dacm.py
===========
Executable runner for the Dynamic Federated DACM Recovery Protocol.
Invokes dacm.execute_dacm_recovery with dynamic client detection and local counting.
"""

import sys
import argparse
from pathlib import Path

# Ensure src is importable
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dacm import execute_dacm_recovery, calculate_dacm_weights, get_client_class_counts, discover_clients


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Federated DACM Recovery"
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default="Hospital_B",
        help="Client ID to exclude/unlearn (e.g. Hospital_B or Hospital_A). Default: Hospital_B.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to starting unlearned model checkpoint.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of local recovery epochs per client (default: 2).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for local recovery training (default: 1e-3).",
    )
    parser.add_argument(
        "--tau-safe",
        type=float,
        default=0.50,
        help="Safety threshold for pneumonia ratio (default: 0.50).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=3.0,
        help="Logarithmic scaling factor (default: 3.0).",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Path where recovered model should be saved.",
    )

    args = parser.parse_args()

    execute_dacm_recovery(
        excluded_client=args.exclude,
        unlearned_model_path=args.model_path,
        tau_safe=args.tau_safe,
        alpha=args.alpha,
        epochs=args.epochs,
        lr=args.lr,
        save_path=args.save_path,
    )
