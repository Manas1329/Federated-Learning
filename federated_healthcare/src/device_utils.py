"""
device_utils.py
===============
Centralized device-selection utility for Federated Healthcare Learning.
Ensures consistent device management across training, DP training,
validation, testing, and model serialization.
"""

import os
import torch
from typing import Optional

def get_device(force_cpu: Optional[bool] = None, verbose: bool = False) -> torch.device:
    """
    Return the torch.device to use according to project configuration.

    Rules:
    - If force_cpu is explicitly passed as True, or env var FORCE_CPU == "1",
      returns torch.device("cpu").
    - If force_cpu is explicitly passed as False, or env var FORCE_CPU == "0",
      permits torch.device("cuda") if CUDA is available, otherwise falls back to CPU.
    """
    if force_cpu is None:
        force_cpu = os.environ.get("FORCE_CPU", "0").strip().lower() in ("1", "true", "yes")

    if force_cpu:
        selected_device = torch.device("cpu")
    else:
        selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        cuda_avail = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "N/A"
        print(f"[DeviceUtils] Selected device: {selected_device} (FORCE_CPU={force_cpu}, CUDA Available={cuda_avail}, GPU={gpu_name})")

    return selected_device
