"""
checkpoint_manager.py
=====================
Model checkpoint management, verification, and atomic persistence.
Supports warm-start validation, parameter shape checks, integrity hashing,
and strict fallback prevention.
"""

import os
import sys
import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model import ChestCNN


def compute_checkpoint_hash(filepath: Path) -> str:
    """Calculate SHA-256 hash of a saved checkpoint file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def save_checkpoint_atomic(
    state_dict_or_model: Any,
    target_path: Path,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Atomically save a PyTorch checkpoint with metadata.
    Writes to a temporary file on the same filesystem before renaming.
    Returns the SHA-256 hash of the saved checkpoint.
    """
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(".tmp")

    if isinstance(state_dict_or_model, nn.Module):
        state_dict = state_dict_or_model.state_dict()
    elif isinstance(state_dict_or_model, dict):
        state_dict = state_dict_or_model
    else:
        raise TypeError(f"Expected nn.Module or dict state_dict, got {type(state_dict_or_model)}")

    payload = {
        "state_dict": state_dict,
        "metadata": metadata or {},
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

    try:
        torch.save(payload, str(temp_path))
        if target_path.exists():
            target_path.unlink()
        temp_path.rename(target_path)
    except Exception:
        # Fallback for systems where rename over existing might fail or cross-device
        if temp_path.exists():
            shutil.move(str(temp_path), str(target_path))

    # Also save metadata as sidecar JSON for quick non-PyTorch inspection
    if metadata:
        meta_json_path = target_path.with_suffix(".json")
        try:
            with open(meta_json_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            print(f"[CheckpointManager] Warning: Failed to write sidecar JSON: {e}")

    return compute_checkpoint_hash(target_path)


def load_raw_state_dict(checkpoint_path: Path, map_location: str = "cpu") -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Load state_dict and metadata from checkpoint, handling both bare state_dicts
    and wrapped dictionary payloads.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")

    loaded = torch.load(str(checkpoint_path), map_location=map_location)
    if isinstance(loaded, dict) and "state_dict" in loaded:
        return loaded["state_dict"], loaded.get("metadata", {})
    elif isinstance(loaded, dict):
        # Could be raw state dict
        return loaded, {}
    else:
        raise ValueError(f"Unrecognized checkpoint format at {checkpoint_path}")


def verify_checkpoint(
    checkpoint_path: Path,
    expected_model: Optional[nn.Module] = None,
    allow_random_fallback: bool = False
) -> Tuple[bool, str, Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Verify checkpoint existence, integrity, architecture, and tensor dimensions.

    Returns:
        (is_valid, error_message, state_dict, metadata)
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        msg = f"Checkpoint not found at: {checkpoint_path}"
        if not allow_random_fallback:
            raise FileNotFoundError(f"[WARM_START_ERROR] {msg} (ALLOW_RANDOM_FALLBACK=0)")
        return False, msg, {}, {}

    try:
        state_dict, metadata = load_raw_state_dict(checkpoint_path, map_location="cpu")
    except Exception as e:
        msg = f"Failed to load or deserialize checkpoint: {e}"
        if not allow_random_fallback:
            raise RuntimeError(f"[WARM_START_ERROR] {msg} (ALLOW_RANDOM_FALLBACK=0)")
        return False, msg, {}, {}

    reference_model = expected_model or ChestCNN()
    ref_keys = set(reference_model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())

    missing_keys = ref_keys - ckpt_keys
    unexpected_keys = ckpt_keys - ref_keys

    if missing_keys or unexpected_keys:
        msg = f"Architecture mismatch! Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}"
        if not allow_random_fallback:
            raise ValueError(f"[WARM_START_ERROR] {msg} (ALLOW_RANDOM_FALLBACK=0)")
        return False, msg, {}, {}

    # Verify tensor shapes
    for k, ref_tensor in reference_model.state_dict().items():
        ckpt_tensor = state_dict[k]
        if ckpt_tensor.shape != ref_tensor.shape:
            msg = f"Shape mismatch for parameter '{k}': expected {ref_tensor.shape}, got {ckpt_tensor.shape}"
            if not allow_random_fallback:
                raise ValueError(f"[WARM_START_ERROR] {msg} (ALLOW_RANDOM_FALLBACK=0)")
            return False, msg, {}, {}

    return True, "Valid", state_dict, metadata
