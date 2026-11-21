"""
utils.py
========
Data loading and partition utilities for Federated Healthcare Learning.
Integrates manifest-based dataset versioning and reproducible
train/validation/test splitting without modifying existing directory structures.
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Union
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from paths import resolve_data_path, MANIFESTS_DIR
from dataset_manifest import DatasetManifestManager, build_manifest_dataloaders


def _canonical_hospital_id(data_path: Union[str, Path], client_id: Optional[str] = None) -> str:
    """Derive standard hospital ID (e.g. 'Hospital_A') from path or client name."""
    if client_id:
        c = client_id.strip()
        parts = c.split("_")
        if len(parts) == 2 and parts[0].lower() == "hospital":
            return f"Hospital_{parts[1].upper()}"
        return c

    p = Path(data_path)
    dir_name = p.name
    parts = dir_name.split("_")
    if len(parts) == 2 and parts[0].lower() == "hospital":
        return f"Hospital_{parts[1].upper()}"
    return dir_name


def load_hospital_data(
    data_path: Union[str, Path],
    batch_size: int = 32,
    img_size: int = 128,
    return_val: bool = True,
    dataset_version: Optional[str] = None,
    split_seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Any:
    """
    Load chest X-ray images for a hospital using its immutable dataset manifest.

    Preserves original folders:
        data_path/
            train/
                NORMAL/
                PNEUMONIA/

    Returns:
        If return_val is True (default):
            (trainloader, validationloader, testloader)
        If return_val is False:
            (trainloader, testloader)
    """
    data_path = Path(data_path)
    version_id = dataset_version or os.environ.get("DATASET_VERSION", "dataset_v001")
    hospital_id = _canonical_hospital_id(data_path, os.environ.get("CLIENT_NAME"))

    manager = DatasetManifestManager(MANIFESTS_DIR)
    manifest_file = manager.get_hospital_manifest_dir(hospital_id) / f"{version_id}.json"

    if manifest_file.exists():
        manifest = manager.load_manifest(hospital_id, version_id)
    else:
        # Create manifest deterministically if not yet present
        print(f"[utils] Manifest {version_id} not found for {hospital_id}. Generating from existing image directory...")
        manifest, _ = manager.create_manifest(
            hospital_id=hospital_id,
            hospital_data_path=data_path,
            version_id=version_id,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            split_seed=split_seed
        )

    trainloader, valloader, testloader = build_manifest_dataloaders(
        hospital_data_path=data_path,
        manifest=manifest,
        batch_size=batch_size,
        img_size=img_size
    )

    if return_val:
        return trainloader, valloader, testloader
    else:
        return trainloader, testloader


def load_partitions(
    client_id: str = "Hospital_A",
    batch_size: int = 32,
    return_val: bool = False,
    dataset_version: Optional[str] = None
) -> Any:
    """
    Load data partitions for a hospital client.

    Returns:
        If return_val is True:
            (trainloader, validationloader, testloader, num_examples)
        If return_val is False (default for backward-compatibility):
            (trainloader, testloader, num_examples)
    """
    data_path = resolve_data_path(None, client_id)
    if return_val:
        trainloader, valloader, testloader = load_hospital_data(
            data_path,
            batch_size=batch_size,
            return_val=True,
            dataset_version=dataset_version
        )
        num_examples = {
            "trainset": len(trainloader.dataset),
            "valset": len(valloader.dataset),
            "testset": len(testloader.dataset)
        }
        return trainloader, valloader, testloader, num_examples
    else:
        trainloader, testloader = load_hospital_data(
            data_path,
            batch_size=batch_size,
            return_val=False,
            dataset_version=dataset_version
        )
        num_examples = {
            "trainset": len(trainloader.dataset),
            "testset": len(testloader.dataset)
        }
        return trainloader, testloader, num_examples
