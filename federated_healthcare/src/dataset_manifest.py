"""
dataset_manifest.py
===================
Dataset manifest and versioning engine for Federated Healthcare.

PRESERVES EXISTING DIRECTORY STRUCTURE:
Does NOT create physical directories or duplicate images.
All partitions (train, validation, test) are tracked via JSON manifests
saved under `federated_healthcare/dataset_manifests/<hospital_id>/`.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, cast
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from paths import MANIFESTS_DIR, resolve_data_path

DEFAULT_CLASS_MAPPING = {
    "NORMAL": 0,
    "PNEUMONIA": 1
}


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file's binary content."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def extract_patient_id(filename: str) -> Optional[str]:
    """
    Attempt to extract patient identifier from filename if present.
    Example: 'person1946_bacteria_4874.jpeg' -> 'person1946'.
    If not recognizable, returns None.
    """
    stem = Path(filename).stem
    if "person" in stem.lower():
        parts = stem.split("_")
        for p in parts:
            if p.lower().startswith("person"):
                return p.lower()
    return None


class ManifestDataset(Dataset):
    """
    PyTorch Dataset that loads images strictly from existing hospital folders
    based on a manifest partition list.
    """
    def __init__(
        self,
        data_root: Path,
        image_entries: List[Dict[str, Any]],
        transform: Optional[transforms.Compose] = None
    ):
        self.data_root = Path(data_root)
        self.image_entries = image_entries
        self.transform = transform or transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

        # Precompute targets for fast O(1) class-distribution queries
        self.targets = [entry["label"] for entry in self.image_entries]

    def __len__(self) -> int:
        return len(self.image_entries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        entry = self.image_entries[idx]
        img_path = self.data_root / entry["relative_path"]

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found at expected location: {img_path}")

        try:
            with Image.open(img_path) as img:
                # Ensure PIL loads the image data into memory before transform
                img = img.convert("L")  # 8-bit pixels, black and white
                if self.transform:
                    tensor_img = self.transform(img)
                else:
                    tensor_img = transforms.ToTensor()(img)
        except Exception as e:
            raise RuntimeError(f"Failed to read image at {img_path}: {e}")

        return cast(torch.Tensor, tensor_img), int(entry["label"])


class DatasetManifestManager:
    """
    Manages immutable logical dataset versions, change detection,
    and train/validation/test splitting for hospital clients.
    """

    def __init__(self, manifests_base_dir: Optional[Path] = None):
        self.base_dir = Path(manifests_base_dir or MANIFESTS_DIR)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_hospital_manifest_dir(self, hospital_id: str) -> Path:
        h_dir = self.base_dir / hospital_id
        h_dir.mkdir(parents=True, exist_ok=True)
        return h_dir

    def list_versions(self, hospital_id: str) -> List[str]:
        h_dir = self.get_hospital_manifest_dir(hospital_id)
        versions = []
        for p in h_dir.glob("dataset_v*.json"):
            versions.append(p.stem)
        return sorted(versions)

    def load_manifest(self, hospital_id: str, version_id: str) -> Dict[str, Any]:
        manifest_path = self.get_hospital_manifest_dir(hospital_id) / f"{version_id}.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest {version_id} not found for {hospital_id} at {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def scan_hospital_images(
        self,
        hospital_data_path: Path,
        class_mapping: Optional[Dict[str, int]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Scan the existing train directory of the hospital folder.
        Returns a dict mapping relative_path -> metadata dict.
        Validates file integrity and rejects corrupted or unsupported files.
        """
        hospital_data_path = Path(hospital_data_path)
        train_path = hospital_data_path / "train"
        if not train_path.exists():
            raise FileNotFoundError(f"Required dataset folder 'train' not found in {hospital_data_path}")

        mapping = class_mapping or DEFAULT_CLASS_MAPPING
        scanned_inventory: Dict[str, Dict[str, Any]] = {}
        seen_hashes: Dict[str, str] = {}

        for class_name, label in mapping.items():
            class_dir = train_path / class_name
            if not class_dir.exists():
                continue

            for file_path in class_dir.iterdir():
                if not file_path.is_file():
                    continue
                # Skip hidden files or OS artifacts (e.g., ._DS_Store, Thumbs.db)
                if file_path.name.startswith(".") or file_path.suffix.lower() not in (
                    ".jpg", ".jpeg", ".png", ".bmp"
                ):
                    continue

                # Verify image integrity
                try:
                    with Image.open(file_path) as test_img:
                        test_img.verify()
                except Exception as e:
                    print(f"[ManifestManager] WARNING: Corrupted or unsupported image skipped: {file_path} ({e})")
                    continue

                # Calculate SHA-256
                rel_path = str(file_path.relative_to(hospital_data_path)).replace("\\", "/")
                file_hash = compute_file_sha256(file_path)

                is_duplicate = file_hash in seen_hashes
                if is_duplicate:
                    duplicate_of = seen_hashes[file_hash]
                else:
                    seen_hashes[file_hash] = rel_path
                    duplicate_of = None

                patient_id = extract_patient_id(file_path.name)

                scanned_inventory[rel_path] = {
                    "relative_path": rel_path,
                    "sha256": file_hash,
                    "class": class_name,
                    "label": label,
                    "patient_id": patient_id or "unknown",
                    "patient_leakage_risk": patient_id is None,
                    "is_duplicate": is_duplicate,
                    "duplicate_of": duplicate_of
                }

        return scanned_inventory

    def detect_changes(
        self,
        current_inventory: Dict[str, Dict[str, Any]],
        previous_manifest: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare current scanned image inventory against previous manifest.
        Identifies new, missing, modified, and duplicate images.
        """
        prev_images = previous_manifest.get("images", {})
        prev_hashes = {info["sha256"]: rel_path for rel_path, info in prev_images.items()}

        new_images = []
        missing_images = []
        modified_images = []
        unchanged_images = []

        for rel_path, current_info in current_inventory.items():
            if rel_path not in prev_images:
                # Check if same hash existed elsewhere
                if current_info["sha256"] in prev_hashes:
                    # Relocated or duplicate
                    new_images.append(rel_path)
                else:
                    new_images.append(rel_path)
            else:
                prev_info = prev_images[rel_path]
                if current_info["sha256"] != prev_info["sha256"]:
                    modified_images.append(rel_path)
                else:
                    unchanged_images.append(rel_path)

        for rel_path in prev_images:
            if rel_path not in current_inventory:
                missing_images.append(rel_path)

        has_changes = bool(new_images or missing_images or modified_images)
        return {
            "has_changes": has_changes,
            "new_count": len(new_images),
            "missing_count": len(missing_images),
            "modified_count": len(modified_images),
            "unchanged_count": len(unchanged_images),
            "new_images": new_images,
            "missing_images": missing_images,
            "modified_images": modified_images
        }

    def create_manifest(
        self,
        hospital_id: str,
        hospital_data_path: Path,
        version_id: str = "dataset_v001",
        parent_version_id: Optional[str] = None,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        split_seed: int = 42,
        force_new: bool = False
    ) -> Tuple[Dict[str, Any], bool]:
        """
        Create a new dataset version manifest or reuse an existing one if unchanged.
        Returns (manifest_dict, is_newly_created).
        """
        hospital_data_path = Path(hospital_data_path)
        current_inventory = self.scan_hospital_images(hospital_data_path)

        if not current_inventory:
            raise RuntimeError(f"No valid images found in hospital dataset path: {hospital_data_path}")

        # Check existing manifest versions
        existing_versions = self.list_versions(hospital_id)
        if version_id in existing_versions and not force_new:
            # Check if current images match the existing manifest
            existing_manifest = self.load_manifest(hospital_id, version_id)
            diff = self.detect_changes(current_inventory, existing_manifest)
            if not diff["has_changes"]:
                return existing_manifest, False

        # If parent version exists, load historical assignments to preserve them
        historical_assignments: Dict[str, str] = {}
        if parent_version_id and parent_version_id in existing_versions:
            parent_manifest = self.load_manifest(hospital_id, parent_version_id)
            for split_name, file_list in parent_manifest.get("splits", {}).items():
                for p in file_list:
                    historical_assignments[p] = split_name
        elif existing_versions:
            # Inherit from the most recent previous version
            latest_prev = existing_versions[-1]
            parent_version_id = latest_prev
            parent_manifest = self.load_manifest(hospital_id, latest_prev)
            for split_name, file_list in parent_manifest.get("splits", {}).items():
                for p in file_list:
                    historical_assignments[p] = split_name

        # Prepare splitting
        # Ratios validation
        total_ratio = train_ratio + val_ratio + test_ratio
        if not np.isclose(total_ratio, 1.0):
            train_ratio = train_ratio / total_ratio
            val_ratio = val_ratio / total_ratio
            test_ratio = test_ratio / total_ratio

        splits: Dict[str, List[str]] = {
            "train": [],
            "validation": [],
            "test": []
        }

        # Step 1: Retain historical assignments for existing images
        unassigned_by_class: Dict[str, List[str]] = {"NORMAL": [], "PNEUMONIA": []}
        for rel_path, info in sorted(current_inventory.items()):
            if rel_path in historical_assignments:
                target_split = historical_assignments[rel_path]
                if target_split in splits:
                    splits[target_split].append(rel_path)
                    info["split"] = target_split
                else:
                    unassigned_by_class[info["class"]].append(rel_path)
            else:
                unassigned_by_class[info["class"]].append(rel_path)

        # Step 2: Deterministically assign new / unassigned images, stratified by class
        rng = np.random.RandomState(split_seed)

        # Check if patient identifiers are available for patient-level grouping
        all_patient_ids = [
            current_inventory[p]["patient_id"]
            for c_files in unassigned_by_class.values()
            for p in c_files
            if current_inventory[p]["patient_id"] != "unknown"
        ]
        has_patient_ids = len(all_patient_ids) > 0 and (len(all_patient_ids) >= 0.5 * len(current_inventory))

        if has_patient_ids:
            # Group unassigned files by patient
            patient_to_files: Dict[str, List[str]] = {}
            for c_files in unassigned_by_class.values():
                for rel_path in c_files:
                    pid = current_inventory[rel_path]["patient_id"]
                    patient_to_files.setdefault(pid, []).append(rel_path)

            unique_patients = sorted(list(patient_to_files.keys()))
            rng.shuffle(unique_patients)

            n_p = len(unique_patients)
            n_val_p = int(round(n_p * val_ratio))
            n_test_p = int(round(n_p * test_ratio))
            n_train_p = n_p - n_val_p - n_test_p

            val_patients = set(unique_patients[:n_val_p])
            test_patients = set(unique_patients[n_val_p:n_val_p + n_test_p])
            train_patients = set(unique_patients[n_val_p + n_test_p:])

            for pid, files in patient_to_files.items():
                if pid in train_patients:
                    target_split = "train"
                elif pid in val_patients:
                    target_split = "validation"
                else:
                    target_split = "test"
                for rel_path in files:
                    splits[target_split].append(rel_path)
                    current_inventory[rel_path]["split"] = target_split
        else:
            # Stratified split per class
            for c_name, files in unassigned_by_class.items():
                sorted_files = sorted(files)
                rng.shuffle(sorted_files)

                n_files = len(sorted_files)
                if n_files == 0:
                    continue

                n_val = int(round(n_files * val_ratio))
                n_test = int(round(n_files * test_ratio))
                n_train = n_files - n_val - n_test

                val_files = sorted_files[:n_val]
                test_files = sorted_files[n_val:n_val + n_test]
                train_files = sorted_files[n_val + n_test:]

                for rel_path in train_files:
                    splits["train"].append(rel_path)
                    current_inventory[rel_path]["split"] = "train"
                for rel_path in val_files:
                    splits["validation"].append(rel_path)
                    current_inventory[rel_path]["split"] = "validation"
                for rel_path in test_files:
                    splits["test"].append(rel_path)
                    current_inventory[rel_path]["split"] = "test"

        # Sort all split file lists for strict reproducibility
        for s in splits:
            splits[s].sort()

        # Compute summary metrics
        counts_by_class = {"NORMAL": 0, "PNEUMONIA": 0}
        counts_by_split = {"train": len(splits["train"]), "validation": len(splits["validation"]), "test": len(splits["test"])}
        split_class_dist = {
            "train": {"NORMAL": 0, "PNEUMONIA": 0},
            "validation": {"NORMAL": 0, "PNEUMONIA": 0},
            "test": {"NORMAL": 0, "PNEUMONIA": 0}
        }

        for rel_path, info in current_inventory.items():
            c_name = info["class"]
            s_name = info["split"]
            counts_by_class[c_name] += 1
            split_class_dist[s_name][c_name] += 1

        manifest: Dict[str, Any] = {
            "hospital_id": hospital_id,
            "dataset_version": version_id,
            "parent_dataset_version": parent_version_id,
            "creation_timestamp": datetime.utcnow().isoformat() + "Z",
            "split_seed": split_seed,
            "ratios": {
                "train": train_ratio,
                "validation": val_ratio,
                "test": test_ratio
            },
            "class_mapping": DEFAULT_CLASS_MAPPING,
            "preprocessing": {
                "channels": 1,
                "img_size": [128, 128],
                "normalization": "Grayscale(1), Resize(128,128), ToTensor()"
            },
            "patient_level_stratification": has_patient_ids,
            "patient_leakage_warning": not has_patient_ids,
            "counts": {
                "total": len(current_inventory),
                "by_class": counts_by_class,
                "by_split": counts_by_split,
                "split_class_distribution": split_class_dist
            },
            "splits": splits,
            "images": current_inventory
        }

        # Calculate manifest content hash for tamper-evidence
        serialized_manifest = json.dumps(manifest, sort_keys=True)
        manifest["manifest_hash"] = hashlib.sha256(serialized_manifest.encode("utf-8")).hexdigest()

        # Save manifest
        h_dir = self.get_hospital_manifest_dir(hospital_id)
        out_path = h_dir / f"{version_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print(f"[ManifestManager] Manifest saved for {hospital_id} ({version_id}) -> {out_path}")
        print(f"  Total images: {len(current_inventory)} | Train: {len(splits['train'])}, Val: {len(splits['validation'])}, Test: {len(splits['test'])}")
        print(f"  Class breakdown: NORMAL={counts_by_class['NORMAL']}, PNEUMONIA={counts_by_class['PNEUMONIA']}")

        return manifest, True


def build_manifest_dataloaders(
    hospital_data_path: Path,
    manifest: Dict[str, Any],
    batch_size: int = 32,
    img_size: int = 128,
    num_workers: int = 0
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Construct separate PyTorch DataLoader objects for train, validation, and test partitions
    strictly using the saved manifest.
    """
    hospital_data_path = Path(hospital_data_path)
    splits = manifest["splits"]
    images_info = manifest["images"]

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    train_entries = [images_info[p] for p in splits["train"] if p in images_info]
    val_entries = [images_info[p] for p in splits["validation"] if p in images_info]
    test_entries = [images_info[p] for p in splits["test"] if p in images_info]

    train_dataset = ManifestDataset(hospital_data_path, train_entries, transform=transform)
    val_dataset = ManifestDataset(hospital_data_path, val_entries, transform=transform)
    test_dataset = ManifestDataset(hospital_data_path, test_entries, transform=transform)

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    valloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return trainloader, valloader, testloader


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create or update dataset manifests for hospital data.")
    parser.add_argument("--hospital", type=str, default="Hospital_A", help="Hospital ID (e.g. Hospital_A or 'all')")
    parser.add_argument("--version", type=str, default="dataset_v001", help="Dataset version string (e.g. dataset_v001)")
    parser.add_argument("--parent", type=str, default=None, help="Parent dataset version to inherit split assignments from")
    parser.add_argument("--train_ratio", type=float, default=0.70, help="Train partition ratio")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation partition ratio")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test partition ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    parser.add_argument("--force_new", action="store_true", help="Force new split generation even if images unchanged")

    args = parser.parse_args()

    manager = DatasetManifestManager()
    hospitals = ["Hospital_A", "Hospital_B", "Hospital_C"] if args.hospital.lower() == "all" else [args.hospital]

    for h in hospitals:
        data_path = resolve_data_path(None, h)
        print(f"\nProcessing manifest for {h} (Data path: {data_path})...")
        manifest, created = manager.create_manifest(
            hospital_id=h,
            hospital_data_path=data_path,
            version_id=args.version,
            parent_version_id=args.parent,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            split_seed=args.seed,
            force_new=args.force_new
        )
        status = "Created new" if created else "Reused existing"
        print(f"[Done] {status} manifest for {h} ({args.version}).")
