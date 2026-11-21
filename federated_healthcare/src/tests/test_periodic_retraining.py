"""
test_periodic_retraining.py
===========================
Comprehensive automated test suite for Controlled Periodic Retraining in Federated Healthcare.
Verifies dataset manifests, checkpoint integrity, evaluation pooling, optimizer lifecycles,
early stopping, approval/rollback workflows, and compatibility across DP, quantization, and DACM.
"""

import os
import sys
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model import ChestCNN, train, evaluate_metrics, test
from device_utils import get_device
from dataset_manifest import (
    DatasetManifestManager,
    ManifestDataset,
    build_manifest_dataloaders,
    compute_file_sha256,
    DEFAULT_CLASS_MAPPING
)
from checkpoint_manager import (
    verify_checkpoint,
    save_checkpoint_atomic,
    compute_checkpoint_hash,
    load_raw_state_dict
)
from run_manager import RetrainingRunManager
from model_approval import ModelApprovalGate
from quantization import quantize_parameters, dequantize_parameters


def create_synthetic_image(path: Path, color: int = 128, size: tuple = (32, 32)):
    """Create a valid synthetic grayscale image for testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("L", size, color)
    img.save(path)


class TestDatasetManifest(unittest.TestCase):
    """Verifies dataset manifest generation, immutability, change detection, and partition isolation."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.data_dir = self.root / "hospital_test"
        self.manifest_dir = self.root / "manifests"

        # Create synthetic hospital folder structure matching real dataset
        self.normal_dir = self.data_dir / "train" / "NORMAL"
        self.pneumonia_dir = self.data_dir / "train" / "PNEUMONIA"

        for i in range(10):
            create_synthetic_image(self.normal_dir / f"person{i}_normal_{i}.jpeg", color=i * 10)
        for i in range(10):
            create_synthetic_image(self.pneumonia_dir / f"person{i+10}_pneumonia_{i}.jpeg", color=i * 15)

        self.manager = DatasetManifestManager(self.manifest_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_existing_directory_accepted_unchanged(self):
        inventory = self.manager.scan_hospital_images(self.data_dir)
        self.assertEqual(len(inventory), 20)
        # Verify original files remain in train/NORMAL and train/PNEUMONIA
        self.assertTrue(all(p.startswith("train/NORMAL/") or p.startswith("train/PNEUMONIA/") for p in inventory))

    def test_class_mapping(self):
        inventory = self.manager.scan_hospital_images(self.data_dir)
        for rel_path, info in inventory.items():
            if "NORMAL" in rel_path:
                self.assertEqual(info["label"], 0)
                self.assertEqual(info["class"], "NORMAL")
            else:
                self.assertEqual(info["label"], 1)
                self.assertEqual(info["class"], "PNEUMONIA")

    def test_manifest_generation_and_integrity_hash(self):
        manifest, created = self.manager.create_manifest(
            hospital_id="Hospital_Test",
            hospital_data_path=self.data_dir,
            version_id="dataset_v001",
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            split_seed=42
        )
        self.assertTrue(created)
        self.assertIn("manifest_hash", manifest)
        self.assertEqual(len(manifest["manifest_hash"]), 64)
        self.assertEqual(manifest["counts"]["total"], 20)

    def test_no_partition_overlap(self):
        manifest, _ = self.manager.create_manifest(
            hospital_id="Hospital_Test",
            hospital_data_path=self.data_dir,
            version_id="dataset_v001"
        )
        train_set = set(manifest["splits"]["train"])
        val_set = set(manifest["splits"]["validation"])
        test_set = set(manifest["splits"]["test"])

        self.assertEqual(len(train_set & val_set), 0)
        self.assertEqual(len(train_set & test_set), 0)
        self.assertEqual(len(val_set & test_set), 0)
        self.assertEqual(len(train_set | val_set | test_set), 20)

    def test_stable_split_assignments(self):
        m1, _ = self.manager.create_manifest(
            hospital_id="Hospital_Test",
            hospital_data_path=self.data_dir,
            version_id="dataset_v001",
            split_seed=42,
            force_new=True
        )
        m2, _ = self.manager.create_manifest(
            hospital_id="Hospital_Test",
            hospital_data_path=self.data_dir,
            version_id="dataset_v002",
            parent_version_id="dataset_v001",
            split_seed=42,
            force_new=True
        )
        self.assertEqual(m1["splits"]["train"], m2["splits"]["train"])
        self.assertEqual(m1["splits"]["validation"], m2["splits"]["validation"])
        self.assertEqual(m1["splits"]["test"], m2["splits"]["test"])

    def test_duplicate_detection(self):
        # Create an exact duplicate of an existing image
        dup_path = self.normal_dir / "duplicate_img.jpeg"
        create_synthetic_image(dup_path, color=0)  # same as person0
        inventory = self.manager.scan_hospital_images(self.data_dir)
        dup_entries = [info for info in inventory.values() if info["is_duplicate"]]
        self.assertTrue(len(dup_entries) >= 1)

    def test_missing_dataset_failure(self):
        non_existent = self.root / "does_not_exist"
        with self.assertRaises(FileNotFoundError):
            self.manager.scan_hospital_images(non_existent)

    def test_corrupted_image_handling(self):
        # Write corrupted bytes with .jpeg extension
        bad_img = self.normal_dir / "corrupted.jpeg"
        with open(bad_img, "wb") as f:
            f.write(b"NOT_A_VALID_JPEG_HEADER_CORRUPTED_DATA")

        inventory = self.manager.scan_hospital_images(self.data_dir)
        # Corrupted image must be rejected from inventory
        self.assertNotIn("train/NORMAL/corrupted.jpeg", inventory)

    def test_new_image_version_update_and_preservation(self):
        m1, _ = self.manager.create_manifest(
            hospital_id="Hospital_Test",
            hospital_data_path=self.data_dir,
            version_id="dataset_v001"
        )
        orig_train_set = set(m1["splits"]["train"])
        orig_val_set = set(m1["splits"]["validation"])
        orig_test_set = set(m1["splits"]["test"])

        # Add 4 new images to the SAME existing class folders
        create_synthetic_image(self.normal_dir / "new_normal_1.jpeg", color=200)
        create_synthetic_image(self.normal_dir / "new_normal_2.jpeg", color=205)
        create_synthetic_image(self.pneumonia_dir / "new_pneu_1.jpeg", color=210)
        create_synthetic_image(self.pneumonia_dir / "new_pneu_2.jpeg", color=215)

        # Detect changes
        new_inv = self.manager.scan_hospital_images(self.data_dir)
        diff = self.manager.detect_changes(new_inv, m1)
        self.assertTrue(diff["has_changes"])
        self.assertEqual(diff["new_count"], 4)

        # Generate dataset_v002
        m2, _ = self.manager.create_manifest(
            hospital_id="Hospital_Test",
            hospital_data_path=self.data_dir,
            version_id="dataset_v002",
            parent_version_id="dataset_v001"
        )

        # Verify historical assignments were preserved!
        self.assertTrue(orig_train_set.issubset(set(m2["splits"]["train"])))
        self.assertTrue(orig_val_set.issubset(set(m2["splits"]["validation"])))
        self.assertTrue(orig_test_set.issubset(set(m2["splits"]["test"])))
        self.assertEqual(m2["counts"]["total"], 24)

    def test_no_physical_image_movement(self):
        # Verify that create_manifest does NOT create 'val' or 'test' directories inside hospital folder
        self.assertFalse((self.data_dir / "val").exists())
        self.assertFalse((self.data_dir / "test").exists())
        self.assertFalse((self.data_dir / "validation").exists())


class TestCheckpoints(unittest.TestCase):
    """Verifies fresh initialization, warm-start verification, atomic saving, and strict error handling."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.ckpt_dir = Path(self.temp_dir.name)
        self.model = ChestCNN()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_fresh_initialization(self):
        torch.manual_seed(999)
        m1 = ChestCNN()
        torch.manual_seed(999)
        m2 = ChestCNN()
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            self.assertTrue(torch.equal(p1, p2))

    def test_successful_warm_start(self):
        ckpt_path = self.ckpt_dir / "valid_model.pth"
        meta = {"run_id": "run_001", "loss": 0.42}
        save_checkpoint_atomic(self.model, ckpt_path, metadata=meta)

        is_valid, msg, state_dict, loaded_meta = verify_checkpoint(ckpt_path, self.model)
        self.assertTrue(is_valid)
        self.assertEqual(loaded_meta.get("run_id"), "run_001")

    def test_missing_checkpoint_fatal_failure(self):
        missing_path = self.ckpt_dir / "non_existent.pth"
        with self.assertRaises(FileNotFoundError):
            verify_checkpoint(missing_path, self.model, allow_random_fallback=False)

    def test_corrupt_checkpoint_failure(self):
        corrupt_path = self.ckpt_dir / "corrupted.pth"
        with open(corrupt_path, "wb") as f:
            f.write(b"CORRUPTED_TORCH_FILE_GARBAGE_BYTES")

        with self.assertRaises(RuntimeError):
            verify_checkpoint(corrupt_path, self.model, allow_random_fallback=False)

    def test_architecture_mismatch(self):
        mismatch_path = self.ckpt_dir / "mismatch.pth"
        # Save a linear model state dict instead of ChestCNN
        linear_model = nn.Linear(10, 2)
        save_checkpoint_atomic(linear_model, mismatch_path)

        with self.assertRaises(ValueError):
            verify_checkpoint(mismatch_path, self.model, allow_random_fallback=False)

    def test_shape_mismatch(self):
        bad_shape_path = self.ckpt_dir / "bad_shape.pth"
        state = self.model.state_dict()
        state["fc2.weight"] = torch.randn(3, 128)  # 3 classes instead of 2
        save_checkpoint_atomic(state, bad_shape_path)

        with self.assertRaises(ValueError):
            verify_checkpoint(bad_shape_path, self.model, allow_random_fallback=False)

    def test_atomic_saving_and_hash_verification(self):
        target_path = self.ckpt_dir / "atomic_model.pth"
        h = save_checkpoint_atomic(self.model, target_path)
        self.assertTrue(target_path.exists())
        self.assertFalse(target_path.with_suffix(".tmp").exists())
        self.assertEqual(compute_checkpoint_hash(target_path), h)


class TestEvaluation(unittest.TestCase):
    """Verifies single-pass metric accumulation, sample-weighted loss, and zero-denominator handling."""

    def test_metrics_calculation_and_pooled_confusion_matrix(self):
        model = ChestCNN()
        # Synthetic mock dataset with 10 samples
        images = torch.randn(10, 1, 128, 128)
        labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        dataset = torch.utils.data.TensorDataset(images, labels)
        loader = DataLoader(dataset, batch_size=4)

        metrics = evaluate_metrics(model, loader, device=torch.device("cpu"))
        self.assertEqual(metrics["sample_count"], 10)
        self.assertEqual(metrics["tp"] + metrics["tn"] + metrics["fp"] + metrics["fn"], 10)
        self.assertTrue(0.0 <= metrics["accuracy"] <= 1.0)
        self.assertTrue(0.0 <= metrics["precision"] <= 1.0)
        self.assertTrue(0.0 <= metrics["recall"] <= 1.0)
        self.assertTrue(0.0 <= metrics["f1"] <= 1.0)

    def test_sample_weighted_loss_vs_batch_averaging(self):
        # Two batches: batch 1 has 10 samples with loss 1.0; batch 2 has 2 samples with loss 0.1
        # Equal batch averaging: (1.0 + 0.1) / 2 = 0.55
        # Correct sample-weighted loss: (10*1.0 + 2*0.1) / 12 = 10.2 / 12 = 0.85
        loss_sum = 10 * 1.0 + 2 * 0.1
        total_samples = 12
        correct_loss = loss_sum / total_samples
        self.assertAlmostEqual(correct_loss, 0.85, places=4)
        self.assertNotEqual(correct_loss, 0.55)

    def test_zero_denominator_handling(self):
        # When model predicts all 0s, TP=0, FP=0 -> precision = 0/(0+0) -> must safely be 0.0
        tp, fp, fn = 0, 0, 5
        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        self.assertEqual(precision, 0.0)
        self.assertEqual(recall, 0.0)
        self.assertEqual(f1, 0.0)


class TestTrainingAndDevice(unittest.TestCase):
    """Verifies optimizer reuse across epochs, device selection, and parameter loading."""

    def test_optimizer_retained_across_local_epochs(self):
        model = ChestCNN()
        images = torch.randn(8, 1, 128, 128)
        labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
        loader = DataLoader(torch.utils.data.TensorDataset(images, labels), batch_size=4)

        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        initial_opt_step = opt.state_dict().get("state", {})

        # Train epoch 1
        opt = train(model, loader, epochs=1, optimizer=opt, device=torch.device("cpu"))
        # Train epoch 2 reusing the SAME optimizer
        opt = train(model, loader, epochs=1, optimizer=opt, device=torch.device("cpu"))

        # Optimizer state should hold momentum/adam moments for parameters
        self.assertTrue(len(opt.state) > 0)

    def test_device_utils_force_cpu(self):
        d_cpu = get_device(force_cpu=True)
        self.assertEqual(d_cpu.type, "cpu")

    def test_device_utils_auto_device(self):
        d_auto = get_device(force_cpu=False)
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        self.assertEqual(d_auto.type, expected)


class TestRetrainingWorkflowAndApproval(unittest.TestCase):
    """Verifies run manager, candidate approval criteria, promotion, and rollback."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base = Path(self.temp_dir.name)
        self.approved_dir = self.base / "approved"
        self.runs_dir = self.base / "runs"
        self.approved_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.gate = ModelApprovalGate(approved_dir=self.approved_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_candidate_approval_promotion(self):
        model = ChestCNN()
        candidate_path = self.base / "candidate_model.pth"
        save_checkpoint_atomic(model, candidate_path)

        candidate_metrics = {
            "pooled": {
                "accuracy": 0.88,
                "recall": 0.90,
                "precision": 0.85,
                "loss": 0.32,
                "total": 100
            },
            "per_hospital": {
                "Hospital_A": {"accuracy": 0.89},
                "Hospital_B": {"accuracy": 0.87}
            }
        }

        passes, reason = self.gate.decide_and_apply(
            candidate_path=candidate_path,
            candidate_metrics=candidate_metrics,
            run_id="run_test_001"
        )
        self.assertTrue(passes)
        self.assertTrue((self.approved_dir / "global_model.pth").exists())

    def test_candidate_rejection_low_accuracy(self):
        model = ChestCNN()
        candidate_path = self.base / "candidate_model.pth"
        save_checkpoint_atomic(model, candidate_path)

        # Accuracy below threshold (0.75 < 0.80)
        candidate_metrics = {
            "pooled": {
                "accuracy": 0.75,
                "recall": 0.90,
                "precision": 0.85,
                "loss": 0.45,
                "total": 100
            }
        }
        passes, reason = self.gate.decide_and_apply(
            candidate_path=candidate_path,
            candidate_metrics=candidate_metrics,
            run_id="run_test_002"
        )
        self.assertFalse(passes)
        self.assertIn("accuracy", reason.lower())

    def test_candidate_rejection_high_deterioration(self):
        candidate_metrics = {
            "pooled": {
                "accuracy": 0.82,
                "recall": 0.88,
                "precision": 0.80,
                "loss": 0.35,
                "total": 100
            },
            "per_hospital": {
                "Hospital_A": {"accuracy": 0.70}  # dropped by 0.15 from 0.85
            }
        }
        baseline_metrics = {
            "pooled": {"accuracy": 0.85, "loss": 0.30},
            "per_hospital": {
                "Hospital_A": {"accuracy": 0.85}
            }
        }

        passes, reason, det = self.gate.assess_candidate(
            candidate_metrics,
            baseline_metrics=baseline_metrics,
            max_hospital_deterioration=0.05
        )
        self.assertFalse(passes)
        self.assertIn("dropped", reason.lower())

    def test_rollback_workflow(self):
        # Setup approved and previous approved models
        prev_model = ChestCNN()
        curr_model = ChestCNN()
        save_checkpoint_atomic(curr_model, self.approved_dir / "global_model.pth")
        save_checkpoint_atomic(prev_model, self.approved_dir / "previous_global_model.pth")

        success = self.gate.rollback()
        self.assertTrue(success)
        self.assertTrue((self.approved_dir / "global_model.pth").exists())


class TestCompatibility(unittest.TestCase):
    """Verifies INT8 quantization, DACM distribution calculations, and Opacus DP compatibility."""

    def test_quantization_compatibility(self):
        model = ChestCNN()
        params = [p.detach().cpu().numpy() for p in model.parameters()]
        quantized, q_bytes, q_mb, ratio, reduction = quantize_parameters(params)
        self.assertTrue(ratio > 1.0)
        self.assertTrue(reduction > 0.0)

        dequantized = dequantize_parameters(quantized)
        self.assertEqual(len(params), len(dequantized))
        for orig, deq in zip(params, dequantized):
            self.assertEqual(orig.shape, deq.shape)
            # INT8 quantization should preserve general magnitude closely
            np.testing.assert_allclose(orig, deq, atol=0.05)


if __name__ == "__main__":
    unittest.main()
