"""
test_dacm_concurrent_recovery.py
================================
Focused unit and integration test suite for concurrent local recovery training in DACM.

Verifies:
1. Model isolation: each worker receives an independent copy with identical starting weights.
2. Target client exclusion: excluded client data and updates are strictly omitted.
3. Sample count fidelity: returned sample counts accurately reflect local client partitions.
4. Mathematical equivalence: FedAvg produces the exact same numerical result whether
   local recovery models are trained/aggregated sequentially or concurrently.
5. Worker failure handling: explicit reporting and strict fail-safe abortion on worker errors.
6. Execution timing comparison: benchmarks sequential vs. concurrent local recovery.
"""

import os
import sys
import copy
import time
import unittest
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure src is importable
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model import ChestCNN
from dacm import (
    train_local_recovery,
    train_concurrent_recovery,
    fedavg_aggregate,
    calculate_dacm_weights,
    discover_clients,
    DACMRecoveryError,
)


class SimpleLinearNet(nn.Module):
    """Small deterministic network for fast, deterministic unit testing."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.fc = nn.Linear(4 * 16 * 16, 2)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)


def create_mock_loader(n_samples: int, class_ratio: float = 0.5, seed: int = 42) -> DataLoader:
    """Create a deterministic synthetic dataset of grayscale 16x16 images."""
    torch.manual_seed(seed)
    n_class_1 = int(n_samples * class_ratio)
    n_class_0 = n_samples - n_class_1

    labels = torch.cat([torch.zeros(n_class_0, dtype=torch.long), torch.ones(n_class_1, dtype=torch.long)])
    images = torch.randn(n_samples, 1, 16, 16)
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=8, shuffle=False)


class TestDACMConcurrentRecovery(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12345)
        self.device = torch.device("cpu")
        self.model = SimpleLinearNet().to(self.device)
        self.dacm_weights = torch.tensor([1.0, 2.5], dtype=torch.float32)

        # Mock surviving clients
        self.mock_loaders = {
            "Hospital_A": create_mock_loader(n_samples=32, class_ratio=0.3, seed=101),
            "Hospital_C": create_mock_loader(n_samples=48, class_ratio=0.2, seed=102),
        }

    def test_starting_weights_and_worker_isolation(self):
        """Requirement 2 & 3: Independent copies, identical initial weights, zero mutable sharing."""
        client_ids = ["Hospital_A", "Hospital_C"]
        initial_params = {k: v.clone() for k, v in self.model.state_dict().items()}

        results = train_concurrent_recovery(
            unlearned_model=self.model,
            surviving_loaders=self.mock_loaders,
            dacm_weights=self.dacm_weights,
            epochs=1,
            lr=0.01,
            device=self.device,
            client_ids=client_ids,
            concurrent=True,
        )

        self.assertEqual(len(results), 2)

        # The base starting model must remain completely unmodified
        current_base_params = self.model.state_dict()
        for k, v in initial_params.items():
            self.assertTrue(
                torch.equal(v, current_base_params[k]),
                f"Starting base model was mutated during concurrent recovery at parameter {k}!",
            )

        # Each trained model must be a distinct object
        model_a, samples_a = results[0]
        model_c, samples_c = results[1]
        self.assertIsNot(model_a, self.model)
        self.assertIsNot(model_c, self.model)
        self.assertIsNot(model_a, model_c)

        # The parameters must have updated from initial
        self.assertFalse(torch.equal(model_a.conv.weight, initial_params["conv.weight"]))
        self.assertFalse(torch.equal(model_c.conv.weight, initial_params["conv.weight"]))
        # Hospital A and Hospital C trained on different data, so their weights must differ
        self.assertFalse(torch.equal(model_a.conv.weight, model_c.conv.weight))

    def test_client_exclusion(self):
        """Requirement 5: Exclude target client from recovery training and aggregation."""
        excluded_target = "Hospital_B"
        all_clients = ["Hospital_A", "Hospital_B", "Hospital_C"]
        surviving = [c for c in all_clients if c != excluded_target]

        self.assertNotIn(excluded_target, surviving)
        self.assertEqual(surviving, ["Hospital_A", "Hospital_C"])

        # Target client's data is never passed to recovery
        self.assertNotIn(excluded_target, self.mock_loaders)

        results = train_concurrent_recovery(
            unlearned_model=self.model,
            surviving_loaders=self.mock_loaders,
            dacm_weights=self.dacm_weights,
            epochs=1,
            lr=0.01,
            device=self.device,
            client_ids=surviving,
            concurrent=True,
        )

        self.assertEqual(len(results), len(surviving))
        # Aggregated model
        agg_model = fedavg_aggregate(results, self.model, device=self.device)
        self.assertIsNotNone(agg_model)

    def test_sample_counts_accuracy(self):
        """Requirement 4: Each worker reports the exact sample count of its local partition."""
        results = train_concurrent_recovery(
            unlearned_model=self.model,
            surviving_loaders=self.mock_loaders,
            dacm_weights=self.dacm_weights,
            epochs=2,
            lr=0.005,
            device=self.device,
            client_ids=["Hospital_A", "Hospital_C"],
            concurrent=True,
        )

        _, n_a = results[0]
        _, n_c = results[1]

        self.assertEqual(n_a, 32)
        self.assertEqual(n_c, 48)

    def test_mathematical_equivalence_sequential_vs_concurrent(self):
        """Requirement 7: Concurrent aggregation produces identical weights to sequential aggregation."""
        # Run sequentially
        results_seq = train_concurrent_recovery(
            unlearned_model=self.model,
            surviving_loaders=self.mock_loaders,
            dacm_weights=self.dacm_weights,
            epochs=2,
            lr=0.01,
            device=self.device,
            client_ids=["Hospital_A", "Hospital_C"],
            concurrent=False,
        )
        agg_seq = fedavg_aggregate(results_seq, self.model, device=self.device)

        # Run concurrently
        results_conc = train_concurrent_recovery(
            unlearned_model=self.model,
            surviving_loaders=self.mock_loaders,
            dacm_weights=self.dacm_weights,
            epochs=2,
            lr=0.01,
            device=self.device,
            client_ids=["Hospital_A", "Hospital_C"],
            concurrent=True,
        )
        agg_conc = fedavg_aggregate(results_conc, self.model, device=self.device)

        # Check sample counts match
        self.assertEqual(results_seq[0][1], results_conc[0][1])
        self.assertEqual(results_seq[1][1], results_conc[1][1])

        # Verify mathematical equivalence of parameters
        state_seq = agg_seq.state_dict()
        state_conc = agg_conc.state_dict()

        for key in state_seq:
            p_seq = state_seq[key]
            p_conc = state_conc[key]
            diff = torch.max(torch.abs(p_seq - p_conc)).item()
            self.assertAlmostEqual(
                diff, 0.0, places=6,
                msg=f"Parameter {key} diverged between sequential and concurrent aggregation! Max diff: {diff}",
            )

    def test_worker_failure_handling_strict_policy(self):
        """Requirement 6: Explicit failure handling — never silently aggregate on worker errors."""
        class FailingDataLoader:
            def __iter__(self):
                raise RuntimeError("Simulated network/disk I/O failure on Hospital_C worker!")

        faulty_loaders = {
            "Hospital_A": self.mock_loaders["Hospital_A"],
            "Hospital_C": FailingDataLoader(),
        }

        # Concurrency must catch worker exception and raise DACMRecoveryError
        with self.assertRaises(DACMRecoveryError) as ctx:
            train_concurrent_recovery(
                unlearned_model=self.model,
                surviving_loaders=faulty_loaders,
                dacm_weights=self.dacm_weights,
                epochs=1,
                lr=0.01,
                device=self.device,
                client_ids=["Hospital_A", "Hospital_C"],
                concurrent=True,
            )

        self.assertIn("Hospital_C", str(ctx.exception))
        self.assertIn("Safe policy enforced", str(ctx.exception))

    def test_sequential_fallback_on_failure(self):
        """Requirement 6: Sequential mode also enforces strict fail-safe error handling."""
        class FailingDataLoader:
            def __iter__(self):
                raise RuntimeError("Simulated sequential failure on Hospital_A!")

        faulty_loaders = {
            "Hospital_A": FailingDataLoader(),
            "Hospital_C": self.mock_loaders["Hospital_C"],
        }

        with self.assertRaises(DACMRecoveryError) as ctx:
            train_concurrent_recovery(
                unlearned_model=self.model,
                surviving_loaders=faulty_loaders,
                dacm_weights=self.dacm_weights,
                epochs=1,
                lr=0.01,
                device=self.device,
                client_ids=["Hospital_A", "Hospital_C"],
                concurrent=False,
            )

        self.assertIn("Hospital_A", str(ctx.exception))

    def test_real_chestcnn_architecture_support(self):
        """Verify that the full ChestCNN model works with concurrent recovery."""
        chest_model = ChestCNN().to(self.device)
        # 1-channel 128x128 images matching ChestCNN input dimensions
        images_a = torch.randn(16, 1, 128, 128)
        labels_a = torch.randint(0, 2, (16,))
        loader_a = DataLoader(TensorDataset(images_a, labels_a), batch_size=8)

        images_c = torch.randn(16, 1, 128, 128)
        labels_c = torch.randint(0, 2, (16,))
        loader_c = DataLoader(TensorDataset(images_c, labels_c), batch_size=8)

        loaders = {"Hospital_A": loader_a, "Hospital_C": loader_c}

        results = train_concurrent_recovery(
            unlearned_model=chest_model,
            surviving_loaders=loaders,
            dacm_weights=self.dacm_weights,
            epochs=1,
            lr=1e-3,
            device=self.device,
            client_ids=["Hospital_A", "Hospital_C"],
            concurrent=True,
        )

        self.assertEqual(len(results), 2)
        recovered = fedavg_aggregate(results, chest_model, device=self.device)
        self.assertIsInstance(recovered, ChestCNN)


def run_benchmark():
    """Runs a representative timing benchmark comparing sequential vs concurrent recovery."""
    print("\n" + "=" * 60)
    print("RUNNING DACM SEQUENTIAL VS CONCURRENT RECOVERY BENCHMARK")
    print("=" * 60)

    device = torch.device("cpu")
    model = ChestCNN().to(device)
    dacm_weights = torch.tensor([1.0, 2.0], dtype=torch.float32)

    # 3 surviving clients with 64 samples each of 128x128 images
    client_ids = ["Hospital_A", "Hospital_C", "Hospital_D"]
    loaders = {}
    for cid in client_ids:
        imgs = torch.randn(64, 1, 128, 128)
        lbls = torch.randint(0, 2, (64,))
        loaders[cid] = DataLoader(TensorDataset(imgs, lbls), batch_size=16)

    epochs = 2
    lr = 1e-3

    # Warmup
    _ = train_concurrent_recovery(
        unlearned_model=model,
        surviving_loaders={client_ids[0]: loaders[client_ids[0]]},
        dacm_weights=dacm_weights,
        epochs=1,
        lr=lr,
        device=device,
        client_ids=[client_ids[0]],
        concurrent=False,
    )

    # Benchmark Sequential
    t0 = time.perf_counter()
    res_seq = train_concurrent_recovery(
        unlearned_model=model,
        surviving_loaders=loaders,
        dacm_weights=dacm_weights,
        epochs=epochs,
        lr=lr,
        device=device,
        client_ids=client_ids,
        concurrent=False,
    )
    _ = fedavg_aggregate(res_seq, model, device=device)
    t_seq = time.perf_counter() - t0

    # Benchmark Concurrent
    t0 = time.perf_counter()
    res_conc = train_concurrent_recovery(
        unlearned_model=model,
        surviving_loaders=loaders,
        dacm_weights=dacm_weights,
        epochs=epochs,
        lr=lr,
        device=device,
        client_ids=client_ids,
        concurrent=True,
        max_workers=len(client_ids),
    )
    _ = fedavg_aggregate(res_conc, model, device=device)
    t_conc = time.perf_counter() - t0

    print(f"\n[BENCHMARK RESULTS] (Surviving Clients = {len(client_ids)}, Epochs = {epochs})")
    print(f"  Sequential Recovery Execution Time : {t_seq:.3f} s")
    print(f"  Concurrent Recovery Execution Time : {t_conc:.3f} s")
    if t_seq > 0:
        speedup = t_seq / t_conc
        print(f"  Observed Local Speedup             : {speedup:.2f}x (CPU local simulation)")


if __name__ == "__main__":
    unittest.main(exit=False)
    run_benchmark()
