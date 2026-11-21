"""
model.py
========
ChestCNN architecture, training, and evaluation for Federated Healthcare Learning.
Provides device-consistent execution, per-sample loss accumulation, and
single-pass validation/test metrics computation.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus import PrivacyEngine

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from device_utils import get_device

# Central default device
device = get_device()

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


class ChestCNN(nn.Module):
    """Convolutional Neural Network for binary chest X-ray pneumonia classification."""
    def __init__(self):
        super(ChestCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # For 128x128 input
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train(
    net: nn.Module,
    trainloader: torch.utils.data.DataLoader,
    epochs: int,
    class_weight: float = 1.0,
    device: Optional[torch.device] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr: float = 0.001
):
    """
    Standard supervised local training.
    The optimizer is created once and reused across all local epochs within this fit call,
    then reset at the beginning of the next federated round.
    """
    target_device = device or get_device()

    if class_weight > 1.0:
        weights = torch.tensor([1.0, float(class_weight)], dtype=torch.float32).to(target_device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    if optimizer is None:
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    net.to(target_device)
    net.train()

    for _ in range(epochs):
        for images, labels in trainloader:
            images = images.to(target_device)
            labels = labels.to(target_device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            del outputs, loss, images, labels

        if target_device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return optimizer


def train_dp(
    private_net: nn.Module,
    optimizer: torch.optim.Optimizer,
    trainloader: torch.utils.data.DataLoader,
    privacy_engine: PrivacyEngine,
    epochs: int = 1,
    delta: float = 1e-5,
    class_weight: float = 1.0,
    device: Optional[torch.device] = None
) -> float:
    """
    DP-SGD training using Opacus PrivacyEngine.
    Global model parameters are loaded before training, and the private model
    and optimizer remain connected across local epochs.
    """
    target_device = device or get_device()

    if class_weight > 1.0:
        weights = torch.tensor([1.0, float(class_weight)], dtype=torch.float32).to(target_device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    private_net.to(target_device)
    private_net.train()

    for _ in range(epochs):
        for images, labels in trainloader:
            images = images.to(target_device)
            labels = labels.to(target_device)

            optimizer.zero_grad()
            outputs = private_net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            del outputs, loss, images, labels

        if target_device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    epsilon = privacy_engine.get_epsilon(delta=delta)
    return float(epsilon)


def evaluate_metrics(
    net: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Single-pass comprehensive evaluation.
    Accumulates per-example loss and computes confusion matrix components
    (TP, TN, FP, FN) to safely calculate pooled global metrics.
    """
    target_device = device or get_device()
    criterion = nn.CrossEntropyLoss(reduction="sum")

    net.to(target_device)
    net.eval()

    correct = 0
    total = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    loss_sum = 0.0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(target_device)
            labels = labels.to(target_device)

            outputs = net(images)
            batch_loss = criterion(outputs, labels).item()
            loss_sum += batch_loss

            _, predicted = torch.max(outputs, 1)

            batch_size = labels.size(0)
            total += batch_size
            correct += int((predicted == labels).sum().item())

            # Binary classification: NORMAL=0 (Negative), PNEUMONIA=1 (Positive)
            tp += int(((predicted == 1) & (labels == 1)).sum().item())
            tn += int(((predicted == 0) & (labels == 0)).sum().item())
            fp += int(((predicted == 1) & (labels == 0)).sum().item())
            fn += int(((predicted == 0) & (labels == 1)).sum().item())

            del outputs, images, labels

    sample_count = total
    avg_loss = (loss_sum / sample_count) if sample_count > 0 else 0.0
    accuracy = (correct / total) if total > 0 else 0.0

    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "correct": int(correct),
        "total": int(total),
        "sample_count": int(sample_count),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "loss_sum": float(loss_sum),
        "loss": float(avg_loss),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }


def test(
    net: nn.Module,
    testloader: torch.utils.data.DataLoader,
    device: Optional[torch.device] = None
) -> Tuple[float, float]:
    """
    Backward-compatible evaluation function returning (loss, accuracy).
    Uses per-example loss accumulation.
    """
    metrics = evaluate_metrics(net, testloader, device=device)
    return metrics["loss"], metrics["accuracy"]


# Prevent pytest from collecting this backwards-compatible production helper
# when another test module imports it.
test.__test__ = False
