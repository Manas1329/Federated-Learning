"""
model_approval.py
=================
Candidate model evaluation, approval gating, atomic promotion, and rollback workflow.
Enforces configurable experimental safeguards before promoting a candidate checkpoint
to the approved model registry.
"""

import os
import sys
import csv
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import torch

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from paths import APPROVED_MODELS_DIR, resolve_data_path
from model import ChestCNN, evaluate_metrics
from checkpoint_manager import load_raw_state_dict, save_checkpoint_atomic
from device_utils import get_device


class ModelApprovalGate:
    """
    Evaluates candidate checkpoints against baseline models and configurable thresholds.
    Handles atomic promotion, archiving of prior approved models, and audit logging.
    """

    def __init__(self, approved_dir: Optional[Path] = None):
        self.approved_dir = Path(approved_dir or APPROVED_MODELS_DIR)
        self.approved_dir.mkdir(parents=True, exist_ok=True)

        self.approved_model_path = self.approved_dir / "global_model.pth"
        self.previous_model_path = self.approved_dir / "previous_global_model.pth"
        self.approval_log_path = self.approved_dir / "approval_log.csv"

        self._ensure_log_exists()

    def _ensure_log_exists(self):
        if not self.approval_log_path.exists():
            with open(self.approval_log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "candidate_path",
                    "run_id",
                    "decision",  # PROMOTED or REJECTED
                    "reason",
                    "candidate_accuracy",
                    "candidate_recall",
                    "candidate_precision",
                    "candidate_loss",
                    "baseline_accuracy",
                    "baseline_loss",
                    "max_hospital_deterioration"
                ])

    def evaluate_checkpoint_on_data(
        self,
        checkpoint_path: Path,
        hospital_ids: list[str],
        split: str = "test",
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """Evaluate a checkpoint on specified hospital data split."""
        from utils import load_hospital_data

        target_device = device or get_device()
        state_dict, metadata = load_raw_state_dict(checkpoint_path, map_location="cpu")

        model = ChestCNN()
        model.load_state_dict(state_dict)
        model.to(target_device)
        model.eval()

        hospital_metrics = {}
        pooled_tp = 0
        pooled_tn = 0
        pooled_fp = 0
        pooled_fn = 0
        pooled_loss_sum = 0.0
        pooled_total = 0

        for h_id in hospital_ids:
            data_path = resolve_data_path(None, h_id)
            trainloader, valloader, testloader = load_hospital_data(data_path, batch_size=32)
            loader = testloader if split == "test" else valloader

            m = evaluate_metrics(model, loader, device=target_device)
            hospital_metrics[h_id] = m

            pooled_tp += m["tp"]
            pooled_tn += m["tn"]
            pooled_fp += m["fp"]
            pooled_fn += m["fn"]
            pooled_loss_sum += m["loss_sum"]
            pooled_total += m["total"]

        pooled_acc = (pooled_tp + pooled_tn) / pooled_total if pooled_total > 0 else 0.0
        pooled_prec = pooled_tp / (pooled_tp + pooled_fp) if (pooled_tp + pooled_fp) > 0 else 0.0
        pooled_rec = pooled_tp / (pooled_tp + pooled_fn) if (pooled_tp + pooled_fn) > 0 else 0.0
        pooled_f1 = (2 * pooled_prec * pooled_rec / (pooled_prec + pooled_rec)) if (pooled_prec + pooled_rec) > 0 else 0.0
        pooled_loss = pooled_loss_sum / pooled_total if pooled_total > 0 else 0.0

        return {
            "pooled": {
                "accuracy": pooled_acc,
                "precision": pooled_prec,
                "recall": pooled_rec,
                "f1": pooled_f1,
                "loss": pooled_loss,
                "total": pooled_total
            },
            "per_hospital": hospital_metrics
        }

    def assess_candidate(
        self,
        candidate_metrics: Dict[str, Any],
        baseline_metrics: Optional[Dict[str, Any]] = None,
        min_accuracy: float = 0.80,
        min_recall: float = 0.85,
        min_precision: float = 0.70,
        max_hospital_deterioration: float = 0.05,
        max_val_loss_increase: float = 0.10
    ) -> Tuple[bool, str, float]:
        """
        Check candidate against experimental thresholds and baseline metrics.
        Returns: (passes_criteria, reason, max_observed_deterioration)
        """
        cand_p = candidate_metrics["pooled"]
        acc = cand_p["accuracy"]
        rec = cand_p["recall"]
        prec = cand_p["precision"]
        loss = cand_p["loss"]

        if acc < min_accuracy:
            return False, f"Candidate accuracy {acc:.4f} below threshold {min_accuracy:.4f}", 0.0
        if rec < min_recall:
            return False, f"Candidate PNEUMONIA recall {rec:.4f} below threshold {min_recall:.4f}", 0.0
        if prec < min_precision:
            return False, f"Candidate precision {prec:.4f} below threshold {min_precision:.4f}", 0.0

        max_deterioration = 0.0
        if baseline_metrics:
            base_p = baseline_metrics["pooled"]
            if loss > base_p["loss"] + max_val_loss_increase:
                return False, f"Loss increase ({loss:.4f} vs base {base_p['loss']:.4f}) exceeds {max_val_loss_increase:.4f}", 0.0

            # Per-hospital deterioration check
            for h_id, cand_h in candidate_metrics.get("per_hospital", {}).items():
                if h_id in baseline_metrics.get("per_hospital", {}):
                    base_h_acc = baseline_metrics["per_hospital"][h_id]["accuracy"]
                    cand_h_acc = cand_h["accuracy"]
                    diff = base_h_acc - cand_h_acc
                    if diff > max_deterioration:
                        max_deterioration = diff
                    if diff > max_hospital_deterioration:
                        return False, f"Accuracy at {h_id} dropped by {diff:.4f} (exceeds allowed {max_hospital_deterioration:.4f})", max_deterioration

        return True, "All criteria satisfied", max_deterioration

    def decide_and_apply(
        self,
        candidate_path: Path,
        candidate_metrics: Dict[str, Any],
        run_id: str,
        baseline_metrics: Optional[Dict[str, Any]] = None,
        criteria: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, str]:
        """
        Evaluate candidate and apply promotion or rejection.
        """
        candidate_path = Path(candidate_path)
        crit = criteria or {}
        min_acc = crit.get("min_accuracy", float(os.environ.get("MODEL_MIN_ACCURACY", "0.80")))
        min_rec = crit.get("min_recall", float(os.environ.get("MODEL_MIN_PNEUMONIA_RECALL", "0.85")))
        min_prec = crit.get("min_precision", float(os.environ.get("MODEL_MIN_PRECISION", "0.70")))
        max_det = crit.get("max_hospital_deterioration", float(os.environ.get("MODEL_MAX_HOSPITAL_DETERIORATION", "0.05")))
        max_loss_inc = crit.get("max_loss_increase", float(os.environ.get("MODEL_MAX_VAL_LOSS_INCREASE", "0.10")))

        passes, reason, observed_det = self.assess_candidate(
            candidate_metrics,
            baseline_metrics=baseline_metrics,
            min_accuracy=min_acc,
            min_recall=min_rec,
            min_precision=min_prec,
            max_hospital_deterioration=max_det,
            max_val_loss_increase=max_loss_inc
        )

        cand_p = candidate_metrics["pooled"]
        base_acc = baseline_metrics["pooled"]["accuracy"] if baseline_metrics else ""
        base_loss = baseline_metrics["pooled"]["loss"] if baseline_metrics else ""

        if passes:
            # Promote candidate:
            # 1. If approved model exists, archive as previous_global_model.pth
            if self.approved_model_path.exists():
                shutil.copy2(str(self.approved_model_path), str(self.previous_model_path))

            # 2. Copy candidate atomically to approved_model_path
            state_dict, metadata = load_raw_state_dict(candidate_path)
            meta = dict(metadata)
            meta["promoted_at"] = datetime.utcnow().isoformat() + "Z"
            meta["promoted_from_run"] = run_id
            meta["candidate_metrics"] = cand_p
            save_checkpoint_atomic(state_dict, self.approved_model_path, metadata=meta)

            decision = "PROMOTED"
            print(f"\n[APPROVAL] Candidate {candidate_path} PROMOTED to Approved Global Model -> {self.approved_model_path}")
        else:
            decision = "REJECTED"
            print(f"\n[APPROVAL] Candidate {candidate_path} REJECTED. Reason: {reason}. Retaining prior approved model.")

        # Log decision
        with open(self.approval_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.utcnow().isoformat() + "Z",
                str(candidate_path),
                run_id,
                decision,
                reason,
                cand_p["accuracy"],
                cand_p["recall"],
                cand_p["precision"],
                cand_p["loss"],
                base_acc,
                base_loss,
                observed_det
            ])

        return passes, reason

    def rollback(self) -> bool:
        """Rollback to the previous approved model if available."""
        if not self.previous_model_path.exists():
            print("[ROLLBACK ERROR] No previous approved model exists to roll back to.")
            return False

        shutil.copy2(str(self.previous_model_path), str(self.approved_model_path))
        print(f"[ROLLBACK] Successfully restored previous approved model: {self.previous_model_path} -> {self.approved_model_path}")

        with open(self.approval_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.utcnow().isoformat() + "Z",
                str(self.previous_model_path),
                "MANUAL_ROLLBACK",
                "ROLLBACK_RESTORED",
                "Restored previous approved model",
                "", "", "", "", "", "", ""
            ])
        return True
