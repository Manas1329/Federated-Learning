import os
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class GlobalRoundMetric:
    round: int
    accuracy: float
    loss: float
    accuracy_percent: float


@dataclass
class ClientRoundMetric:
    client: str
    round: int
    record_type: str
    accuracy: Optional[float]
    loss: Optional[float]
    training_time_sec: Optional[float]
    epoch_time_sec: Optional[float]
    payload_size_mb: Optional[float]
    quantized_payload_size_mb: Optional[float]
    compression_ratio: Optional[float]
    device: Optional[str]
    epsilon: Optional[float]


@dataclass
class RoundInfo:
    round: int
    successful_clients: int
    failed_clients: int
    aggregation_time_sec: float
    total_round_time_sec: float


class MetricsReader:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._resolve_dirs()

    def _resolve_dirs(self):
        project_root = Path(__file__).resolve().parent.parent.parent
        self.RESULTS_DIR = project_root / "federated_healthcare" / "dashboard" / "results"
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def _get_suffix_dir(self, suffix: str) -> Path:
        return self.RESULTS_DIR / suffix

    def list_suffixes(self) -> List[str]:
        if not self.RESULTS_DIR.exists():
            return []
        return sorted([
            d.name for d in self.RESULTS_DIR.iterdir()
            if d.is_dir() and d.name in ("a_pure", "b_quantized", "c_dp")
        ])

    def read_global_metrics(self, suffix: str = "a_pure") -> List[GlobalRoundMetric]:
        suffix_dir = self._get_suffix_dir(suffix)
        metrics_file = suffix_dir / f"metrics_{suffix}.csv"
        results = []
        if not metrics_file.exists():
            return results
        try:
            with open(metrics_file, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        r = int(row.get("Round", 0))
                        acc_raw = float(row.get("Accuracy", 0))
                        loss = float(row.get("Loss", 0))
                        acc_pct = acc_raw * 100 if acc_raw <= 1.0 else acc_raw
                        results.append(GlobalRoundMetric(
                            round=r,
                            accuracy=acc_raw,
                            loss=loss,
                            accuracy_percent=acc_pct
                        ))
                    except (ValueError, TypeError):
                        continue
        except Exception:
            pass
        return results

    def read_round_info(self, suffix: str = "a_pure") -> List[RoundInfo]:
        suffix_dir = self._get_suffix_dir(suffix)
        round_file = suffix_dir / f"round_metrics_{suffix}.csv"
        results = []
        if not round_file.exists():
            return results
        try:
            with open(round_file, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        results.append(RoundInfo(
                            round=int(row.get("Round", 0)),
                            successful_clients=int(row.get("Successful_Clients", 0)),
                            failed_clients=int(row.get("Failed_Clients", 0)),
                            aggregation_time_sec=float(row.get("Aggregation_Time_sec", 0)),
                            total_round_time_sec=float(row.get("Total_Round_Time_sec", 0))
                        ))
                    except (ValueError, TypeError):
                        continue
        except Exception:
            pass
        return results

    def read_client_metrics(self, suffix: str, client_name: str) -> List[ClientRoundMetric]:
        suffix_dir = self._get_suffix_dir(suffix)
        csv_file = suffix_dir / f"{client_name}_{suffix}.csv"
        results = []
        if not csv_file.exists():
            return results
        try:
            with open(csv_file, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        def _f(v):
                            if v is None or v == "":
                                return None
                            try:
                                return float(v)
                            except (ValueError, TypeError):
                                return None
                        def _i(v):
                            if v is None or v == "":
                                return None
                            try:
                                return int(float(v))
                            except (ValueError, TypeError):
                                return None
                        results.append(ClientRoundMetric(
                            client=row.get("client", client_name),
                            round=int(row.get("round", 0) or 0),
                            record_type=row.get("record_type", ""),
                            accuracy=_f(row.get("accuracy")),
                            loss=_f(row.get("loss")),
                            training_time_sec=_f(row.get("training_time_sec")),
                            epoch_time_sec=_f(row.get("epoch_time_sec")),
                            payload_size_mb=_f(row.get("payload_size_mb")),
                            quantized_payload_size_mb=_f(row.get("quantized_payload_size_mb")),
                            compression_ratio=_f(row.get("compression_ratio")),
                            device=row.get("device"),
                            epsilon=_f(row.get("epsilon"))
                        ))
                    except (ValueError, TypeError):
                        continue
        except Exception:
            pass
        return results

    def get_latest_global_metrics(self, suffix: str = "a_pure") -> Optional[GlobalRoundMetric]:
        metrics = self.read_global_metrics(suffix)
        return metrics[-1] if metrics else None

    def get_current_suffix(self) -> str:
        suffixes = self.list_suffixes()
        if "c_dp" in suffixes:
            return "c_dp"
        if "b_quantized" in suffixes:
            return "b_quantized"
        return "a_pure"

    def get_active_suffix_from_env(self) -> str:
        use_dp = os.environ.get("USE_DP", "0") == "1"
        use_quant = os.environ.get("USE_QUANTIZATION", "1") == "1"
        if use_dp:
            return "c_dp"
        if use_quant:
            return "b_quantized"
        return "a_pure"

    def get_training_curve_data(self, suffix: Optional[str] = None) -> Dict[str, Any]:
        if suffix is None:
            suffix = self.get_active_suffix_from_env()
        global_metrics = self.read_global_metrics(suffix)
        round_info = self.read_round_info(suffix)
        return {
            "suffix": suffix,
            "rounds": [m.round for m in global_metrics],
            "accuracies": [round(m.accuracy_percent, 2) for m in global_metrics],
            "losses": [round(m.loss, 4) for m in global_metrics],
            "latest": {
                "round": global_metrics[-1].round if global_metrics else 0,
                "accuracy": global_metrics[-1].accuracy_percent if global_metrics else 0,
                "loss": global_metrics[-1].loss if global_metrics else 0
            },
            "round_info": [
                {
                    "round": r.round,
                    "successful": r.successful_clients,
                    "failed": r.failed_clients,
                    "time_sec": round(r.total_round_time_sec, 2)
                }
                for r in round_info
            ]
        }

    def get_hospital_training_curve(self, hospital_code: str,
                                     suffix: Optional[str] = None) -> Dict[str, Any]:
        if suffix is None:
            suffix = self.get_active_suffix_from_env()
        client_metrics = self.read_client_metrics(suffix, hospital_code)
        training_rows = [r for r in client_metrics if r.record_type == "training"]
        eval_rows = [r for r in client_metrics if r.record_type == "evaluation"]
        rounds_seen = sorted(set(r.round for r in client_metrics))
        eval_by_round = {r.round: r for r in eval_rows}
        return {
            "suffix": suffix,
            "hospital_code": hospital_code,
            "rounds": rounds_seen,
            "training_times": [
                next((round(r.training_time_sec, 2) for r in training_rows if r.round == rn), None)
                for rn in rounds_seen
            ],
            "accuracies": [
                round(eval_by_round[rn].accuracy * 100, 2)
                if rn in eval_by_round and eval_by_round[rn].accuracy is not None
                else None
                for rn in rounds_seen
            ],
            "losses": [
                round(eval_by_round[rn].loss, 4)
                if rn in eval_by_round and eval_by_round[rn].loss is not None
                else None
                for rn in rounds_seen
            ],
            "payload_sizes_mb": [
                next((round(r.payload_size_mb, 3) for r in training_rows if r.round == rn and r.payload_size_mb is not None), None)
                for rn in rounds_seen
            ],
            "quantized_sizes_mb": [
                next((round(r.quantized_payload_size_mb, 3) for r in training_rows if r.round == rn and r.quantized_payload_size_mb is not None), None)
                for rn in rounds_seen
            ]
        }


def get_metrics_reader() -> MetricsReader:
    return MetricsReader()
