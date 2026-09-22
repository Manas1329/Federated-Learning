import os
import sys
import subprocess
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class ProcessInfo:
    process_id: str
    process_type: str
    hospital_id: Optional[int] = None
    hospital_code: Optional[str] = None
    started_at: datetime = field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    status: str = "starting"
    return_code: Optional[int] = None
    log_buffer: List[str] = field(default_factory=list)
    subscribers: List[Callable] = field(default_factory=list)

    def to_dict(self):
        return {
            "process_id": self.process_id,
            "process_type": self.process_type,
            "hospital_id": self.hospital_id,
            "hospital_code": self.hospital_code,
            "started_at": self.started_at.isoformat() + "Z",
            "ended_at": self.ended_at.isoformat() + "Z"
            if self.ended_at else None,
            "status": self.status,
            "return_code": self.return_code,
            "log_tail": self.log_buffer[-100:]
            if len(self.log_buffer) > 100 else self.log_buffer
        }


class ProcessManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self.processes: Dict[str, ProcessInfo] = {}
        self._subprocesses: Dict[str, subprocess.Popen] = {}
        self._threads: Dict[str, threading.Thread] = {}
        self._global_callbacks: List[
            Callable[[ProcessInfo, str], None]
        ] = []
        self._python_bin = sys.executable
        self._reset_seed_cmd = (
            "from webapp.database.init_db import init_db; "
            "from webapp.database.connection import Base, engine; "
            "Base.metadata.drop_all(bind=engine); "
            "Base.metadata.create_all(bind=engine); "
            "init_db(reset_first=True)"
        )
        self._seed_cmd = (
            "from webapp.database.init_db import init_db; "
            "from webapp.database.connection import Base, engine; "
            "Base.metadata.create_all(bind=engine); "
            "init_db(reset_first=False)"
        )

    def register_global_callback(
        self,
        callback: Callable[[ProcessInfo, str], None]
    ):
        self._global_callbacks.append(callback)

    def _resolve_paths(self):
        project_root = Path(__file__).resolve().parent.parent.parent
        src_dir = project_root / "federated_healthcare"
        src_pkg = src_dir / "src"
        return src_dir, project_root, src_pkg

    def _build_env(
        self,
        extra_env: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:

        env = os.environ.copy()
        src_dir, project_root, src_pkg = self._resolve_paths()

        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = (
                str(project_root)
                + os.pathsep
                + str(src_dir)
                + os.pathsep
                + env["PYTHONPATH"]
            )
        else:
            env["PYTHONPATH"] = str(project_root) + os.pathsep + str(src_dir)

        try:
            from .runtime_settings import get_runtime_settings

            runtime = get_runtime_settings()
            for k, v in runtime.as_env_overrides().items():
                env[k] = v
        except Exception:
            pass

        if extra_env:
            for k, v in extra_env.items():
                env[k] = "" if v is None else str(v)

        return env

    def _stream_output(self, process_id: str, pipe, stream_name: str):
        proc_info = self.processes[process_id]

        try:
            for raw_line in iter(pipe.readline, ""):
                if raw_line:
                    line = raw_line.rstrip("\n").rstrip("\r")

                    if line:
                        timestamp = datetime.utcnow().isoformat() + "Z"
                        log_entry = (
                            f"[{timestamp}] "
                            f"[{stream_name}] "
                            f"{line}"
                        )

                        proc_info.log_buffer.append(log_entry)

                        if len(proc_info.log_buffer) > 2000:
                            proc_info.log_buffer = (
                                proc_info.log_buffer[-2000:]
                            )

                        for cb in proc_info.subscribers:
                            try:
                                cb(proc_info, log_entry)
                            except Exception:
                                pass

                        for gcb in self._global_callbacks:
                            try:
                                gcb(proc_info, log_entry)
                            except Exception:
                                pass

        except Exception as e:
            err_line = f"[Stream error: {e}]"
            proc_info.log_buffer.append(err_line)

        finally:
            try:
                pipe.close()
            except Exception:
                pass

    def _monitor_process(self, process_id: str):
        proc = self._subprocesses.get(process_id)
        proc_info = self.processes.get(process_id)

        if not proc or not proc_info:
            return

        proc_info.status = "running"

        try:
            return_code = proc.wait()

            proc_info.return_code = return_code
            proc_info.ended_at = datetime.utcnow()
            proc_info.status = (
                "completed" if return_code == 0 else "failed"
            )

            end_msg = (
                f"Process {process_id} ended with code {return_code}"
            )

            proc_info.log_buffer.append(
                f"[{datetime.utcnow().isoformat()}Z] "
                f"[SYSTEM] {end_msg}"
            )

            for cb in proc_info.subscribers:
                try:
                    cb(proc_info, end_msg)
                except Exception:
                    pass

            for gcb in self._global_callbacks:
                try:
                    gcb(proc_info, end_msg)
                except Exception:
                    pass

        except Exception as e:
            proc_info.status = "error"
            proc_info.ended_at = datetime.utcnow()
            proc_info.log_buffer.append(
                f"[Monitor error: {e}]"
            )

    def start_split_data(
        self,
        extra_env: Optional[Dict[str, str]] = None
    ) -> ProcessInfo:

        src_dir, project_root, src_pkg = self._resolve_paths()
        script_path = src_dir / "split_data.py"

        process_id = f"split-{uuid.uuid4().hex[:8]}"

        proc_info = ProcessInfo(
            process_id=process_id,
            process_type="split_data"
        )

        self.processes[process_id] = proc_info
        env = self._build_env(extra_env)

        try:
            proc = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=str(project_root),
                text=True,
                bufsize=1
            )

            self._subprocesses[process_id] = proc

            t_out = threading.Thread(
                target=self._stream_output,
                args=(process_id, proc.stdout, "OUT"),
                daemon=True
            )

            t_err = threading.Thread(
                target=self._stream_output,
                args=(process_id, proc.stderr, "ERR"),
                daemon=True
            )

            t_out.start()
            t_err.start()

            t_mon = threading.Thread(
                target=self._monitor_process,
                args=(process_id,),
                daemon=True
            )

            t_mon.start()

            self._threads[process_id] = t_mon
            proc_info.status = "running"

        except Exception as e:
            proc_info.status = "failed"
            proc_info.log_buffer.append(
                f"[Launch error: {e}]"
            )
            raise

        return proc_info

    def start_server(
        self,
        target_clients: int = 3,
        min_clients: int = 2,
        num_rounds: int = 20,
        extra_env: Optional[Dict[str, str]] = None
    ) -> ProcessInfo:

        src_dir, project_root, src_pkg = self._resolve_paths()
        script_path = src_pkg / "server.py"

        existing_server = self.get_processes_by_type("fl_server")

        for p in existing_server:
            if p.status in ("running", "starting"):
                raise RuntimeError(
                    f"FL Server already running "
                    f"(pid {p.process_id})"
                )

        process_id = f"server-{uuid.uuid4().hex[:8]}"

        proc_info = ProcessInfo(
            process_id=process_id,
            process_type="fl_server"
        )

        self.processes[process_id] = proc_info

        env = self._build_env(extra_env)
        env["TARGET_CLIENTS"] = str(target_clients)
        env["MIN_CLIENTS"] = str(min_clients)
        env["NUM_ROUNDS"] = str(num_rounds)

        try:
            proc = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=str(project_root),
                text=True,
                bufsize=1
            )

            self._subprocesses[process_id] = proc

            t_out = threading.Thread(
                target=self._stream_output,
                args=(process_id, proc.stdout, "OUT"),
                daemon=True
            )

            t_err = threading.Thread(
                target=self._stream_output,
                args=(process_id, proc.stderr, "ERR"),
                daemon=True
            )

            t_out.start()
            t_err.start()

            t_mon = threading.Thread(
                target=self._monitor_process,
                args=(process_id,),
                daemon=True
            )

            t_mon.start()

            self._threads[process_id] = t_mon
            proc_info.status = "running"

        except Exception as e:
            proc_info.status = "failed"
            proc_info.log_buffer.append(
                f"[Launch error: {e}]"
            )
            raise

        return proc_info

    def start_client(
        self,
        hospital_code: str,
        hospital_id: Optional[int] = None,
        data_path: Optional[str] = None,
        server_address: str = "localhost:8080",
        use_quantization: bool = True,
        use_dp: bool = False,
        force_cpu: bool = False,
        extra_env: Optional[Dict[str, str]] = None
    ) -> ProcessInfo:

        src_dir, project_root, src_pkg = self._resolve_paths()
        script_path = src_pkg / "client.py"

        process_id = (
            f"client-{hospital_code.lower()}-"
            f"{uuid.uuid4().hex[:6]}"
        )

        proc_info = ProcessInfo(
            process_id=process_id,
            process_type="fl_client",
            hospital_id=hospital_id,
            hospital_code=hospital_code
        )

        self.processes[process_id] = proc_info

        env = self._build_env(extra_env)

        env["CLIENT_NAME"] = hospital_code
        env["SERVER_ADDRESS"] = server_address
        env["USE_QUANTIZATION"] = (
            "1" if use_quantization else "0"
        )
        env["USE_DP"] = "1" if use_dp else "0"
        env["FORCE_CPU"] = "1" if force_cpu else "0"

        if data_path:
            env["DATA_PATH"] = str(data_path)
        else:
            data_dir = (
                project_root
                / "data"
                / hospital_code.lower()
            )
            env["DATA_PATH"] = str(data_dir)

        try:
            proc = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=str(project_root),
                text=True,
                bufsize=1
            )

            self._subprocesses[process_id] = proc

            t_out = threading.Thread(
                target=self._stream_output,
                args=(process_id, proc.stdout, "OUT"),
                daemon=True
            )

            t_err = threading.Thread(
                target=self._stream_output,
                args=(process_id, proc.stderr, "ERR"),
                daemon=True
            )

            t_out.start()
            t_err.start()

            t_mon = threading.Thread(
                target=self._monitor_process,
                args=(process_id,),
                daemon=True
            )

            t_mon.start()

            self._threads[process_id] = t_mon
            proc_info.status = "running"

        except Exception as e:
            proc_info.status = "failed"
            proc_info.log_buffer.append(
                f"[Launch error: {e}]"
            )
            raise

        return proc_info

    def start_evaluate(
        self,
        extra_env: Optional[Dict[str, str]] = None
    ) -> ProcessInfo:

        src_dir, project_root, src_pkg = self._resolve_paths()
        script_path = src_pkg / "evaluate.py"

        process_id = f"eval-{uuid.uuid4().hex[:8]}"

        proc_info = ProcessInfo(
            process_id=process_id,
            process_type="evaluate"
        )

        self.processes[process_id] = proc_info
        env = self._build_env(extra_env)

        try:
            proc = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=str(project_root),
                text=True,
                bufsize=1
            )

            self._subprocesses[process_id] = proc

            t_out = threading.Thread(
                target=self._stream_output,
                args=(process_id, proc.stdout, "OUT"),
                daemon=True
            )

            t_err = threading.Thread(
                target=self._stream_output,
                args=(process_id, proc.stderr, "ERR"),
                daemon=True
            )

            t_out.start()
            t_err.start()

            t_mon = threading.Thread(
                target=self._monitor_process,
                args=(process_id,),
                daemon=True
            )

            t_mon.start()

            self._threads[process_id] = t_mon
            proc_info.status = "running"

        except Exception as e:
            proc_info.status = "failed"
            proc_info.log_buffer.append(
                f"[Launch error: {e}]"
            )
            raise

        return proc_info

    def initialize_global_model(self) -> ProcessInfo:

        src_dir, project_root, src_pkg = self._resolve_paths()

        models_dir = src_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        process_id = f"init-model-{uuid.uuid4().hex[:8]}"

        proc_info = ProcessInfo(
            process_id=process_id,
            process_type="init_model"
        )

        self.processes[process_id] = proc_info

        try:
            import torch

            sys.path.insert(0, str(src_dir))
            sys.path.insert(0, str(project_root))

            from federated_healthcare.src.model import ChestCNN
            from federated_healthcare.src.paths import MODELS_DIR

            proc_info.log_buffer.append(
                f"[{datetime.utcnow().isoformat()}Z] "
                f"[SYSTEM] Initializing global ChestCNN architecture..."
            )

            model = ChestCNN()

            proc_info.log_buffer.append(
                f"[{datetime.utcnow().isoformat()}Z] "
                f"[SYSTEM] Model architecture created"
            )

            total_params = sum(
                p.numel() for p in model.parameters()
            )

            trainable_params = sum(
                p.numel()
                for p in model.parameters()
                if p.requires_grad
            )

            proc_info.log_buffer.append(
                f"[{datetime.utcnow().isoformat()}Z] "
                f"[SYSTEM] Parameters: "
                f"{total_params:,} total, "
                f"{trainable_params:,} trainable"
            )

            for suffix, path_suffix in [
                ("a_pure", "global_model_a_pure.pth"),
                ("b_quantized", "global_model_b_quantized.pth"),
                ("c_dp", "global_model_c_dp.pth")
            ]:

                model_path = MODELS_DIR / path_suffix

                torch.save(
                    model.state_dict(),
                    str(model_path)
                )

                proc_info.log_buffer.append(
                    f"[{datetime.utcnow().isoformat()}Z] "
                    f"[SYSTEM] Saved initial weights -> "
                    f"{model_path}"
                )

            proc_info.status = "completed"
            proc_info.ended_at = datetime.utcnow()
            proc_info.return_code = 0

        except Exception as e:
            proc_info.status = "failed"
            proc_info.ended_at = datetime.utcnow()
            proc_info.log_buffer.append(
                f"[Init error: {e}]"
            )
            raise

        return proc_info

    def stop_process(self, process_id: str) -> bool:

        proc = self._subprocesses.get(process_id)
        proc_info = self.processes.get(process_id)

        if not proc or not proc_info:
            return False

        if proc_info.status in (
            "completed",
            "failed",
            "stopped"
        ):
            return True

        try:
            proc.terminate()

            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)

        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

        proc_info.status = "stopped"
        proc_info.ended_at = datetime.utcnow()

        proc_info.log_buffer.append(
            f"[{datetime.utcnow().isoformat()}Z] "
            f"[SYSTEM] Process terminated by user"
        )

        return True

    def get_process(
        self,
        process_id: str
    ) -> Optional[ProcessInfo]:

        return self.processes.get(process_id)

    def get_processes_by_type(
        self,
        process_type: str
    ) -> List[ProcessInfo]:

        return [
            p for p in self.processes.values()
            if p.process_type == process_type
        ]

    def get_processes_by_hospital(
        self,
        hospital_id: int
    ) -> List[ProcessInfo]:

        return [
            p for p in self.processes.values()
            if p.hospital_id == hospital_id
        ]

    def list_processes(self) -> List[ProcessInfo]:
        return list(self.processes.values())

    def subscribe(
        self,
        process_id: str,
        callback: Callable[[ProcessInfo, str], None]
    ) -> bool:

        proc_info = self.processes.get(process_id)

        if not proc_info:
            return False

        proc_info.subscribers.append(callback)
        return True

    def _launch_script(
        self,
        script_path: Path,
        process_id: str,
        process_type: str,
        hospital_id: Optional[int] = None,
        hospital_code: Optional[str] = None,
        script_args: Optional[List[str]] = None,
        extra_env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None
    ) -> ProcessInfo:
        src_dir, project_root, src_pkg = self._resolve_paths()
        effective_cwd = cwd if cwd else str(project_root)

        proc_info = ProcessInfo(
            process_id=process_id,
            process_type=process_type,
            hospital_id=hospital_id,
            hospital_code=hospital_code
        )
        self.processes[process_id] = proc_info
        env = self._build_env(extra_env)

        command = [self._python_bin, str(script_path)]
        if script_args:
            command.extend([str(a) for a in script_args])

        try:
            proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=effective_cwd,
                text=True,
                bufsize=1,
                shell=False
            )
        except Exception as e:
            proc_info.status = "failed"
            proc_info.log_buffer.append(f"[Launch error: {e}]")
            raise

        self._subprocesses[process_id] = proc
        t_out = threading.Thread(
            target=self._stream_output,
            args=(process_id, proc.stdout, "OUT"),
            daemon=True
        )
        t_err = threading.Thread(
            target=self._stream_output,
            args=(process_id, proc.stderr, "ERR"),
            daemon=True
        )
        t_mon = threading.Thread(
            target=self._monitor_process,
            args=(process_id,),
            daemon=True
        )
        t_out.start()
        t_err.start()
        t_mon.start()
        self._threads[process_id] = t_mon
        proc_info.status = "running"
        return proc_info

    def start_seed_database(
        self,
        reset_first: bool = False,
        extra_env: Optional[Dict[str, str]] = None
    ) -> ProcessInfo:
        src_dir, project_root, src_pkg = self._resolve_paths()
        process_id = f"seed-db-{uuid.uuid4().hex[:8]}"
        script = self._reset_seed_cmd if reset_first else self._seed_cmd
        return self._launch_snippet(
            snippet=script,
            process_id=process_id,
            process_type="init_db",
            extra_env=extra_env,
            cwd=str(project_root)
        )

    def _launch_snippet(
        self,
        snippet: str,
        process_id: str,
        process_type: str,
        hospital_id: Optional[int] = None,
        hospital_code: Optional[str] = None,
        extra_env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None
    ) -> ProcessInfo:
        _, project_root, _ = self._resolve_paths()
        effective_cwd = cwd if cwd else str(project_root)
        proc_info = ProcessInfo(
            process_id=process_id,
            process_type=process_type,
            hospital_id=hospital_id,
            hospital_code=hospital_code
        )
        self.processes[process_id] = proc_info
        env = self._build_env(extra_env)
        command = [self._python_bin, "-c", snippet]

        try:
            proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=effective_cwd,
                text=True,
                bufsize=1,
                shell=False
            )
        except Exception as e:
            proc_info.status = "failed"
            proc_info.log_buffer.append(f"[Launch error: {e}]")
            raise

        self._subprocesses[process_id] = proc
        t_out = threading.Thread(
            target=self._stream_output,
            args=(process_id, proc.stdout, "OUT"),
            daemon=True
        )
        t_err = threading.Thread(
            target=self._stream_output,
            args=(process_id, proc.stderr, "ERR"),
            daemon=True
        )
        t_mon = threading.Thread(
            target=self._monitor_process,
            args=(process_id,),
            daemon=True
        )
        t_out.start()
        t_err.start()
        t_mon.start()
        self._threads[process_id] = t_mon
        proc_info.status = "running"
        return proc_info

    def start_evaluate_comparison(
        self,
        extra_env: Optional[Dict[str, str]] = None
    ) -> ProcessInfo:
        _, _, src_pkg = self._resolve_paths()
        return self._launch_script(
            script_path=src_pkg / "evaluate_comparison.py",
            process_id=f"eval-cmp-{uuid.uuid4().hex[:8]}",
            process_type="evaluate_comparison",
            extra_env=extra_env
        )

    def start_graph(
        self,
        extra_env: Optional[Dict[str, str]] = None
    ) -> ProcessInfo:
        _, _, src_pkg = self._resolve_paths()
        return self._launch_script(
            script_path=src_pkg / "graph.py",
            process_id=f"graph-{uuid.uuid4().hex[:8]}",
            process_type="graph",
            extra_env=extra_env
        )

    def start_comparison_graph(
        self,
        extra_env: Optional[Dict[str, str]] = None
    ) -> ProcessInfo:
        _, _, src_pkg = self._resolve_paths()
        return self._launch_script(
            script_path=src_pkg / "comparison_graph.py",
            process_id=f"cmp-graph-{uuid.uuid4().hex[:8]}",
            process_type="comparison_graph",
            extra_env=extra_env
        )

    def start_comparison_table(
        self,
        extra_env: Optional[Dict[str, str]] = None
    ) -> ProcessInfo:
        _, _, src_pkg = self._resolve_paths()
        return self._launch_script(
            script_path=src_pkg / "comparison.py",
            process_id=f"cmp-table-{uuid.uuid4().hex[:8]}",
            process_type="comparison_table",
            extra_env=extra_env
        )

    def start_parse_results(
        self,
        extra_env: Optional[Dict[str, str]] = None
    ) -> ProcessInfo:
        _, _, src_pkg = self._resolve_paths()
        return self._launch_script(
            script_path=src_pkg / "scripts" / "parse_results.py",
            process_id=f"parse-{uuid.uuid4().hex[:8]}",
            process_type="parse_results",
            extra_env=extra_env
        )

    def start_aggregate_results(
        self,
        extra_env: Optional[Dict[str, str]] = None
    ) -> ProcessInfo:
        _, _, src_pkg = self._resolve_paths()
        return self._launch_script(
            script_path=src_pkg / "scripts" / "aggregate_results.py",
            process_id=f"aggregate-{uuid.uuid4().hex[:8]}",
            process_type="aggregate_results",
            extra_env=extra_env
        )

    def start_generate_figures(
        self,
        extra_env: Optional[Dict[str, str]] = None
    ) -> ProcessInfo:
        _, _, src_pkg = self._resolve_paths()
        return self._launch_script(
            script_path=src_pkg / "scripts" / "generate_figures.py",
            process_id=f"figures-{uuid.uuid4().hex[:8]}",
            process_type="generate_figures",
            extra_env=extra_env
        )

    def start_run_experiment(
        self,
        exp: str,
        runs: int = 1,
        num_rounds: Optional[int] = None,
        extra_env: Optional[Dict[str, str]] = None
    ) -> ProcessInfo:
        _, _, src_pkg = self._resolve_paths()
        args = ["--exp", str(exp), "--runs", str(int(runs))]
        merged_env: Dict[str, str] = {}
        if num_rounds is not None:
            merged_env["NUM_ROUNDS"] = str(int(num_rounds))
        if extra_env:
            merged_env.update({k: ("" if v is None else str(v)) for k, v in extra_env.items()})
        return self._launch_script(
            script_path=src_pkg / "scripts" / "run_experiments.py",
            process_id=f"exp-{exp.lower()}-{uuid.uuid4().hex[:6]}",
            process_type="run_experiment",
            script_args=args,
            extra_env=merged_env
        )

    def get_logs(
        self,
        process_id: str,
        tail: Optional[int] = None
    ) -> List[str]:

        proc_info = self.processes.get(process_id)

        if not proc_info:
            return []

        if tail:
            return proc_info.log_buffer[-tail:]

        return proc_info.log_buffer


def get_process_manager() -> ProcessManager:
    return ProcessManager()