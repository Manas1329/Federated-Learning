import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional


_RUNTIME_SETTINGS_FILE_NAME = ".dashboard_runtime_settings.json"


class RuntimeSettingsStore:
    _instance: Optional["RuntimeSettingsStore"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._initialized = False
                    cls._instance = inst
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.file_path = self.project_root / _RUNTIME_SETTINGS_FILE_NAME
        self._data_lock = threading.Lock()
        self._settings: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        try:
            if self.file_path.exists():
                text = self.file_path.read_text(encoding="utf-8")
                if text.strip():
                    loaded = json.loads(text)
                    if isinstance(loaded, dict):
                        self._settings = {str(k): v for k, v in loaded.items()}
        except Exception:
            self._settings = {}

    def save(self) -> None:
        with self._data_lock:
            try:
                self.file_path.write_text(
                    json.dumps(self._settings, indent=2, sort_keys=True),
                    encoding="utf-8"
                )
            except Exception:
                raise

    def get_all(self) -> Dict[str, Any]:
        with self._data_lock:
            return dict(self._settings)

    def get(self, key: str, default: Any = None) -> Any:
        with self._data_lock:
            return self._settings.get(key, default)

    def update(self, values: Dict[str, Any], persist: bool = True) -> Dict[str, Any]:
        with self._data_lock:
            normalized = {str(k): v for k, v in values.items()}
            self._settings.update(normalized)
            if persist:
                self.save()
            return dict(self._settings)

    def reset(self, persist: bool = True) -> Dict[str, Any]:
        with self._data_lock:
            self._settings = {}
            if persist:
                try:
                    if self.file_path.exists():
                        self.file_path.unlink()
                except Exception:
                    pass
            return {}

    def as_env_overrides(self) -> Dict[str, str]:
        result: Dict[str, str] = {}
        with self._data_lock:
            for k, v in self._settings.items():
                if v is None:
                    continue
                if isinstance(v, bool):
                    result[k] = "1" if v else "0"
                else:
                    result[k] = str(v)
        return result


def get_runtime_settings() -> RuntimeSettingsStore:
    return RuntimeSettingsStore()
