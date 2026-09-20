import os
import re
import csv
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime


_SAFE_FILENAME_RE = re.compile(r'[^\w\-\. ]+', re.UNICODE)
_DISALLOWED_PATH_CHARS = set('<>|"*?\n\r\t')
_EXT_MIME = {
    '.csv': 'text/csv',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.gif': 'image/gif',
    '.pdf': 'application/pdf',
    '.md': 'text/markdown',
    '.txt': 'text/plain',
    '.log': 'text/plain',
    '.json': 'application/json',
    '.pth': 'application/octet-stream',
    '.pt': 'application/octet-stream',
    '.db': 'application/octet-stream',
    '.html': 'text/html',
    '.svg': 'image/svg+xml',
}
_VIEWABLE_EXT = {'.csv', '.png', '.jpg', '.jpeg', '.gif', '.md', '.txt', '.log', '.json', '.svg'}
_DOWNLOADABLE_EXT = _EXT_MIME.keys()


@dataclass
class WorkspaceRoot:
    key: str
    path: Path
    role: str
    label: str
    writable: bool = False


@dataclass
class FileEntry:
    name: str
    path: str
    rel_path: str
    is_dir: bool
    size_bytes: int
    modified_at: str
    extension: str
    viewable: bool
    downloadable: bool
    mime: str


class WorkspaceManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, '_initialized', False):
            return
        self._initialized = True
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self._setup_roots()
        self._setup_upload_roots()

    def _setup_roots(self):
        pr = self.project_root
        roots: Dict[str, WorkspaceRoot] = {}

        data_root = pr / "data"
        data_root.mkdir(parents=True, exist_ok=True)
        roots["data"] = WorkspaceRoot(
            key="data", path=data_root, role="either",
            label="Dataset Directory", writable=True
        )

        models_root = pr / "federated_healthcare" / "models"
        models_root.mkdir(parents=True, exist_ok=True)
        roots["models"] = WorkspaceRoot(
            key="models", path=models_root, role="admin",
            label="Global Model Checkpoints", writable=False
        )

        dashboard_root = pr / "federated_healthcare" / "dashboard" / "results"
        dashboard_root.mkdir(parents=True, exist_ok=True)
        roots["results"] = WorkspaceRoot(
            key="results", path=dashboard_root, role="admin",
            label="Dashboard Results & Metrics", writable=False
        )

        plots_root = pr / "federated_healthcare" / "dashboard" / "plots"
        plots_root.mkdir(parents=True, exist_ok=True)
        roots["plots"] = WorkspaceRoot(
            key="plots", path=plots_root, role="admin",
            label="Training Plots & Figures", writable=False
        )

        reports_root = pr / "federated_healthcare" / "dashboard" / "classification_reports"
        reports_root.mkdir(parents=True, exist_ok=True)
        roots["reports"] = WorkspaceRoot(
            key="reports", path=reports_root, role="admin",
            label="Classification Reports", writable=False
        )

        figures_root = pr / "federated_healthcare" / "dashboard" / "plots" / "figures"
        figures_root.mkdir(parents=True, exist_ok=True)
        roots["figures"] = WorkspaceRoot(
            key="figures", path=figures_root, role="admin",
            label="Publication Figures", writable=False
        )

        experiments_root = dashboard_root / "ADSM_results" / "experiments"
        experiments_root.mkdir(parents=True, exist_ok=True)
        roots["experiments"] = WorkspaceRoot(
            key="experiments", path=experiments_root, role="admin",
            label="Experiment Results", writable=False
        )

        admin_upload = pr / "uploads" / "admin"
        admin_upload.mkdir(parents=True, exist_ok=True)
        roots["admin_uploads"] = WorkspaceRoot(
            key="admin_uploads", path=admin_upload, role="admin",
            label="Admin Uploads", writable=True
        )

        hospital_upload_base = pr / "uploads" / "hospitals"
        hospital_upload_base.mkdir(parents=True, exist_ok=True)
        roots["hospital_uploads"] = WorkspaceRoot(
            key="hospital_uploads", path=hospital_upload_base, role="hospital",
            label="Per-Hospital Uploads", writable=True
        )

        self._roots = roots
        self._upload_base = pr / "uploads"
        self._hospital_upload_base = hospital_upload_base

    def _setup_upload_roots(self):
        self._upload_base.mkdir(parents=True, exist_ok=True)
        (self._upload_base / "admin").mkdir(exist_ok=True)
        self._hospital_upload_base.mkdir(parents=True, exist_ok=True)

    def get_hospital_upload_dir(self, hospital_id: int, hospital_code: str = "") -> Path:
        safe_code = _SAFE_FILENAME_RE.sub('_', hospital_code).strip('_').lower() or f"hospital_{hospital_id}"
        d = self._hospital_upload_base / f"{hospital_id:04d}_{safe_code}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def list_roots_for_role(self, role: str, hospital_id: Optional[int] = None) -> List[Dict[str, Any]]:
        role = role.lower()
        result = []
        for r in self._roots.values():
            ok = (r.role == "either")
            ok = ok or (r.role == role)
            if r.key == "hospital_uploads" and role == "hospital" and hospital_id is not None:
                d = self.get_hospital_upload_dir(hospital_id, "")
                result.append({
                    "key": f"hospital_{hospital_id}_uploads",
                    "path": str(d),
                    "label": f"My Uploads ({d.name})",
                    "writable": True
                })
                continue
            if r.role == "hospital" and role != "hospital":
                continue
            if r.key == "hospital_uploads" and role != "admin":
                continue
            if ok:
                result.append({
                    "key": r.key,
                    "path": str(r.path),
                    "label": r.label,
                    "writable": r.writable
                })
        return result

    def _validate_escape_attempt(self, user_path: str) -> Tuple[bool, Optional[str]]:
        if user_path is None:
            return False, "Path is None"
        if any(ch in user_path for ch in _DISALLOWED_PATH_CHARS):
            return False, "Path contains disallowed characters"
        if ".." in Path(user_path).parts:
            return False, "Directory traversal not allowed"
        return True, None

    def resolve_and_validate(
        self,
        root_key: str,
        rel_path: str,
        role: str,
        hospital_id: Optional[int] = None,
        require_exists: bool = False
    ) -> Tuple[Optional[Path], Optional[str]]:

        role = role.lower()
        ok, err = self._validate_escape_attempt(rel_path or "")
        if not ok:
            return None, err

        if root_key.startswith("hospital_") and root_key.endswith("_uploads") and role == "hospital":
            parts = root_key.split("_")
            try:
                hid = int(parts[1])
            except Exception:
                return None, "Invalid hospital upload root"
            if hospital_id is None or hid != hospital_id:
                return None, "Access denied to upload root"
            root = self.get_hospital_upload_dir(hid, "")
        else:
            root_obj = self._roots.get(root_key)
            if not root_obj:
                return None, "Unknown root key"
            if root_obj.role not in ("either", role):
                if not (root_obj.role == "hospital" and role == "hospital"):
                    return None, "Access denied"
            if root_key == "hospital_uploads" and role != "admin":
                return None, "Access denied"
            root = root_obj.path

        rel = Path(rel_path) if rel_path else Path("")
        if rel.is_absolute():
            return None, "Absolute paths not allowed"

        resolved = (root / rel).resolve()
        root_resolved = root.resolve()
        try:
            resolved.relative_to(root_resolved)
        except ValueError:
            return None, "Path escape detected"

        if require_exists and not resolved.exists():
            return None, "Path does not exist"
        return resolved, None

    def sanitize_filename(self, filename: str) -> str:
        if not filename:
            return "unnamed"
        name = os.path.basename(filename).replace('\x00', '')
        name = _SAFE_FILENAME_RE.sub('_', name)
        name = name.strip('. ')
        if not name:
            name = "unnamed"
        if len(name) > 200:
            base, ext = os.path.splitext(name)
            name = base[:180] + ext
        return name

    def list_directory(
        self,
        root_key: str,
        rel_path: str,
        role: str,
        hospital_id: Optional[int] = None
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:

        resolved, err = self.resolve_and_validate(
            root_key, rel_path, role, hospital_id, require_exists=True
        )
        if err:
            return None, err
        if not resolved.is_dir():
            return None, "Not a directory"

        entries: List[FileEntry] = []
        root_obj = self._roots.get(root_key)
        if root_obj:
            base_root = root_obj.path.resolve()
        elif root_key.startswith("hospital_") and root_key.endswith("_uploads"):
            parts = root_key.split("_")
            hid = int(parts[1])
            base_root = self.get_hospital_upload_dir(hid, "").resolve()
        else:
            base_root = self.project_root.resolve()

        try:
            for p in sorted(resolved.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
                try:
                    resolved_child = p.resolve()
                    resolved_child.relative_to(base_root)
                except ValueError:
                    continue
                is_dir = p.is_dir()
                ext = p.suffix.lower()
                stat = p.stat() if not is_dir else None
                size = stat.st_size if stat else 0
                modified = datetime.fromtimestamp(
                    stat.st_mtime if stat else p.stat().st_mtime
                ).isoformat() + "Z" if not is_dir else ""
                if is_dir:
                    try:
                        s = p.stat()
                        modified = datetime.fromtimestamp(s.st_mtime).isoformat() + "Z"
                    except Exception:
                        modified = ""
                try:
                    rel = str(resolved_child.relative_to(base_root)).replace("\\", "/")
                    if rel == ".":
                        rel = ""
                except Exception:
                    rel = p.name
                entries.append(FileEntry(
                    name=p.name,
                    path=str(resolved_child),
                    rel_path=rel,
                    is_dir=is_dir,
                    size_bytes=size,
                    modified_at=modified,
                    extension=ext,
                    viewable=(not is_dir and ext in _VIEWABLE_EXT),
                    downloadable=(not is_dir and ext in _DOWNLOADABLE_EXT),
                    mime=_EXT_MIME.get(ext, "application/octet-stream")
                ))
        except PermissionError:
            return None, "Permission denied"

        parent_rel = ""
        if rel_path:
            parent = Path(rel_path).parent
            parent_rel = str(parent).replace("\\", "/")
            if parent_rel == ".":
                parent_rel = ""

        return {
            "root_key": root_key,
            "rel_path": rel_path,
            "parent_rel_path": parent_rel,
            "entries": [e.__dict__ for e in entries],
            "directory": str(resolved)
        }, None

    def get_file_for_download(
        self,
        root_key: str,
        rel_path: str,
        role: str,
        hospital_id: Optional[int] = None
    ) -> Tuple[Optional[Tuple[Path, str, str]], Optional[str]]:
        resolved, err = self.resolve_and_validate(
            root_key, rel_path, role, hospital_id, require_exists=True
        )
        if err:
            return None, err
        if not resolved.is_file():
            return None, "Not a file"
        ext = resolved.suffix.lower()
        if ext not in _DOWNLOADABLE_EXT:
            return None, "File type not allowed for download"
        mime = _EXT_MIME.get(ext, "application/octet-stream")
        return (resolved, resolved.name, mime), None

    def get_file_for_view(
        self,
        root_key: str,
        rel_path: str,
        role: str,
        hospital_id: Optional[int] = None
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        resolved, err = self.resolve_and_validate(
            root_key, rel_path, role, hospital_id, require_exists=True
        )
        if err:
            return None, err
        if not resolved.is_file():
            return None, "Not a file"
        ext = resolved.suffix.lower()
        if ext not in _VIEWABLE_EXT:
            return None, "File type not viewable inline"

        try:
            if ext == ".csv":
                columns = []
                rows = []
                with open(resolved, "r", newline="", encoding="utf-8", errors="replace") as f:
                    reader = csv.reader(f)
                    for i, row in enumerate(reader):
                        if i == 0:
                            columns = row
                        else:
                            rows.append(row)
                            if len(rows) >= 500:
                                break
                return {
                    "type": "csv",
                    "columns": columns,
                    "rows": rows,
                    "truncated": len(rows) >= 500,
                    "filename": resolved.name,
                    "size_bytes": resolved.stat().st_size
                }, None
            elif ext in (".md", ".txt", ".log"):
                text = resolved.read_text(encoding="utf-8", errors="replace")
                lines = text.splitlines()
                truncated = False
                if len(lines) > 2000:
                    lines = lines[-2000:]
                    truncated = True
                return {
                    "type": "text",
                    "content": "\n".join(lines),
                    "content_kind": "markdown" if ext == ".md" else "plain",
                    "truncated": truncated,
                    "filename": resolved.name,
                    "size_bytes": resolved.stat().st_size
                }, None
            elif ext == ".json":
                with open(resolved, "r", encoding="utf-8", errors="replace") as f:
                    data = json.load(f)
                return {
                    "type": "json",
                    "data": data,
                    "filename": resolved.name,
                    "size_bytes": resolved.stat().st_size
                }, None
            elif ext in (".png", ".jpg", ".jpeg", ".gif", ".svg"):
                return {
                    "type": "image",
                    "root_key": root_key,
                    "rel_path": rel_path,
                    "filename": resolved.name,
                    "size_bytes": resolved.stat().st_size,
                    "mime": _EXT_MIME.get(ext, "image/png")
                }, None
        except Exception as e:
            return None, f"Failed to read file: {e}"

        return None, "Unsupported view operation"

    def save_uploaded_file(
        self,
        filename: str,
        content_bytes: bytes,
        role: str,
        hospital_id: Optional[int] = None,
        sub_folder: Optional[str] = None
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        role = role.lower()
        safe_name = self.sanitize_filename(filename)
        if role == "admin":
            target = self._roots["admin_uploads"].path
        elif role == "hospital" and hospital_id is not None:
            target = self.get_hospital_upload_dir(hospital_id, "")
        else:
            return None, "Invalid role for upload"

        if sub_folder:
            ok, err = self._validate_escape_attempt(sub_folder)
            if not ok:
                return None, err
            sub_path = Path(sub_folder)
            if sub_path.is_absolute():
                return None, "Absolute subfolder not allowed"
            target_sub = (target / sub_path).resolve()
            try:
                target_sub.relative_to(target.resolve())
            except ValueError:
                return None, "Subfolder escape detected"
            target = target_sub
            target.mkdir(parents=True, exist_ok=True)

        base, ext = os.path.splitext(safe_name)
        final_name = safe_name
        counter = 1
        while (target / final_name).exists():
            final_name = f"{base}_{counter}{ext}"
            counter += 1
        final_path = target / final_name
        try:
            final_path.write_bytes(content_bytes)
        except Exception as e:
            return None, f"Write failed: {e}"

        size = final_path.stat().st_size
        sha = hashlib.sha256(content_bytes).hexdigest()
        rel = ""
        try:
            rel = str(final_path.relative_to(target)).replace("\\", "/")
        except Exception:
            rel = final_name
        return {
            "filename": final_name,
            "full_path": str(final_path),
            "rel_path": rel,
            "size_bytes": size,
            "uploaded_at": datetime.utcnow().isoformat() + "Z",
            "sha256": sha,
            "extension": ext.lower()
        }, None


def get_workspace_manager() -> WorkspaceManager:
    return WorkspaceManager()
