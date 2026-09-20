import json
import asyncio
from datetime import datetime
from typing import Dict, Set, Optional, List
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from sqlalchemy.orm import Session

from ..database.connection import get_db
from ..core.security import decode_access_token
from ..core.process_manager import get_process_manager, ProcessManager, ProcessInfo
from ..core.metrics_reader import get_metrics_reader, MetricsReader
from ..models.db_models import User, UserRole

router = APIRouter(tags=["WebSocket"])


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {
            "admin": set(),
            "hospital": {},
            "process": {}
        }

    async def connect_admin(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections["admin"].add(websocket)

    async def connect_hospital(self, websocket: WebSocket, hospital_id: int):
        await websocket.accept()
        hid = str(hospital_id)
        if hid not in self.active_connections["hospital"]:
            self.active_connections["hospital"][hid] = set()
        self.active_connections["hospital"][hid].add(websocket)

    async def connect_process(self, websocket: WebSocket, process_id: str):
        await websocket.accept()
        if process_id not in self.active_connections["process"]:
            self.active_connections["process"][process_id] = set()
        self.active_connections["process"][process_id].add(websocket)

    def disconnect_admin(self, websocket: WebSocket):
        self.active_connections["admin"].discard(websocket)

    def disconnect_hospital(self, websocket: WebSocket, hospital_id: int):
        hid = str(hospital_id)
        if hid in self.active_connections["hospital"]:
            self.active_connections["hospital"][hid].discard(websocket)

    def disconnect_process(self, websocket: WebSocket, process_id: str):
        if process_id in self.active_connections["process"]:
            self.active_connections["process"][process_id].discard(websocket)

    async def broadcast_admin(self, message: dict):
        dead = set()
        for conn in list(self.active_connections["admin"]):
            try:
                await conn.send_json(message)
            except Exception:
                dead.add(conn)
        for d in dead:
            self.active_connections["admin"].discard(d)

    async def broadcast_hospital(self, hospital_id: int, message: dict):
        hid = str(hospital_id)
        if hid not in self.active_connections["hospital"]:
            return
        dead = set()
        for conn in list(self.active_connections["hospital"][hid]):
            try:
                await conn.send_json(message)
            except Exception:
                dead.add(conn)
        for d in dead:
            self.active_connections["hospital"][hid].discard(d)

    async def broadcast_process(self, process_id: str, message: dict):
        if process_id not in self.active_connections["process"]:
            return
        dead = set()
        for conn in list(self.active_connections["process"][process_id]):
            try:
                await conn.send_json(message)
            except Exception:
                dead.add(conn)
        for d in dead:
            self.active_connections["process"][process_id].discard(d)


manager = ConnectionManager()
pm: ProcessManager = get_process_manager()
mr: MetricsReader = get_metrics_reader()


def _setup_process_callbacks():
    def _on_log(proc_info: ProcessInfo, log_line: str):
        msg = {
            "type": "process_log",
            "process_id": proc_info.process_id,
            "process_type": proc_info.process_type,
            "hospital_id": proc_info.hospital_id,
            "hospital_code": proc_info.hospital_code,
            "status": proc_info.status,
            "line": log_line,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        asyncio.create_task(manager.broadcast_process(proc_info.process_id, msg))
        if proc_info.process_type == "fl_server":
            asyncio.create_task(manager.broadcast_admin({
                **msg,
                "scope": "server_logs"
            }))
        if proc_info.hospital_id:
            asyncio.create_task(manager.broadcast_hospital(proc_info.hospital_id, {
                **msg,
                "scope": "hospital_logs"
            }))

    def _on_status_change(proc_info: ProcessInfo, log_line: str):
        if any(kw in log_line for kw in ("ended with code", "terminated by user", "Process")):
            status_msg = {
                "type": "process_status",
                "process_id": proc_info.process_id,
                "process_type": proc_info.process_type,
                "hospital_id": proc_info.hospital_id,
                "hospital_code": proc_info.hospital_code,
                "status": proc_info.status,
                "return_code": proc_info.return_code,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
            asyncio.create_task(manager.broadcast_process(proc_info.process_id, status_msg))
            if proc_info.process_type == "fl_server":
                asyncio.create_task(manager.broadcast_admin(status_msg))
            if proc_info.hospital_id:
                asyncio.create_task(manager.broadcast_hospital(proc_info.hospital_id, status_msg))

    def _combined(proc_info: ProcessInfo, log_line: str):
        _on_log(proc_info, log_line)
        _on_status_change(proc_info, log_line)

    pm.register_global_callback(_combined)


_setup_process_callbacks()


def _get_user_from_token(token: Optional[str], db: Session) -> Optional[User]:
    if not token:
        return None
    token = token.replace("Bearer ", "")
    payload = decode_access_token(token)
    if not payload:
        return None
    username = payload.get("sub")
    if not username:
        return None
    user = db.query(User).filter(User.username == username).first()
    if user and user.is_active:
        return user
    return None


@router.websocket("/ws/admin")
async def websocket_admin(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    user = _get_user_from_token(token, db)
    if not user or user.role != UserRole.ADMIN:
        await websocket.close(code=4001, reason="Unauthorized")
        return
    await manager.connect_admin(websocket)
    try:
        hello = {
            "type": "welcome",
            "scope": "admin",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "processes": [p.to_dict() for p in pm.list_processes()]
        }
        await websocket.send_json(hello)
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                action = msg.get("action")
                if action == "subscribe_process":
                    pid = msg.get("process_id")
                    if pid:
                        await manager.connect_process(websocket, pid)
                        proc = pm.get_process(pid)
                        if proc:
                            await websocket.send_json({
                                "type": "process_history",
                                "process_id": pid,
                                "logs": pm.get_logs(pid, tail=200)
                            })
                elif action == "refresh_metrics":
                    suffix = msg.get("suffix")
                    curve = mr.get_training_curve_data(suffix)
                    await websocket.send_json({
                        "type": "metrics_update",
                        "scope": "global",
                        "data": curve
                    })
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        manager.disconnect_admin(websocket)
    except Exception:
        manager.disconnect_admin(websocket)


@router.websocket("/ws/hospital/{hospital_id}")
async def websocket_hospital(
    websocket: WebSocket,
    hospital_id: int,
    token: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    user = _get_user_from_token(token, db)
    if not user or user.role != UserRole.HOSPITAL or user.hospital_id != hospital_id:
        await websocket.close(code=4001, reason="Unauthorized")
        return
    await manager.connect_hospital(websocket, hospital_id)
    try:
        hospital_procs = pm.get_processes_by_hospital(hospital_id)
        hello = {
            "type": "welcome",
            "scope": "hospital",
            "hospital_id": hospital_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "processes": [p.to_dict() for p in hospital_procs]
        }
        await websocket.send_json(hello)
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                action = msg.get("action")
                if action == "subscribe_process":
                    pid = msg.get("process_id")
                    if pid:
                        proc = pm.get_process(pid)
                        if proc and proc.hospital_id == hospital_id:
                            await manager.connect_process(websocket, pid)
                            await websocket.send_json({
                                "type": "process_history",
                                "process_id": pid,
                                "logs": pm.get_logs(pid, tail=200)
                            })
                elif action == "refresh_metrics":
                    suffix = msg.get("suffix")
                    hospital_code = msg.get("hospital_code")
                    if hospital_code:
                        curve = mr.get_hospital_training_curve(hospital_code, suffix)
                        await websocket.send_json({
                            "type": "metrics_update",
                            "scope": "hospital",
                            "hospital_code": hospital_code,
                            "data": curve
                        })
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        manager.disconnect_hospital(websocket, hospital_id)
    except Exception:
        manager.disconnect_hospital(websocket, hospital_id)


@router.websocket("/ws/process/{process_id}")
async def websocket_process(
    websocket: WebSocket,
    process_id: str,
    token: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    user = _get_user_from_token(token, db)
    if not user:
        await websocket.close(code=4001, reason="Unauthorized")
        return
    proc = pm.get_process(process_id)
    if not proc:
        await websocket.close(code=4004, reason="Process not found")
        return
    if user.role != UserRole.ADMIN:
        if proc.hospital_id and user.hospital_id != proc.hospital_id:
            await websocket.close(code=4003, reason="Forbidden")
            return
        if not proc.hospital_id:
            await websocket.close(code=4003, reason="Forbidden")
            return
    await manager.connect_process(websocket, process_id)
    try:
        history = pm.get_logs(process_id, tail=300)
        await websocket.send_json({
            "type": "process_history",
            "process_id": process_id,
            "logs": history,
            "process": proc.to_dict()
        })
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect_process(websocket, process_id)
    except Exception:
        manager.disconnect_process(websocket, process_id)
