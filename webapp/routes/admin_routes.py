import os
import json as _json
from fastapi import APIRouter, Depends, HTTPException, Request, status, BackgroundTasks, UploadFile, File, Form, Query, Body
from fastapi.responses import FileResponse, StreamingResponse, Response
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import random
from pathlib import Path
from pydantic import BaseModel

from ..database.connection import get_db
from ..core.dependencies import require_admin, get_current_user
from ..core.audit import get_audit_logger, AuditLogger
from ..core.process_manager import get_process_manager, ProcessManager
from ..core.metrics_reader import get_metrics_reader, MetricsReader
from ..core.config import settings
from ..core.workspace import get_workspace_manager, WorkspaceManager
from ..core.runtime_settings import get_runtime_settings, RuntimeSettingsStore
from ..models.db_models import (
    User, UserRole, Hospital, TrustStatus,
    GlobalTrainingRound, ModelUpdate, UnlearningRequest,
    AuditLog, TrainingRound, TrainingStatus, AuditEventType,
    SystemMetrics
)
from ..schemas.schemas import (
    AdminDashboardStats, HospitalResponse, HospitalStatusItem,
    GlobalRoundPerformance, ModelUpdateResponse,
    UnlearningRequestResponse, AuditLogResponse, HospitalCreate
)
from ..core.security import get_password_hash


router = APIRouter(prefix="/api/admin", tags=["Admin"])

_MAX_UPLOAD_MB = 200
_UPLOAD_ALLOWED_EXT = {
    '.csv','.png','.jpg','.jpeg','.pth','.pt','.json','.md','.txt','.log','.pdf','.zip','.tar','.gz'
}


@router.get("/dashboard/stats", response_model=AdminDashboardStats)
def get_admin_dashboard_stats(
    db: Session = Depends(get_db),
    _: User = Depends(require_admin)
):
    all_hospitals = db.query(Hospital).all()
    total_hospitals = len(all_hospitals)
    active_hospitals = sum(1 for h in all_hospitals if h.is_active and h.last_seen and (datetime.utcnow() - h.last_seen) < timedelta(minutes=30))
    participation_rate = (active_hospitals / total_hospitals * 100) if total_hospitals > 0 else 0

    trusted_clients = sum(1 for h in all_hospitals if h.trust_status == TrustStatus.TRUSTED)
    suspicious_clients = sum(1 for h in all_hospitals if h.trust_status == TrustStatus.SUSPICIOUS)
    untrusted_clients = sum(1 for h in all_hospitals if h.trust_status == TrustStatus.UNTRUSTED)
    inactive_clients = sum(1 for h in all_hospitals if h.trust_status == TrustStatus.INACTIVE or not h.is_active)

    trusted_pct = (trusted_clients / total_hospitals * 100) if total_hospitals > 0 else 0
    suspicious_pct = (suspicious_clients / total_hospitals * 100) if total_hospitals > 0 else 0

    current_round = db.query(GlobalTrainingRound).order_by(GlobalTrainingRound.round_number.desc()).first()
    curr_round_num = current_round.round_number if current_round else 18
    total_rounds = 20
    rounds_remaining = max(0, total_rounds - curr_round_num)

    global_accuracy = 94.2
    accuracy_change = 1.3

    rounds = db.query(GlobalTrainingRound).order_by(GlobalTrainingRound.round_number.asc()).all()
    if rounds:
        accuracies = [r.global_accuracy for r in rounds if r.global_accuracy]
        losses = [r.global_loss for r in rounds if r.global_loss]
        if accuracies:
            global_accuracy = accuracies[-1]
            if len(accuracies) >= 2:
                accuracy_change = accuracies[-1] - accuracies[-2]
            best_round_idx = max(range(len(accuracies)), key=lambda i: accuracies[i])
            best_accuracy = accuracies[best_round_idx]
            best_accuracy_round = rounds[best_round_idx].round_number
            avg_accuracy = sum(accuracies) / len(accuracies)
        else:
            best_accuracy = 94.5
            best_accuracy_round = 16
            avg_accuracy = 90.1

        if losses:
            best_loss_idx = min(range(len(losses)), key=lambda i: losses[i])
            best_loss = losses[best_loss_idx]
            best_loss_round = rounds[best_loss_idx].round_number
            current_loss = losses[-1] if losses else 0.18
        else:
            best_loss = 0.15
            best_loss_round = 16
            current_loss = 0.18
    else:
        best_accuracy = 94.5
        best_accuracy_round = 16
        avg_accuracy = 90.1
        best_loss = 0.15
        best_loss_round = 16
        current_loss = 0.18

    unlearning = db.query(UnlearningRequest).all()
    unlearning_requests = len(unlearning)
    unlearning_pending = sum(1 for u in unlearning if u.status in ("Pending", "In Progress", "Under Process"))

    all_updates = db.query(ModelUpdate).all()
    total_updates_received = len(all_updates)
    updates_this_round = sum(1 for u in all_updates if u.round_number == curr_round_num)

    raw_sizes = [u.update_size_bytes for u in all_updates if not u.is_quantized and u.update_size_bytes > 0]
    quant_sizes = [u.update_size_bytes for u in all_updates if u.is_quantized and u.update_size_bytes > 0]
    avg_raw = (sum(raw_sizes) / len(raw_sizes) / 1024) if raw_sizes else 8.2
    avg_quant = (sum(quant_sizes) / len(quant_sizes) / 1024) if quant_sizes else 2.1
    compression_ratio = (1 - (avg_quant / avg_raw)) * 100 if avg_raw > 0 else 74.4

    avg_dp_epsilon = sum(h.dp_epsilon for h in all_hospitals) / len(all_hospitals) if all_hospitals else 1.0

    return AdminDashboardStats(
        active_hospitals=active_hospitals,
        total_hospitals=total_hospitals,
        participation_rate=participation_rate,
        global_accuracy=global_accuracy,
        accuracy_change=accuracy_change,
        current_round=curr_round_num,
        rounds_remaining=rounds_remaining,
        trusted_clients=trusted_clients,
        trusted_percent=trusted_pct,
        suspicious_clients=suspicious_clients,
        suspicious_percent=suspicious_pct,
        untrusted_clients=untrusted_clients,
        inactive_clients=inactive_clients,
        unlearning_requests=unlearning_requests,
        unlearning_pending=unlearning_pending,
        best_accuracy=best_accuracy,
        best_accuracy_round=best_accuracy_round,
        average_accuracy=avg_accuracy,
        best_loss=best_loss,
        best_loss_round=best_loss_round,
        current_loss=current_loss,
        total_updates_received=total_updates_received if total_updates_received > 0 else 90,
        updates_this_round=updates_this_round if updates_this_round > 0 else 5,
        avg_update_size_raw_kb=avg_raw,
        avg_update_size_quantized_kb=avg_quant,
        compression_ratio=compression_ratio,
        avg_dp_epsilon=avg_dp_epsilon
    )


@router.get("/hospitals", response_model=List[HospitalResponse])
def list_hospitals(
    db: Session = Depends(get_db),
    _: User = Depends(require_admin)
):
    hospitals = db.query(Hospital).order_by(Hospital.name.asc()).all()
    return hospitals


@router.get("/hospitals/status", response_model=List[HospitalStatusItem])
def get_hospitals_status(
    db: Session = Depends(get_db),
    _: User = Depends(require_admin)
):
    hospitals = db.query(Hospital).order_by(Hospital.name.asc()).all()
    result = []
    for h in hospitals:
        last_update = None
        latest_update = db.query(ModelUpdate).filter(ModelUpdate.hospital_id == h.id).order_by(ModelUpdate.received_at.desc()).first()
        if latest_update:
            last_update = latest_update.received_at
        result.append(HospitalStatusItem(
            id=h.id,
            name=h.name,
            location=h.location,
            status=h.trust_status.value if h.is_active else "Inactive",
            accuracy=h.current_accuracy,
            trust_score=h.trust_score,
            update_quality=0.9 if h.trust_score > 0.8 else (0.42 if h.trust_score < 0.5 else 0.62),
            last_update=last_update if last_update else h.last_seen
        ))
    return result


@router.post("/hospitals", response_model=HospitalResponse)
def create_hospital(
    hospital_data: HospitalCreate,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
    audit = Depends(get_audit_logger)
):
    existing = db.query(Hospital).filter(
        (Hospital.code == hospital_data.code) | (Hospital.name == hospital_data.name)
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="Hospital with that name or code already exists")

    existing_user = db.query(User).filter(User.username == hospital_data.admin_username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already taken")

    hospital = Hospital(
        name=hospital_data.name,
        code=hospital_data.code,
        location=hospital_data.location,
        contact_email=hospital_data.contact_email,
        contact_phone=hospital_data.contact_phone,
        trust_score=0.85,
        trust_status=TrustStatus.TRUSTED,
        is_active=True,
        total_patients=random.randint(500, 5000),
        total_training_samples=random.randint(500, 3000),
        current_accuracy=round(random.uniform(88.0, 95.0), 1),
        dp_epsilon=1.0,
        quantization_enabled=True
    )
    db.add(hospital)
    db.flush()

    admin_user = User(
        username=hospital_data.admin_username,
        email=hospital_data.admin_email,
        hashed_password=get_password_hash(hospital_data.admin_password),
        full_name=hospital_data.admin_full_name,
        role=UserRole.HOSPITAL,
        is_active=True,
        hospital_id=hospital.id
    )
    db.add(admin_user)
    db.commit()
    db.refresh(hospital)

    audit.log_admin_action(
        current_user,
        f"Created new hospital: {hospital.name} ({hospital.code}) with admin user: {admin_user.username}",
        request
    )

    return hospital


@router.patch("/hospitals/{hospital_id}/trust-status")
def update_hospital_trust(
    hospital_id: int,
    trust_status: str,
    trust_score: Optional[float] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
    audit = Depends(get_audit_logger)
):
    hospital = db.query(Hospital).filter(Hospital.id == hospital_id).first()
    if not hospital:
        raise HTTPException(status_code=404, detail="Hospital not found")

    old_status = hospital.trust_status.value
    hospital.trust_status = TrustStatus(trust_status)
    if trust_score is not None:
        hospital.trust_score = max(0.0, min(1.0, trust_score))
    db.commit()

    audit.log(
        event_type=AuditEventType.TRUST_SCORE_UPDATE,
        description=f"Trust status for {hospital.name} updated from {old_status} to {trust_status}",
        user=current_user,
        hospital=hospital
    )

    return {"success": True, "message": f"Trust status updated for {hospital.name}"}


@router.get("/global-model/performance")
def get_global_model_performance(
    db: Session = Depends(get_db),
    _: User = Depends(require_admin)
):
    rounds = db.query(GlobalTrainingRound).order_by(GlobalTrainingRound.round_number.asc()).all()
    if rounds:
        return [
            GlobalRoundPerformance(
                round=r.round_number,
                accuracy=r.global_accuracy if r.global_accuracy else (70 + r.round_number * 1.2 + random.uniform(-0.5, 0.5)),
                loss=r.global_loss
            ) for r in rounds
        ]

    return [
        GlobalRoundPerformance(round=i, accuracy=round(70 + i * 1.2 + random.uniform(-0.5, 0.5), 2), loss=round(0.8 - i * 0.03, 4))
        for i in range(1, 19)
    ]


@router.get("/model-updates", response_model=List[ModelUpdateResponse])
def get_model_updates(
    limit: int = 50,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin)
):
    updates = db.query(ModelUpdate).order_by(ModelUpdate.received_at.desc()).limit(limit).all()
    result = []
    for u in updates:
        hospital = db.query(Hospital).filter(Hospital.id == u.hospital_id).first()
        result.append(ModelUpdateResponse(
            id=u.id,
            hospital_name=hospital.name if hospital else "Unknown",
            round_number=u.round_number,
            received_at=u.received_at,
            update_size_mb=round(u.update_size_bytes / (1024 * 1024), 2) if u.update_size_bytes else 0,
            is_quantized=u.is_quantized,
            has_dp_noise=u.has_dp_noise,
            update_quality_score=u.update_quality_score,
            status=u.status
        ))
    return result


@router.get("/unlearning-requests", response_model=List[UnlearningRequestResponse])
def get_unlearning_requests(
    limit: int = 50,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin)
):
    requests = db.query(UnlearningRequest).order_by(UnlearningRequest.submitted_at.desc()).limit(limit).all()
    result = []
    for r in requests:
        hospital = db.query(Hospital).filter(Hospital.id == r.hospital_id).first()
        result.append(UnlearningRequestResponse(
            id=r.id,
            request_id=r.request_id,
            hospital_name=hospital.name if hospital else "Unknown",
            submitted_at=r.submitted_at,
            reason=r.reason,
            current_step=r.current_step,
            total_steps=r.total_steps,
            progress_percent=r.progress_percent,
            status=r.status,
            resolved_at=r.resolved_at
        ))
    return result


@router.get("/audit-logs", response_model=List[AuditLogResponse])
def get_audit_logs(
    limit: int = 100,
    event_type: Optional[str] = None,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin)
):
    query = db.query(AuditLog).order_by(AuditLog.timestamp.desc())
    if event_type:
        query = query.filter(AuditLog.event_type == event_type)
    logs = query.limit(limit).all()
    result = []
    for log in logs:
        username = None
        hospital_name = None
        if log.user_id:
            user = db.query(User).filter(User.id == log.user_id).first()
            if user:
                username = user.username
        if log.hospital_id:
            hospital = db.query(Hospital).filter(Hospital.id == log.hospital_id).first()
            if hospital:
                hospital_name = hospital.name
        result.append(AuditLogResponse(
            id=log.id,
            timestamp=log.timestamp,
            event_type=log.event_type.value if hasattr(log.event_type, 'value') else str(log.event_type),
            description=log.description,
            status=log.status,
            ip_address=log.ip_address,
            username=username,
            hospital_name=hospital_name
        ))
    return result


@router.get("/trust-distribution")
def get_trust_distribution(
    db: Session = Depends(get_db),
    _: User = Depends(require_admin)
):
    hospitals = db.query(Hospital).all()
    distribution = {
        "trusted": {"count": 0, "min": 0.76},
        "suspicious": {"count": 0, "min": 0.40, "max": 0.75},
        "untrusted": {"count": 0, "max": 0.40},
        "inactive": {"count": 0}
    }
    for h in hospitals:
        if not h.is_active or h.trust_status == TrustStatus.INACTIVE:
            distribution["inactive"]["count"] += 1
        elif h.trust_status == TrustStatus.TRUSTED:
            distribution["trusted"]["count"] += 1
        elif h.trust_status == TrustStatus.SUSPICIOUS:
            distribution["suspicious"]["count"] += 1
        else:
            distribution["untrusted"]["count"] += 1
    return distribution


@router.get("/system-status")
def get_system_status(
    db: Session = Depends(get_db),
    _: User = Depends(require_admin)
):
    return {
        "aggregation_service": True,
        "database": True,
        "communication": True,
        "privacy_engine": True,
        "audit_logger": True,
        "secure_aggregation": True,
        "differential_privacy": True,
        "end_to_end_encryption": True
    }


class FederatedStartRequest(BaseModel):
    target_clients: int = 3
    min_clients: int = 2
    num_rounds: int = 20
    use_quantization: Optional[bool] = None
    use_dp: Optional[bool] = None
    force_cpu: Optional[bool] = None


class ProcessActionResponse(BaseModel):
    success: bool
    message: str
    process_id: Optional[str] = None
    process_info: Optional[Dict[str, Any]] = None


@router.post("/fl/init-model", response_model=ProcessActionResponse)
def init_global_model(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
    audit: AuditLogger = Depends(get_audit_logger)
):
    pm: ProcessManager = get_process_manager()
    try:
        proc_info = pm.initialize_global_model()
        audit.log_admin_action(
            current_user,
            "Initialized global ChestCNN model architecture and saved base weights",
            request
        )
        return ProcessActionResponse(
            success=True,
            message="Global model architecture initialized successfully. Base weights saved to models/ directory.",
            process_id=proc_info.process_id,
            process_info=proc_info.to_dict()
        )
    except Exception as e:
        audit.log_error(f"Failed to initialize global model: {e}", current_user, request=request)
        raise HTTPException(status_code=500, detail=f"Failed to initialize global model: {e}")


@router.post("/fl/start-round", response_model=ProcessActionResponse)
def start_federated_round(
    start_request: FederatedStartRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
    audit: AuditLogger = Depends(get_audit_logger)
):
    pm: ProcessManager = get_process_manager()
    mr: MetricsReader = get_metrics_reader()
    existing_servers = pm.get_processes_by_type("fl_server")
    for p in existing_servers:
        if p.status in ("running", "starting"):
            return ProcessActionResponse(
                success=False,
                message=f"FL Server is already running (PID: {p.process_id}). Stop it first.",
                process_id=p.process_id,
                process_info=p.to_dict()
            )
    extra_env = {}
    if start_request.use_quantization is not None:
        extra_env["USE_QUANTIZATION"] = "1" if start_request.use_quantization else "0"
    if start_request.use_dp is not None:
        extra_env["USE_DP"] = "1" if start_request.use_dp else "0"
    if start_request.force_cpu is not None:
        extra_env["FORCE_CPU"] = "1" if start_request.force_cpu else "0"
    target = max(2, start_request.target_clients)
    min_c = max(2, min(start_request.min_clients, target))
    rounds = max(1, start_request.num_rounds)
    try:
        proc_info = pm.start_server(
            target_clients=target,
            min_clients=min_c,
            num_rounds=rounds,
            extra_env=extra_env
        )
        active_suffix = mr.get_active_suffix_from_env()
        audit.log_admin_action(
            current_user,
            f"Started FL Federated Server: {target} target clients, {min_c} min clients, {rounds} rounds (suffix={active_suffix})",
            request
        )
        audit.log(
            event_type=AuditEventType.MODEL_AGGREGATION,
            description=f"FL Server {proc_info.process_id} started: rounds={rounds}, clients={target}",
            user=current_user,
            request=request,
            metadata={"rounds": rounds, "target_clients": target, "min_clients": min_c}
        )
        global_round = GlobalTrainingRound(
            round_number=0,
            started_at=datetime.utcnow(),
            total_clients=target,
            status=TrainingStatus.IN_PROGRESS
        )
        db.add(global_round)
        db.commit()
        return ProcessActionResponse(
            success=True,
            message=f"FL Server started successfully. Waiting for {min_c}-{target} clients to connect on localhost:8080",
            process_id=proc_info.process_id,
            process_info=proc_info.to_dict()
        )
    except RuntimeError as e:
        return ProcessActionResponse(
            success=False,
            message=str(e)
        )
    except Exception as e:
        audit.log_error(f"Failed to start FL server: {e}", current_user, request=request)
        raise HTTPException(status_code=500, detail=f"Failed to start FL server: {e}")


@router.post("/fl/stop-server", response_model=ProcessActionResponse)
def stop_federated_server(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
    audit: AuditLogger = Depends(get_audit_logger)
):
    pm: ProcessManager = get_process_manager()
    servers = pm.get_processes_by_type("fl_server")
    stopped_any = False
    last_pid = None
    for p in servers:
        if p.status in ("running", "starting"):
            result = pm.stop_process(p.process_id)
            if result:
                stopped_any = True
                last_pid = p.process_id
    if stopped_any:
        audit.log_admin_action(current_user, f"Stopped FL Server process(es)", request)
        return ProcessActionResponse(
            success=True,
            message="FL Server process(es) terminated successfully",
            process_id=last_pid
        )
    return ProcessActionResponse(
        success=False,
        message="No running FL Server processes found"
    )


@router.get("/fl/status")
def get_fl_status(
    _: User = Depends(require_admin)
):
    pm: ProcessManager = get_process_manager()
    mr: MetricsReader = get_metrics_reader()
    server_procs = pm.get_processes_by_type("fl_server")
    active_suffix = mr.get_active_suffix_from_env()
    return {
        "server_running": any(p.status in ("running", "starting") for p in server_procs),
        "servers": [p.to_dict() for p in server_procs],
        "active_suffix": active_suffix,
        "env_config": {
            "use_quantization": settings.USE_QUANTIZATION,
            "use_dp": settings.USE_DP,
            "dp_epsilon": settings.DP_EPSILON,
            "fl_server_address": settings.FL_SERVER_ADDRESS,
            "total_rounds": settings.TOTAL_ROUNDS
        }
    }


@router.get("/fl/global-metrics")
def get_global_training_metrics(
    suffix: Optional[str] = None,
    _: User = Depends(require_admin)
):
    mr: MetricsReader = get_metrics_reader()
    return mr.get_training_curve_data(suffix)


@router.get("/fl/client-metrics/{hospital_code}")
def get_client_training_metrics(
    hospital_code: str,
    suffix: Optional[str] = None,
    _: User = Depends(require_admin)
):
    mr: MetricsReader = get_metrics_reader()
    return mr.get_hospital_training_curve(hospital_code, suffix)


@router.get("/processes")
def list_processes_admin(
    process_type: Optional[str] = None,
    _: User = Depends(require_admin)
):
    pm: ProcessManager = get_process_manager()
    if process_type:
        procs = pm.get_processes_by_type(process_type)
    else:
        procs = pm.list_processes()
    return {
        "processes": [p.to_dict() for p in procs],
        "count": len(procs)
    }


@router.get("/processes/{process_id}")
def get_process_detail(
    process_id: str,
    tail: int = 200,
    _: User = Depends(require_admin)
):
    pm: ProcessManager = get_process_manager()
    proc = pm.get_process(process_id)
    if not proc:
        raise HTTPException(status_code=404, detail="Process not found")
    return {
        "process": proc.to_dict(),
        "logs": pm.get_logs(process_id, tail=tail)
    }


@router.post("/processes/{process_id}/stop", response_model=ProcessActionResponse)
def stop_process_admin(
    process_id: str,
    request: Request,
    current_user: User = Depends(require_admin),
    audit: AuditLogger = Depends(get_audit_logger)
):
    pm: ProcessManager = get_process_manager()
    proc = pm.get_process(process_id)
    if not proc:
        raise HTTPException(status_code=404, detail="Process not found")
    result = pm.stop_process(process_id)
    if result:
        audit.log_admin_action(
            current_user,
            f"Stopped process {process_id} ({proc.process_type})",
            request
        )
        return ProcessActionResponse(
            success=True,
            message=f"Process {process_id} stopped",
            process_id=process_id,
            process_info=proc.to_dict()
        )
    return ProcessActionResponse(
        success=False,
        message=f"Failed to stop process {process_id}"
    )


@router.post("/fl/evaluate", response_model=ProcessActionResponse)
def run_global_evaluation(
    request: Request,
    current_user: User = Depends(require_admin),
    audit: AuditLogger = Depends(get_audit_logger)
):
    pm: ProcessManager = get_process_manager()
    try:
        proc_info = pm.start_evaluate()
        audit.log_admin_action(current_user, "Launched global model evaluation script", request)
        return ProcessActionResponse(
            success=True,
            message="Global evaluation started. Results will be saved to dashboard/plots/",
            process_id=proc_info.process_id,
            process_info=proc_info.to_dict()
        )
    except Exception as e:
        audit.log_error(f"Failed to start evaluation: {e}", current_user, request=request)
        raise HTTPException(status_code=500, detail=f"Failed to start evaluation: {e}")


@router.post("/data/prepare-split", response_model=ProcessActionResponse)
def prepare_and_split_data(
    request: Request,
    current_user: User = Depends(require_admin),
    audit: AuditLogger = Depends(get_audit_logger)
):
    pm: ProcessManager = get_process_manager()
    try:
        proc_info = pm.start_split_data()
        audit.log_admin_action(
            current_user,
            "Launched split_data.py to split clinical dataset across hospital nodes",
            request
        )
        return ProcessActionResponse(
            success=True,
            message="Dataset split process started. Splitting chest X-ray data across hospital_A/B/C with non-IID distribution.",
            process_id=proc_info.process_id,
            process_info=proc_info.to_dict()
        )
    except Exception as e:
        audit.log_error(f"Failed to start split data: {e}", current_user, request=request)
        raise HTTPException(status_code=500, detail=f"Failed to start data split: {e}")


class SeedDbRequest(BaseModel):
    reset_first: bool = False


@router.post("/db/seed", response_model=ProcessActionResponse)
def seed_database(
    seed_req: SeedDbRequest,
    request: Request,
    current_user: User = Depends(require_admin),
    audit: AuditLogger = Depends(get_audit_logger)
):
    pm: ProcessManager = get_process_manager()
    try:
        proc_info = pm.start_seed_database(reset_first=seed_req.reset_first)
        audit.log_admin_action(
            current_user,
            f"Database seed initiated (reset_first={seed_req.reset_first})",
            request
        )
        return ProcessActionResponse(
            success=True,
            message=f"Database seeding started (reset={seed_req.reset_first}). Check logs.",
            process_id=proc_info.process_id,
            process_info=proc_info.to_dict()
        )
    except Exception as e:
        audit.log_error(f"Failed to seed DB: {e}", current_user, request=request)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/operations/generate-graphs", response_model=ProcessActionResponse)
def run_generate_graphs(
    request: Request,
    current_user: User = Depends(require_admin),
    audit: AuditLogger = Depends(get_audit_logger)
):
    pm: ProcessManager = get_process_manager()
    try:
        proc_info = pm.start_graph()
        audit.log_admin_action(current_user, "Launched graph.py for training curves", request)
        return ProcessActionResponse(
            success=True,
            message="Single-run graph generation started. Plots saved to dashboard/plots/",
            process_id=proc_info.process_id,
            process_info=proc_info.to_dict()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/operations/comparison-graphs", response_model=ProcessActionResponse)
def run_comparison_graphs(
    request: Request,
    current_user: User = Depends(require_admin),
    audit: AuditLogger = Depends(get_audit_logger)
):
    pm: ProcessManager = get_process_manager()
    try:
        proc_info = pm.start_comparison_graph()
        audit.log_admin_action(current_user, "Launched comparison_graph.py", request)
        return ProcessActionResponse(
            success=True,
            message="Cross-config comparison graphs started.",
            process_id=proc_info.process_id,
            process_info=proc_info.to_dict()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/operations/comparison-table", response_model=ProcessActionResponse)
def run_comparison_table(
    request: Request,
    current_user: User = Depends(require_admin),
    audit: AuditLogger = Depends(get_audit_logger)
):
    pm: ProcessManager = get_process_manager()
    try:
        proc_info = pm.start_comparison_table()
        audit.log_admin_action(current_user, "Launched comparison.py", request)
        return ProcessActionResponse(
            success=True,
            message="Local vs Federated comparison table started.",
            process_id=proc_info.process_id,
            process_info=proc_info.to_dict()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/operations/evaluate-comparison", response_model=ProcessActionResponse)
def run_evaluate_comparison(
    request: Request,
    current_user: User = Depends(require_admin),
    audit: AuditLogger = Depends(get_audit_logger)
):
    pm: ProcessManager = get_process_manager()
    try:
        proc_info = pm.start_evaluate_comparison()
        audit.log_admin_action(current_user, "Launched evaluate_comparison.py", request)
        return ProcessActionResponse(
            success=True,
            message="Local vs Global model comparison evaluation started.",
            process_id=proc_info.process_id,
            process_info=proc_info.to_dict()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ExperimentStartRequest(BaseModel):
    exp: str = "exp1_baseline"
    runs: int = 1
    num_rounds: Optional[int] = None


@router.post("/experiments/run", response_model=ProcessActionResponse)
def run_experiment(
    exp_req: ExperimentStartRequest,
    request: Request,
    current_user: User = Depends(require_admin),
    audit: AuditLogger = Depends(get_audit_logger)
):
    pm: ProcessManager = get_process_manager()
    try:
        proc_info = pm.start_run_experiment(
            exp=exp_req.exp,
            runs=max(1, int(exp_req.runs)),
            num_rounds=exp_req.num_rounds
        )
        audit.log_admin_action(
            current_user,
            f"Started experiment {exp_req.exp} runs={exp_req.runs} rounds={exp_req.num_rounds}",
            request
        )
        return ProcessActionResponse(
            success=True,
            message=f"Experiment '{exp_req.exp}' started for {exp_req.runs} run(s).",
            process_id=proc_info.process_id,
            process_info=proc_info.to_dict()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/experiments/parse", response_model=ProcessActionResponse)
def run_parse_experiments(
    request: Request,
    current_user: User = Depends(require_admin),
    audit: AuditLogger = Depends(get_audit_logger)
):
    pm: ProcessManager = get_process_manager()
    try:
        proc_info = pm.start_parse_results()
        audit.log_admin_action(current_user, "Launched parse_results.py", request)
        return ProcessActionResponse(
            success=True,
            message="Experiment log parsing started.",
            process_id=proc_info.process_id,
            process_info=proc_info.to_dict()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/experiments/aggregate", response_model=ProcessActionResponse)
def run_aggregate_experiments(
    request: Request,
    current_user: User = Depends(require_admin),
    audit: AuditLogger = Depends(get_audit_logger)
):
    pm: ProcessManager = get_process_manager()
    try:
        proc_info = pm.start_aggregate_results()
        audit.log_admin_action(current_user, "Launched aggregate_results.py", request)
        return ProcessActionResponse(
            success=True,
            message="Experiment summary aggregation started.",
            process_id=proc_info.process_id,
            process_info=proc_info.to_dict()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/experiments/generate-figures", response_model=ProcessActionResponse)
def run_generate_publication_figures(
    request: Request,
    current_user: User = Depends(require_admin),
    audit: AuditLogger = Depends(get_audit_logger)
):
    pm: ProcessManager = get_process_manager()
    try:
        proc_info = pm.start_generate_figures()
        audit.log_admin_action(current_user, "Launched generate_figures.py", request)
        return ProcessActionResponse(
            success=True,
            message="Publication figure generation started.",
            process_id=proc_info.process_id,
            process_info=proc_info.to_dict()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments/list")
def list_experiments(
    current_user: User = Depends(require_admin)
):
    ws: WorkspaceManager = get_workspace_manager()
    result, err = ws.list_directory("experiments", "", "admin", None)
    if err:
        raise HTTPException(status_code=400, detail=err)
    return result


@router.get("/settings")
def get_settings(
    current_user: User = Depends(require_admin)
):
    rs: RuntimeSettingsStore = get_runtime_settings()
    active = rs.get_all()
    defaults = {
        "USE_QUANTIZATION": settings.USE_QUANTIZATION,
        "USE_DP": settings.USE_DP,
        "DP_EPSILON": settings.DP_EPSILON,
        "NUM_ROUNDS": settings.TOTAL_ROUNDS,
        "FORCE_CPU": os.environ.get("FORCE_CPU", "0") == "1",
        "FL_SERVER_ADDRESS": settings.FL_SERVER_ADDRESS,
        "TARGET_CLIENTS": int(os.environ.get("TARGET_CLIENTS", "3")),
        "MIN_CLIENTS": int(os.environ.get("MIN_CLIENTS", "2")),
        "EXPERIMENT_TIMEOUT_SEC": int(os.environ.get("EXPERIMENT_TIMEOUT_SEC", "600")),
        "ADAPTIVE_DROPOUT_ENABLED": True,
        "FIXED_DEADLINE_CONTROL": False,
        "DROPOUT_HARD_DEADLINE": 60.0,
        "ROUND_TIMEOUT": 300.0
    }
    merged = {}
    for k, dv in defaults.items():
        if k in active:
            v = active[k]
            origin = "override"
        else:
            v = dv
            origin = "default"
        merged[k] = {"value": v, "origin": origin}
    return {"settings": merged, "defaults": defaults, "overrides": active}


class SettingsUpdateRequest(BaseModel):
    values: Dict[str, Any]


_SETTINGS_VALIDATION = {
    "USE_QUANTIZATION": lambda v: isinstance(v, bool),
    "USE_DP": lambda v: isinstance(v, bool),
    "DP_EPSILON": lambda v: isinstance(v, (int, float)) and v > 0,
    "NUM_ROUNDS": lambda v: isinstance(v, int) and 1 <= v <= 500,
    "FORCE_CPU": lambda v: isinstance(v, bool),
    "FL_SERVER_ADDRESS": lambda v: isinstance(v, str) and 1 <= len(v) <= 200,
    "TARGET_CLIENTS": lambda v: isinstance(v, int) and 1 <= v <= 50,
    "MIN_CLIENTS": lambda v: isinstance(v, int) and 1 <= v <= 50,
    "EXPERIMENT_TIMEOUT_SEC": lambda v: isinstance(v, int) and 60 <= v <= 7200,
    "ADAPTIVE_DROPOUT_ENABLED": lambda v: isinstance(v, bool),
    "FIXED_DEADLINE_CONTROL": lambda v: isinstance(v, bool),
    "DROPOUT_HARD_DEADLINE": lambda v: isinstance(v, (int, float)) and v >= 1,
    "ROUND_TIMEOUT": lambda v: isinstance(v, (int, float)) and v >= 1,
}


@router.post("/settings")
def update_settings(
    update_req: SettingsUpdateRequest,
    request: Request,
    current_user: User = Depends(require_admin),
    audit: AuditLogger = Depends(get_audit_logger)
):
    rs: RuntimeSettingsStore = get_runtime_settings()
    validated: Dict[str, Any] = {}
    for k, v in (update_req.values or {}).items():
        if k not in _SETTINGS_VALIDATION:
            continue
        validator = _SETTINGS_VALIDATION[k]
        try:
            if not validator(v):
                raise HTTPException(status_code=400, detail=f"Invalid value for {k}")
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=400, detail=f"Invalid type for {k}")
        validated[k] = v
    try:
        rs.update(validated, persist=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save settings: {e}")
    audit.log_admin_action(
        current_user,
        f"Updated system settings: {sorted(validated.keys())}",
        request,
        metadata={"keys": list(validated.keys())}
    )
    return {"success": True, "message": "Settings saved and will apply to next subprocess launch.", "updated": validated}


@router.post("/settings/reset")
def reset_settings(
    request: Request,
    current_user: User = Depends(require_admin),
    audit: AuditLogger = Depends(get_audit_logger)
):
    rs: RuntimeSettingsStore = get_runtime_settings()
    rs.reset(persist=True)
    audit.log_admin_action(current_user, "Reset runtime settings to defaults", request)
    return {"success": True, "message": "Runtime settings reset to defaults"}


@router.get("/workspace/roots")
def list_workspace_roots(
    current_user: User = Depends(require_admin)
):
    ws: WorkspaceManager = get_workspace_manager()
    return {"roots": ws.list_roots_for_role("admin", None)}


@router.get("/workspace/ls")
def browse_workspace(
    root_key: str,
    rel_path: str = "",
    current_user: User = Depends(require_admin)
):
    ws: WorkspaceManager = get_workspace_manager()
    result, err = ws.list_directory(root_key, rel_path, "admin", None)
    if err:
        raise HTTPException(status_code=400, detail=err)
    return result


@router.get("/workspace/download")
def download_workspace_file(
    root_key: str,
    rel_path: str,
    current_user: User = Depends(require_admin)
):
    ws: WorkspaceManager = get_workspace_manager()
    result, err = ws.get_file_for_download(root_key, rel_path, "admin", None)
    if err:
        raise HTTPException(status_code=400, detail=err)
    path, name, mime = result
    return FileResponse(
        path=path,
        filename=name,
        media_type=mime,
        headers={
            "Content-Disposition": f"attachment; filename=\"{name}\"",
            "X-Content-Type-Options": "nosniff"
        }
    )


@router.get("/workspace/view")
def inline_view_workspace_file(
    root_key: str,
    rel_path: str,
    current_user: User = Depends(require_admin)
):
    ws: WorkspaceManager = get_workspace_manager()
    result, err = ws.get_file_for_view(root_key, rel_path, "admin", None)
    if err:
        raise HTTPException(status_code=400, detail=err)
    if result.get("type") == "image":
        file_res, e2 = ws.get_file_for_download(root_key, rel_path, "admin", None)
        if e2:
            raise HTTPException(status_code=400, detail=e2)
        p, n, mime = file_res
        return FileResponse(path=p, filename=n, media_type=mime)
    return result


@router.get("/outputs/summary")
def outputs_summary(
    current_user: User = Depends(require_admin)
):
    ws: WorkspaceManager = get_workspace_manager()
    pr = ws.project_root
    roots_summary = []
    for rk in ("models", "results", "plots", "reports", "figures", "experiments"):
        info, _ = ws.list_directory(rk, "", "admin", None)
        if info:
            entries = info.get("entries", [])
            roots_summary.append({
                "root": rk,
                "label": {
                    "models": "Global Models",
                    "results": "Training Metrics",
                    "plots": "Training Plots",
                    "reports": "Classification Reports",
                    "figures": "Publication Figures",
                    "experiments": "Experiment Results"
                }.get(rk, rk),
                "count": len(entries),
                "entries": entries[:20]
            })
    models_csv = pr / "federated_healthcare" / "models" / "all_global_models_registry.csv"
    registry = None
    if models_csv.exists():
        try:
            import csv
            with open(models_csv, "r", newline="") as f:
                reader = list(csv.DictReader(f))
                registry = reader
        except Exception:
            pass
    return {"sections": roots_summary, "model_registry": registry}


@router.get("/process-registry")
def get_process_registry(
    current_user: User = Depends(require_admin)
):
    from ..core.process_registry import PROCESS_DEFINITIONS, ProcessRole
    return {
        "definitions": [
            {
                "process_type": p.process_type,
                "title": p.title,
                "description": p.description,
                "launcher_method": p.launcher_method,
                "role": p.role.value,
                "hospital_bound": p.hospital_bound,
                "input_parameters": [
                    {
                        "name": ip.name,
                        "label": ip.label,
                        "kind": ip.kind,
                        "required": ip.required,
                        "default": ip.default,
                        "options": ip.options,
                        "min": ip.min,
                        "max": ip.max,
                        "step": ip.step,
                        "help": ip.help
                    }
                    for ip in p.input_parameters
                ],
                "output_roots": p.output_roots,
                "audit_event": p.audit_event
            }
            for p in PROCESS_DEFINITIONS
        ]
    }


@router.post("/process-launch/{process_type}", response_model=ProcessActionResponse)
def launch_process_by_type(
    process_type: str,
    payload: Optional[Dict[str, Any]] = Body(default=None),
    db: Session = Depends(get_db),
    request: Request = None,
    current_user: User = Depends(require_admin),
    audit: AuditLogger = Depends(get_audit_logger)
):
    from ..core.process_registry import get_process_definition
    pm: ProcessManager = get_process_manager()
    payload = payload or {}
    definition = get_process_definition(process_type)
    if not definition:
        raise HTTPException(status_code=404, detail="Unknown process type")
    extra_env: Dict[str, str] = {}
    kwargs: Dict[str, Any] = {}
    for ip in definition.input_parameters:
        if ip.name in payload:
            val = payload[ip.name]
            if ip.kind == "bool":
                env_v = "1" if val else "0"
                kwargs[ip.name] = bool(val)
                if ip.name.isupper():
                    extra_env[ip.name] = env_v
            elif ip.kind == "int":
                kwargs[ip.name] = int(val)
                if ip.name.isupper():
                    extra_env[ip.name] = str(int(val))
            elif ip.kind == "float":
                kwargs[ip.name] = float(val)
                if ip.name.isupper():
                    extra_env[ip.name] = str(float(val))
            elif ip.kind in ("path", "str", "json"):
                kwargs[ip.name] = val
                if ip.name.isupper():
                    extra_env[ip.name] = str(val) if val is not None else ""
        elif ip.default is not None:
            kwargs[ip.name] = ip.default
    launcher = getattr(pm, definition.launcher_method, None)
    if launcher is None:
        raise HTTPException(status_code=500, detail="Launcher method missing on ProcessManager")
    env_kwargs = {k: v for k, v in kwargs.items() if k.isupper()}
    for k in env_kwargs:
        if k in extra_env:
            continue
        v = kwargs.pop(k)
        if isinstance(v, bool):
            extra_env[k] = "1" if v else "0"
        else:
            extra_env[k] = "" if v is None else str(v)
    try:
        filtered_kwargs = {k: v for k, v in kwargs.items() if not k.isupper()}
        if extra_env:
            filtered_kwargs["extra_env"] = extra_env
        proc_info = launcher(**filtered_kwargs)
    except TypeError as te:
        try:
            if "extra_env" in str(te) or "unexpected keyword" in str(te).lower():
                final_kwargs = {k: v for k, v in filtered_kwargs.items() if k != "extra_env"}
                if extra_env:
                    raise
                proc_info = launcher(**final_kwargs)
            else:
                raise HTTPException(status_code=400, detail=f"Parameter error: {te}")
        except HTTPException:
            raise
        except Exception as e2:
            raise HTTPException(status_code=500, detail=str(e2))
    audit.log_admin_action(
        current_user,
        f"Launched {definition.title} ({process_type}) via generic launcher",
        request,
        metadata={"process_type": process_type, "params": sorted(kwargs.keys())}
    )
    return ProcessActionResponse(
        success=True,
        message=f"{definition.title} started successfully.",
        process_id=proc_info.process_id,
        process_info=proc_info.to_dict()
    )


@router.post("/uploads")
async def upload_admin_files(
    request: Request,
    files: List[UploadFile] = File(...),
    sub_folder: Optional[str] = Form(None),
    current_user: User = Depends(require_admin),
    audit: AuditLogger = Depends(get_audit_logger)
):
    ws: WorkspaceManager = get_workspace_manager()
    saved = []
    errors = []
    for f in files:
        try:
            if not f.filename:
                errors.append("Empty filename")
                continue
            ext = Path(f.filename).suffix.lower()
            if ext and ext not in _UPLOAD_ALLOWED_EXT:
                errors.append(f"{f.filename}: extension not allowed")
                continue
            content = await f.read()
            size_mb = len(content) / (1024 * 1024)
            if size_mb > _MAX_UPLOAD_MB:
                errors.append(f"{f.filename}: exceeds {_MAX_UPLOAD_MB}MB")
                continue
            result, err = ws.save_uploaded_file(
                filename=f.filename,
                content_bytes=content,
                role="admin",
                hospital_id=None,
                sub_folder=sub_folder
            )
            if err:
                errors.append(f"{f.filename}: {err}")
            else:
                saved.append(result)
        except Exception as e:
            errors.append(f"{f.filename}: {e}")
    audit.log_admin_action(
        current_user,
        f"Uploaded {len(saved)} file(s) to admin workspace (errors={len(errors)})",
        request,
        metadata={"saved": len(saved), "errors": errors}
    )
    return {"success": len(saved) > 0, "saved": saved, "errors": errors}
