import json
from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
from fastapi import Request, Depends

from ..database.connection import get_db
from ..models.db_models import AuditLog, AuditEventType, User, Hospital


class AuditLogger:
    def __init__(self, db: Session):
        self.db = db

    def log(
        self,
        event_type: AuditEventType,
        description: str,
        status: str = "Success",
        user: Optional[User] = None,
        hospital: Optional[Hospital] = None,
        request: Optional[Request] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AuditLog:
        ip_address = None
        user_agent = None
        if request:
            try:
                ip_address = request.client.host if request.client else None
            except Exception:
                pass
            user_agent = request.headers.get("user-agent")

        metadata_json = None
        if metadata:
            try:
                metadata_json = json.dumps(metadata)
            except Exception:
                pass

        audit_entry = AuditLog(
            timestamp=datetime.utcnow(),
            event_type=event_type,
            description=description,
            status=status,
            ip_address=ip_address,
            user_agent=user_agent,
            metadata_json=metadata_json,
            user_id=user.id if user else None,
            hospital_id=hospital.id if hospital else None
        )
        self.db.add(audit_entry)
        self.db.commit()
        self.db.refresh(audit_entry)
        return audit_entry

    def log_login(self, user: User, request: Request, status: str = "Success"):
        return self.log(
            event_type=AuditEventType.LOGIN,
            description=f"User '{user.username}' logged in successfully" if status == "Success" else f"Failed login attempt for '{user.username}'",
            status=status,
            user=user,
            hospital=user.hospital if user.hospital_id else None,
            request=request
        )

    def log_logout(self, user: User, request: Request):
        return self.log(
            event_type=AuditEventType.LOGOUT,
            description=f"User '{user.username}' logged out",
            user=user,
            hospital=user.hospital if user.hospital_id else None,
            request=request
        )

    def log_training_start(self, user: User, hospital: Hospital, round_num: int, request: Request = None):
        return self.log(
            event_type=AuditEventType.TRAINING_START,
            description=f"Local training started at {hospital.name} for round {round_num}",
            user=user,
            hospital=hospital,
            request=request,
            metadata={"round": round_num, "hospital": hospital.name}
        )

    def log_training_complete(self, user: User, hospital: Hospital, round_num: int, accuracy: float, request: Request = None):
        return self.log(
            event_type=AuditEventType.TRAINING_COMPLETE,
            description=f"Local training completed at {hospital.name} for round {round_num} with accuracy {accuracy:.2f}%",
            user=user,
            hospital=hospital,
            request=request,
            metadata={"round": round_num, "accuracy": accuracy, "hospital": hospital.name}
        )

    def log_model_upload(self, user: User, hospital: Hospital, round_num: int, size_mb: float, request: Request = None):
        return self.log(
            event_type=AuditEventType.MODEL_UPLOAD,
            description=f"Model update ({size_mb:.1f} MB) sent from {hospital.name} for round {round_num}",
            user=user,
            hospital=hospital,
            request=request,
            metadata={"round": round_num, "size_mb": size_mb, "hospital": hospital.name}
        )

    def log_model_aggregation(self, round_num: int, num_clients: int, global_accuracy: float):
        return self.log(
            event_type=AuditEventType.MODEL_AGGREGATION,
            description=f"Global model aggregated for round {round_num} with {num_clients} clients. Accuracy: {global_accuracy:.2f}%",
            metadata={"round": round_num, "num_clients": num_clients, "accuracy": global_accuracy}
        )

    def log_unlearning_request(self, user: User, hospital: Hospital, request_id: str, request: Request = None):
        return self.log(
            event_type=AuditEventType.UNLEARNING_REQUEST,
            description=f"Unlearning request {request_id} submitted by {hospital.name}",
            user=user,
            hospital=hospital,
            request=request,
            metadata={"request_id": request_id, "hospital": hospital.name}
        )

    def log_privacy_check(self, user: User, hospital: Hospital, request: Request = None):
        return self.log(
            event_type=AuditEventType.PRIVACY_CHECK,
            description=f"Privacy check completed for {hospital.name}",
            user=user,
            hospital=hospital,
            request=request,
            metadata={"hospital": hospital.name, "dp_epsilon": hospital.dp_epsilon}
        )

    def log_admin_action(self, admin_user: User, action_description: str, request: Request = None, status: str = "Success"):
        return self.log(
            event_type=AuditEventType.ADMIN_ACTION,
            description=f"Admin action: {action_description}",
            status=status,
            user=admin_user,
            request=request
        )

    def log_error(self, description: str, user: Optional[User] = None, hospital: Optional[Hospital] = None, request: Request = None, metadata: Optional[Dict[str, Any]] = None):
        return self.log(
            event_type=AuditEventType.ERROR,
            description=description,
            status="Error",
            user=user,
            hospital=hospital,
            request=request,
            metadata=metadata
        )


def get_audit_logger(db: Session = Depends(get_db)):
    return AuditLogger(db)
