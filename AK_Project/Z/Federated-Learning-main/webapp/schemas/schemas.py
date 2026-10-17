from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from typing import Optional, List, Dict, Any
from ..models.db_models import UserRole, TrustStatus, TrainingStatus, AuditEventType


PN = {"protected_namespaces": ()}


class Token(BaseModel):
    model_config = ConfigDict(**PN)
    access_token: str
    token_type: str = "bearer"
    role: str
    username: str
    hospital_id: Optional[int] = None
    hospital_name: Optional[str] = None
    full_name: Optional[str] = None


class TokenData(BaseModel):
    model_config = ConfigDict(**PN)
    username: Optional[str] = None
    role: Optional[str] = None
    hospital_id: Optional[int] = None


class UserLogin(BaseModel):
    model_config = ConfigDict(**PN)
    username: str = Field(..., min_length=3, max_length=100)
    password: str = Field(..., min_length=6, max_length=255)


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, **PN)
    id: int
    username: str
    email: Optional[str]
    full_name: str
    role: str
    is_active: bool
    hospital_id: Optional[int]
    created_at: datetime
    last_login: Optional[datetime]


class HospitalBase(BaseModel):
    model_config = ConfigDict(**PN)
    name: str
    code: str
    location: Optional[str] = None
    contact_email: Optional[str] = None
    contact_phone: Optional[str] = None


class HospitalCreate(HospitalBase):
    admin_username: str
    admin_password: str
    admin_full_name: str
    admin_email: Optional[str] = None


class HospitalResponse(HospitalBase):
    model_config = ConfigDict(from_attributes=True, **PN)
    id: int
    trust_score: float
    trust_status: str
    is_active: bool
    total_patients: int
    total_training_samples: int
    current_accuracy: float
    dp_epsilon: float
    quantization_enabled: bool
    created_at: datetime
    last_seen: Optional[datetime]


class HospitalDashboardStats(BaseModel):
    model_config = ConfigDict(**PN)
    local_accuracy: float
    accuracy_change: float
    training_status: str
    current_round: int
    total_rounds: int
    next_round_minutes: Optional[int]
    privacy_enabled: bool
    dp_epsilon: float
    data_sent_mb: float
    trust_score: float
    trust_status: str
    local_epochs: int
    training_loss: float
    model_accuracy: float


class AdminDashboardStats(BaseModel):
    model_config = ConfigDict(**PN)
    active_hospitals: int
    total_hospitals: int
    participation_rate: float
    global_accuracy: float
    accuracy_change: float
    current_round: int
    rounds_remaining: int
    trusted_clients: int
    trusted_percent: float
    suspicious_clients: int
    suspicious_percent: float
    untrusted_clients: int
    inactive_clients: int
    unlearning_requests: int
    unlearning_pending: int
    best_accuracy: float
    best_accuracy_round: int
    average_accuracy: float
    best_loss: float
    best_loss_round: int
    current_loss: float
    total_updates_received: int
    updates_this_round: int
    avg_update_size_raw_kb: float
    avg_update_size_quantized_kb: float
    compression_ratio: float
    avg_dp_epsilon: float


class RoundPerformance(BaseModel):
    model_config = ConfigDict(**PN)
    round: int
    training_accuracy: float
    validation_accuracy: float


class GlobalRoundPerformance(BaseModel):
    model_config = ConfigDict(**PN)
    round: int
    accuracy: float
    loss: Optional[float] = None


class HospitalStatusItem(BaseModel):
    model_config = ConfigDict(**PN)
    id: int
    name: str
    location: Optional[str]
    status: str
    accuracy: float
    trust_score: float
    update_quality: float
    last_update: Optional[datetime]


class AuditLogResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, **PN)
    id: int
    timestamp: datetime
    event_type: str
    description: str
    status: str
    ip_address: Optional[str]
    username: Optional[str]
    hospital_name: Optional[str]


class AuditLogCreate(BaseModel):
    model_config = ConfigDict(**PN)
    event_type: AuditEventType
    description: str
    status: str = "Success"
    metadata: Optional[Dict[str, Any]] = None


class ModelUpdateResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, **PN)
    id: int
    hospital_name: str
    round_number: int
    received_at: datetime
    update_size_mb: float
    is_quantized: bool
    has_dp_noise: bool
    update_quality_score: float
    status: str


class UnlearningRequestCreate(BaseModel):
    model_config = ConfigDict(**PN)
    reason: str
    patient_identifiers: Optional[str] = None


class UnlearningRequestResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, **PN)
    id: int
    request_id: str
    hospital_name: str
    submitted_at: datetime
    reason: str
    current_step: int
    total_steps: int
    progress_percent: float
    status: str
    resolved_at: Optional[datetime]


class TrainingStartRequest(BaseModel):
    model_config = ConfigDict(**PN)
    local_epochs: int = 5
    use_dp: Optional[bool] = None
    use_quantization: Optional[bool] = None


class TrainingActionResponse(BaseModel):
    model_config = ConfigDict(**PN)
    success: bool
    message: str
    training_round_id: Optional[int] = None
