from datetime import datetime
from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Float, Text, Enum
from sqlalchemy.orm import relationship
import enum

from ..database.connection import Base


class UserRole(str, enum.Enum):
    ADMIN = "admin"
    HOSPITAL = "hospital"


class TrustStatus(str, enum.Enum):
    TRUSTED = "Trusted"
    SUSPICIOUS = "Suspicious"
    UNTRUSTED = "Untrusted"
    INACTIVE = "Inactive"


class TrainingStatus(str, enum.Enum):
    NOT_STARTED = "Not Started"
    IN_PROGRESS = "In Progress"
    COMPLETED = "Completed"
    FAILED = "Failed"


class AuditEventType(str, enum.Enum):
    LOGIN = "Login"
    LOGOUT = "Logout"
    TRAINING_START = "Training Start"
    TRAINING_COMPLETE = "Training Complete"
    TRAINING_FAILED = "Training Failed"
    MODEL_UPLOAD = "Model Upload"
    MODEL_AGGREGATION = "Model Aggregation"
    WEIGHTS_SENT = "Weights Sent"
    UNLEARNING_REQUEST = "Unlearning Request"
    PRIVACY_CHECK = "Privacy Check"
    TRUST_SCORE_UPDATE = "Trust Score Update"
    ADMIN_ACTION = "Admin Action"
    SYSTEM_EVENT = "System Event"
    DATA_ACTION = "Data Action"
    UPLOAD = "Upload"
    ERROR = "Error"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    role = Column(Enum(UserRole), default=UserRole.HOSPITAL, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_login = Column(DateTime)

    hospital_id = Column(Integer, ForeignKey("hospitals.id"), nullable=True)
    hospital = relationship("Hospital", back_populates="users")
    audit_logs = relationship("AuditLog", back_populates="user")


class Hospital(Base):
    __tablename__ = "hospitals"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, nullable=False)
    code = Column(String(50), unique=True, nullable=False)
    location = Column(String(255))
    contact_email = Column(String(255))
    contact_phone = Column(String(50))
    trust_score = Column(Float, default=0.85, nullable=False)
    trust_status = Column(Enum(TrustStatus), default=TrustStatus.TRUSTED, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    total_patients = Column(Integer, default=0)
    total_training_samples = Column(Integer, default=0)
    current_accuracy = Column(Float, default=0.0)
    dp_epsilon = Column(Float, default=1.0)
    quantization_enabled = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_seen = Column(DateTime)

    users = relationship("User", back_populates="hospital")
    training_rounds = relationship("TrainingRound", back_populates="hospital")
    model_updates = relationship("ModelUpdate", back_populates="hospital")
    unlearning_requests = relationship("UnlearningRequest", back_populates="hospital")
    local_activities = relationship("AuditLog", back_populates="hospital")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)
    event_type = Column(Enum(AuditEventType), nullable=False)
    description = Column(Text, nullable=False)
    status = Column(String(50), default="Success")
    ip_address = Column(String(100))
    user_agent = Column(String(500))
    metadata_json = Column(Text)

    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    user = relationship("User", back_populates="audit_logs")

    hospital_id = Column(Integer, ForeignKey("hospitals.id"), nullable=True)
    hospital = relationship("Hospital", back_populates="local_activities")


class GlobalTrainingRound(Base):
    __tablename__ = "global_training_rounds"

    id = Column(Integer, primary_key=True, index=True)
    round_number = Column(Integer, unique=True, nullable=False)
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime)
    total_clients = Column(Integer, default=0)
    participating_clients = Column(Integer, default=0)
    global_accuracy = Column(Float)
    global_loss = Column(Float)
    best_accuracy = Column(Float)
    avg_update_size = Column(Float)
    status = Column(Enum(TrainingStatus), default=TrainingStatus.NOT_STARTED, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    client_updates = relationship("ModelUpdate", back_populates="global_round")


class TrainingRound(Base):
    __tablename__ = "training_rounds"

    id = Column(Integer, primary_key=True, index=True)
    round_number = Column(Integer, nullable=False)
    hospital_id = Column(Integer, ForeignKey("hospitals.id"), nullable=False)
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime)
    local_epochs = Column(Integer, default=5)
    training_loss = Column(Float)
    training_accuracy = Column(Float)
    validation_loss = Column(Float)
    validation_accuracy = Column(Float)
    num_samples = Column(Integer, default=0)
    status = Column(Enum(TrainingStatus), default=TrainingStatus.NOT_STARTED, nullable=False)
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    hospital = relationship("Hospital", back_populates="training_rounds")


class ModelUpdate(Base):
    __tablename__ = "model_updates"

    id = Column(Integer, primary_key=True, index=True)
    hospital_id = Column(Integer, ForeignKey("hospitals.id"), nullable=False)
    global_round_id = Column(Integer, ForeignKey("global_training_rounds.id"), nullable=True)
    round_number = Column(Integer, nullable=False)
    received_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    update_size_bytes = Column(Integer, default=0)
    is_quantized = Column(Boolean, default=False)
    has_dp_noise = Column(Boolean, default=False)
    dp_epsilon_used = Column(Float)
    update_quality_score = Column(Float, default=1.0)
    contribution_weight = Column(Float, default=1.0)
    status = Column(String(50), default="Received")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    hospital = relationship("Hospital", back_populates="model_updates")
    global_round = relationship("GlobalTrainingRound", back_populates="client_updates")


class UnlearningRequest(Base):
    __tablename__ = "unlearning_requests"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String(50), unique=True, nullable=False)
    hospital_id = Column(Integer, ForeignKey("hospitals.id"), nullable=False)
    submitted_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    reason = Column(Text, nullable=False)
    patient_identifiers = Column(Text)
    current_step = Column(Integer, default=1)
    total_steps = Column(Integer, default=5)
    progress_percent = Column(Float, default=0.0)
    status = Column(String(50), default="Pending")
    resolved_at = Column(DateTime)
    resolution_notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    hospital = relationship("Hospital", back_populates="unlearning_requests")


class SystemMetrics(Base):
    __tablename__ = "system_metrics"

    id = Column(Integer, primary_key=True, index=True)
    metric_name = Column(String(100), unique=True, nullable=False)
    metric_value = Column(Text)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
