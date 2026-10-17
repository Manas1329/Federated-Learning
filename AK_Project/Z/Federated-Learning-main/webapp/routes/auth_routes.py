from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import JSONResponse, Response
from sqlalchemy.orm import Session
from datetime import datetime

from ..database.connection import get_db
from ..core.config import settings
from ..core.security import verify_password, create_access_token
from ..core.dependencies import get_current_user
from ..core.audit import get_audit_logger, AuditLogger
from ..models.db_models import User, UserRole, Hospital
from ..schemas.schemas import UserLogin, Token


router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@router.post("/login", response_model=Token)
def login(
    form_data: UserLogin,
    request: Request,
    db: Session = Depends(get_db),
    audit = Depends(get_audit_logger)
):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        if user:
            audit.log_login(user, request, status="Failed")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated"
        )

    if user.hospital_id:
        hospital = db.query(Hospital).filter(Hospital.id == user.hospital_id).first()
        if hospital and not hospital.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Hospital account is deactivated"
            )
        hospital_name = hospital.name if hospital else None
    else:
        hospital_name = None

    user.last_login = datetime.utcnow()
    db.commit()

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        subject={
            "sub": user.username,
            "role": user.role.value,
            "hospital_id": user.hospital_id
        },
        expires_delta=access_token_expires
    )

    audit.log_login(user, request)

    return Token(
        access_token=access_token,
        token_type="bearer",
        role=user.role.value,
        username=user.username,
        hospital_id=user.hospital_id,
        hospital_name=hospital_name,
        full_name=user.full_name
    )


@router.post("/logout")
def logout(
    request: Request,
    current_user: User = Depends(get_current_user),
    audit = Depends(get_audit_logger)
):
    audit.log_logout(current_user, request)
    return {"success": True, "message": "Logged out successfully"}


@router.get("/me")
def get_current_user_info(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    hospital_name = None
    if current_user.hospital_id:
        hospital = db.query(Hospital).filter(Hospital.id == current_user.hospital_id).first()
        if hospital:
            hospital_name = hospital.name

    return {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "role": current_user.role.value,
        "hospital_id": current_user.hospital_id,
        "hospital_name": hospital_name,
        "is_active": current_user.is_active,
        "last_login": current_user.last_login,
        "created_at": current_user.created_at
    }
