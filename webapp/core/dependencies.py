from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from typing import Optional

from ..database.connection import get_db
from ..core.security import decode_access_token
from ..models.db_models import User, UserRole, Hospital
from ..core.audit import AuditLogger, get_audit_logger


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)


async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not token:
        raise credentials_exception

    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception

    username: str = payload.get("sub")
    if username is None:
        raise credentials_exception

    user = db.query(User).filter(User.username == username).first()
    if user is None or not user.is_active:
        raise credentials_exception

    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


def require_admin(
    current_user: User = Depends(get_current_active_user)
) -> User:
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


def require_hospital(
    current_user: User = Depends(get_current_active_user)
) -> User:
    if current_user.role != UserRole.HOSPITAL:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Hospital access required"
        )
    if not current_user.hospital_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Hospital account not properly configured"
        )
    return current_user


def get_hospital_for_user(
    current_user: User = Depends(require_hospital),
    db: Session = Depends(get_db)
) -> Hospital:
    hospital = db.query(Hospital).filter(Hospital.id == current_user.hospital_id).first()
    if not hospital:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Hospital not found"
        )
    if not hospital.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Hospital account is inactive"
        )
    return hospital


async def get_current_user_from_cookie(
    request: Request,
    db: Session = Depends(get_db)
) -> Optional[User]:
    token = request.cookies.get("access_token")
    if not token:
        return None
    token = token.replace("Bearer ", "")
    payload = decode_access_token(token)
    if payload is None:
        return None
    username: str = payload.get("sub")
    if username is None:
        return None
    user = db.query(User).filter(User.username == username).first()
    if user is None or not user.is_active:
        return None
    return user
