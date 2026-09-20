from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from pathlib import Path

from ..database.connection import get_db
from ..core.dependencies import get_current_user_from_cookie
from ..models.db_models import User, UserRole


BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

router = APIRouter(tags=["Views"])


def _ensure_role(current_user, required_role: str):
    if not current_user:
        return False
    if required_role == "admin" and current_user.role != UserRole.ADMIN:
        return False
    if required_role == "hospital" and current_user.role != UserRole.HOSPITAL:
        return False
    return True


@router.get("/", response_class=HTMLResponse)
async def root(request: Request, current_user: User = Depends(get_current_user_from_cookie)):
    if not current_user:
        return RedirectResponse(url="/login")
    if current_user.role == UserRole.ADMIN:
        return RedirectResponse(url="/admin/dashboard")
    return RedirectResponse(url="/hospital/dashboard")


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, current_user: User = Depends(get_current_user_from_cookie)):
    if current_user:
        if current_user.role == UserRole.ADMIN:
            return RedirectResponse(url="/admin/dashboard")
        return RedirectResponse(url="/hospital/dashboard")
    return templates.TemplateResponse("login.html", {"request": request})


@router.get("/admin")
async def admin_base():
    return RedirectResponse(url="/admin/dashboard")


@router.get("/admin/dashboard", response_class=HTMLResponse)
async def admin_dashboard(
    request: Request,
    current_user: User = Depends(get_current_user_from_cookie)
):
    if not _ensure_role(current_user, "admin"):
        return RedirectResponse(url="/login")
    return templates.TemplateResponse("admin_dashboard.html", {
        "request": request,
        "user": current_user,
        "current_datetime": getattr(request.state, "current_datetime", "")
    })


@router.get("/admin/hospitals", response_class=HTMLResponse)
async def admin_hospitals(
    request: Request,
    current_user: User = Depends(get_current_user_from_cookie)
):
    if not _ensure_role(current_user, "admin"):
        return RedirectResponse(url="/login")
    return templates.TemplateResponse("admin_hospitals.html", {
        "request": request,
        "user": current_user
    })


@router.get("/admin/global-model", response_class=HTMLResponse)
async def admin_global_model(
    request: Request,
    current_user: User = Depends(get_current_user_from_cookie)
):
    if not _ensure_role(current_user, "admin"):
        return RedirectResponse(url="/login")
    return templates.TemplateResponse("admin_global_model.html", {
        "request": request,
        "user": current_user
    })


@router.get("/admin/training", response_class=HTMLResponse)
async def admin_training(
    request: Request,
    current_user: User = Depends(get_current_user_from_cookie)
):
    if not _ensure_role(current_user, "admin"):
        return RedirectResponse(url="/login")
    return templates.TemplateResponse("admin_training.html", {
        "request": request,
        "user": current_user
    })


@router.get("/admin/trust", response_class=HTMLResponse)
async def admin_trust(
    request: Request,
    current_user: User = Depends(get_current_user_from_cookie)
):
    if not _ensure_role(current_user, "admin"):
        return RedirectResponse(url="/login")
    return templates.TemplateResponse("admin_trust.html", {
        "request": request,
        "user": current_user
    })


@router.get("/admin/unlearning", response_class=HTMLResponse)
async def admin_unlearning(
    request: Request,
    current_user: User = Depends(get_current_user_from_cookie)
):
    if not _ensure_role(current_user, "admin"):
        return RedirectResponse(url="/login")
    return templates.TemplateResponse("admin_unlearning.html", {
        "request": request,
        "user": current_user
    })


@router.get("/admin/audit", response_class=HTMLResponse)
async def admin_audit(
    request: Request,
    current_user: User = Depends(get_current_user_from_cookie)
):
    if not _ensure_role(current_user, "admin"):
        return RedirectResponse(url="/login")
    return templates.TemplateResponse("admin_audit.html", {
        "request": request,
        "user": current_user
    })


@router.get("/admin/settings", response_class=HTMLResponse)
async def admin_settings(
    request: Request,
    current_user: User = Depends(get_current_user_from_cookie)
):
    if not _ensure_role(current_user, "admin"):
        return RedirectResponse(url="/login")
    return templates.TemplateResponse("admin_settings.html", {
        "request": request,
        "user": current_user
    })


@router.get("/admin/operations", response_class=HTMLResponse)
async def admin_operations(
    request: Request,
    current_user: User = Depends(get_current_user_from_cookie)
):
    if not _ensure_role(current_user, "admin"):
        return RedirectResponse(url="/login")
    return templates.TemplateResponse("admin_operations.html", {
        "request": request,
        "user": current_user
    })


@router.get("/admin/outputs", response_class=HTMLResponse)
async def admin_outputs(
    request: Request,
    current_user: User = Depends(get_current_user_from_cookie)
):
    if not _ensure_role(current_user, "admin"):
        return RedirectResponse(url="/login")
    return templates.TemplateResponse("admin_outputs.html", {
        "request": request,
        "user": current_user
    })


@router.get("/admin/experiments", response_class=HTMLResponse)
async def admin_experiments(
    request: Request,
    current_user: User = Depends(get_current_user_from_cookie)
):
    if not _ensure_role(current_user, "admin"):
        return RedirectResponse(url="/login")
    return templates.TemplateResponse("admin_experiments.html", {
        "request": request,
        "user": current_user
    })


@router.get("/hospital")
async def hospital_base():
    return RedirectResponse(url="/hospital/dashboard")


@router.get("/hospital/dashboard", response_class=HTMLResponse)
async def hospital_dashboard(
    request: Request,
    current_user: User = Depends(get_current_user_from_cookie)
):
    if not _ensure_role(current_user, "hospital"):
        return RedirectResponse(url="/login")
    return templates.TemplateResponse("hospital_dashboard.html", {
        "request": request,
        "user": current_user
    })


@router.get("/hospital/training", response_class=HTMLResponse)
async def hospital_training(
    request: Request,
    current_user: User = Depends(get_current_user_from_cookie)
):
    if not _ensure_role(current_user, "hospital"):
        return RedirectResponse(url="/login")
    return templates.TemplateResponse("hospital_training.html", {
        "request": request,
        "user": current_user
    })


@router.get("/hospital/performance", response_class=HTMLResponse)
async def hospital_performance(
    request: Request,
    current_user: User = Depends(get_current_user_from_cookie)
):
    if not _ensure_role(current_user, "hospital"):
        return RedirectResponse(url="/login")
    return templates.TemplateResponse("hospital_performance.html", {
        "request": request,
        "user": current_user
    })


@router.get("/hospital/privacy", response_class=HTMLResponse)
async def hospital_privacy(
    request: Request,
    current_user: User = Depends(get_current_user_from_cookie)
):
    if not _ensure_role(current_user, "hospital"):
        return RedirectResponse(url="/login")
    return templates.TemplateResponse("hospital_privacy.html", {
        "request": request,
        "user": current_user
    })


@router.get("/hospital/unlearning", response_class=HTMLResponse)
async def hospital_unlearning(
    request: Request,
    current_user: User = Depends(get_current_user_from_cookie)
):
    if not _ensure_role(current_user, "hospital"):
        return RedirectResponse(url="/login")
    return templates.TemplateResponse("hospital_unlearning.html", {
        "request": request,
        "user": current_user
    })


@router.get("/hospital/activity", response_class=HTMLResponse)
async def hospital_activity(
    request: Request,
    current_user: User = Depends(get_current_user_from_cookie)
):
    if not _ensure_role(current_user, "hospital"):
        return RedirectResponse(url="/login")
    return templates.TemplateResponse("hospital_activity.html", {
        "request": request,
        "user": current_user
    })


@router.get("/hospital/settings", response_class=HTMLResponse)
async def hospital_settings(
    request: Request,
    current_user: User = Depends(get_current_user_from_cookie)
):
    if not _ensure_role(current_user, "hospital"):
        return RedirectResponse(url="/login")
    return templates.TemplateResponse("hospital_settings.html", {
        "request": request,
        "user": current_user
    })


@router.get("/hospital/outputs", response_class=HTMLResponse)
async def hospital_outputs(
    request: Request,
    current_user: User = Depends(get_current_user_from_cookie)
):
    if not _ensure_role(current_user, "hospital"):
        return RedirectResponse(url="/login")
    return templates.TemplateResponse("hospital_outputs.html", {
        "request": request,
        "user": current_user
    })
