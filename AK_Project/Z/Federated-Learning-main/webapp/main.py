import sys
import os
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "federated_healthcare"))

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import inspect

from .core.config import settings
from .database.connection import Base, engine, SessionLocal
from .routes import auth_routes, admin_routes, hospital_routes, view_routes, ws_routes
from .core.audit import get_audit_logger, AuditLogger
from .models.db_models import AuditEventType, SystemMetrics
from .core.process_manager import get_process_manager

DB_PATH = Path(settings.DATABASE_URL.replace("sqlite:///", ""))
DB_PARENT = DB_PATH.parent


def ensure_db_initialized():
    if not DB_PARENT.exists():
        DB_PARENT.mkdir(parents=True, exist_ok=True)
    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    if not table_names or "users" not in table_names:
        print("📦 Creating database tables...")
        Base.metadata.create_all(bind=engine)
        print("✅ Tables created.")
        print("🌱 Running seed script...")
        try:
            from .database.init_db import init_db
            init_db(reset_first=False)
            print("✅ Seed complete.")
        except Exception as exc:
            print(f"⚠️  Seed failed during startup: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_db_initialized()
    db = SessionLocal()
    try:
        try:
            audit = get_audit_logger(db)
            audit.log(
                event_type=AuditEventType.SYSTEM_EVENT,
                description=f"Federated Healthcare AI Platform v{settings.APP_VERSION} started successfully",
                status="Success"
            )
        except Exception:
            pass
        yield
        try:
            audit = get_audit_logger(db)
            audit.log(
                event_type=AuditEventType.SYSTEM_EVENT,
                description=f"Platform shutdown initiated",
                status="Success"
            )
        except Exception:
            pass
    finally:
        db.close()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Secure Federated Learning Platform for Healthcare AI with Multi-Tenancy",
    lifespan=lifespan,
    debug=settings.DEBUG
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.middleware("http")
async def add_current_time(request: Request, call_next):
    request.state.current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    response = await call_next(request)
    return response


@app.get("/api/health")
async def health_check():
    pm = get_process_manager()
    server_procs = pm.get_processes_by_type("fl_server")
    running_server = any(p.status == "running" for p in server_procs)
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "services": {
            "database": True,
            "authentication": True,
            "audit_logger": True,
            "websocket_server": True,
            "process_manager": True,
            "federated_controller": "active" if running_server else "standby"
        },
        "fl_server": {
            "running": running_server,
            "processes": [p.to_dict() for p in server_procs]
        }
    }


app.include_router(view_routes.router)
app.include_router(auth_routes.router)
app.include_router(admin_routes.router)
app.include_router(hospital_routes.router)
app.include_router(ws_routes.router)


@app.get("/api/processes")
async def list_all_processes():
    pm = get_process_manager()
    return {
        "processes": [p.to_dict() for p in pm.list_processes()]
    }


def run():
    import uvicorn
    print(f"\n{'=' * 70}")
    print(f"  🏥 {settings.APP_NAME} v{settings.APP_VERSION}")
    print(f"{'=' * 70}")
    print(f"  🌐 Central URL:   http://localhost:8000/")
    print(f"  🔧 API Docs:      http://localhost:8000/docs")
    print(f"  🛡️  Admin Login:   admin / admin123")
    print(f"  🏥 Hospitals:     hospital_a through hospital_f")
    print(f"                    (password: hospital123)")
    print(f"{'=' * 70}\n")
    uvicorn.run(
        "webapp.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info"
    )


if __name__ == "__main__":
    run()
