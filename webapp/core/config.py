import os
from pathlib import Path
from datetime import timedelta


BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "database"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_FILE = DB_DIR / "federated_healthcare.db"


class Settings:
    APP_NAME: str = "Federated Healthcare AI Platform"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = os.getenv("DEBUG", "1") == "1"

    SECRET_KEY: str = os.getenv(
        "SECRET_KEY",
        "federated-healthcare-ai-super-secret-key-change-in-production-2024"
    )
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 8

    _db_path = os.getenv("DB_FILE", str(DB_FILE.resolve()))
    _db_path_posix = Path(_db_path).resolve().as_posix()
    if os.name == "nt":
        DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{_db_path_posix}")
    else:
        DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:////{_db_path_posix}")

    USE_QUANTIZATION: bool = os.getenv("USE_QUANTIZATION", "1") == "1"
    USE_DP: bool = os.getenv("USE_DP", "0") == "1"
    DP_EPSILON: float = float(os.getenv("DP_EPSILON", "1.0"))
    TOTAL_ROUNDS: int = int(os.getenv("NUM_ROUNDS", "20"))

    ADMIN_USERNAME: str = os.getenv("ADMIN_USERNAME", "admin")
    ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "admin123")
    ADMIN_EMAIL: str = os.getenv("ADMIN_EMAIL", "admin@federated.ai")

    FL_SERVER_ADDRESS: str = os.getenv("FL_SERVER_ADDRESS", "localhost:8080")


settings = Settings()
