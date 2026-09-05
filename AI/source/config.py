import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=_env_path, override=False)
except ImportError:
    pass

DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "sentineldb")
DB_USER = os.environ.get("DB_USER", "sentinel")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")

_url_from_env = os.environ.get("DATABASE_URL", "")

DATABASE_URL = _url_from_env or (
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

FLASK_SECRET_KEY = os.environ.get(
    "FLASK_SECRET_KEY",
    "change_me_before_production"
)

PG_WEBCAM_COOLDOWN = int(
    os.environ.get("PG_WEBCAM_COOLDOWN", "30")
)