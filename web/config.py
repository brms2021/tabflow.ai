"""Application configuration loaded from environment."""

import secrets
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "TabFlow.ai"
    secret_key: str = secrets.token_urlsafe(32)
    database_url: str = "sqlite+aiosqlite:///./tabflow.db"
    upload_dir: Path = Path("uploads")
    max_upload_mb: int = 100

    # Pipeline defaults
    default_model: str = "htdemucs_ft"
    default_tuning: str = "standard"
    default_transcriber: str = "basic-pitch"
    enable_llm: bool = True

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
