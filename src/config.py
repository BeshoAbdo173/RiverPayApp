"""Configuration settings loaded from environment variables and .env."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings for RiverPay RAG assistant."""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str = ""
    llm_model: str = "gemini-3.5-flash-lite"
    llm_temperature: float = 0.0
    embedding_model: str = "all-MiniLM-L6-v2"
    chroma_persist_dir: str = str(BASE_DIR / "chroma_db")
    chroma_collection: str = "riverpay_kb"
    top_k: int = 5

    data_dir: str = str(BASE_DIR / "data")
    output_dir: str = str(BASE_DIR / "output")


settings = Settings()
