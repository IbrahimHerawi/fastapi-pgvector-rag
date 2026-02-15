"""Application configuration via environment variables."""

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        case_sensitive=True,
        extra="ignore",
    )

    # Application
    APP_ENV: str = "dev"
    API_V1_PREFIX: str = "/api/v1"
    API_KEY: Optional[str] = None

    # Infrastructure defaults aligned with local Docker service names.
    DATABASE_URL: str = "postgresql+psycopg://postgres:postgres@postgres:5432/rag"
    TEST_DATABASE_URL: Optional[str] = None
    OLLAMA_BASE_URL: str = "http://ollama:11434"

    # Models
    OLLAMA_CHAT_MODEL: str = "llama3.1:8b"
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text"
    EMBED_DIM: int = 768

    # Ingestion/chunking
    CHUNK_MAX_CHARS: int = 1200
    CHUNK_OVERLAP_CHARS: int = 200
    MAX_DOC_CHARS: int = 200000

    # Retrieval/runtime
    RETRIEVE_TOP_K: int = 5
    REQUEST_TIMEOUT_S: int = 30


def get_settings() -> Settings:
    return Settings()
