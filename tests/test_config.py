import pytest
from pydantic import ValidationError

from rag_api.core.config import Settings


ENV_KEYS = (
    "APP_ENV",
    "DATABASE_URL",
    "TEST_DATABASE_URL",
    "OLLAMA_BASE_URL",
    "OLLAMA_CHAT_MODEL",
    "OLLAMA_EMBED_MODEL",
    "EMBED_DIM",
    "CHUNK_MAX_CHARS",
    "CHUNK_OVERLAP_CHARS",
    "RETRIEVE_TOP_K",
    "REQUEST_TIMEOUT_S",
    "MAX_DOC_CHARS",
    "API_KEY",
)


def test_settings_defaults_load(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    settings = Settings()

    assert settings.APP_ENV == "dev"
    assert settings.DATABASE_URL == "postgresql+psycopg://postgres:postgres@postgres:5432/rag"
    assert settings.TEST_DATABASE_URL is None
    assert settings.OLLAMA_BASE_URL == "http://ollama:11434"
    assert settings.OLLAMA_CHAT_MODEL == "llama3.1:8b"
    assert settings.OLLAMA_EMBED_MODEL == "nomic-embed-text"
    assert settings.EMBED_DIM == 768
    assert settings.CHUNK_MAX_CHARS == 1200
    assert settings.CHUNK_OVERLAP_CHARS == 200
    assert settings.RETRIEVE_TOP_K == 5
    assert settings.REQUEST_TIMEOUT_S == 30
    assert settings.MAX_DOC_CHARS == 200000
    assert settings.API_KEY is None


def test_settings_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://user:pass@localhost:5432/custom")
    monkeypatch.setenv("TEST_DATABASE_URL", "postgresql+psycopg://user:pass@localhost:5432/custom_test")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("OLLAMA_CHAT_MODEL", "qwen2.5:7b")
    monkeypatch.setenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large")
    monkeypatch.setenv("EMBED_DIM", "1024")
    monkeypatch.setenv("CHUNK_MAX_CHARS", "2048")
    monkeypatch.setenv("CHUNK_OVERLAP_CHARS", "256")
    monkeypatch.setenv("RETRIEVE_TOP_K", "8")
    monkeypatch.setenv("REQUEST_TIMEOUT_S", "9")
    monkeypatch.setenv("MAX_DOC_CHARS", "50000")
    monkeypatch.setenv("API_KEY", "secret-token")

    settings = Settings()

    assert settings.APP_ENV == "test"
    assert settings.DATABASE_URL == "postgresql+psycopg://user:pass@localhost:5432/custom"
    assert settings.TEST_DATABASE_URL == "postgresql+psycopg://user:pass@localhost:5432/custom_test"
    assert settings.OLLAMA_BASE_URL == "http://localhost:11434"
    assert settings.OLLAMA_CHAT_MODEL == "qwen2.5:7b"
    assert settings.OLLAMA_EMBED_MODEL == "mxbai-embed-large"
    assert settings.EMBED_DIM == 1024
    assert settings.CHUNK_MAX_CHARS == 2048
    assert settings.CHUNK_OVERLAP_CHARS == 256
    assert settings.RETRIEVE_TOP_K == 8
    assert settings.REQUEST_TIMEOUT_S == 9
    assert settings.MAX_DOC_CHARS == 50000
    assert settings.API_KEY == "secret-token"


def test_settings_invalid_types_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REQUEST_TIMEOUT_S", "not-an-int")

    with pytest.raises(ValidationError):
        Settings()
