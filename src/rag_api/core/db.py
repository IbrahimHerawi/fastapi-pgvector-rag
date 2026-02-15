"""Database placeholders for rag_api."""

from rag_api.core.config import get_settings


def get_database_url() -> str:
    return get_settings().DATABASE_URL
