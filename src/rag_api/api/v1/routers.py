"""Versioned API routers."""

from fastapi import APIRouter

from rag_api.core.config import get_settings

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    settings = get_settings()
    return {"status": "ok", "app_env": settings.APP_ENV}

