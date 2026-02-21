"""Versioned API routers."""

from fastapi import APIRouter, Depends

from rag_api.api.deps import require_api_key
from rag_api.core.config import get_settings

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.get("/health")
async def health() -> dict[str, str]:
    settings = get_settings()
    return {"status": "ok", "app_env": settings.APP_ENV}
