"""FastAPI application bootstrap."""

from fastapi import FastAPI

from rag_api.api.v1.routers import router
from rag_api.core.config import get_settings

settings = get_settings()
app = FastAPI(title="RAG API")
app.include_router(router, prefix=settings.API_V1_PREFIX)
