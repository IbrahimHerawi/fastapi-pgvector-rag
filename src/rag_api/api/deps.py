"""API dependencies."""

from fastapi import Header, HTTPException, status

from rag_api.core.config import get_settings


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Conditionally enforce X-API-Key based on configured API_KEY."""
    expected_api_key = get_settings().API_KEY
    if not expected_api_key:
        return

    if x_api_key != expected_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
