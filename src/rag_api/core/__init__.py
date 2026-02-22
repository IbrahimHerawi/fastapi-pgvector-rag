"""Core package for shared configuration, database, logging, and errors."""

from rag_api.core.config import Settings, get_settings
from rag_api.core.errors import (
    APIError,
    BadRequest,
    ConfigurationError,
    ExternalServiceUnavailable,
    NotFound,
    RagAPIError,
    register_exception_handlers,
)

__all__ = [
    "APIError",
    "BadRequest",
    "ConfigurationError",
    "ExternalServiceUnavailable",
    "NotFound",
    "RagAPIError",
    "Settings",
    "get_settings",
    "register_exception_handlers",
]
