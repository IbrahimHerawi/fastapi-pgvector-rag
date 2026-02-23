"""Structured logging utilities for rag_api."""

from __future__ import annotations

import contextvars
import json
import logging
import re
from contextvars import Token
from typing import Any

_REQUEST_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id",
    default=None,
)
_REQUEST_ID_DEFAULT = "-"
_SAFE_VALUE = re.compile(r"^[A-Za-z0-9_.:/-]+$")


def get_request_id() -> str | None:
    """Return the request ID from the current execution context."""
    return _REQUEST_ID.get()


def set_request_id(request_id: str | None) -> Token[str | None]:
    """Set the request ID for the current execution context."""
    return _REQUEST_ID.set(request_id)


def reset_request_id(token: Token[str | None]) -> None:
    """Reset the request ID using a token returned by set_request_id."""
    _REQUEST_ID.reset(token)


def clear_request_id() -> None:
    """Clear any request ID from the current execution context."""
    _REQUEST_ID.set(None)


def _serialize_value(value: Any) -> str:
    if value is None:
        return _REQUEST_ID_DEFAULT
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)

    text = str(value)
    if _SAFE_VALUE.fullmatch(text):
        return text
    return json.dumps(text, ensure_ascii=True)


class RequestIdFilter(logging.Filter):
    """Attach request_id from context to each log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id() or _REQUEST_ID_DEFAULT
        return True


class StructuredFormatter(logging.Formatter):
    """Format logs as key=value fields."""

    def format(self, record: logging.LogRecord) -> str:
        request_id = getattr(record, "request_id", None) or get_request_id() or _REQUEST_ID_DEFAULT
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "request_id": request_id,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)
        return " ".join(
            f"{key}={_serialize_value(value)}" for key, value in payload.items()
        )


def _ensure_request_id_filter(handler: logging.Handler) -> None:
    if any(isinstance(log_filter, RequestIdFilter) for log_filter in handler.filters):
        return
    handler.addFilter(RequestIdFilter())


def configure_logging(
    level: int | str = logging.INFO,
    *,
    logger: logging.Logger | None = None,
) -> None:
    """Configure structured logging for an existing logger hierarchy."""
    target_logger = logger if logger is not None else logging.getLogger()
    target_logger.setLevel(level)

    if not target_logger.handlers:
        target_logger.addHandler(logging.StreamHandler())

    formatter = StructuredFormatter()
    for handler in target_logger.handlers:
        handler.setFormatter(formatter)
        _ensure_request_id_filter(handler)
