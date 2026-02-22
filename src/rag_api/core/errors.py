"""Custom error hierarchy and FastAPI exception handlers for rag_api."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from rag_api.core.logging import get_request_id

REQUEST_ID_HEADER = "X-Request-ID"
_DEFAULT_HTTP_MESSAGE = "Request failed."
_DEFAULT_VALIDATION_MESSAGE = "Request validation failed."
_HTTP_422_STATUS = getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422)
_STATUS_TO_CODE = {
    status.HTTP_400_BAD_REQUEST: "bad_request",
    status.HTTP_401_UNAUTHORIZED: "unauthorized",
    status.HTTP_403_FORBIDDEN: "forbidden",
    status.HTTP_404_NOT_FOUND: "not_found",
    status.HTTP_405_METHOD_NOT_ALLOWED: "method_not_allowed",
    status.HTTP_409_CONFLICT: "conflict",
    _HTTP_422_STATUS: "validation_error",
    status.HTTP_500_INTERNAL_SERVER_ERROR: "internal_server_error",
    status.HTTP_502_BAD_GATEWAY: "bad_gateway",
    status.HTTP_503_SERVICE_UNAVAILABLE: "external_service_unavailable",
}


class RagAPIError(Exception):
    """Base exception for rag_api."""


class APIError(RagAPIError):
    """Base API error that can be converted to a stable HTTP payload."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code = "internal_server_error"
    default_message = "Internal server error."

    def __init__(self, message: str | None = None, *, code: str | None = None) -> None:
        self.message = self.default_message if message is None else message
        self.code = self.error_code if code is None else code
        super().__init__(self.message)


class NotFound(APIError):
    status_code = status.HTTP_404_NOT_FOUND
    error_code = "not_found"
    default_message = "Resource not found."


class BadRequest(APIError):
    status_code = status.HTTP_400_BAD_REQUEST
    error_code = "bad_request"
    default_message = "Invalid request."


class ExternalServiceUnavailable(APIError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    error_code = "external_service_unavailable"
    default_message = "External service unavailable."


class ConfigurationError(RagAPIError):
    """Raised when required configuration is missing or invalid."""


def _resolve_request_id(request: Request) -> str:
    request_id = getattr(request.state, "request_id", None)
    if isinstance(request_id, str) and request_id.strip():
        return request_id

    header_request_id = request.headers.get(REQUEST_ID_HEADER)
    if header_request_id and header_request_id.strip():
        request.state.request_id = header_request_id
        return header_request_id

    context_request_id = get_request_id()
    if context_request_id and context_request_id.strip():
        request.state.request_id = context_request_id
        return context_request_id

    generated_request_id = str(uuid4())
    request.state.request_id = generated_request_id
    return generated_request_id


def _detail_to_message(detail: Any, default_message: str) -> str:
    if isinstance(detail, str):
        message = detail.strip()
        if message:
            return message
        return default_message

    if isinstance(detail, dict):
        message_value = detail.get("message")
        if isinstance(message_value, str):
            message = message_value.strip()
            if message:
                return message
        return default_message

    return default_message


def _error_payload(*, code: str, message: str, request_id: str) -> dict[str, str]:
    return {
        "code": code,
        "message": message,
        "request_id": request_id,
    }


def register_exception_handlers(app: FastAPI) -> None:
    """Register app-level exception handlers with a stable response shape."""

    @app.exception_handler(APIError)
    async def _handle_api_error(request: Request, exc: APIError) -> JSONResponse:
        request_id = _resolve_request_id(request)
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(
                code=exc.code,
                message=exc.message,
                request_id=request_id,
            ),
            headers={REQUEST_ID_HEADER: request_id},
        )

    @app.exception_handler(HTTPException)
    async def _handle_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
        request_id = _resolve_request_id(request)
        status_code = exc.status_code
        code = _STATUS_TO_CODE.get(status_code, "http_error")
        message = _detail_to_message(exc.detail, _DEFAULT_HTTP_MESSAGE)
        return JSONResponse(
            status_code=status_code,
            content=_error_payload(
                code=code,
                message=message,
                request_id=request_id,
            ),
            headers={REQUEST_ID_HEADER: request_id},
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_request_validation_error(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        request_id = _resolve_request_id(request)
        first_error = exc.errors()[0] if exc.errors() else {}
        validation_message = first_error.get("msg")
        message = _DEFAULT_VALIDATION_MESSAGE
        if isinstance(validation_message, str) and validation_message.strip():
            message = f"Invalid request: {validation_message}"

        return JSONResponse(
            status_code=_HTTP_422_STATUS,
            content=_error_payload(
                code="validation_error",
                message=message,
                request_id=request_id,
            ),
            headers={REQUEST_ID_HEADER: request_id},
        )
