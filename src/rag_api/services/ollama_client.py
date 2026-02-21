"""Async Ollama client wrapper for embeddings and chat."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx

from rag_api.core.config import get_settings
from rag_api.core.errors import RagAPIError


class OllamaClientError(RagAPIError):
    """Raised when Ollama requests fail or return invalid data."""


class OllamaClient:
    """Thin async wrapper around Ollama HTTP APIs."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        chat_model: str | None = None,
        embed_model: str | None = None,
        embed_dim: int | None = None,
        timeout_s: float | None = None,
        batch_size: int = 32,
        max_retries: int = 2,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        settings = get_settings()
        self._chat_model = settings.OLLAMA_CHAT_MODEL if chat_model is None else chat_model
        self._embed_model = settings.OLLAMA_EMBED_MODEL if embed_model is None else embed_model
        self._embed_dim = settings.EMBED_DIM if embed_dim is None else embed_dim
        self._max_retries = max_retries
        self._attempt_count = self._max_retries + 1
        self._batch_size = batch_size

        if self._batch_size <= 0:
            msg = "batch_size must be greater than 0"
            raise ValueError(msg)
        if self._max_retries < 0:
            msg = "max_retries must be >= 0"
            raise ValueError(msg)
        if self._embed_dim <= 0:
            msg = "embed_dim must be greater than 0"
            raise ValueError(msg)

        resolved_base_url = (settings.OLLAMA_BASE_URL if base_url is None else base_url).rstrip("/")
        resolved_timeout = timeout_s if timeout_s is not None else float(settings.REQUEST_TIMEOUT_S)

        if client is None:
            self._client = httpx.AsyncClient(base_url=resolved_base_url, timeout=resolved_timeout)
            self._owns_client = True
        else:
            self._client = client
            self._owns_client = False

    async def __aenter__(self) -> OllamaClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            payload = {"model": self._embed_model, "input": batch}
            data = await self._post_json_with_retries(path="/api/embed", payload=payload)
            vectors.extend(self._extract_and_validate_embeddings(data=data, expected_count=len(batch)))

        return vectors

    async def chat(self, messages: list[dict[str, Any]]) -> str:
        payload = {"model": self._chat_model, "messages": messages, "stream": False}
        data = await self._post_json_with_retries(path="/api/chat", payload=payload)

        message = data.get("message")
        if not isinstance(message, dict):
            msg = "Ollama chat response missing 'message' object."
            raise OllamaClientError(msg)

        content = message.get("content")
        if not isinstance(content, str):
            msg = "Ollama chat response missing 'message.content' string."
            raise OllamaClientError(msg)

        return content

    async def _post_json_with_retries(self, *, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None

        for _attempt in range(1, self._attempt_count + 1):
            try:
                response = await self._client.post(path, json=payload)
                if response.status_code >= 500:
                    msg = f"Ollama request to {path} failed with status {response.status_code}."
                    raise httpx.HTTPStatusError(msg, request=response.request, response=response)

                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    msg = f"Ollama response for {path} was not a JSON object."
                    raise OllamaClientError(msg)
                return data
            except httpx.TransportError as exc:
                last_error = exc
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                if status_code >= 500:
                    last_error = exc
                else:
                    detail = exc.response.text.strip()
                    msg = f"Ollama request to {path} failed with status {status_code}."
                    if detail:
                        msg = f"{msg} Detail: {detail}"
                    raise OllamaClientError(msg) from exc
            except ValueError as exc:
                msg = f"Ollama response for {path} could not be decoded as JSON."
                raise OllamaClientError(msg) from exc

        if isinstance(last_error, httpx.HTTPStatusError):
            status_code = last_error.response.status_code
            msg = (
                f"Ollama request to {path} failed after {self._attempt_count} attempts "
                f"with status {status_code}."
            )
            raise OllamaClientError(msg) from last_error

        msg = (
            f"Ollama request to {path} failed after {self._attempt_count} attempts "
            "due to transport/timeout error."
        )
        raise OllamaClientError(msg) from last_error

    def _extract_and_validate_embeddings(
        self, *, data: dict[str, Any], expected_count: int
    ) -> list[list[float]]:
        embeddings = data.get("embeddings")
        if embeddings is None and "embedding" in data:
            embeddings = [data["embedding"]]

        if not isinstance(embeddings, list):
            msg = "Ollama embed response missing 'embeddings' list."
            raise OllamaClientError(msg)

        if len(embeddings) != expected_count:
            msg = (
                f"Ollama embed response count mismatch: expected {expected_count}, "
                f"got {len(embeddings)}."
            )
            raise OllamaClientError(msg)

        vectors: list[list[float]] = []
        for idx, vector in enumerate(embeddings):
            if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)):
                msg = f"Ollama embedding at index {idx} was not a numeric list."
                raise OllamaClientError(msg)

            if len(vector) != self._embed_dim:
                msg = (
                    f"Ollama embedding dimension mismatch at index {idx}: "
                    f"expected {self._embed_dim}, got {len(vector)}."
                )
                raise OllamaClientError(msg)

            converted: list[float] = []
            for value in vector:
                if not isinstance(value, (int, float)):
                    msg = f"Ollama embedding at index {idx} contained a non-numeric value."
                    raise OllamaClientError(msg)
                converted.append(float(value))

            vectors.append(converted)

        return vectors
