"""Run an end-to-end local demo against the API.

Flow:
1) Create a document
2) Poll ingestion status until done
3) Ask a question
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import httpx

PENDING_STATUSES = {"pending", "processing"}
DONE_STATUS = "done"
FAILED_STATUS = "failed"


def _print_json(label: str, payload: dict[str, Any]) -> None:
    print(f"\n{label}:")
    print(json.dumps(payload, indent=2, sort_keys=True))


def _build_headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local RAG demo flow.")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000/api/v1",
        help="Base API URL (default: http://127.0.0.1:8000/api/v1).",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("API_KEY"),
        help="Optional X-API-Key header value. Defaults to API_KEY env var.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=60.0,
        help="Polling timeout for ingestion status.",
    )
    parser.add_argument(
        "--poll-interval-s",
        type=float,
        default=1.0,
        help="Polling interval for ingestion status.",
    )
    parser.add_argument(
        "--question",
        default="How does this project generate embeddings?",
        help="Question to send to /ask.",
    )
    return parser.parse_args()


def _ensure_json_response(response: httpx.Response) -> dict[str, Any]:
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object, got: {type(payload).__name__}")
    return payload


def _create_document(
    client: httpx.Client,
    *,
    base_url: str,
    headers: dict[str, str],
) -> str:
    payload = {
        "title": "Demo document",
        "source": "scripts/demo/run_demo.py",
        "content": (
            "This project stores embeddings in pgvector and uses an Ollama-backed "
            "worker to chunk and embed documents."
        ),
    }
    response = client.post(f"{base_url}/documents", headers=headers, json=payload)
    body = _ensure_json_response(response)
    _print_json("Create document response", body)

    document_id = body.get("document_id")
    if not isinstance(document_id, str) or not document_id:
        raise RuntimeError("Create response did not include document_id.")
    return document_id


def _wait_for_ingestion(
    client: httpx.Client,
    *,
    base_url: str,
    headers: dict[str, str],
    document_id: str,
    timeout_s: float,
    poll_interval_s: float,
) -> None:
    deadline = time.monotonic() + timeout_s

    while time.monotonic() < deadline:
        response = client.get(f"{base_url}/documents/{document_id}/status", headers=headers)
        body = _ensure_json_response(response)
        _print_json("Status response", body)

        status = body.get("status")
        if status == DONE_STATUS:
            return
        if status == FAILED_STATUS:
            error = body.get("error", "unknown error")
            raise RuntimeError(f"Ingestion failed: {error}")
        if status not in PENDING_STATUSES:
            raise RuntimeError(f"Unexpected ingestion status: {status}")

        time.sleep(poll_interval_s)

    raise TimeoutError(f"Timed out waiting for ingestion after {timeout_s:.1f}s")


def _ask_question(
    client: httpx.Client,
    *,
    base_url: str,
    headers: dict[str, str],
    question: str,
) -> None:
    response = client.post(
        f"{base_url}/ask",
        headers=headers,
        json={"question": question},
    )
    body = _ensure_json_response(response)
    _print_json("Ask response", body)


def main() -> int:
    args = _parse_args()
    headers = _build_headers(args.api_key)

    try:
        with httpx.Client(timeout=30.0) as client:
            document_id = _create_document(client, base_url=args.base_url, headers=headers)
            _wait_for_ingestion(
                client,
                base_url=args.base_url,
                headers=headers,
                document_id=document_id,
                timeout_s=args.timeout_s,
                poll_interval_s=args.poll_interval_s,
            )
            _ask_question(
                client,
                base_url=args.base_url,
                headers=headers,
                question=args.question,
            )
    except Exception as exc:
        print(f"\nDemo failed: {exc}", file=sys.stderr)
        return 1

    print("\nDemo completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
