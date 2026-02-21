import json

import httpx
import pytest

from rag_api.services.ollama_client import OllamaClient, OllamaClientError


pytestmark = pytest.mark.asyncio


async def test_embed_texts_success_returns_vectors_and_batches_requests(
    respx_mock: object,
) -> None:
    route = respx_mock.post("http://ollama.test/api/embed").mock(
        side_effect=[
            httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]]}),
            httpx.Response(200, json={"embeddings": [[2.1, 2.2, 2.3]]}),
        ]
    )
    client = OllamaClient(
        base_url="http://ollama.test",
        embed_model="embed-test-model",
        embed_dim=3,
        batch_size=2,
    )

    try:
        vectors = await client.embed_texts(["alpha", "beta", "gamma"])
    finally:
        await client.aclose()

    assert vectors == [[0.1, 0.2, 0.3], [1.1, 1.2, 1.3], [2.1, 2.2, 2.3]]
    assert route.call_count == 2

    first_payload = json.loads(route.calls[0].request.content)
    second_payload = json.loads(route.calls[1].request.content)
    assert first_payload == {"model": "embed-test-model", "input": ["alpha", "beta"]}
    assert second_payload == {"model": "embed-test-model", "input": ["gamma"]}


async def test_embed_texts_wrong_dimension_raises_clear_error(respx_mock: object) -> None:
    respx_mock.post("http://ollama.test/api/embed").mock(
        return_value=httpx.Response(200, json={"embeddings": [[0.1, 0.2]]})
    )
    client = OllamaClient(
        base_url="http://ollama.test",
        embed_model="embed-test-model",
        embed_dim=3,
    )

    try:
        with pytest.raises(OllamaClientError, match=r"dimension mismatch.*expected 3, got 2"):
            await client.embed_texts(["alpha"])
    finally:
        await client.aclose()


async def test_embed_texts_timeout_retries_then_fails_cleanly(respx_mock: object) -> None:
    route = respx_mock.post("http://ollama.test/api/embed").mock(
        side_effect=httpx.ReadTimeout("timed out")
    )
    client = OllamaClient(
        base_url="http://ollama.test",
        embed_model="embed-test-model",
        embed_dim=3,
        max_retries=2,
    )

    try:
        with pytest.raises(OllamaClientError, match=r"failed after 3 attempts"):
            await client.embed_texts(["alpha"])
    finally:
        await client.aclose()

    assert route.call_count == 3


async def test_chat_success_returns_message_content(respx_mock: object) -> None:
    route = respx_mock.post("http://ollama.test/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "Hello from model."}},
        )
    )
    client = OllamaClient(
        base_url="http://ollama.test",
        chat_model="chat-test-model",
    )

    try:
        answer = await client.chat([{"role": "user", "content": "Hi"}])
    finally:
        await client.aclose()

    assert answer == "Hello from model."
    assert route.call_count == 1
    payload = json.loads(route.calls[0].request.content)
    assert payload == {
        "model": "chat-test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": False,
    }


async def test_chat_server_500_retries_then_raises(respx_mock: object) -> None:
    route = respx_mock.post("http://ollama.test/api/chat").mock(
        return_value=httpx.Response(500, json={"error": "internal"})
    )
    client = OllamaClient(
        base_url="http://ollama.test",
        chat_model="chat-test-model",
        max_retries=2,
    )

    try:
        with pytest.raises(OllamaClientError, match=r"status 500"):
            await client.chat([{"role": "user", "content": "Hi"}])
    finally:
        await client.aclose()

    assert route.call_count == 3
