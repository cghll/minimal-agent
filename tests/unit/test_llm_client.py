import json

import httpx
import pytest

from minimal_agent.domain import LLMUnavailable
from minimal_agent.llm.client import OpenAICompatibleClient


@pytest.mark.asyncio
async def test_client_sends_compatible_chat_completion_request() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"kind":"final_answer"}'}}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = OpenAICompatibleClient(
            base_url="https://llm.example/v1",
            api_key="secret-key",
            model="test-model",
            http_client=http_client,
            retry_delays=(0, 0),
        )
        result = await client.generate([{"role": "user", "content": "hello"}])

    assert result == '{"kind":"final_answer"}'
    assert captured[0].url == "https://llm.example/v1/chat/completions"
    assert captured[0].headers["authorization"] == "Bearer secret-key"
    payload = json.loads(captured[0].content)
    assert payload == {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }


@pytest.mark.asyncio
async def test_client_retries_429_then_returns_content() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(429, json={"error": {"message": "rate limited"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = OpenAICompatibleClient(
            base_url="https://llm.example/v1",
            api_key="key",
            model="model",
            http_client=http_client,
            retry_delays=(0, 0),
        )
        result = await client.generate([])

    assert result == "ok"
    assert attempts == 3


@pytest.mark.asyncio
async def test_client_does_not_retry_bad_requests() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(400, json={"error": {"message": "bad request"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = OpenAICompatibleClient(
            base_url="https://llm.example/v1",
            api_key="key",
            model="model",
            http_client=http_client,
            retry_delays=(0, 0),
        )
        with pytest.raises(LLMUnavailable, match="400"):
            await client.generate([])

    assert attempts == 1
