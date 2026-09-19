from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Protocol

import httpx

from minimal_agent.domain import JsonObject, LLMUnavailable


class LLMClient(Protocol):
    async def generate(self, messages: list[JsonObject]) -> str: ...


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 30.0,
        http_client: httpx.AsyncClient | None = None,
        retry_delays: Sequence[float] = (0.25, 0.75),
    ) -> None:
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._api_key = api_key
        self._model = model
        self._client = http_client or httpx.AsyncClient(timeout=timeout_seconds)
        self._owns_client = http_client is None
        self._retry_delays = tuple(retry_delays)

    async def generate(self, messages: list[JsonObject]) -> str:
        attempts = len(self._retry_delays) + 1
        for attempt in range(attempts):
            try:
                response = await self._client.post(
                    self._url,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json={
                        "model": self._model,
                        "messages": messages,
                        "response_format": {"type": "json_object"},
                        "temperature": 0,
                    },
                )
                if response.status_code not in {429, 500, 502, 503, 504}:
                    response.raise_for_status()
                elif attempt < attempts - 1:
                    await asyncio.sleep(self._retry_delays[attempt])
                    continue
                else:
                    response.raise_for_status()
                payload = response.json()
                content = payload["choices"][0]["message"]["content"]
                if not isinstance(content, str) or not content:
                    raise ValueError("response content is empty")
                return content
            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
                retryable = isinstance(exc, httpx.RequestError)
                if retryable and attempt < attempts - 1:
                    await asyncio.sleep(self._retry_delays[attempt])
                    continue
                status = (
                    exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
                )
                detail = f"status {status}" if status is not None else type(exc).__name__
                raise LLMUnavailable(f"LLM request failed: {detail}") from exc
        raise LLMUnavailable("LLM request failed")

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()
