import os

import httpx
import pytest

from minimal_agent.api import create_default_app

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.getenv("RUN_LIVE_LLM_TESTS") != "1",
        reason="live LLM tests require RUN_LIVE_LLM_TESTS=1",
    ),
]


@pytest.mark.asyncio
async def test_real_llm_can_choose_calculator() -> None:
    transport = httpx.ASGITransport(app=create_default_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = (await client.post("/v1/sessions", json={"user_id": "live-user"})).json()[
            "session_id"
        ]
        response = await client.post(
            f"/v1/sessions/{session_id}/messages",
            json={"user_id": "live-user", "content": "请计算 (17+5)*3"},
        )
        payload = response.json()
        trace = await client.get(f"/v1/runs/{payload['run_id']}", params={"user_id": "live-user"})

    assert response.status_code == 200
    assert "66" in payload["answer"]
    assert any(
        call["name"] == "calculator"
        for step in trace.json()["steps"]
        for call in step["tool_calls"]
    )


@pytest.mark.asyncio
async def test_real_llm_recalls_task_in_followup() -> None:
    transport = httpx.ASGITransport(app=create_default_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = (await client.post("/v1/sessions", json={"user_id": "live-user"})).json()[
            "session_id"
        ]
        first = await client.post(
            f"/v1/sessions/{session_id}/messages",
            json={"user_id": "live-user", "content": "记下待办：周五提交周报"},
        )
        second = await client.post(
            f"/v1/sessions/{session_id}/messages",
            json={"user_id": "live-user", "content": "我刚才记了什么？"},
        )

    assert first.status_code == second.status_code == 200
    assert "周五" in second.json()["answer"]
    assert "周报" in second.json()["answer"]
