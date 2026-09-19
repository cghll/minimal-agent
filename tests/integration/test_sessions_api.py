import json
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from minimal_agent.api import create_app
from minimal_agent.context import ContextBuilder
from minimal_agent.domain import Message
from minimal_agent.llm.parser import DecisionParser
from minimal_agent.memory.sqlite_repository import SQLiteRepository
from minimal_agent.runtime import AgentRuntime
from minimal_agent.tools.registry import ToolRegistry
from minimal_agent.tools.task_list import TaskListTool
from minimal_agent.tracing import SQLiteTraceRecorder


class ScriptedLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses

    async def generate(self, messages: list[dict[str, object]]) -> str:
        return self.responses.pop(0)


class StubSummarizer:
    async def summarize(self, previous_summary: str, messages: list[Message]) -> str:
        return "summary"


def tool_decision(action: str, text: str | None = None) -> str:
    arguments: dict[str, object] = {"action": action}
    if text:
        arguments["text"] = text
    return json.dumps(
        {
            "kind": "tool_calls",
            "reasoning_summary": "use tasks",
            "tool_calls": [
                {"call_id": f"{action}-{text}", "name": "task_list", "arguments": arguments}
            ],
            "final_answer": None,
        }
    )


def final_decision(answer: str) -> str:
    return json.dumps(
        {
            "kind": "final_answer",
            "reasoning_summary": "done",
            "tool_calls": [],
            "final_answer": answer,
        }
    )


def build_app(tmp_path: Path) -> FastAPI:
    database = tmp_path / "api.db"
    repository = SQLiteRepository(database)
    repository.initialize()
    trace = SQLiteTraceRecorder(database)
    trace.initialize()
    registry = ToolRegistry([TaskListTool()])
    llm = ScriptedLLM(
        [
            tool_decision("add", "准备雨伞"),
            final_decision("窗口一已记录"),
            tool_decision("add", "提交周报"),
            final_decision("窗口二已记录"),
            tool_decision("list"),
            final_decision("你的待办是准备雨伞"),
            tool_decision("list"),
            final_decision("你的待办是提交周报"),
        ]
    )
    builder = ContextBuilder(
        repository=repository,
        registry=registry,
        summarizer=StubSummarizer(),
        system_prompt="protocol",
        char_budget=10_000,
    )
    runtime = AgentRuntime(
        llm=llm,
        parser=DecisionParser(),
        registry=registry,
        repository=repository,
        context_builder=builder,
        trace=trace,
        max_steps=6,
    )
    return create_app(runtime=runtime, repository=repository, trace=trace)


@pytest.mark.asyncio
async def test_two_windows_keep_messages_and_tasks_isolated(tmp_path: Path) -> None:
    transport = httpx.ASGITransport(app=build_app(tmp_path))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = (await client.post("/v1/sessions", json={"user_id": "user-a"})).json()["session_id"]
        second = (await client.post("/v1/sessions", json={"user_id": "user-a"})).json()[
            "session_id"
        ]

        response1 = await client.post(
            f"/v1/sessions/{first}/messages",
            json={"user_id": "user-a", "content": "查天气并提醒带伞"},
        )
        response2 = await client.post(
            f"/v1/sessions/{second}/messages",
            json={"user_id": "user-a", "content": "帮我记周报待办"},
        )
        followup1 = await client.post(
            f"/v1/sessions/{first}/messages",
            json={"user_id": "user-a", "content": "刚才的待办是什么"},
        )
        followup2 = await client.post(
            f"/v1/sessions/{second}/messages",
            json={"user_id": "user-a", "content": "刚才的待办是什么"},
        )

    assert response1.status_code == response2.status_code == 200
    assert "准备雨伞" in followup1.json()["answer"]
    assert "提交周报" not in followup1.json()["answer"]
    assert "提交周报" in followup2.json()["answer"]
    assert "准备雨伞" not in followup2.json()["answer"]


@pytest.mark.asyncio
async def test_messages_and_trace_can_be_queried_by_owner(tmp_path: Path) -> None:
    transport = httpx.ASGITransport(app=build_app(tmp_path))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = (await client.post("/v1/sessions", json={"user_id": "u1"})).json()[
            "session_id"
        ]
        sent = await client.post(
            f"/v1/sessions/{session_id}/messages",
            json={"user_id": "u1", "content": "remember umbrella"},
        )

        messages = await client.get(f"/v1/sessions/{session_id}/messages", params={"user_id": "u1"})
        trace = await client.get(f"/v1/runs/{sent.json()['run_id']}", params={"user_id": "u1"})
        forbidden = await client.get(
            f"/v1/runs/{sent.json()['run_id']}", params={"user_id": "someone-else"}
        )

    assert messages.status_code == 200
    assert [item["role"] for item in messages.json()["messages"]] == ["user", "tool", "assistant"]
    assert trace.status_code == 200
    assert len(trace.json()["steps"]) == 2
    assert forbidden.status_code == 404


@pytest.mark.asyncio
async def test_message_validation_rejects_empty_content(tmp_path: Path) -> None:
    transport = httpx.ASGITransport(app=build_app(tmp_path))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = (await client.post("/v1/sessions", json={"user_id": "u1"})).json()[
            "session_id"
        ]

        response = await client.post(
            f"/v1/sessions/{session_id}/messages",
            json={"user_id": "u1", "content": ""},
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_session_id_validation_rejects_oversized_path(tmp_path: Path) -> None:
    transport = httpx.ASGITransport(app=build_app(tmp_path))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"/v1/sessions/{'x' * 129}/messages",
            json={"user_id": "u1", "content": "hello"},
        )

    assert response.status_code == 422
