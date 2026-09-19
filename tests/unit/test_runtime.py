import json
from pathlib import Path
from typing import Any

import pytest

from minimal_agent.context import ContextBuilder
from minimal_agent.domain import MaxStepsExceeded, Message
from minimal_agent.llm.parser import DecisionParser
from minimal_agent.memory.sqlite_repository import SQLiteRepository
from minimal_agent.runtime import AgentRuntime
from minimal_agent.tools.calculator import CalculatorTool
from minimal_agent.tools.registry import ToolRegistry


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.seen_messages: list[list[dict[str, object]]] = []

    async def generate(self, messages: list[dict[str, object]]) -> str:
        self.seen_messages.append(messages)
        return self.responses.pop(0)


class StubSummarizer:
    async def summarize(self, previous_summary: str, messages: list[Message]) -> str:
        return "summary"


class RecordingTrace:
    def __init__(self) -> None:
        self.steps: list[tuple[int, str]] = []
        self.status: str | None = None

    def start_run(self, user_id: str, session_id: str) -> str:
        return "run-1"

    def record_step(self, run_id: str, step_number: int, decision: Any, duration_ms: int) -> None:
        self.steps.append((step_number, decision.kind))

    def finish_run(self, run_id: str, status: str, error_code: str | None = None) -> None:
        self.status = status

    def record_tool_result(
        self, run_id: str, step_number: int, call_id: str, result: dict[str, object]
    ) -> None:
        del run_id, step_number, call_id, result


def tool_decision(name: str, arguments: dict[str, object], call_id: str = "c1") -> str:
    return json.dumps(
        {
            "kind": "tool_calls",
            "reasoning_summary": "需要工具",
            "tool_calls": [{"call_id": call_id, "name": name, "arguments": arguments}],
            "final_answer": None,
        },
        ensure_ascii=False,
    )


def final_decision(answer: str) -> str:
    return json.dumps(
        {
            "kind": "final_answer",
            "reasoning_summary": "可以回答",
            "tool_calls": [],
            "final_answer": answer,
        },
        ensure_ascii=False,
    )


def build_runtime(
    tmp_path: Path, llm: FakeLLM, max_steps: int = 6
) -> tuple[AgentRuntime, SQLiteRepository, RecordingTrace]:
    repository = SQLiteRepository(tmp_path / "runtime.db")
    repository.initialize()
    registry = ToolRegistry([CalculatorTool()])
    context_builder = ContextBuilder(
        repository=repository,
        registry=registry,
        summarizer=StubSummarizer(),
        system_prompt="protocol",
        char_budget=10_000,
    )
    trace = RecordingTrace()
    runtime = AgentRuntime(
        llm=llm,
        parser=DecisionParser(),
        registry=registry,
        repository=repository,
        context_builder=context_builder,
        trace=trace,
        max_steps=max_steps,
    )
    return runtime, repository, trace


@pytest.mark.asyncio
async def test_runtime_returns_direct_answer(tmp_path: Path) -> None:
    runtime, repository, trace = build_runtime(tmp_path, FakeLLM([final_decision("你好")]))

    result = await runtime.run("u1", "s1", "你好")

    assert result.answer == "你好"
    assert result.step_count == 1
    assert [message.role for message in repository.list_messages("u1", "s1")] == [
        "user",
        "assistant",
    ]
    assert trace.status == "completed"


@pytest.mark.asyncio
async def test_runtime_executes_tool_then_answers(tmp_path: Path) -> None:
    llm = FakeLLM([tool_decision("calculator", {"expression": "6*7"}), final_decision("结果是 42")])
    runtime, repository, trace = build_runtime(tmp_path, llm)

    result = await runtime.run("u1", "s1", "6 乘 7 是多少")

    assert result.answer == "结果是 42"
    assert result.step_count == 2
    stored = repository.list_messages("u1", "s1")
    assert stored[1].role == "tool"
    assert stored[1].content == {"ok": True, "value": 42}
    assert "42" in str(llm.seen_messages[1])
    assert trace.steps == [(1, "tool_calls"), (2, "final_answer")]


@pytest.mark.asyncio
async def test_runtime_returns_tool_error_to_model(tmp_path: Path) -> None:
    llm = FakeLLM(
        [tool_decision("calculator", {"expression": "bad()"}), final_decision("无法计算")]
    )
    runtime, repository, _ = build_runtime(tmp_path, llm)

    await runtime.run("u1", "s1", "计算 bad")

    tool_message = repository.list_messages("u1", "s1")[1]
    assert tool_message.content["ok"] is False
    assert tool_message.content["error"]["code"] == "tool_validation_error"


@pytest.mark.asyncio
async def test_runtime_repairs_invalid_model_output_once(tmp_path: Path) -> None:
    llm = FakeLLM(["not json", final_decision("修复成功")])
    runtime, _, _ = build_runtime(tmp_path, llm)

    result = await runtime.run("u1", "s1", "hello")

    assert result.answer == "修复成功"
    assert any("valid JSON" in str(message["content"]) for message in llm.seen_messages[1])


@pytest.mark.asyncio
async def test_runtime_stops_at_maximum_steps(tmp_path: Path) -> None:
    llm = FakeLLM([tool_decision("calculator", {"expression": "1+1"}, f"c{i}") for i in range(3)])
    runtime, _, trace = build_runtime(tmp_path, llm, max_steps=3)

    with pytest.raises(MaxStepsExceeded) as caught:
        await runtime.run("u1", "s1", "keep going")

    assert caught.value.run_id == "run-1"
    assert trace.status == "failed"
