from __future__ import annotations

import json
import time
from typing import Protocol

from minimal_agent.context import ContextBuilder
from minimal_agent.domain import (
    AgentDecision,
    AgentError,
    InvalidModelOutput,
    JsonObject,
    MaxStepsExceeded,
    Message,
    RunResult,
    ToolError,
)
from minimal_agent.llm.client import LLMClient
from minimal_agent.llm.parser import DecisionParser
from minimal_agent.memory.sqlite_repository import SQLiteRepository
from minimal_agent.tools.base import ToolContext
from minimal_agent.tools.registry import ToolRegistry


class TraceRecorder(Protocol):
    def start_run(self, user_id: str, session_id: str) -> str: ...

    def record_step(
        self, run_id: str, step_number: int, decision: AgentDecision, duration_ms: int
    ) -> None: ...

    def finish_run(self, run_id: str, status: str, error_code: str | None = None) -> None: ...

    def record_tool_result(
        self, run_id: str, step_number: int, call_id: str, result: JsonObject
    ) -> None: ...


class AgentRuntime:
    def __init__(
        self,
        *,
        llm: LLMClient,
        parser: DecisionParser,
        registry: ToolRegistry,
        repository: SQLiteRepository,
        context_builder: ContextBuilder,
        trace: TraceRecorder,
        max_steps: int,
    ) -> None:
        self._llm = llm
        self._parser = parser
        self._registry = registry
        self._repository = repository
        self._context_builder = context_builder
        self._trace = trace
        self._max_steps = max_steps

    async def run(self, user_id: str, session_id: str, user_input: str) -> RunResult:
        run_id = self._trace.start_run(user_id, session_id)
        failure_counts: dict[str, int] = {}
        try:
            self._repository.append_message(user_id, session_id, Message.user(user_input))
            for step_number in range(1, self._max_steps + 1):
                started = time.perf_counter()
                messages = await self._context_builder.build(user_id, session_id)
                decision = await self._generate_decision(messages)
                duration_ms = int((time.perf_counter() - started) * 1000)
                self._trace.record_step(run_id, step_number, decision, duration_ms)

                if decision.kind == "final_answer":
                    answer = decision.final_answer or ""
                    self._repository.append_message(user_id, session_id, Message.assistant(answer))
                    self._trace.finish_run(run_id, "completed")
                    return RunResult(answer=answer, step_count=step_number, run_id=run_id)

                for call in decision.tool_calls:
                    signature = json.dumps(
                        {"name": call.name, "arguments": call.arguments},
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    try:
                        result = await self._registry.execute(
                            call.name,
                            call.arguments,
                            ToolContext(
                                user_id=user_id,
                                session_id=session_id,
                                repository=self._repository,
                            ),
                        )
                        failure_counts.pop(signature, None)
                    except AgentError as exc:
                        failure_counts[signature] = failure_counts.get(signature, 0) + 1
                        result = _tool_error(exc.code, str(exc))
                    except Exception as exc:
                        failure_counts[signature] = failure_counts.get(signature, 0) + 1
                        result = _tool_error("tool_execution_error", type(exc).__name__)
                    self._repository.append_message(user_id, session_id, Message.tool(call, result))
                    self._trace.record_tool_result(run_id, step_number, call.call_id, result)
                    if failure_counts.get(signature, 0) >= 2:
                        raise ToolError("The same tool call failed twice")

            error = MaxStepsExceeded(run_id=run_id, limit=self._max_steps)
            self._trace.finish_run(run_id, "failed", error.code)
            raise error
        except Exception as exc:
            if not isinstance(exc, MaxStepsExceeded):
                error_code = exc.code if isinstance(exc, AgentError) else "internal_error"
                self._trace.finish_run(run_id, "failed", error_code)
            raise

    async def _generate_decision(self, messages: list[JsonObject]) -> AgentDecision:
        raw = await self._llm.generate(messages)
        try:
            return self._parser.parse(raw)
        except InvalidModelOutput as exc:
            repair_messages = [
                *messages,
                {
                    "role": "system",
                    "content": (
                        f"你上一次的输出无效：{exc}。"
                        "请返回一个符合要求决策格式的有效 JSON 对象。"
                    ),
                },
            ]
            repaired = await self._llm.generate(repair_messages)
            return self._parser.parse(repaired)


def _tool_error(code: str, message: str) -> JsonObject:
    return {"ok": False, "error": {"code": code, "message": message}}
