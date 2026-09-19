from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

JsonObject = dict[str, Any]


class AgentError(Exception):
    code = "agent_error"


class InvalidModelOutput(AgentError):
    code = "invalid_model_output"


class ToolError(AgentError):
    code = "tool_error"


class ToolNotFoundError(ToolError):
    code = "tool_not_found"


class ToolValidationError(ToolError):
    code = "tool_validation_error"


class DuplicateToolError(ToolError):
    code = "duplicate_tool"


class MaxStepsExceeded(AgentError):
    code = "max_steps_exceeded"

    def __init__(self, run_id: str, limit: int) -> None:
        self.run_id = run_id
        self.limit = limit
        super().__init__(f"Agent exceeded the maximum of {limit} steps")


class LLMUnavailable(AgentError):
    code = "llm_unavailable"


class SessionNotFoundError(AgentError):
    code = "session_not_found"


@dataclass(frozen=True, slots=True)
class ToolCall:
    call_id: str
    name: str
    arguments: JsonObject


@dataclass(frozen=True, slots=True)
class AgentDecision:
    kind: Literal["tool_calls", "final_answer"]
    reasoning_summary: str
    tool_calls: tuple[ToolCall, ...] = ()
    final_answer: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "final_answer" and not self.final_answer:
            raise ValueError("final_answer is required for a final decision")
        if self.kind == "final_answer" and self.tool_calls:
            raise ValueError("final decisions cannot contain tool calls")
        if self.kind == "tool_calls" and not self.tool_calls:
            raise ValueError("tool_calls decisions require at least one call")
        if self.kind == "tool_calls" and self.final_answer is not None:
            raise ValueError("tool_calls decisions cannot contain final_answer")


@dataclass(frozen=True, slots=True)
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str | JsonObject
    name: str | None = None
    tool_call_id: str | None = None
    id: int | None = field(default=None, compare=False)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str) -> Message:
        return cls(role="assistant", content=content)

    @classmethod
    def tool(cls, call: ToolCall, content: JsonObject) -> Message:
        return cls(
            role="tool",
            content=content,
            name=call.name,
            tool_call_id=call.call_id,
        )


@dataclass(frozen=True, slots=True)
class RunResult:
    answer: str
    step_count: int
    run_id: str
