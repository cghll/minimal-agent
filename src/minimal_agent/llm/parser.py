from __future__ import annotations

import json
from typing import Any

from minimal_agent.domain import AgentDecision, InvalidModelOutput, ToolCall


class DecisionParser:
    def parse(self, raw: str) -> AgentDecision:
        text = _strip_fence(raw.strip())
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise InvalidModelOutput("Model output must be valid JSON") from exc
        if not isinstance(payload, dict):
            raise InvalidModelOutput("Model output must be a JSON object")
        try:
            return _parse_payload(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidModelOutput(f"Invalid decision: {exc}") from exc


def _strip_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) < 3 or lines[-1].strip() != "```":
        return text
    return "\n".join(lines[1:-1]).strip()


def _parse_payload(payload: dict[str, Any]) -> AgentDecision:
    allowed = {"kind", "reasoning_summary", "tool_calls", "final_answer"}
    extras = set(payload) - allowed
    if extras:
        raise ValueError(f"unsupported fields: {sorted(extras)}")
    kind = payload["kind"]
    if kind not in {"tool_calls", "final_answer"}:
        raise ValueError("kind must be tool_calls or final_answer")
    summary = payload["reasoning_summary"]
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("reasoning_summary must be a non-empty string")
    raw_calls = payload.get("tool_calls", [])
    if not isinstance(raw_calls, list) or len(raw_calls) > 4:
        raise ValueError("tool_calls must contain at most four calls")
    calls: list[ToolCall] = []
    call_ids: set[str] = set()
    for item in raw_calls:
        if not isinstance(item, dict) or set(item) != {"call_id", "name", "arguments"}:
            raise ValueError("each tool call requires call_id, name, and arguments")
        call_id = item["call_id"]
        name = item["name"]
        arguments = item["arguments"]
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("call_id must be a non-empty string")
        if call_id in call_ids:
            raise ValueError("call_id values must be unique")
        if not isinstance(name, str) or not name or not isinstance(arguments, dict):
            raise ValueError("tool name must be a string and arguments must be an object")
        call_ids.add(call_id)
        calls.append(ToolCall(call_id=call_id, name=name, arguments=arguments))
    answer = payload.get("final_answer")
    if answer is not None and not isinstance(answer, str):
        raise ValueError("final_answer must be a string or null")
    try:
        return AgentDecision(
            kind=kind,
            reasoning_summary=summary.strip(),
            tool_calls=tuple(calls),
            final_answer=answer.strip() if isinstance(answer, str) else None,
        )
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
