import json

import pytest

from minimal_agent.domain import InvalidModelOutput
from minimal_agent.llm.parser import DecisionParser


def test_parser_accepts_tool_decision_inside_markdown_fence() -> None:
    raw = """```json
    {"kind":"tool_calls","reasoning_summary":"需要计算", "tool_calls":[
      {"call_id":"c1","name":"calculator","arguments":{"expression":"2+2"}}
    ],"final_answer":null}
    ```"""

    decision = DecisionParser().parse(raw)

    assert decision.tool_calls[0].name == "calculator"
    assert decision.tool_calls[0].arguments == {"expression": "2+2"}


def test_parser_accepts_final_answer() -> None:
    decision = DecisionParser().parse(
        json.dumps(
            {
                "kind": "final_answer",
                "reasoning_summary": "信息足够",
                "tool_calls": [],
                "final_answer": "答案是 4",
            },
            ensure_ascii=False,
        )
    )

    assert decision.final_answer == "答案是 4"


@pytest.mark.parametrize(
    "payload",
    [
        {
            "kind": "final_answer",
            "reasoning_summary": "冲突",
            "tool_calls": [{"call_id": "c1", "name": "search", "arguments": {}}],
            "final_answer": "完成",
        },
        {"kind": "tool_calls", "reasoning_summary": "空", "tool_calls": [], "final_answer": None},
        {
            "kind": "tool_calls",
            "reasoning_summary": "重复",
            "tool_calls": [
                {"call_id": "c1", "name": "search", "arguments": {}},
                {"call_id": "c1", "name": "search", "arguments": {}},
            ],
            "final_answer": None,
        },
    ],
)
def test_parser_rejects_invalid_decisions(payload: dict[str, object]) -> None:
    with pytest.raises(InvalidModelOutput):
        DecisionParser().parse(json.dumps(payload, ensure_ascii=False))


def test_parser_rejects_non_json_text() -> None:
    with pytest.raises(InvalidModelOutput, match="valid JSON"):
        DecisionParser().parse("I will use a tool")
