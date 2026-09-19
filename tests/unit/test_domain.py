import pytest

from minimal_agent.domain import AgentDecision, Message, ToolCall


def test_agent_decision_requires_answer_for_final_kind() -> None:
    with pytest.raises(ValueError, match="final_answer"):
        AgentDecision(kind="final_answer", reasoning_summary="done")


def test_tool_message_keeps_call_identity() -> None:
    call = ToolCall(call_id="call-1", name="calculator", arguments={"expression": "2+2"})

    message = Message.tool(call, {"ok": True, "value": 4})

    assert message.role == "tool"
    assert message.name == "calculator"
    assert message.tool_call_id == "call-1"
    assert message.content == {"ok": True, "value": 4}
