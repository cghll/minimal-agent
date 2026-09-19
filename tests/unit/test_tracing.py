import sqlite3
from pathlib import Path

import pytest

from minimal_agent.domain import AgentDecision, ToolCall
from minimal_agent.tracing import SQLiteTraceRecorder


def test_trace_records_run_steps_and_completion(tmp_path: Path) -> None:
    recorder = SQLiteTraceRecorder(tmp_path / "trace.db")
    recorder.initialize()
    run_id = recorder.start_run("u1", "s1")
    recorder.record_step(
        run_id,
        1,
        AgentDecision(
            kind="tool_calls",
            reasoning_summary="需要查询",
            tool_calls=(ToolCall("c1", "search", {"query": "weather"}),),
        ),
        17,
    )
    recorder.record_step(
        run_id,
        2,
        AgentDecision(
            kind="final_answer",
            reasoning_summary="可以回答",
            final_answer="sunny",
        ),
        8,
    )
    recorder.finish_run(run_id, "completed")

    trace = recorder.get_trace(run_id)

    assert trace["status"] == "completed"
    assert trace["user_id"] == "u1"
    assert [step["decision_kind"] for step in trace["steps"]] == ["tool_calls", "final_answer"]
    assert trace["steps"][0]["duration_ms"] == 17


def test_trace_redacts_sensitive_tool_arguments(tmp_path: Path) -> None:
    recorder = SQLiteTraceRecorder(tmp_path / "trace.db")
    recorder.initialize()
    run_id = recorder.start_run("u1", "s1")
    recorder.record_step(
        run_id,
        1,
        AgentDecision(
            kind="tool_calls",
            reasoning_summary="authenticate",
            tool_calls=(
                ToolCall(
                    "c1",
                    "custom",
                    {"token": "top-secret", "nested": {"password": "hidden", "safe": "visible"}},
                ),
            ),
        ),
        1,
    )

    serialized = str(recorder.get_trace(run_id))

    assert "top-secret" not in serialized
    assert "hidden" not in serialized
    assert "visible" in serialized
    assert "***" in serialized


def test_trace_recorder_closes_connections_after_each_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_connect = sqlite3.connect
    opened: list[sqlite3.Connection] = []

    def tracking_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        connection = real_connect(*args, **kwargs)
        opened.append(connection)
        return connection

    monkeypatch.setattr("minimal_agent.tracing.sqlite3.connect", tracking_connect)
    recorder = SQLiteTraceRecorder(tmp_path / "connections.db")

    try:
        recorder.initialize()
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            opened[0].execute("SELECT 1")
    finally:
        for connection in opened:
            connection.close()
