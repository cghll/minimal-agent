from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from minimal_agent.domain import AgentDecision

_SENSITIVE_KEYS = {"token", "password", "secret", "api_key", "authorization"}


class SQLiteTraceRecorder:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    finished_at TEXT,
                    error_code TEXT
                );
                CREATE TABLE IF NOT EXISTS run_steps (
                    run_id TEXT NOT NULL,
                    step_number INTEGER NOT NULL,
                    decision_kind TEXT NOT NULL,
                    reasoning_summary TEXT NOT NULL,
                    tool_calls_json TEXT NOT NULL,
                    tool_results_json TEXT NOT NULL DEFAULT '[]',
                    duration_ms INTEGER NOT NULL,
                    error_code TEXT,
                    PRIMARY KEY(run_id, step_number),
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                );
                """
            )

    def start_run(self, user_id: str, session_id: str) -> str:
        run_id = str(uuid.uuid4())
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO runs(run_id, user_id, session_id, status) VALUES (?, ?, ?, 'running')",
                (run_id, user_id, session_id),
            )
        return run_id

    def record_step(
        self,
        run_id: str,
        step_number: int,
        decision: AgentDecision,
        duration_ms: int,
    ) -> None:
        calls = [
            {
                "call_id": call.call_id,
                "name": call.name,
                "arguments": _redact(call.arguments),
            }
            for call in decision.tool_calls
        ]
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO run_steps(
                       run_id, step_number, decision_kind, reasoning_summary,
                       tool_calls_json, tool_results_json, duration_ms
                   ) VALUES (?, ?, ?, ?, ?, '[]', ?)""",
                (
                    run_id,
                    step_number,
                    decision.kind,
                    _decision_label(decision),
                    json.dumps(calls, ensure_ascii=False),
                    duration_ms,
                ),
            )

    def record_tool_result(
        self, run_id: str, step_number: int, call_id: str, result: dict[str, Any]
    ) -> None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT tool_results_json FROM run_steps WHERE run_id = ? AND step_number = ?",
                (run_id, step_number),
            ).fetchone()
            if row is None:
                return
            results = json.loads(row["tool_results_json"])
            results.append({"call_id": call_id, **_redact(result)})
            error = result.get("error")
            error_code = error.get("code") if isinstance(error, dict) else None
            connection.execute(
                """UPDATE run_steps SET tool_results_json = ?, error_code = COALESCE(?, error_code)
                   WHERE run_id = ? AND step_number = ?""",
                (json.dumps(results, ensure_ascii=False), error_code, run_id, step_number),
            )

    def finish_run(self, run_id: str, status: str, error_code: str | None = None) -> None:
        with self._connect() as connection:
            connection.execute(
                """UPDATE runs SET status = ?, error_code = ?,
                       finished_at = CURRENT_TIMESTAMP WHERE run_id = ?""",
                (status, error_code, run_id),
            )

    def get_trace(self, run_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            run = connection.execute(
                """SELECT run_id, user_id, session_id, status, started_at,
                          finished_at, error_code FROM runs WHERE run_id = ?""",
                (run_id,),
            ).fetchone()
            if run is None:
                raise KeyError(run_id)
            rows = connection.execute(
                """SELECT step_number, decision_kind, reasoning_summary,
                          tool_calls_json, tool_results_json, duration_ms, error_code
                   FROM run_steps WHERE run_id = ? ORDER BY step_number""",
                (run_id,),
            ).fetchall()
        result = dict(run)
        result["steps"] = [_deserialize_step(row) for row in rows]
        return result

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "***" if key.casefold() in _SENSITIVE_KEYS else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _deserialize_step(row: sqlite3.Row) -> dict[str, Any]:
    step = dict(row)
    step["tool_calls"] = json.loads(step.pop("tool_calls_json"))
    step["tool_results"] = json.loads(step.pop("tool_results_json"))
    return step


def _decision_label(decision: AgentDecision) -> str:
    return (
        "model requested tool calls"
        if decision.kind == "tool_calls"
        else "model returned final answer"
    )
