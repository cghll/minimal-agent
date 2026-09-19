import sqlite3
from pathlib import Path

import pytest

from minimal_agent.domain import Message, SessionNotFoundError, ToolCall
from minimal_agent.memory.sqlite_repository import SQLiteRepository


@pytest.fixture
def repository(tmp_path: Path) -> SQLiteRepository:
    repo = SQLiteRepository(tmp_path / "agent.db")
    repo.initialize()
    return repo


def test_sessions_do_not_share_messages(repository: SQLiteRepository) -> None:
    repository.append_message("user-a", "window-1", Message.user("查天气"))
    repository.append_message("user-a", "window-2", Message.user("写周报"))

    assert [m.content for m in repository.list_messages("user-a", "window-1")] == ["查天气"]
    assert [m.content for m in repository.list_messages("user-a", "window-2")] == ["写周报"]


def test_same_session_id_cannot_be_read_by_another_user(repository: SQLiteRepository) -> None:
    repository.append_message("user-a", "shared-name", Message.user("private"))

    with pytest.raises(SessionNotFoundError):
        repository.list_messages("user-b", "shared-name")


def test_tool_message_round_trips_structured_content(repository: SQLiteRepository) -> None:
    call = ToolCall(call_id="c1", name="calculator", arguments={"expression": "2+2"})
    repository.append_message("u1", "s1", Message.tool(call, {"ok": True, "value": 4}))

    stored = repository.list_messages("u1", "s1")[0]

    assert stored.content == {"ok": True, "value": 4}
    assert stored.name == "calculator"
    assert stored.tool_call_id == "c1"
    assert stored.id is not None


def test_summary_tracks_last_compressed_message(repository: SQLiteRepository) -> None:
    repository.append_message("u1", "s1", Message.user("old"))
    old_id = repository.list_messages("u1", "s1")[0].id

    repository.update_summary("u1", "s1", "用户讨论过 old", old_id or 0)

    session = repository.get_session("u1", "s1")
    assert session.summary == "用户讨论过 old"
    assert session.summarized_through_message_id == old_id


def test_tasks_are_isolated_and_can_be_completed(repository: SQLiteRepository) -> None:
    first = repository.add_task("u1", "s1", "first")
    repository.add_task("u1", "s2", "second")

    completed = repository.complete_task("u1", "s1", int(first["id"]))

    assert completed["status"] == "completed"
    assert [task["text"] for task in repository.list_tasks("u1", "s1")] == ["first"]
    assert [task["text"] for task in repository.list_tasks("u1", "s2")] == ["second"]


def test_repository_closes_connections_after_each_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_connect = sqlite3.connect
    opened: list[sqlite3.Connection] = []

    def tracking_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        connection = real_connect(*args, **kwargs)
        opened.append(connection)
        return connection

    monkeypatch.setattr("minimal_agent.memory.sqlite_repository.sqlite3.connect", tracking_connect)
    repository = SQLiteRepository(tmp_path / "connections.db")

    try:
        repository.initialize()
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            opened[0].execute("SELECT 1")
    finally:
        for connection in opened:
            connection.close()
