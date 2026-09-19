from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from minimal_agent.domain import Message, SessionNotFoundError


@dataclass(frozen=True, slots=True)
class SessionState:
    user_id: str
    session_id: str
    summary: str
    summarized_through_message_id: int


class SQLiteRepository:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA foreign_keys = ON;
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    summarized_through_message_id INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content_json TEXT NOT NULL,
                    name TEXT,
                    tool_call_id TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session
                    ON messages(user_id, session_id, id);

                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                );
                CREATE INDEX IF NOT EXISTS idx_tasks_session
                    ON tasks(user_id, session_id, id);
                """
            )

    def create_session(self, user_id: str, session_id: str) -> SessionState:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT user_id FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row is not None and row["user_id"] != user_id:
                raise SessionNotFoundError(f"Session not found: {session_id}")
            connection.execute(
                "INSERT OR IGNORE INTO sessions(user_id, session_id) VALUES (?, ?)",
                (user_id, session_id),
            )
        return self.get_session(user_id, session_id)

    def get_session(self, user_id: str, session_id: str) -> SessionState:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT user_id, session_id, summary, summarized_through_message_id
                   FROM sessions WHERE user_id = ? AND session_id = ?""",
                (user_id, session_id),
            ).fetchone()
        if row is None:
            raise SessionNotFoundError(f"Session not found: {session_id}")
        return SessionState(**dict(row))

    def append_message(self, user_id: str, session_id: str, message: Message) -> int:
        self.create_session(user_id, session_id)
        with self._connect() as connection:
            cursor = connection.execute(
                """INSERT INTO messages(user_id, session_id, role, content_json, name, tool_call_id)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    user_id,
                    session_id,
                    message.role,
                    json.dumps(message.content, ensure_ascii=False),
                    message.name,
                    message.tool_call_id,
                ),
            )
            connection.execute(
                "UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
                (session_id,),
            )
            return int(cursor.lastrowid)

    def list_messages(self, user_id: str, session_id: str, *, after_id: int = 0) -> list[Message]:
        self.get_session(user_id, session_id)
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT id, role, content_json, name, tool_call_id FROM messages
                   WHERE user_id = ? AND session_id = ? AND id > ? ORDER BY id""",
                (user_id, session_id, after_id),
            ).fetchall()
        return [
            Message(
                id=row["id"],
                role=row["role"],
                content=json.loads(row["content_json"]),
                name=row["name"],
                tool_call_id=row["tool_call_id"],
            )
            for row in rows
        ]

    def update_summary(
        self, user_id: str, session_id: str, summary: str, through_message_id: int
    ) -> None:
        self.get_session(user_id, session_id)
        with self._connect() as connection:
            connection.execute(
                """UPDATE sessions
                   SET summary = ?, summarized_through_message_id = ?,
                       updated_at = CURRENT_TIMESTAMP
                   WHERE user_id = ? AND session_id = ?""",
                (summary, through_message_id, user_id, session_id),
            )

    def add_task(self, user_id: str, session_id: str, text: str) -> dict[str, object]:
        self.create_session(user_id, session_id)
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO tasks(user_id, session_id, text) VALUES (?, ?, ?)",
                (user_id, session_id, text),
            )
            task_id = int(cursor.lastrowid)
        return {"id": task_id, "text": text, "status": "pending"}

    def list_tasks(self, user_id: str, session_id: str) -> list[dict[str, object]]:
        self.get_session(user_id, session_id)
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT id, text, status FROM tasks
                   WHERE user_id = ? AND session_id = ? ORDER BY id""",
                (user_id, session_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def complete_task(self, user_id: str, session_id: str, task_id: int) -> dict[str, object]:
        self.get_session(user_id, session_id)
        with self._connect() as connection:
            cursor = connection.execute(
                """UPDATE tasks SET status = 'completed'
                   WHERE id = ? AND user_id = ? AND session_id = ?""",
                (task_id, user_id, session_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(task_id)
            row = connection.execute(
                "SELECT id, text, status FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
        return dict(row)

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
