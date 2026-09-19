from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from minimal_agent.domain import JsonObject


@dataclass(frozen=True, slots=True)
class ToolContext:
    user_id: str
    session_id: str
    repository: Any


class Tool(Protocol):
    name: str
    description: str
    parameters: JsonObject

    async def execute(self, arguments: JsonObject, context: ToolContext) -> JsonObject: ...
