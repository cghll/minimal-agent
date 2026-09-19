from __future__ import annotations

from dataclasses import asdict, dataclass

from minimal_agent.domain import JsonObject
from minimal_agent.tools.base import ToolContext


@dataclass(frozen=True, slots=True)
class SearchItem:
    title: str
    snippet: str
    url: str


class SearchTool:
    name = "search"
    description = "搜索确定性的本地知识索引。"
    parameters: JsonObject = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, index: dict[str, list[SearchItem]] | None = None) -> None:
        self._index = index or {}

    async def execute(self, arguments: JsonObject, context: ToolContext) -> JsonObject:
        del context
        query = str(arguments["query"]).strip().casefold()
        matches: list[SearchItem] = []
        for key, items in self._index.items():
            if query == key.casefold() or query in key.casefold() or key.casefold() in query:
                matches.extend(items)
        return {
            "ok": True,
            "query": arguments["query"],
            "results": [asdict(item) for item in matches],
        }
