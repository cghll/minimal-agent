from dataclasses import dataclass, field

import pytest

from minimal_agent.domain import ToolValidationError
from minimal_agent.tools.base import ToolContext
from minimal_agent.tools.calculator import CalculatorTool
from minimal_agent.tools.search import SearchItem, SearchTool
from minimal_agent.tools.task_list import TaskListTool


@dataclass
class MemoryTaskRepository:
    tasks: dict[tuple[str, str], list[dict[str, object]]] = field(default_factory=dict)

    def add_task(self, user_id: str, session_id: str, text: str) -> dict[str, object]:
        bucket = self.tasks.setdefault((user_id, session_id), [])
        task = {"id": len(bucket) + 1, "text": text, "status": "pending"}
        bucket.append(task)
        return task

    def list_tasks(self, user_id: str, session_id: str) -> list[dict[str, object]]:
        return list(self.tasks.get((user_id, session_id), []))

    def complete_task(self, user_id: str, session_id: str, task_id: int) -> dict[str, object]:
        for task in self.tasks.get((user_id, session_id), []):
            if task["id"] == task_id:
                task["status"] = "completed"
                return task
        raise KeyError(task_id)


def context(repo: MemoryTaskRepository, session_id: str = "s1") -> ToolContext:
    return ToolContext(user_id="u1", session_id=session_id, repository=repo)


@pytest.mark.asyncio
async def test_calculator_evaluates_basic_arithmetic() -> None:
    result = await CalculatorTool().execute(
        {"expression": "(2 + 3) * 4"}, context(MemoryTaskRepository())
    )

    assert result == {"ok": True, "value": 20}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "expression",
    ["__import__('os').system('echo unsafe')", "(1).__class__", "sum([1, 2])"],
)
async def test_calculator_rejects_unsafe_syntax(expression: str) -> None:
    with pytest.raises(ToolValidationError, match="unsupported"):
        await CalculatorTool().execute({"expression": expression}, context(MemoryTaskRepository()))


@pytest.mark.asyncio
async def test_search_returns_deterministic_matches() -> None:
    tool = SearchTool(
        {"上海明天天气": [SearchItem(title="上海天气", snippet="晴，26C", url="mock://weather")]}
    )

    result = await tool.execute({"query": "上海明天天气"}, context(MemoryTaskRepository()))

    assert result["results"] == [
        {"title": "上海天气", "snippet": "晴，26C", "url": "mock://weather"}
    ]


@pytest.mark.asyncio
async def test_task_list_is_isolated_by_session() -> None:
    repo = MemoryTaskRepository()
    tool = TaskListTool()
    await tool.execute({"action": "add", "text": "窗口一待办"}, context(repo, "window-1"))
    await tool.execute({"action": "add", "text": "窗口二待办"}, context(repo, "window-2"))

    first = await tool.execute({"action": "list"}, context(repo, "window-1"))
    second = await tool.execute({"action": "list"}, context(repo, "window-2"))

    assert [task["text"] for task in first["tasks"]] == ["窗口一待办"]
    assert [task["text"] for task in second["tasks"]] == ["窗口二待办"]
