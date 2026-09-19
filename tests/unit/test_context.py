from pathlib import Path

import pytest

from minimal_agent.context import ContextBuilder, JSONSummarizer
from minimal_agent.domain import Message
from minimal_agent.memory.sqlite_repository import SQLiteRepository
from minimal_agent.tools.calculator import CalculatorTool
from minimal_agent.tools.registry import ToolRegistry


class RecordingSummarizer:
    def __init__(self, result: str = "earlier conversation summary", fails: bool = False) -> None:
        self.result = result
        self.fails = fails
        self.seen: list[Message] = []

    async def summarize(self, previous_summary: str, messages: list[Message]) -> str:
        self.seen = messages
        if self.fails:
            raise RuntimeError("summary unavailable")
        return self.result


class SummaryLLM:
    def __init__(self) -> None:
        self.messages: list[dict[str, object]] = []

    async def generate(self, messages: list[dict[str, object]]) -> str:
        self.messages = messages
        return '{"summary":"用户需要在周五提交周报"}'


@pytest.fixture
def repository(tmp_path: Path) -> SQLiteRepository:
    repo = SQLiteRepository(tmp_path / "context.db")
    repo.initialize()
    return repo


def total_chars(messages: list[dict[str, object]]) -> int:
    return sum(len(str(message["content"])) for message in messages)


@pytest.mark.asyncio
async def test_context_orders_protocol_catalog_summary_and_history(
    repository: SQLiteRepository,
) -> None:
    repository.append_message("u1", "s1", Message.user("old question"))
    old_id = repository.list_messages("u1", "s1")[0].id or 0
    repository.update_summary("u1", "s1", "old summary", old_id)
    repository.append_message("u1", "s1", Message.user("latest question"))
    builder = ContextBuilder(
        repository=repository,
        registry=ToolRegistry([CalculatorTool()]),
        summarizer=RecordingSummarizer(),
        system_prompt="protocol",
        char_budget=10_000,
    )

    messages = await builder.build("u1", "s1")

    assert messages[0] == {"role": "system", "content": "protocol"}
    assert messages[1]["role"] == "system"
    assert "calculator" in str(messages[1]["content"])
    assert messages[2] == {"role": "system", "content": "Conversation summary:\nold summary"}
    assert messages[-1] == {"role": "user", "content": "latest question"}


@pytest.mark.asyncio
async def test_over_budget_history_is_summarized_and_keeps_latest_eight(
    repository: SQLiteRepository,
) -> None:
    for index in range(20):
        repository.append_message("u1", "s1", Message.user(f"message-{index}-" + "x" * 120))
    summarizer = RecordingSummarizer()
    builder = ContextBuilder(
        repository=repository,
        registry=ToolRegistry(),
        summarizer=summarizer,
        system_prompt="protocol",
        char_budget=1_500,
    )

    messages = await builder.build("u1", "s1")

    assert len(summarizer.seen) == 12
    assert repository.get_session("u1", "s1").summary == "earlier conversation summary"
    assert "message-12" in str(messages)
    assert "message-19" in str(messages)
    assert "message-11" not in str(messages)
    assert total_chars(messages) <= 1_500


@pytest.mark.asyncio
async def test_summary_failure_uses_deterministic_fallback(
    repository: SQLiteRepository,
) -> None:
    for index in range(12):
        repository.append_message("u1", "s1", Message.user(f"fact-{index}-" + "y" * 100))
    builder = ContextBuilder(
        repository=repository,
        registry=ToolRegistry(),
        summarizer=RecordingSummarizer(fails=True),
        system_prompt="protocol",
        char_budget=1_100,
    )

    messages = await builder.build("u1", "s1")

    summary = repository.get_session("u1", "s1").summary
    assert summary.startswith("Fallback summary:")
    assert "fact-0" in summary
    assert "fact-11" in str(messages)


@pytest.mark.asyncio
async def test_json_summarizer_returns_summary_from_llm() -> None:
    llm = SummaryLLM()
    summarizer = JSONSummarizer(llm)

    result = await summarizer.summarize("先前摘要", [Message.user("记住周五提交周报")])

    assert result == "用户需要在周五提交周报"
    assert "先前摘要" in str(llm.messages)
    assert "周五提交周报" in str(llm.messages)
