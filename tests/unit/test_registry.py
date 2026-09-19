from dataclasses import dataclass

import pytest

from minimal_agent.domain import DuplicateToolError, ToolNotFoundError, ToolValidationError
from minimal_agent.tools.base import ToolContext
from minimal_agent.tools.calculator import CalculatorTool
from minimal_agent.tools.registry import ToolRegistry


@dataclass
class StubRepository:
    pass


@pytest.fixture
def context() -> ToolContext:
    return ToolContext(user_id="u1", session_id="s1", repository=StubRepository())


def test_catalog_exposes_name_description_and_schema() -> None:
    registry = ToolRegistry([CalculatorTool()])

    assert registry.catalog() == [
        {
            "name": "calculator",
            "description": "Evaluate a basic arithmetic expression safely.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
                "additionalProperties": False,
            },
        }
    ]


def test_registry_rejects_duplicate_names() -> None:
    registry = ToolRegistry([CalculatorTool()])

    with pytest.raises(DuplicateToolError):
        registry.register(CalculatorTool())


@pytest.mark.asyncio
async def test_registry_rejects_invalid_arguments(context: ToolContext) -> None:
    registry = ToolRegistry([CalculatorTool()])

    with pytest.raises(ToolValidationError, match="expression"):
        await registry.execute("calculator", {"expression": 7}, context)


@pytest.mark.asyncio
async def test_registry_rejects_unknown_tools(context: ToolContext) -> None:
    registry = ToolRegistry()

    with pytest.raises(ToolNotFoundError, match="missing"):
        await registry.execute("missing", {}, context)
