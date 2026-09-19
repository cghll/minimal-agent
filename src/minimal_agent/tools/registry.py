from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from minimal_agent.domain import (
    DuplicateToolError,
    JsonObject,
    ToolNotFoundError,
    ToolValidationError,
)
from minimal_agent.tools.base import Tool, ToolContext


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise DuplicateToolError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def catalog(self) -> list[JsonObject]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in self._tools.values()
        ]

    async def execute(self, name: str, arguments: JsonObject, context: ToolContext) -> JsonObject:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(f"Unknown tool: {name}")
        _validate_schema(arguments, tool.parameters, "arguments")
        return await tool.execute(arguments, context)


def _validate_schema(value: Any, schema: JsonObject, path: str) -> None:
    expected_type = schema.get("type")
    type_map: dict[str, type | tuple[type, ...]] = {
        "object": dict,
        "array": list,
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
    }
    python_type = type_map.get(str(expected_type))
    if python_type and (
        not isinstance(value, python_type)
        or isinstance(value, bool)
        and expected_type in {"integer", "number"}
    ):
        raise ToolValidationError(f"{path} must be {expected_type}")

    if "enum" in schema and value not in schema["enum"]:
        raise ToolValidationError(f"{path} must be one of {schema['enum']}")

    if expected_type == "object" and isinstance(value, dict):
        properties = schema.get("properties", {})
        for required in schema.get("required", []):
            if required not in value:
                raise ToolValidationError(f"{path}.{required} is required")
        if schema.get("additionalProperties") is False:
            extras = set(value) - set(properties)
            if extras:
                raise ToolValidationError(f"{path} has unsupported fields: {sorted(extras)}")
        for key, item in value.items():
            if key in properties:
                _validate_schema(item, properties[key], f"{path}.{key}")

    if expected_type == "array" and isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value):
            _validate_schema(item, schema["items"], f"{path}[{index}]")
