from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from typing import Any

from minimal_agent.domain import JsonObject, ToolValidationError
from minimal_agent.tools.base import ToolContext

BinaryOperator = Callable[[Any, Any], Any]
UnaryOperator = Callable[[Any], Any]

_BINARY_OPERATORS: dict[type[ast.operator], BinaryOperator] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], UnaryOperator] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class CalculatorTool:
    name = "calculator"
    description = "Evaluate a basic arithmetic expression safely."
    parameters: JsonObject = {
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
        "additionalProperties": False,
    }

    async def execute(self, arguments: JsonObject, context: ToolContext) -> JsonObject:
        del context
        expression = str(arguments["expression"])
        if len(expression) > 200:
            raise ToolValidationError("expression is too long")
        try:
            tree = ast.parse(expression, mode="eval")
            value = _evaluate(tree.body)
        except (SyntaxError, TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
            if isinstance(exc, ToolValidationError):
                raise
            raise ToolValidationError(f"invalid expression: {exc}") from exc
        return {"ok": True, "value": value}


def _evaluate(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 12:
            raise ToolValidationError("unsupported exponent")
        return _BINARY_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate(node.operand))
    raise ToolValidationError(f"unsupported syntax: {type(node).__name__}")
