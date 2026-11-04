"""A safe scalar calculator with no Python code execution."""

from __future__ import annotations

import ast
import math
import operator
from typing import Any, Callable

from pydantic import Field

from ..errors import ToolError
from .base import Tool, ToolArgs, ToolContext

MAX_EXPRESSION_LENGTH = 1_000
MAX_AST_NODES = 200
MAX_DEPTH = 50
MAX_INT_BITS = 4_096
MAX_FLOAT = 1e300

_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
_COMPARE_OPS: dict[type[ast.cmpop], Callable[[Any, Any], bool]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}
_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "floor": math.floor,
    "ceil": math.ceil,
}
_CONSTANTS: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}


def _check_number(value: Any) -> int | float | bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value.bit_length() > MAX_INT_BITS:
            raise ToolError("integer result is too large")
        return value
    if isinstance(value, float):
        if not math.isfinite(value) or abs(value) > MAX_FLOAT:
            raise ToolError("floating-point result is out of range")
        return value
    if isinstance(value, complex):
        raise ToolError("complex results are not supported")
    raise ToolError("expression did not produce a number")


def _eval_node(node: ast.AST, depth: int = 0) -> int | float | bool:
    if depth > MAX_DEPTH:
        raise ToolError("expression is nested too deeply")
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, depth + 1)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (bool, int, float)):
            return _check_number(node.value)
        raise ToolError("only numeric literals are allowed")
    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise ToolError(f"unknown name {node.id!r}")
    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand, depth + 1)
        if isinstance(node.op, ast.UAdd):
            return _check_number(+operand)
        if isinstance(node.op, ast.USub):
            return _check_number(-operand)
        if isinstance(node.op, ast.Not):
            return not operand
        raise ToolError("unsupported unary operator")
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, depth + 1)
        right = _eval_node(node.right, depth + 1)
        try:
            if isinstance(node.op, ast.Pow):
                if isinstance(left, int) and isinstance(right, int) and abs(right) > 1_000:
                    raise ToolError("exponent is too large")
                result = left**right
            else:
                operation = _BIN_OPS.get(type(node.op))
                if operation is None:
                    raise ToolError("unsupported binary operator")
                result = operation(left, right)
        except ToolError:
            raise
        except ZeroDivisionError as exc:
            raise ToolError("division by zero") from exc
        except (OverflowError, ValueError, TypeError) as exc:
            raise ToolError(f"arithmetic error: {exc}") from exc
        return _check_number(result)
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, depth + 1)
        for op, comparator in zip(node.ops, node.comparators):
            right = _eval_node(comparator, depth + 1)
            operation = _COMPARE_OPS.get(type(op))
            if operation is None:
                raise ToolError("unsupported comparison operator")
            if not operation(left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            for value in node.values:
                if not _eval_node(value, depth + 1):
                    return False
            return True
        if isinstance(node.op, ast.Or):
            for value in node.values:
                if _eval_node(value, depth + 1):
                    return True
            return False
        raise ToolError("unsupported boolean operator")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ToolError("only direct calls to whitelisted functions are allowed")
        function = _FUNCTIONS.get(node.func.id)
        if function is None:
            raise ToolError(f"unknown function {node.func.id!r}")
        if node.keywords:
            raise ToolError("keyword arguments are not allowed")
        arguments = [_eval_node(arg, depth + 1) for arg in node.args]
        try:
            return _check_number(function(*arguments))
        except ToolError:
            raise
        except ZeroDivisionError as exc:
            raise ToolError("division by zero") from exc
        except (OverflowError, ValueError, TypeError) as exc:
            raise ToolError(f"arithmetic error: {exc}") from exc
    raise ToolError(f"unsupported expression element: {type(node).__name__}")


def safe_eval_expression(expression: str) -> int | float | bool:
    """Evaluate a restricted arithmetic expression without ``eval``/``exec``."""

    if not isinstance(expression, str) or not expression.strip():
        raise ToolError("expression must be a non-empty string")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ToolError(f"expression is longer than {MAX_EXPRESSION_LENGTH} characters")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"invalid expression: {exc.msg}") from exc
    if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
        raise ToolError("expression has too many syntax nodes")
    return _eval_node(tree)


class CalculateArgs(ToolArgs):
    expression: str = Field(
        description="Arithmetic expression, e.g. '(167 * 28 + 1.9) / 3'"
    )


def calculate(ctx: ToolContext, args: CalculateArgs) -> dict[str, Any]:
    value = safe_eval_expression(args.expression)
    return {
        "expression": args.expression,
        "result": value,
        "repr": repr(value),
        "type": type(value).__name__,
    }


TOOLS: list[Tool] = [
    Tool(
        name="calculate",
        description=(
            "Evaluate a safe scalar arithmetic expression without code execution. "
            "Supports + - * / // % **, parentheses, comparisons, booleans, pi/e/tau, "
            "and abs/min/max/round/sqrt/exp/log/log2/log10/sin/cos/tan/floor/ceil. "
            "Useful for exact ratios, deltas, and scaling factors."
        ),
        args_model=CalculateArgs,
        handler=calculate,
    )
]
