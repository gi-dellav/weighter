from __future__ import annotations

import pytest

from weighter.errors import ToolError
from weighter.tools.calculate import safe_eval_expression


def test_basic_arithmetic() -> None:
    assert safe_eval_expression("167 * 28 + 1.9") == 4677.9
    assert safe_eval_expression("(2 + 3) * 4") == 20
    assert safe_eval_expression("7 // 2") == 3
    assert safe_eval_expression("2 ** 10") == 1024
    assert safe_eval_expression("1 < 2 < 3") is True


def test_functions_and_constants() -> None:
    assert safe_eval_expression("round(sqrt(2), 3)") == 1.414
    assert safe_eval_expression("max(3, 7) - min(3, 7)") == 4
    assert safe_eval_expression("sin(pi / 2)") == pytest.approx(1.0)
    assert safe_eval_expression("abs(-4)") == 4


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo hi')",
        "open('x')",
        "x + 1",
        "(1).__class__",
        "[1, 2][0]",
        "(lambda: 1)()",
        "[x for x in range(3)]",
        "sqrt(-1)",
        "1 / 0",
        "2 ** 10000",
        "9 ** 9 ** 9",
    ],
)
def test_rejects_unsafe_or_invalid_expressions(expression: str) -> None:
    with pytest.raises(ToolError):
        safe_eval_expression(expression)


def test_rejects_long_expressions() -> None:
    with pytest.raises(ToolError):
        safe_eval_expression("1 + " * 600 + "1")
