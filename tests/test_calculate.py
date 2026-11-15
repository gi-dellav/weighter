from __future__ import annotations

import pytest
import torch

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


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("not 0", True),
        ("+2", 2),
        ("-2", -2),
        ("2 < 1", False),
        ("True and False", False),
        ("True or False", True),
        ("pi > e", True),
        ("tau", pytest.approx(6.283185307179586)),
        ("floor(2.9)", 2),
        ("ceil(2.1)", 3),
        ("log(exp(1))", pytest.approx(1.0)),
        ("log2(8)", 3.0),
        ("log10(100)", 2.0),
        ("tan(0)", 0.0),
        ("cos(0)", 1.0),
        ("1 % 2", 1),
        ("-7 // 2", -4),
        ("round(2.5)", 2),
        ("min(1, 2, 3)", 1),
        ("max(1, 2)", 2),
    ],
)
def test_more_valid_expressions(expression: str, expected) -> None:
    assert safe_eval_expression(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "",
        "   ",
        "1 << 2",
        "[1, 2]",
        "{1, 2}",
        "'text'",
        "None",
        "~1",
        "1 is 1",
        "round(1, ndigits=2)",
        "1 // 0",
        "1 % 0",
        "0 ** -1",
        "1e300 * 1e300",
        "-" * 60 + "1",
        "1 + " * 150 + "1",
        "2 ** 1001",
        ("9" * 800) + " ** 2",
    ],
)
def test_more_invalid_expressions(expression: str) -> None:
    with pytest.raises(ToolError):
        safe_eval_expression(expression)


def test_safe_eval_rejects_non_string_input() -> None:
    with pytest.raises(ToolError, match="non-empty string"):
        safe_eval_expression(None)  # type: ignore[arg-type]
    with pytest.raises(ToolError, match="non-empty string"):
        safe_eval_expression(123)  # type: ignore[arg-type]


def test_safe_eval_reports_syntax_and_unknown_names() -> None:
    with pytest.raises(ToolError, match="invalid expression"):
        safe_eval_expression("1 +")
    with pytest.raises(ToolError, match="unknown name"):
        safe_eval_expression("unknown + 1")
    with pytest.raises(ToolError, match="unknown function"):
        safe_eval_expression("unknown(1)")
    with pytest.raises(ToolError, match="arithmetic error"):
        safe_eval_expression("min()")


def test_calculate_tool_handler() -> None:
    from weighter.tools.base import ProbeBudget, ToolContext
    from weighter.config import HarnessConfig
    from weighter.tools.calculate import CalculateArgs, calculate
    from weighter.weights import WeightStore

    context = ToolContext(
        store=WeightStore(torch.nn.Linear(2, 2)),
        config=HarnessConfig(llm="fake"),
        probe=None,
        probe_budget=ProbeBudget(),
    )
    result = calculate(context, CalculateArgs(expression="1 / 2"))
    assert result == {
        "expression": "1 / 2",
        "result": 0.5,
        "repr": "0.5",
        "type": "float",
    }
