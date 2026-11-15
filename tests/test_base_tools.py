from __future__ import annotations

import pytest
import torch

from weighter.config import HarnessConfig
from weighter.errors import ConfigurationError, ToolError
from weighter.tools.base import (
    ProbeBudget,
    Tool,
    ToolArgs,
    ToolContext,
    tool,
    validate_tool,
    validate_tool_name,
)
from weighter.weights import WeightStore


def _context(*, probe=None, remaining: int | None = None) -> ToolContext:
    return ToolContext(
        store=WeightStore(torch.nn.Linear(2, 2)),
        config=HarnessConfig(llm="fake"),
        probe=probe,
        probe_budget=ProbeBudget(remaining=remaining),
    )


def test_probe_budget_limited_and_unlimited() -> None:
    budget = ProbeBudget(remaining=2)
    assert budget.consume() == 1
    assert budget.consume() == 0
    with pytest.raises(ToolError, match="exhausted"):
        budget.consume()

    unlimited = ProbeBudget(remaining=None)
    assert unlimited.consume() is None
    assert unlimited.consume() is None


def test_tool_context_run_probe_errors_and_wrapping() -> None:
    with pytest.raises(ToolError, match="no probe"):
        _context().run_probe({})

    def failing(module, payload):
        raise ValueError("bad probe")

    with pytest.raises(ToolError, match="probe failed: ValueError: bad probe"):
        _context(probe=failing, remaining=1).run_probe({})

    def tool_error(module, payload):
        raise ToolError("preserved")

    with pytest.raises(ToolError, match="preserved"):
        _context(probe=tool_error, remaining=1).run_probe({})


def test_tool_context_run_probe_returns_value_and_consumes_budget() -> None:
    seen: list[dict] = []

    def probe(module, payload):
        seen.append(payload)
        return {"echo": payload}

    context = _context(probe=probe, remaining=1)
    assert context.run_probe({"x": 1}) == {"echo": {"x": 1}}
    assert seen == [{"x": 1}]
    assert context.probe_budget.remaining == 0


def test_validate_tool_name_rules() -> None:
    validate_tool_name("read_tensor")
    validate_tool_name("a-b")
    validate_tool_name("A" * 64)
    for bad in ["", "has space", "a" * 65, "b@d", None]:
        with pytest.raises(ConfigurationError):
            validate_tool_name(bad)  # type: ignore[arg-type]


def test_validate_tool_rejects_wrong_objects() -> None:
    with pytest.raises(ConfigurationError, match="expected weighter.Tool"):
        validate_tool(object())  # type: ignore[arg-type]

    class NotCallable:
        pass

    bad_handler = Tool(
        name="bad",
        description="Bad.",
        args_model=ToolArgs,
        handler=NotCallable(),  # type: ignore[arg-type]
    )
    with pytest.raises(ConfigurationError, match="callable"):
        validate_tool(bad_handler)

    class NotToolArgs:
        pass

    bad_model = Tool(
        name="bad",
        description="Bad.",
        args_model=NotToolArgs,  # type: ignore[arg-type]
        handler=lambda ctx, args: {},
    )
    with pytest.raises(ConfigurationError, match="subclass ToolArgs"):
        validate_tool(bad_model)


def test_tool_from_function_builds_schema_and_handler() -> None:
    seen: list[tuple[ToolContext, str, int]] = []

    def echo(ctx: ToolContext, label: str, count: int = 2) -> dict:
        """Echo a label."""
        seen.append((ctx, label, count))
        return {"label": label, "count": count}

    built = Tool.from_function(echo)
    assert built.name == "echo"
    assert built.description == "Echo a label."
    assert built.mutates is False
    schema = built.schema()
    function = schema["function"]
    assert "title" not in function["parameters"]
    assert function["parameters"]["required"] == ["label"]

    context = _context()
    args = built.args_model(label="hi")
    assert built.handler(context, args) == {"label": "hi", "count": 2}
    assert seen[0][0] is context


def _no_parameters() -> dict:
    return {}


def _wrong_context(other, value) -> dict:
    return {}


def _varargs(ctx: ToolContext, *values: int) -> dict:
    return {"values": values}


def _kwargs(ctx: ToolContext, **values: int) -> dict:
    return {"values": values}


@pytest.mark.parametrize(
    "func, match",
    [
        (_no_parameters, "first parameter"),
        (_wrong_context, "ToolContext"),
        (_varargs, r"\*args or \*\*kwargs"),
        (_kwargs, r"\*args or \*\*kwargs"),
    ],
)
def test_tool_signature_validation(func, match: str) -> None:
    with pytest.raises(ConfigurationError, match=match):
        Tool.from_function(func)


def test_tool_signature_requires_annotations() -> None:
    def missing(ctx: ToolContext, value) -> dict:
        return {"value": value}

    with pytest.raises(ConfigurationError, match="type annotation"):
        Tool.from_function(missing)


def test_tool_keyword_only_context_is_rejected() -> None:
    def keyword_only(*, ctx: ToolContext) -> dict:
        return {}

    with pytest.raises(ConfigurationError, match="keyword-only"):
        Tool.from_function(keyword_only)


def test_tool_description_fallback_and_overrides() -> None:
    def undocumented(ctx: ToolContext, value: int = 1) -> dict:
        return {"value": value}

    built = Tool.from_function(undocumented)
    assert built.description == "Run the undocumented tool."

    renamed = tool(
        undocumented,
        name="renamed",
        description="Custom description.",
        mutates=True,
    )
    assert isinstance(renamed, Tool)
    assert renamed.name == "renamed"
    assert renamed.description == "Custom description."
    assert renamed.mutates is True


def test_tool_decorator_with_arguments_and_explicit_model() -> None:
    class EchoArgs(ToolArgs):
        text: str

    @tool(name="explicit_echo", args_model=EchoArgs)
    def echo(ctx: ToolContext, text: str = "ignored") -> dict:
        return {"text": text}

    assert echo.name == "explicit_echo"
    assert echo.args_model is EchoArgs
    result = echo.handler(_context(), EchoArgs(text="hello"))
    assert result == {"text": "hello"}


def test_tool_from_function_rejects_non_callable() -> None:
    with pytest.raises(ConfigurationError, match="callable"):
        Tool.from_function(5)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="callable"):
        tool(5)  # type: ignore[arg-type]
