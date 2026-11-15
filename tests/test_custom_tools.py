from __future__ import annotations

import json

import pytest
import torch

from weighter import (
    ConfigurationError,
    Harness,
    Tool,
    ToolArgs,
    ToolContext,
    tool,
)


def tool_call(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def response(content: str | None = None, tool_calls: list[dict] | None = None) -> dict:
    message: dict = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "choices": [{"message": message}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


class ScriptedCompletion:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("no scripted completion left")
        return self.responses.pop(0)


@pytest.fixture()
def module() -> torch.nn.Linear:
    return torch.nn.Linear(3, 2)


def test_decorator_infers_schema_and_executes(module: torch.nn.Linear) -> None:
    seen: list[dict] = []

    @tool
    def echo_context(ctx: ToolContext, label: str, count: int = 2) -> dict:
        """Echo a label and the harness extras."""
        seen.append(ctx.extras)
        return {"label": label, "count": count, "keys": sorted(ctx.extras)}

    fake = ScriptedCompletion(
        [
            response(
                tool_calls=[
                    tool_call("c1", "echo_context", {"label": "hello", "count": 4})
                ]
            ),
            response("Done."),
        ]
    )
    harness = Harness(
        module,
        llm="fake/model",
        tools=[echo_context],
        tool_extras={"base": 3},
        completion_fn=fake,
    )
    result = harness.prompt("use the custom tool")

    assert result.tool_calls[0].ok is True
    assert result.tool_calls[0].result["data"] == {
        "label": "hello",
        "count": 4,
        "keys": ["base"],
    }
    assert seen == [{"base": 3}]
    assert "echo_context" in harness.tools

    schemas = {item["function"]["name"]: item for item in fake.calls[0]["tools"]}
    parameters = schemas["echo_context"]["function"]["parameters"]
    assert schemas["echo_context"]["function"]["description"] == (
        "Echo a label and the harness extras."
    )
    assert parameters["required"] == ["label"]
    assert parameters["properties"]["count"]["default"] == 2


def test_explicit_tool_executes(module: torch.nn.Linear) -> None:
    class RecordArgs(ToolArgs):
        value: int

    def record(ctx: ToolContext, args: RecordArgs) -> dict:
        return {"value": args.value, "module": type(ctx.store.module).__name__}

    custom = Tool(
        name="record_value",
        description="Record an integer.",
        args_model=RecordArgs,
        handler=record,
    )
    fake = ScriptedCompletion(
        [
            response(tool_calls=[tool_call("c1", "record_value", {"value": 7})]),
            response("Done."),
        ]
    )
    harness = Harness(
        module,
        llm="fake/model",
        tools=[custom],
        completion_fn=fake,
    )
    result = harness.prompt("record it")

    assert result.tool_calls[0].ok is True
    assert result.tool_calls[0].result["data"] == {
        "value": 7,
        "module": "Linear",
    }
    assert harness.tools["record_value"] is custom


def test_custom_tool_results_and_errors_are_json_safe(module: torch.nn.Linear) -> None:
    @tool
    def make_tensor(ctx: ToolContext) -> torch.Tensor:
        """Return a tensor."""
        return torch.ones(2, 3)

    @tool
    def boom(ctx: ToolContext) -> dict:
        """Always fail."""
        raise ValueError("kaboom")

    fake = ScriptedCompletion(
        [
            response(tool_calls=[tool_call("c1", "make_tensor", {})]),
            response(tool_calls=[tool_call("c2", "boom", {})]),
            response("Done."),
        ]
    )
    harness = Harness(
        module,
        llm="fake/model",
        tools=[make_tensor, boom],
        completion_fn=fake,
    )
    result = harness.prompt("call both")

    assert result.tool_calls[0].ok is True
    tensor_data = result.tool_calls[0].result["data"]
    assert tensor_data.shape == (2, 3)
    tool_messages = [m for m in harness.messages if m["role"] == "tool"]
    assert '"type":"tensor"' in tool_messages[0]["content"]

    assert result.tool_calls[1].ok is False
    assert result.tool_calls[1].result["error_type"] == "ValueError"
    assert "kaboom" in result.tool_calls[1].result["error"]


def test_include_and_exclude_builtins(module: torch.nn.Linear) -> None:
    @tool
    def custom(ctx: ToolContext) -> dict:
        """A custom tool."""
        return {"ok": True}

    fake = ScriptedCompletion([])
    harness = Harness(
        module,
        llm="fake/model",
        tools=[custom],
        exclude_tools=["run_probe"],
        completion_fn=fake,
    )
    assert "run_probe" not in harness.tools
    assert "custom" in harness.tools

    limited = Harness(
        module,
        llm="fake/model",
        tools=[custom],
        include_tools=["read_tensor", "calculate"],
        completion_fn=fake,
    )
    assert set(limited.tools) == {"custom", "read_tensor", "calculate"}

    only_custom = Harness(
        module,
        llm="fake/model",
        tools=[custom],
        include_tools=[],
        completion_fn=fake,
    )
    assert set(only_custom.tools) == {"custom"}


def test_unknown_builtin_names_raise(module: torch.nn.Linear) -> None:
    fake = ScriptedCompletion([])
    with pytest.raises(ConfigurationError, match="unknown built-in tool"):
        Harness(
            module,
            llm="fake/model",
            include_tools=["not_a_tool"],
            completion_fn=fake,
        )
    with pytest.raises(ConfigurationError, match="unknown built-in tool"):
        Harness(
            module,
            llm="fake/model",
            exclude_tools=["not_a_tool"],
            completion_fn=fake,
        )


def test_duplicate_and_colliding_names(module: torch.nn.Linear) -> None:
    @tool
    def duplicate(ctx: ToolContext) -> dict:
        """Duplicate."""
        return {}

    @tool(name="calculate")
    def override_calculate(ctx: ToolContext) -> dict:
        """Replace the calculator."""
        return {"custom": True}

    fake = ScriptedCompletion([])
    with pytest.raises(ConfigurationError, match="duplicate custom tool"):
        Harness(
            module,
            llm="fake/model",
            tools=[duplicate, duplicate],
            completion_fn=fake,
        )
    with pytest.raises(ConfigurationError, match="collides with a built-in tool"):
        Harness(
            module,
            llm="fake/model",
            tools=[override_calculate],
            completion_fn=fake,
        )

    harness = Harness(
        module,
        llm="fake/model",
        tools=[override_calculate],
        allow_tool_override=True,
        completion_fn=fake,
    )
    assert harness.tools["calculate"] is override_calculate


def test_read_only_hides_mutating_custom_tools(module: torch.nn.Linear) -> None:
    @tool(mutates=True)
    def mutating(ctx: ToolContext) -> dict:
        """A mutating custom tool."""
        return {}

    @tool
    def read_only(ctx: ToolContext) -> dict:
        """A read-only custom tool."""
        return {}

    fake = ScriptedCompletion([])
    harness = Harness(
        module,
        llm="fake/model",
        edit_policy="read_only",
        tools=[mutating, read_only],
        completion_fn=fake,
    )
    assert "mutating" not in harness.tools
    assert "read_only" in harness.tools

    normal = Harness(
        module,
        llm="fake/model",
        tools=[mutating],
        completion_fn=fake,
    )
    assert "mutating" in normal.tools


def test_confirm_policy_checks_custom_mutating_tools(module: torch.nn.Linear) -> None:
    @tool(mutates=True)
    def mutating(ctx: ToolContext) -> dict:
        """A mutating custom tool."""
        return {}

    fake = ScriptedCompletion(
        [
            response(tool_calls=[tool_call("c1", "mutating", {})]),
            response("Done."),
        ]
    )
    harness = Harness(
        module,
        llm="fake/model",
        edit_policy="confirm",
        tools=[mutating],
        completion_fn=fake,
    )
    result = harness.prompt("mutate")

    assert result.tool_calls[0].ok is False
    assert "confirm_callback" in result.tool_calls[0].result["error"]


def test_invalid_tool_definitions_raise(module: torch.nn.Linear) -> None:
    def missing_context(value: int) -> dict:
        return {"value": value}

    def missing_annotation(ctx: ToolContext, value) -> dict:
        return {"value": value}

    def varargs(ctx: ToolContext, *values: int) -> dict:
        return {"values": values}

    with pytest.raises(ConfigurationError, match="ToolContext"):
        tool(missing_context)
    with pytest.raises(ConfigurationError, match="type annotation"):
        tool(missing_annotation)
    with pytest.raises(ConfigurationError, match=r"\*args or \*\*kwargs"):
        tool(varargs)
    with pytest.raises(ConfigurationError, match="invalid tool name"):
        tool(missing_annotation, name="bad name!")

    class NotToolArgs:
        pass

    bad = Tool(
        name="bad_tool",
        description="Bad.",
        args_model=NotToolArgs,  # type: ignore[arg-type]
        handler=lambda ctx, args: {},
    )
    fake = ScriptedCompletion([])
    with pytest.raises(ConfigurationError, match="args_model"):
        Harness(
            module,
            llm="fake/model",
            tools=[bad],
            completion_fn=fake,
        )


def test_tool_extras_are_shared_and_mutable(module: torch.nn.Linear) -> None:
    @tool
    def bump(ctx: ToolContext, step: int = 1) -> dict:
        """Increment shared state."""
        ctx.extras["count"] = ctx.extras.get("count", 0) + step
        return {"count": ctx.extras["count"]}

    fake = ScriptedCompletion(
        [
            response(tool_calls=[tool_call("c1", "bump", {"step": 2})]),
            response(tool_calls=[tool_call("c2", "bump", {})]),
            response("Done."),
        ]
    )
    harness = Harness(
        module,
        llm="fake/model",
        tools=[bump],
        tool_extras={"count": 0},
        completion_fn=fake,
    )
    result = harness.prompt("bump twice")

    assert result.tool_calls[0].result["data"]["count"] == 2
    assert result.tool_calls[1].result["data"]["count"] == 3
    assert harness._tool_context().extras["count"] == 3
