"""Tool primitives and registry support."""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Type, get_type_hints

from pydantic import BaseModel, ConfigDict, create_model

from ..config import HarnessConfig
from ..errors import ConfigurationError, ToolError
from ..types import ConfirmCallback, ProbeFn
from ..weights import WeightStore

_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class ToolArgs(BaseModel):
    """Base class for tool arguments; unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid")


@dataclass(slots=True)
class ProbeBudget:
    remaining: int | None = None

    def consume(self) -> int | None:
        if self.remaining is None:
            return None
        if self.remaining <= 0:
            raise ToolError("probe budget exhausted")
        self.remaining -= 1
        return self.remaining


@dataclass(slots=True)
class ToolContext:
    store: WeightStore
    config: HarnessConfig
    probe: ProbeFn | None
    probe_budget: ProbeBudget
    confirm: ConfirmCallback | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def run_probe(self, payload: dict[str, Any]) -> Any:
        if self.probe is None:
            raise ToolError("no probe callback is configured")
        self.probe_budget.consume()
        try:
            return self.probe(self.store.module, payload)
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(f"probe failed: {type(exc).__name__}: {exc}") from exc


def validate_tool_name(name: str) -> None:
    if not isinstance(name, str) or not _TOOL_NAME_RE.match(name):
        raise ConfigurationError(
            f"invalid tool name {name!r}; use letters, digits, '_' or '-' "
            "(max 64 characters)"
        )


def validate_tool(tool: Tool) -> None:
    if not isinstance(tool, Tool):
        raise ConfigurationError(
            f"expected weighter.Tool instances, got {type(tool).__name__}"
        )
    validate_tool_name(tool.name)
    if not callable(tool.handler):
        raise ConfigurationError(f"tool {tool.name!r} handler must be callable")
    if not (isinstance(tool.args_model, type) and issubclass(tool.args_model, ToolArgs)):
        raise ConfigurationError(
            f"tool {tool.name!r} args_model must subclass ToolArgs"
        )


def _function_description(func: Callable[..., Any]) -> str:
    doc = inspect.getdoc(func)
    if doc:
        return doc.strip()
    return f"Run the {func.__name__} tool."


def _args_model_from_signature(func: Callable[..., Any]) -> Type[ToolArgs]:
    signature = inspect.signature(func)
    try:
        hints = get_type_hints(func)
    except Exception as exc:
        raise ConfigurationError(
            f"could not resolve type hints for tool {func.__name__!r}: {exc}"
        ) from exc

    parameters = list(signature.parameters.values())
    if not parameters:
        raise ConfigurationError(
            f"tool {func.__name__!r} must take a ToolContext as its first parameter"
        )
    first = parameters[0]
    first_hint = hints.get(first.name, first.annotation)
    if first.name not in {"ctx", "context"} and first_hint is not ToolContext:
        raise ConfigurationError(
            f"tool {func.__name__!r} first parameter must be a ToolContext "
            "named 'ctx' or 'context'"
        )
    if first.kind is inspect.Parameter.KEYWORD_ONLY:
        raise ConfigurationError(
            f"tool {func.__name__!r} ToolContext parameter must not be keyword-only"
        )

    fields: dict[str, Any] = {}
    for parameter in parameters[1:]:
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            raise ConfigurationError(
                f"tool {func.__name__!r} does not support *args or **kwargs"
            )
        annotation = hints.get(parameter.name, parameter.annotation)
        if annotation is inspect.Parameter.empty:
            raise ConfigurationError(
                f"tool argument {parameter.name!r} needs a type annotation"
            )
        default = (
            ...
            if parameter.default is inspect.Parameter.empty
            else parameter.default
        )
        fields[parameter.name] = (annotation, default)

    return create_model(f"{func.__name__}Args", __base__=ToolArgs, **fields)


@dataclass(slots=True)
class Tool:
    name: str
    description: str
    args_model: Type[ToolArgs]
    handler: Callable[[ToolContext, Any], Any]
    mutates: bool = False

    def schema(self) -> dict[str, Any]:
        parameters = self.args_model.model_json_schema()
        parameters.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }

    @classmethod
    def from_function(
        cls,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        mutates: bool = False,
        args_model: Type[ToolArgs] | None = None,
    ) -> "Tool":
        if not callable(func):
            raise ConfigurationError("tool handler must be callable")
        tool_name = name or func.__name__
        validate_tool_name(tool_name)
        model = args_model or _args_model_from_signature(func)

        def handler(ctx: ToolContext, args: ToolArgs) -> Any:
            values = {name: getattr(args, name) for name in type(args).model_fields}
            return func(ctx, **values)

        return cls(
            name=tool_name,
            description=description or _function_description(func),
            args_model=model,
            handler=handler,
            mutates=mutates,
        )


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    mutates: bool = False,
    args_model: Type[ToolArgs] | None = None,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Turn a typed function into a :class:`Tool`.

    Use as ``@tool`` or ``@tool(name=..., description=..., mutates=True)``.
    The first parameter must be a :class:`ToolContext`; remaining parameters
    become the JSON-schema arguments.
    """

    def decorator(target: Callable[..., Any]) -> Tool:
        return Tool.from_function(
            target,
            name=name,
            description=description,
            mutates=mutates,
            args_model=args_model,
        )

    if func is None:
        return decorator
    return decorator(func)
