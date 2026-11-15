"""Tool registry for the weight-space harness."""

from __future__ import annotations

from collections.abc import Collection, Iterable

from ..errors import ConfigurationError
from .ablate import TOOLS as ABLATE_TOOLS
from .base import Tool, ToolArgs, ToolContext, tool, validate_tool
from .calculate import TOOLS as CALCULATE_TOOLS
from .edit import TOOLS as EDIT_TOOLS
from .history import TOOLS as HISTORY_TOOLS
from .inspect import TOOLS as INSPECT_TOOLS
from .linalg import TOOLS as LINALG_TOOLS
from .probe import TOOLS as PROBE_TOOLS, default_probe

__all__ = [
    "Tool",
    "ToolArgs",
    "ToolContext",
    "build_tools",
    "default_probe",
    "tool",
]

_BUILTIN_TOOLS: tuple[Tool, ...] = (
    *INSPECT_TOOLS,
    *LINALG_TOOLS,
    *CALCULATE_TOOLS,
    *HISTORY_TOOLS,
    *EDIT_TOOLS,
    *ABLATE_TOOLS,
    *PROBE_TOOLS,
)


def build_tools(
    *,
    allow_writes: bool = True,
    undo_available: bool = True,
    probe_available: bool = True,
    extra_tools: Iterable[Tool] = (),
    include: Collection[str] | None = None,
    exclude: Collection[str] = (),
    allow_override: bool = False,
) -> dict[str, Tool]:
    tools: list[Tool] = [*INSPECT_TOOLS, *LINALG_TOOLS, *CALCULATE_TOOLS]
    for history_tool in HISTORY_TOOLS:
        if history_tool.mutates and not (allow_writes and undo_available):
            continue
        tools.append(history_tool)
    tools.extend(EDIT_TOOLS)
    tools.extend(ABLATE_TOOLS)
    if probe_available:
        tools.extend(PROBE_TOOLS)
    if not allow_writes:
        tools = [built_in for built_in in tools if not built_in.mutates]

    include_names = None if include is None else set(include)
    exclude_names = set(exclude)
    known_names = {built_in.name for built_in in _BUILTIN_TOOLS}
    unknown = (include_names or set()) | exclude_names
    unknown -= known_names
    if unknown:
        raise ConfigurationError(
            f"unknown built-in tool(s): {', '.join(sorted(unknown))}"
        )

    if include_names is not None:
        tools = [built_in for built_in in tools if built_in.name in include_names]
    tools = [built_in for built_in in tools if built_in.name not in exclude_names]

    registry = {built_in.name: built_in for built_in in tools}
    seen: set[str] = set()
    for custom in extra_tools:
        validate_tool(custom)
        if custom.name in seen:
            raise ConfigurationError(f"duplicate custom tool {custom.name!r}")
        seen.add(custom.name)
        if custom.mutates and not allow_writes:
            continue
        if custom.name in registry and not allow_override:
            raise ConfigurationError(
                f"custom tool {custom.name!r} collides with a built-in tool; "
                "pass allow_tool_override=True to replace it"
            )
        registry[custom.name] = custom
    return registry
