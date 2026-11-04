"""Tool registry for the weight-space harness."""

from __future__ import annotations

from .base import Tool, ToolContext
from .edit import TOOLS as EDIT_TOOLS
from .history import TOOLS as HISTORY_TOOLS
from .inspect import TOOLS as INSPECT_TOOLS
from .linalg import TOOLS as LINALG_TOOLS
from .probe import TOOLS as PROBE_TOOLS, default_probe

__all__ = [
    "Tool",
    "ToolContext",
    "build_tools",
    "default_probe",
]


def build_tools(
    *,
    allow_writes: bool = True,
    undo_available: bool = True,
    probe_available: bool = True,
) -> dict[str, Tool]:
    tools: list[Tool] = [*INSPECT_TOOLS, *LINALG_TOOLS]
    for tool in HISTORY_TOOLS:
        if tool.mutates and not (allow_writes and undo_available):
            continue
        tools.append(tool)
    if allow_writes:
        tools.extend(EDIT_TOOLS)
    if probe_available:
        tools.extend(PROBE_TOOLS)
    return {tool.name: tool for tool in tools}
