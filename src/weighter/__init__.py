"""Weighter: a weight-space programming AI harness for PyTorch modules."""

from __future__ import annotations

from .config import HarnessConfig
from .errors import (
    ConfigurationError,
    EditError,
    LLMError,
    ToolError,
    WeighterError,
)
from .harness import Harness
from .tools import Tool, ToolArgs, ToolContext, default_probe, tool
from .types import EditPolicy, PromptResult, ToolCallRecord, Usage

__version__ = "1.0.0"

__all__ = [
    "ConfigurationError",
    "EditError",
    "EditPolicy",
    "Harness",
    "HarnessConfig",
    "LLMError",
    "PromptResult",
    "Tool",
    "ToolArgs",
    "ToolCallRecord",
    "ToolContext",
    "ToolError",
    "Usage",
    "WeighterError",
    "default_probe",
    "tool",
]
