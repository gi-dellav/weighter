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
from .tools import default_probe
from .types import EditPolicy, PromptResult, ToolCallRecord, Usage

__version__ = "0.1.0"

__all__ = [
    "Harness",
    "HarnessConfig",
    "PromptResult",
    "ToolCallRecord",
    "Usage",
    "EditPolicy",
    "default_probe",
    "WeighterError",
    "ConfigurationError",
    "ToolError",
    "EditError",
    "LLMError",
]
