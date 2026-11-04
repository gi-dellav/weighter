"""Configuration objects for the harness."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from .errors import ConfigurationError
from .types import ConfirmCallback, EditPolicy, ProbeFn

TensorFilter = Callable[[str], bool] | str | re.Pattern[str] | None


def _compile_filter(tensor_filter: TensorFilter) -> Callable[[str], bool] | None:
    if tensor_filter is None:
        return None
    if callable(tensor_filter):
        return tensor_filter
    if isinstance(tensor_filter, str):
        tensor_filter = re.compile(tensor_filter)
    if isinstance(tensor_filter, re.Pattern):
        return lambda name: bool(tensor_filter.search(name))
    raise ConfigurationError("tensor_filter must be None, a callable, or a regex string")


@dataclass(slots=True)
class HarnessConfig:
    """Runtime configuration for :class:`weighter.Harness`.

    The LLM id is a LiteLLM model string such as ``openai/gpt-4.1-mini`` or
    ``anthropic/claude-sonnet-4``.  If it is omitted, ``WEIGHTER_LLM`` is used.
    """

    llm: str | None = None
    api_key: str | None = None
    api_base: str | None = None
    temperature: float | None = None
    request_timeout: float | None = None
    completion_kwargs: Mapping[str, Any] = field(default_factory=dict)
    system_prompt: str | None = None

    probe: ProbeFn | None = None
    probe_budget: int | None = None
    edit_policy: EditPolicy = "auto"
    confirm_callback: ConfirmCallback | None = None
    include_buffers: bool = False
    tensor_filter: TensorFilter = None
    undo: bool = True
    max_undo_bytes: int = 512 * 1024 * 1024
    max_edit_elements: int = 1_000_000
    max_read_elements: int = 2_048
    max_tool_output_chars: int = 50_000
    max_steps: int = 16
    max_history_messages: int | None = None
    stream: bool = False

    def __post_init__(self) -> None:
        self.llm = self.llm or os.environ.get("WEIGHTER_LLM")
        if self.edit_policy not in {"auto", "confirm", "read_only"}:
            raise ConfigurationError("edit_policy must be 'auto', 'confirm', or 'read_only'")
        if self.probe_budget is not None and self.probe_budget < 0:
            raise ConfigurationError("probe_budget must be >= 0 or None")
        if self.max_steps < 1:
            raise ConfigurationError("max_steps must be >= 1")
        if self.max_read_elements < 1:
            raise ConfigurationError("max_read_elements must be >= 1")
        if self.max_edit_elements < 1:
            raise ConfigurationError("max_edit_elements must be >= 1")
        if self.max_undo_bytes < 0:
            raise ConfigurationError("max_undo_bytes must be >= 0")
        if self.max_history_messages is not None and self.max_history_messages < 1:
            raise ConfigurationError("max_history_messages must be >= 1 or None")
        self.tensor_filter = _compile_filter(self.tensor_filter)
        self.completion_kwargs = dict(self.completion_kwargs)
