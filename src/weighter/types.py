"""Shared public types for the harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import torch

ProbeFn = Callable[[torch.nn.Module, dict[str, Any]], Any]
ConfirmCallback = Callable[["EditProposal"], bool]
EditPolicy = Literal["auto", "confirm", "read_only"]
HarnessEvent = dict[str, Any]


@dataclass(slots=True)
class EditProposal:
    """Description of a proposed atomic weight edit."""

    ops: list[dict[str, Any]]
    targets: list[str]
    selection_elements: int
    note: str | None = None


@dataclass(slots=True)
class ToolCallRecord:
    """One tool call executed during a prompt."""

    id: str
    name: str
    arguments: dict[str, Any]
    ok: bool
    result: dict[str, Any]
    duration: float = 0.0


@dataclass(slots=True)
class Usage:
    """Accumulated token usage."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens

    @classmethod
    def from_object(cls, usage: Any) -> "Usage":
        if usage is None:
            return cls()
        get = usage.get if isinstance(usage, dict) else lambda key, default=None: getattr(usage, key, default)
        prompt = int(get("prompt_tokens", 0) or 0)
        completion = int(get("completion_tokens", 0) or 0)
        total = int(get("total_tokens", 0) or (prompt + completion))
        return cls(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)


@dataclass(slots=True)
class PromptResult:
    """Result returned by :meth:`Harness.prompt`."""

    text: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    revisions: list[str] = field(default_factory=list)
    stop_reason: str = "complete"

    def __str__(self) -> str:
        return self.text
