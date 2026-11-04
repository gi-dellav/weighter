"""Tool primitives and registry support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Type

from pydantic import BaseModel, ConfigDict

from ..config import HarnessConfig
from ..errors import ToolError
from ..types import ConfirmCallback, ProbeFn
from ..weights import WeightStore


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


@dataclass(slots=True)
class Tool:
    name: str
    description: str
    args_model: Type[ToolArgs]
    handler: Callable[[ToolContext, Any], dict[str, Any]]
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
