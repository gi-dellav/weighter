"""Structured weight-writing tools."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from ..weights import summarize_diff
from .base import Tool, ToolArgs, ToolContext


class EditOp(ToolArgs):
    op: Literal["set", "add", "scale", "fill", "copy", "project_out", "random", "clamp"]
    name: str = Field(description="Tensor to edit")
    index: list[Any] | None = Field(
        default=None,
        description='Per-dimension selection, e.g. [null, "0:32"]; null/omitted means all',
    )
    values: Any = Field(default=None, description="Values for set/add")
    value: Any = Field(default=None, description="Scalar for fill")
    factor: float | None = Field(default=None, description="Scale factor")
    src: str | None = Field(default=None, description="Source tensor for copy")
    src_index: list[Any] | None = None
    direction: list[float] | None = Field(
        default=None, description="1D vector for project_out"
    )
    axis: int = Field(default=-1, description="Axis for project_out")
    scale: float = Field(default=1.0, description="Scale for add/project_out")
    distribution: Literal["normal", "uniform"] = "normal"
    std: float = Field(default=0.02, ge=0.0)
    low: float = -0.1
    high: float = 0.1
    seed: int | None = None
    min: float | None = None
    max: float | None = None


class ApplyEditsArgs(ToolArgs):
    edits: list[EditOp] = Field(min_length=1, description="Atomic list of weight edits")
    note: str | None = Field(default=None, description="Short rationale for the revision")


def apply_edits(ctx: ToolContext, args: ApplyEditsArgs) -> dict[str, Any]:
    ops = [edit.model_dump(exclude_none=True) for edit in args.edits]
    revision = ctx.store.apply(
        ops,
        note=args.note,
        max_edit_elements=ctx.config.max_edit_elements,
        confirm=ctx.confirm,
    )
    result: dict[str, Any] = {
        "revision": revision.info(),
        "changed_tensors": revision.tensors,
        "hint": "Use weight_diff, revision_history, or undo_edits to inspect or revert.",
    }
    try:
        result["diff"] = summarize_diff(
            ctx.store.diff(since="last"),
            module_elements=ctx.store.total_elements(),
        )
    except Exception:
        result["diff"] = None
    return result


TOOLS: list[Tool] = [
    Tool(
        name="apply_edits",
        description=(
            "Apply one atomic transaction of structured weight edits. Ops: set, add, scale, "
            "fill, copy, project_out (subtract a direction along an axis), random, clamp. "
            "Every op accepts a per-dimension index. The transaction is all-or-nothing and "
            "creates an undoable revision."
        ),
        args_model=ApplyEditsArgs,
        handler=apply_edits,
        mutates=True,
    )
]
