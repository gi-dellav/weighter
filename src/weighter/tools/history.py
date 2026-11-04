"""Revision history and rollback tools."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from ..errors import ToolError
from ..weights import summarize_diff
from .base import Tool, ToolArgs, ToolContext


class RevisionHistoryArgs(ToolArgs):
    include_diff: bool = Field(default=False, description="Include a current-vs-baseline diff")
    diff_since: str | None = Field(
        default=None,
        description="Diff baseline: 'initial', 'last', or a revision id",
    )
    limit: int = Field(default=50, ge=1, le=500)


class WeightDiffArgs(ToolArgs):
    since: str = Field(
        default="initial",
        description="Baseline: 'initial', 'last', or a revision id such as 'r3'",
    )
    tensor_prefix: str | None = Field(
        default=None, description="Only include tensors whose names start with this"
    )
    limit: int = Field(default=50, ge=1, le=500)
    summary_only: bool = False


class UndoEditsArgs(ToolArgs):
    n: int = Field(default=1, ge=1, le=100, description="Number of revisions to roll back")
    all_revisions: bool = Field(default=False, description="Undo all revisions")


def revision_history(ctx: ToolContext, args: RevisionHistoryArgs) -> dict[str, Any]:
    revisions = ctx.store.revision_infos()
    selected = revisions[-args.limit :]
    result: dict[str, Any] = {
        "count": len(revisions),
        "returned": len(selected),
        "revisions": selected,
        "dirty_tensors": ctx.store.dirty_tensors(),
    }
    if args.include_diff:
        try:
            diffs = ctx.store.diff(since=args.diff_since)
            result["diff"] = diffs
            result["diff_summary"] = summarize_diff(
                diffs, module_elements=ctx.store.total_elements()
            )
        except Exception as exc:
            result["diff_error"] = str(exc)
    return result


def weight_diff(ctx: ToolContext, args: WeightDiffArgs) -> dict[str, Any]:
    try:
        diffs = ctx.store.diff(since=args.since, tensor_prefix=args.tensor_prefix)
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"could not compute diff: {exc}") from exc
    summary = summarize_diff(diffs, module_elements=ctx.store.total_elements())
    result: dict[str, Any] = {
        "since": args.since,
        "count": len(diffs),
        "summary": summary,
    }
    if not args.summary_only:
        result["tensors"] = diffs[: args.limit]
        result["returned"] = len(result["tensors"])
        result["truncated"] = len(diffs) > args.limit
    return result


def undo_edits(ctx: ToolContext, args: UndoEditsArgs) -> dict[str, Any]:
    if not ctx.store.revisions:
        return {"undone": [], "remaining_revisions": 0, "dirty_tensors": []}
    n = len(ctx.store.revisions) if args.all_revisions else args.n
    undone = ctx.store.undo(n)
    return {
        "undone": [revision.info() for revision in undone],
        "remaining_revisions": len(ctx.store.revisions),
        "dirty_tensors": ctx.store.dirty_tensors(),
    }


TOOLS: list[Tool] = [
    Tool(
        name="revision_history",
        description="List recent weight revisions, dirty tensors, and optionally the current diff.",
        args_model=RevisionHistoryArgs,
        handler=revision_history,
    ),
    Tool(
        name="weight_diff",
        description=(
            "Compare current weights against a baseline. Use since='initial' for all edits, "
            "since='last' for the most recent revision, or a revision id such as 'r3'. "
            "Returns global change summary and per-tensor delta statistics."
        ),
        args_model=WeightDiffArgs,
        handler=weight_diff,
    ),
    Tool(
        name="undo_edits",
        description="Roll back the most recent revision(s), restoring exact prior values.",
        args_model=UndoEditsArgs,
        handler=undo_edits,
        mutates=True,
    ),
]
