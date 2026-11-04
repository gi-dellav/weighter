"""Revision history and rollback tools."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from .base import Tool, ToolArgs, ToolContext


class RevisionHistoryArgs(ToolArgs):
    include_diff: bool = Field(default=False, description="Include current-vs-initial diff")
    limit: int = Field(default=50, ge=1, le=500)


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
            result["diff"] = ctx.store.diff()
        except Exception as exc:
            result["diff_error"] = str(exc)
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
        name="undo_edits",
        description="Roll back the most recent revision(s), restoring exact prior values.",
        args_model=UndoEditsArgs,
        handler=undo_edits,
        mutates=True,
    ),
]
