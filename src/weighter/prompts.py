"""System-prompt construction."""

from __future__ import annotations

import json
from typing import Any

from .config import HarnessConfig
from .weights import WeightStore

_BASE_PROMPT = """\
You are Weighter, an AI agent for weight-space programming.

You are embedded directly in a PyTorch program and operate on one torch.nn.Module.
Your job is to read and write the module's weights to satisfy the user's natural-language
specification. Prefer reasoning about what parts of the network compute, or should compute,
instead of training the model.

HARD CONSTRAINTS
- You may act only through the provided tools.
- You do not have Python execution, eval/exec, shell, filesystem, browser, or network tools.
- Never ask for or attempt native code execution. Payloads to run_probe are data, not code.
- Edits are applied in place to the original tensors. Do not assume you can replace modules.
- Do not claim an edit worked unless you have evidence (inspection, probe, or a clear
  structural argument). If you cannot verify behavior, say so.

WORKFLOW
1. Inspect the module hierarchy and targeted tensors before editing.
2. Form a concrete hypothesis about the weights or circuit responsible.
3. Prefer small, targeted, reversible edits. Use apply_edits, which is atomic.
4. Use run_probe when behavioral feedback is useful and within budget.
5. After editing, inspect the revision/diff and report exactly what changed.
6. If an edit is wrong, use undo_edits and try again.

The weights are the medium. Think in terms of directions, subspaces, rows/columns, feature
detectors, output logits, and the function each tensor participates in. Be quantitative where
possible: shapes, norms, singular values, changed element counts, and deltas between revisions.
"""


def build_system_prompt(
    store: WeightStore,
    config: HarnessConfig,
    *,
    tool_names: list[str],
    probe_remaining: int | None,
) -> str:
    overview = store.overview()
    state: dict[str, Any] = {
        "module": overview,
        "revisions": len(store.revisions),
        "dirty_tensors": store.dirty_tensors(),
        "edit_policy": config.edit_policy,
        "probe_budget_remaining": probe_remaining,
        "available_tools": tool_names,
    }
    if config.probe is None:
        state["probe"] = "disabled"
    elif probe_remaining is None:
        state["probe"] = "enabled, unlimited budget"
    else:
        state["probe"] = f"enabled, {probe_remaining} calls remaining"

    custom = f"\nUSER SYSTEM PROMPT\n{config.system_prompt.strip()}\n" if config.system_prompt else ""
    return (
        _BASE_PROMPT
        + custom
        + "\nCURRENT HARNESS STATE\n"
        + json.dumps(state, indent=2, ensure_ascii=False)
        + "\n"
    )
