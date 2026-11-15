from __future__ import annotations

import pytest
import torch

from weighter.config import HarnessConfig
from weighter.errors import ToolError
from weighter.tools.base import ProbeBudget, ToolContext
from weighter.tools.history import (
    RevisionHistoryArgs,
    UndoEditsArgs,
    WeightDiffArgs,
    revision_history,
    undo_edits,
    weight_diff,
)
from weighter.tools.inspect import (
    DescribeModuleArgs,
    ListTensorsArgs,
    ReadTensorArgs,
    TensorStatsArgs,
    describe_module,
    list_tensors,
    read_tensor,
    tensor_stats,
)
from weighter.weights import WeightStore


class Nested(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.ReLU())
        self.register_buffer("counter", torch.zeros(2))


def _context(module: torch.nn.Module | None = None) -> tuple[WeightStore, ToolContext]:
    store = WeightStore(module or Nested(), include_buffers=True)
    context = ToolContext(
        store=store,
        config=HarnessConfig(llm="fake", max_read_elements=2),
        probe=None,
        probe_budget=ProbeBudget(),
    )
    return store, context


def test_describe_module_root_prefix_and_depth() -> None:
    _, context = _context()
    root = describe_module(context, DescribeModuleArgs())
    assert root["overview"]["module_type"] == "Nested"
    assert root["tree"]["name"] == "Nested"
    assert root["tree"]["children"][0]["name"] == "layers"

    shallow = describe_module(context, DescribeModuleArgs(max_depth=0))
    assert "children" not in shallow["tree"]

    prefixed = describe_module(context, DescribeModuleArgs(prefix="layers.0"))
    assert prefixed["tree"]["name"] == "layers.0"
    assert prefixed["tree"]["type"] == "Linear"

    with pytest.raises(ToolError, match="unknown module path"):
        describe_module(context, DescribeModuleArgs(prefix="missing"))


def test_list_tensors_prefix_kind_and_pagination() -> None:
    _, context = _context()
    all_tensors = list_tensors(context, ListTensorsArgs())
    assert all_tensors["total"] == 3
    assert all_tensors["truncated"] is False
    assert all_tensors["tensors"][0]["name"] == "layers.0.weight"
    assert "norm" in all_tensors["tensors"][0]

    prefixed = list_tensors(context, ListTensorsArgs(prefix="layers.0"))
    assert {item["name"] for item in prefixed["tensors"]} == {
        "layers.0.weight",
        "layers.0.bias",
    }

    page = list_tensors(
        context, ListTensorsArgs(offset=1, limit=2, include_stats=False)
    )
    assert page["offset"] == 1
    assert page["returned"] == 2
    assert page["truncated"] is False
    assert "norm" not in page["tensors"][0]

    buffers = list_tensors(context, ListTensorsArgs(kind="buffer"))
    assert [item["name"] for item in buffers["tensors"]] == ["counter"]


def test_read_tensor_and_stats_tools() -> None:
    _, context = _context()
    values = read_tensor(
        context,
        ReadTensorArgs(name="layers.0.weight", index=[0], max_elements=1),
    )
    assert values["returned"] == 1
    assert values["truncated"] is True

    capped = read_tensor(context, ReadTensorArgs(name="layers.0.weight"))
    assert capped["returned"] == 2

    stats = tensor_stats(
        context,
        TensorStatsArgs(name="layers.0.weight", index=[0], sample_limit=10),
    )
    assert stats["selection_shape"] == [2]
    assert stats["count"] == 2


def test_revision_history_empty_and_populated() -> None:
    store, context = _context()
    empty = revision_history(context, RevisionHistoryArgs())
    assert empty["count"] == 0
    assert empty["revisions"] == []

    store.apply(
        [{"op": "fill", "name": "layers.0.bias", "value": 0.0}], note="clear bias"
    )
    history = revision_history(context, RevisionHistoryArgs(include_diff=True))
    assert history["count"] == 1
    assert history["revisions"][0]["note"] == "clear bias"
    assert history["dirty_tensors"] == ["layers.0.bias"]
    assert history["diff_summary"]["total_changed_elements"] == 3

    bad_diff = revision_history(
        context, RevisionHistoryArgs(include_diff=True, diff_since="missing")
    )
    assert "diff_error" in bad_diff

    limited = revision_history(context, RevisionHistoryArgs(limit=1))
    assert limited["returned"] == 1


def test_weight_diff_limit_and_summary_only() -> None:
    store, context = _context()
    store.apply(
        [
            {"op": "fill", "name": "layers.0.weight", "value": 0.0},
            {"op": "fill", "name": "layers.0.bias", "value": 0.0},
        ]
    )
    full = weight_diff(context, WeightDiffArgs(limit=1))
    assert full["count"] == 2
    assert full["returned"] == 1
    assert full["truncated"] is True
    assert full["summary"]["total_changed_elements"] == 9

    summary = weight_diff(context, WeightDiffArgs(summary_only=True))
    assert "tensors" not in summary
    assert summary["summary"]["changed_tensors"] == 2


def test_undo_edits_noop_and_multiple() -> None:
    store, context = _context()
    empty = undo_edits(context, UndoEditsArgs())
    assert empty == {"undone": [], "remaining_revisions": 0, "dirty_tensors": []}

    store.apply([{"op": "fill", "name": "layers.0.bias", "value": 0.0}])
    store.apply([{"op": "fill", "name": "layers.0.weight", "value": 0.0}])
    one = undo_edits(context, UndoEditsArgs())
    assert [item["id"] for item in one["undone"]] == ["r2"]
    assert one["remaining_revisions"] == 1
    assert one["dirty_tensors"] == ["layers.0.bias"]

    rest = undo_edits(context, UndoEditsArgs(all_revisions=True))
    assert [item["id"] for item in rest["undone"]] == ["r1"]
    assert rest["remaining_revisions"] == 0
