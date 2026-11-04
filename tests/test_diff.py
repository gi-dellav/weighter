from __future__ import annotations

import pytest
import torch

from weighter.config import HarnessConfig
from weighter.errors import ToolError
from weighter.tools.base import ProbeBudget, ToolContext
from weighter.tools.history import WeightDiffArgs, weight_diff
from weighter.weights import WeightStore, summarize_diff


def _store() -> WeightStore:
    return WeightStore(torch.nn.Linear(2, 2))


def _context(store: WeightStore) -> ToolContext:
    return ToolContext(
        store=store,
        config=HarnessConfig(llm="fake"),
        probe=None,
        probe_budget=ProbeBudget(),
    )


def test_diff_since_revision() -> None:
    store = _store()
    r1 = store.apply(
        [{"op": "set", "name": "weight", "index": [0, 0], "values": 1.0}]
    )
    r2 = store.apply(
        [{"op": "set", "name": "bias", "index": [0], "values": 2.0}]
    )

    initial = store.diff()
    assert {item["name"] for item in initial} == {"weight", "bias"}

    since_r1 = store.diff(since=r1.id)
    assert [item["name"] for item in since_r1] == ["bias"]
    assert since_r1[0]["changed_elements"] == 1

    since_last = store.diff(since="last")
    assert [item["name"] for item in since_last] == ["bias"]

    assert store.diff(since=r2.id) == []
    with pytest.raises(ToolError):
        store.diff(since="missing")


def test_diff_after_undo() -> None:
    store = _store()
    store.apply([{"op": "fill", "name": "weight", "value": 0.5}])
    store.apply([{"op": "fill", "name": "bias", "value": 0.5}])
    store.undo()
    initial = store.diff()
    assert [item["name"] for item in initial] == ["weight"]


def test_summarize_diff_and_global_fraction() -> None:
    store = _store()
    store.apply([{"op": "fill", "name": "weight", "value": 0.0}])
    diffs = store.diff()
    summary = summarize_diff(diffs, module_elements=store.total_elements())
    assert summary["changed_tensors"] == 1
    assert summary["total_changed_elements"] == 4
    assert summary["module_elements"] == 6
    assert summary["changed_fraction_of_module"] == pytest.approx(4 / 6)
    assert summary["top_changed"][0]["name"] == "weight"


def test_weight_diff_tool() -> None:
    store = _store()
    store.apply([{"op": "set", "name": "weight", "index": [0, 0], "values": 9.0}])
    ctx = _context(store)
    result = weight_diff(ctx, WeightDiffArgs(since="initial"))
    assert result["count"] == 1
    assert result["summary"]["total_changed_elements"] == 1
    assert result["tensors"][0]["name"] == "weight"

    summary_only = weight_diff(ctx, WeightDiffArgs(since="initial", summary_only=True))
    assert "tensors" not in summary_only


def test_weight_diff_prefix() -> None:
    module = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    store = WeightStore(module)
    store.apply([{"op": "fill", "name": "0.weight", "value": 0.0}])
    store.apply([{"op": "fill", "name": "1.weight", "value": 0.0}])
    result = weight_diff(_context(store), WeightDiffArgs(tensor_prefix="0."))
    assert [item["name"] for item in result["tensors"]] == ["0.weight"]
