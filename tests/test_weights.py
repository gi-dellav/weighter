from __future__ import annotations

import pytest
import torch

from weighter.errors import EditError, ToolError
from weighter.weights import WeightStore, parse_index


class TiedModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = torch.nn.Linear(3, 3, bias=True)
        self.second = torch.nn.Linear(3, 3, bias=False)
        self.second.weight = self.first.weight
        self.register_buffer("running", torch.zeros(2))


@pytest.fixture()
def module() -> TiedModule:
    return TiedModule()


def test_index_and_aliases(module: TiedModule) -> None:
    store = WeightStore(module, include_buffers=True)
    aliases = store.resolve("second.weight").names
    assert "first.weight" in aliases and "second.weight" in aliases
    assert store.canonical_name("second.weight") == "first.weight"
    assert "running" in store.names()
    assert parse_index([None, "0:2"], 2) == (slice(None), slice(0, 2, None))
    assert parse_index([0, [1, 2]], 2) == (0, [1, 2])
    with pytest.raises(ToolError):
        store.resolve("missing")


def test_read_and_stats(module: TiedModule) -> None:
    store = WeightStore(module)
    values = store.read_values("first.weight", index=[0, "0:2"], max_elements=10)
    assert values["shape"] == [2]
    assert values["returned"] == 2
    assert values["truncated"] is False
    stats = store.stats("first.weight", index=[0])
    assert stats["selection_shape"] == [3]
    assert stats["count"] == 3
    assert "percentiles" in stats
    limited = store.read_values("first.weight", max_elements=2)
    assert limited["truncated"] is True
    assert limited["returned"] == 2


def test_edit_undo_and_diff(module: TiedModule) -> None:
    store = WeightStore(module)
    original = module.first.weight.detach().clone()
    original_bias = module.first.bias.detach().clone()
    revision = store.apply(
        [
            {"op": "set", "name": "first.weight", "index": [0, 0], "values": 2.0},
            {"op": "add", "name": "first.bias", "index": [0], "values": [0.5]},
        ],
        note="adjust",
    )
    assert revision.id == "r1"
    assert module.first.weight[0, 0] == 2.0
    assert module.first.bias[0].item() == pytest.approx(original_bias[0].item() + 0.5)
    diff = store.diff()
    assert set(diff[0]) >= {"name", "changed_elements", "delta_norm"}
    assert store.revision_infos()[0]["note"] == "adjust"
    store.undo()
    assert torch.equal(module.first.weight, original)
    assert store.diff() == []


def test_tied_alias_edit_targets_canonical_tensor(module: TiedModule) -> None:
    store = WeightStore(module)
    before = module.second.weight.detach().clone()
    revision = store.apply(
        [{"op": "add", "name": "second.weight", "values": 1.0}]
    )
    assert revision.tensors == ["first.weight"]
    assert torch.allclose(module.second.weight, before + 1.0)
    assert torch.equal(module.first.weight, module.second.weight)


def test_atomic_rollback_on_bad_copy(module: TiedModule) -> None:
    store = WeightStore(module)
    original = module.first.weight.detach().clone()
    with pytest.raises(EditError):
        store.apply(
            [
                {"op": "set", "name": "first.weight", "index": [0, 0], "values": 9.0},
                {
                    "op": "copy",
                    "name": "first.bias",
                    "src": "first.weight",
                },
            ]
        )
    assert torch.equal(module.first.weight, original)


def test_project_out_removes_direction(module: TiedModule) -> None:
    store = WeightStore(module)
    direction = module.first.weight[0].detach().clone()
    store.apply(
        [
            {
                "op": "project_out",
                "name": "first.weight",
                "index": [0],
                "direction": direction.tolist(),
            }
        ]
    )
    dot = torch.dot(module.first.weight[0].detach(), direction)
    assert abs(float(dot)) < 1e-5


def test_undo_disabled_blocks_undo_and_diff(module: TiedModule) -> None:
    store = WeightStore(module, undo=False)
    store.apply([{"op": "fill", "name": "first.bias", "value": 0.0}])
    assert store.revision_infos()
    with pytest.raises(ToolError):
        store.undo()
    with pytest.raises(ToolError):
        store.diff()


def test_edit_limit(module: TiedModule) -> None:
    store = WeightStore(module)
    with pytest.raises(EditError):
        store.apply(
            [{"op": "fill", "name": "first.weight", "value": 0.0}],
            max_edit_elements=2,
        )


def test_optimizer_reference_survives_edit() -> None:
    module = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    weight_before = module.weight
    store = WeightStore(module)
    store.apply([{"op": "fill", "name": "weight", "value": 0.25}])
    assert module.weight is weight_before
    loss = module(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()
    assert module.weight is weight_before


def test_buffer_edit_when_enabled() -> None:
    module = torch.nn.BatchNorm1d(2)
    module.register_buffer("custom", torch.zeros(2))
    store = WeightStore(module, include_buffers=True)
    store.apply([{"op": "fill", "name": "custom", "value": 1.0}])
    assert torch.equal(module.custom, torch.ones(2))


def test_undo_all_from_tool() -> None:
    from weighter.tools.base import ProbeBudget, ToolContext
    from weighter.config import HarnessConfig
    from weighter.tools.history import UndoEditsArgs, undo_edits

    module = torch.nn.Linear(2, 1)
    store = WeightStore(module)
    config = HarnessConfig(llm="fake")
    ctx = ToolContext(store=store, config=config, probe=None, probe_budget=ProbeBudget())
    result = undo_edits(ctx, UndoEditsArgs(all_revisions=True))
    assert result["undone"] == []
