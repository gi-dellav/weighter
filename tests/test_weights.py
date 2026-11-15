from __future__ import annotations

import pytest
import torch

from weighter.errors import EditError, ToolError
from weighter.weights import WeightStore, parse_index, summarize_diff


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


def test_parse_index_validation_and_variants() -> None:
    assert parse_index(None, 2) == (slice(None), slice(None))
    assert parse_index([], 2) == (slice(None), slice(None))
    assert parse_index("1", 2) == (1, slice(None))
    assert parse_index("0:2", 2) == (slice(0, 2, None), slice(None))
    assert parse_index("::-1", 1) == (slice(None, None, -1),)
    assert parse_index((0, [1, 2]), 2) == (0, [1, 2])

    invalid = [
        [0, 0, 0],
        [1.5],
        [[1.5]],
        [[True]],
        [[]],
        "abc",
        "0:2:0",
        object(),
    ]
    for index in invalid:
        with pytest.raises(EditError):
            parse_index(index, 2)  # type: ignore[arg-type]


def test_parse_index_slice_step_zero_is_edit_error(module: TiedModule) -> None:
    store = WeightStore(module)
    with pytest.raises(EditError, match="step cannot be zero"):
        store.selection("first.weight", [None, "0:2:0"])
    with pytest.raises(EditError, match="step cannot be zero"):
        store.apply(
            [
                {
                    "op": "fill",
                    "name": "first.weight",
                    "value": 0.0,
                    "index": [None, "::0"],
                }
            ]
        )


def test_all_edit_operations() -> None:
    module = torch.nn.Linear(2, 2)
    with torch.no_grad():
        module.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        module.bias.copy_(torch.tensor([5.0, 6.0]))
    store = WeightStore(module)
    store.apply(
        [
            {"op": "set", "name": "weight", "values": [[0.0, 0.0], [0.0, 0.0]]},
            {
                "op": "add",
                "name": "weight",
                "index": [0],
                "values": [1.0, 1.0],
                "scale": 2.0,
            },
            {
                "op": "scale",
                "name": "weight",
                "index": [1],
                "factor": 0.5,
            },
            {"op": "fill", "name": "bias", "value": 7.0},
            {
                "op": "copy",
                "name": "bias",
                "index": [1],
                "src": "weight",
                "src_index": [0, 1],
            },
            {
                "op": "clamp",
                "name": "weight",
                "min": 1.0,
                "max": 1.5,
            },
        ]
    )
    assert torch.equal(module.weight, torch.tensor([[1.5, 1.5], [1.0, 1.0]]))
    assert torch.equal(module.bias, torch.tensor([7.0, 2.0]))


def test_random_edits_are_seeded_and_bounded() -> None:
    module = torch.nn.Linear(2, 2)
    store = WeightStore(module)
    store.apply(
        [
            {
                "op": "random",
                "name": "weight",
                "seed": 123,
                "distribution": "normal",
                "std": 1.0,
            }
        ]
    )
    first = module.weight.detach().clone()
    store.undo()
    store.apply(
        [
            {
                "op": "random",
                "name": "weight",
                "seed": 123,
                "distribution": "normal",
                "std": 1.0,
            }
        ]
    )
    assert torch.equal(module.weight, first)

    store.undo()
    store.apply(
        [
            {
                "op": "random",
                "name": "weight",
                "seed": 1,
                "distribution": "uniform",
                "low": -0.5,
                "high": 0.5,
            }
        ]
    )
    assert float(module.weight.detach().min()) >= -0.5
    assert float(module.weight.detach().max()) <= 0.5

    with pytest.raises(EditError, match="high must be >= low"):
        store.apply(
            [
                {
                    "op": "random",
                    "name": "weight",
                    "distribution": "uniform",
                    "low": 1.0,
                    "high": 0.0,
                }
            ]
        )
    with pytest.raises(EditError, match="normal' or 'uniform"):
        store.apply(
            [{"op": "random", "name": "weight", "distribution": "poisson"}]
        )


def test_project_out_axis_zero_and_validation() -> None:
    module = torch.nn.Linear(2, 3)
    store = WeightStore(module)
    with torch.no_grad():
        module.weight.fill_(1.0)
    direction = torch.tensor([1.0, 1.0, 1.0])
    store.apply(
        [
            {
                "op": "project_out",
                "name": "weight",
                "axis": 0,
                "direction": direction.tolist(),
            }
        ]
    )
    residual = (module.weight * direction.reshape(3, 1)).sum(dim=0)
    assert torch.allclose(residual, torch.zeros(2), atol=1e-6)

    with pytest.raises(EditError, match="1D vector"):
        store.apply(
            [
                {
                    "op": "project_out",
                    "name": "weight",
                    "direction": [[1.0], [1.0], [1.0]],
                }
            ]
        )
    with pytest.raises(EditError, match="expected 3"):
        store.apply(
            [
                {
                    "op": "project_out",
                    "name": "weight",
                    "axis": 0,
                    "direction": [1.0, 1.0],
                }
            ]
        )
    with pytest.raises(EditError, match="out of range"):
        store.apply(
            [
                {
                    "op": "project_out",
                    "name": "weight",
                    "axis": 5,
                    "direction": [1.0, 1.0],
                }
            ]
        )
    with pytest.raises(EditError, match="zero vector"):
        store.apply(
            [
                {
                    "op": "project_out",
                    "name": "weight",
                    "direction": [0.0, 0.0],
                }
            ]
        )


def test_edit_validation_errors(module: TiedModule) -> None:
    store = WeightStore(module)
    cases = [
        ([{"op": "unknown", "name": "first.weight"}], "unknown edit op"),
        ([{"op": "fill", "value": 0.0}], "needs a tensor 'name'"),
        ([{"op": "set", "name": "first.weight"}], "requires 'values'"),
        ([{"op": "scale", "name": "first.weight"}], "requires 'factor'"),
        ([{"op": "copy", "name": "first.weight"}], "requires 'src'"),
        ([{"op": "project_out", "name": "first.weight"}], "requires 'direction'"),
        ([{"op": "clamp", "name": "first.weight"}], "requires at least one"),
        ("not a dict", "each edit must be an object"),
    ]
    for ops, match in cases:
        with pytest.raises(EditError, match=match):
            store.propose(ops)  # type: ignore[arg-type]


def test_apply_confirm_and_limits() -> None:
    module = torch.nn.Linear(2, 2)
    store = WeightStore(module)
    with pytest.raises(EditError, match="no edits"):
        store.apply([])

    rejected: list = []
    with pytest.raises(EditError, match="rejected"):
        store.apply(
            [{"op": "fill", "name": "weight", "value": 0.0}],
            confirm=lambda proposal: rejected.append(proposal) or False,
        )
    assert rejected[0].targets == ["weight"]

    revision = store.apply(
        [{"op": "fill", "name": "weight", "value": 0.0}],
        confirm=lambda proposal: True,
    )
    assert revision.id == "r1"

    limited = WeightStore(torch.nn.Linear(2, 2), max_undo_bytes=4)
    with pytest.raises(EditError, match="max_undo_bytes"):
        limited.apply([{"op": "fill", "name": "weight", "value": 0.0}])
    limited.apply([{"op": "set", "name": "weight", "index": [0, 0], "values": 1.0}])
    with pytest.raises(EditError, match="max_undo_bytes"):
        limited.apply(
            [{"op": "set", "name": "weight", "index": [0, 1], "values": 1.0}]
        )

    no_undo = WeightStore(torch.nn.Linear(2, 2), undo=False, max_undo_bytes=4)
    with pytest.raises(EditError, match="atomic rollback"):
        no_undo.apply([{"op": "fill", "name": "weight", "value": 0.0}])


def test_stats_edge_cases() -> None:
    module = torch.nn.Module()
    module.register_parameter(
        "values",
        torch.nn.Parameter(torch.tensor([float("nan"), float("inf"), -float("inf"), 1.0, 2.0])),
    )
    store = WeightStore(module)
    stats = store.stats("values")
    assert stats["count"] == 5
    assert stats["nan_count"] == 1
    assert stats["inf_count"] == 2
    assert stats["mean"] == pytest.approx(1.5)
    assert stats["min"] == pytest.approx(1.0)
    assert stats["max"] == pytest.approx(2.0)

    empty = store.stats("values", index=["0:0"])
    assert empty["count"] == 0
    assert empty["mean"] is None
    assert empty["norm"] == 0.0
    assert empty["zero_fraction"] is None

    sampled = store.stats("values", sample_limit=2)
    assert sampled["sampled"] is True
    assert sampled["count"] == 5

    module.register_parameter(
        "complex", torch.nn.Parameter(torch.tensor([1 + 2j, 3 + 4j]))
    )
    complex_stats = WeightStore(module).stats("complex")
    assert complex_stats["count"] == 2
    assert complex_stats["max"] == pytest.approx(5.0)


def test_read_values_non_finite_and_complex() -> None:
    module = torch.nn.Module()
    module.register_parameter(
        "values",
        torch.nn.Parameter(
            torch.tensor([float("nan"), float("inf"), -float("inf"), 2.0])
        ),
    )
    store = WeightStore(module)
    read = store.read_values("values")
    assert read["values"] == ["nan", "inf", "-inf", 2.0]

    module.register_parameter(
        "complex", torch.nn.Parameter(torch.tensor([1 + 2j, 3 + 4j]))
    )
    store = WeightStore(module)
    complex_read = store.read_values("complex")
    assert all(isinstance(value, str) for value in complex_read["values"])


def test_list_refs_filters_and_overview() -> None:
    module = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
    store = WeightStore(module, include_buffers=False)
    assert [ref.name for ref in store.list_refs(prefix="0.")] == [
        "0.weight",
        "0.bias",
    ]
    assert [ref.name for ref in store.list_refs(kind="buffer")] == []

    filtered = WeightStore(module, tensor_filter=lambda name: name.endswith("bias"))
    assert set(filtered.names()) == {"0.bias", "1.bias"}
    assert filtered.resolve("0.bias").kind == "parameter"

    overview = store.overview(max_children=1)
    assert overview["module_type"] == "Sequential"
    assert overview["parameter_tensors"] == 4
    assert len(overview["children"]) == 1
    assert overview["dtypes"] == ["float32"]
    assert overview["devices"] == ["cpu"]
    assert overview["revisions"] == 0


def test_dirty_tensors_are_deduplicated(module: TiedModule) -> None:
    store = WeightStore(module)
    store.apply([{"op": "fill", "name": "first.weight", "value": 0.0}])
    store.apply([{"op": "fill", "name": "first.weight", "value": 1.0}])
    store.apply([{"op": "fill", "name": "first.bias", "value": 1.0}])
    assert store.dirty_tensors() == ["first.weight", "first.bias"]
    assert store.total_elements() == 12


def test_undo_multiple_and_revert_all(module: TiedModule) -> None:
    store = WeightStore(module)
    original = module.first.weight.detach().clone()
    store.apply([{"op": "fill", "name": "first.weight", "value": 0.0}])
    store.apply([{"op": "fill", "name": "first.weight", "value": 1.0}])
    undone = store.undo(5)
    assert [revision.id for revision in undone] == ["r2", "r1"]
    assert torch.equal(module.first.weight, original)
    with pytest.raises(ToolError, match="no revisions"):
        store.undo()
    with pytest.raises(ToolError, match="n must be >= 1"):
        WeightStore(torch.nn.Linear(2, 2)).undo(0)
    assert WeightStore(torch.nn.Linear(2, 2)).revert_all() == []


def test_undo_disabled_revision_bytes_are_zero() -> None:
    store = WeightStore(torch.nn.Linear(2, 2), undo=False)
    revision = store.apply(
        [{"op": "set", "name": "weight", "index": [0, 0], "values": 1.0}],
        max_edit_elements=10,
    )
    assert revision.bytes == 0
    assert store.revision_infos()[0]["bytes"] == 0


def test_diff_baselines_prefix_and_unknown() -> None:
    store = WeightStore(torch.nn.Sequential(torch.nn.Linear(2, 2)))
    store.apply(
        [{"op": "set", "name": "0.weight", "index": [0, 0], "values": 9.0}]
    )
    store.apply(
        [{"op": "set", "name": "0.bias", "index": [0], "values": 9.0}]
    )
    assert {item["name"] for item in store.diff(since="")} == {"0.weight", "0.bias"}
    assert {item["name"] for item in store.diff(since="all")} == {
        "0.weight",
        "0.bias",
    }
    assert [item["name"] for item in store.diff(since="last")] == ["0.bias"]
    assert store.diff(tensor_prefix="missing") == []
    with pytest.raises(ToolError, match="unknown revision"):
        store.diff(since="r99")


def test_diff_unchanged_values_are_zero() -> None:
    store = WeightStore(torch.nn.Linear(2, 2))
    value = float(store.tensor("weight")[0, 0].detach())
    store.apply(
        [{"op": "set", "name": "weight", "index": [0, 0], "values": value}]
    )
    diff = store.diff()
    assert diff[0]["changed_elements"] == 0
    assert diff[0]["delta_norm"] == 0.0


def test_summarize_diff_edges_and_top_n() -> None:
    diffs = [
        {
            "name": "a",
            "changed_elements": 1,
            "total_elements": 2,
            "delta_norm": 1.0,
            "original_norm": 2.0,
            "relative_delta": 0.5,
            "max_abs_delta": 1.0,
        },
        {
            "name": "b",
            "changed_elements": 3,
            "total_elements": 4,
            "delta_norm": "bad",
            "original_norm": "nan",
            "relative_delta": 0.0,
            "max_abs_delta": "nan",
        },
    ]
    summary = summarize_diff(diffs, module_elements=0, top_n=1)
    assert summary["changed_tensors"] == 2
    assert summary["total_changed_elements"] == 4
    assert summary["dirty_tensor_elements"] == 6
    assert summary["global_delta_norm"] == 1.0
    assert summary["max_abs_delta"] == 1.0
    assert summary["changed_fraction_of_module"] == 0.0
    assert [item["name"] for item in summary["top_changed"]] == ["b"]

    no_module = summarize_diff([])
    assert no_module["changed_tensors"] == 0
    assert "module_elements" not in no_module


def test_more_index_component_errors() -> None:
    store = WeightStore(torch.nn.Linear(2, 2))
    cases = [
        ([True], "boolean indices"),
        (["1:2:3:4"], "invalid slice string"),
        (["a:b"], "invalid slice string"),
        ([["1", 2]], "index lists must contain integers"),
        ("not-an-index", "invalid index string"),
    ]
    for index, match in cases:
        with pytest.raises(EditError, match=match):
            store.selection("weight", index)  # type: ignore[arg-type]


def test_set_values_validation_errors() -> None:
    store = WeightStore(torch.nn.Linear(2, 2))
    with pytest.raises(EditError, match="could not convert"):
        store.apply(
            [{"op": "set", "name": "weight", "values": object()}]
        )
    with pytest.raises(EditError, match="complex values"):
        store.apply(
            [{"op": "set", "name": "weight", "values": [1 + 2j]}]
        )
    with pytest.raises(EditError, match="does not broadcast"):
        store.apply(
            [
                {
                    "op": "set",
                    "name": "weight",
                    "index": [0],
                    "values": [1.0, 2.0, 3.0],
                }
            ]
        )


def test_random_requires_floating_point() -> None:
    module = torch.nn.Module()
    module.register_parameter(
        "integers",
        torch.nn.Parameter(torch.tensor([1, 2], dtype=torch.int64), requires_grad=False),
    )
    store = WeightStore(module)
    with pytest.raises(EditError, match="floating-point"):
        store.apply(
            [{"op": "random", "name": "integers", "seed": 1}]
        )
