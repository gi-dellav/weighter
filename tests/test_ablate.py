from __future__ import annotations

import pytest
import torch

from weighter.config import HarnessConfig
from weighter.errors import ToolError
from weighter.tools.ablate import AblateUnitsArgs, ablate_units
from weighter.tools.base import ProbeBudget, ToolContext
from weighter.weights import WeightStore


def _context(module: torch.nn.Module) -> tuple[WeightStore, ToolContext]:
    store = WeightStore(module)
    context = ToolContext(
        store=store,
        config=HarnessConfig(llm="fake"),
        probe=None,
        probe_budget=ProbeBudget(),
    )
    return store, context


def test_linear_output_ablation_and_undo() -> None:
    module = torch.nn.Linear(4, 3)
    with torch.no_grad():
        module.weight.fill_(1.0)
        module.bias.fill_(1.0)
    store, context = _context(module)
    weight = module.weight.detach().clone()
    bias = module.bias.detach().clone()
    result = ablate_units(context, AblateUnitsArgs(module_path="", units=[1]))
    assert result["unit_count"] == 1
    assert torch.all(module.weight[1] == 0)
    assert module.bias[1] == 0
    assert torch.equal(module.weight[0], weight[0])
    assert module.bias[0] == bias[0]
    assert result["diff"]["total_changed_elements"] == 5
    store.undo()
    assert torch.equal(module.weight, weight)
    assert torch.equal(module.bias, bias)


def test_linear_input_ablation() -> None:
    module = torch.nn.Linear(4, 3)
    _, context = _context(module)
    bias = module.bias.detach().clone()
    ablate_units(context, AblateUnitsArgs(module_path="", units=[2], axis="input"))
    assert torch.all(module.weight[:, 2] == 0)
    assert torch.equal(module.bias, bias)


def test_linear_scale_mode_without_bias() -> None:
    module = torch.nn.Linear(3, 2)
    _, context = _context(module)
    original = module.weight.detach().clone()
    ablate_units(
        context,
        AblateUnitsArgs(
            module_path="",
            units=[0],
            mode="scale",
            factor=0.5,
            include_bias=False,
        ),
    )
    assert torch.allclose(module.weight[0], original[0] * 0.5)
    assert torch.equal(module.weight[1], original[1])


def test_layer_norm_ablation() -> None:
    module = torch.nn.LayerNorm(5)
    _, context = _context(module)
    ablate_units(context, AblateUnitsArgs(module_path="", units=[1, 3]))
    assert module.weight[1] == 0 and module.weight[3] == 0
    assert module.bias[1] == 0 and module.bias[3] == 0
    assert module.weight[2] != 0


def test_conv_output_and_input_ablation() -> None:
    module = torch.nn.Conv2d(3, 4, 3)
    _, context = _context(module)
    ablate_units(context, AblateUnitsArgs(module_path="", units=[2]))
    assert torch.all(module.weight[2] == 0)
    assert module.bias[2] == 0

    module2 = torch.nn.Conv2d(3, 4, 3)
    _, context2 = _context(module2)
    bias = module2.bias.detach().clone()
    ablate_units(context2, AblateUnitsArgs(module_path="", units=[1], axis="input"))
    assert torch.all(module2.weight[:, 1] == 0)
    assert torch.equal(module2.bias, bias)


def test_grouped_conv_input_axis_is_rejected() -> None:
    module = torch.nn.Conv2d(4, 4, 3, groups=2)
    _, context = _context(module)
    with pytest.raises(ToolError):
        ablate_units(context, AblateUnitsArgs(module_path="", units=[1], axis="input"))


def test_units_are_normalized_and_validated() -> None:
    module = torch.nn.Linear(3, 4)
    _, context = _context(module)
    result = ablate_units(context, AblateUnitsArgs(module_path="", units=[-1, 0, 0]))
    assert result["units"] == [3, 0]
    with pytest.raises(ToolError):
        ablate_units(context, AblateUnitsArgs(module_path="", units=[4]))


def test_unsupported_module_has_actionable_error() -> None:
    module = torch.nn.Sequential(torch.nn.ReLU())
    _, context = _context(module)
    with pytest.raises(ToolError, match="use apply_edits"):
        ablate_units(context, AblateUnitsArgs(module_path="0", units=[0]))


def test_multi_dimensional_layer_norm_ablation() -> None:
    module = torch.nn.LayerNorm((2, 3))
    with torch.no_grad():
        module.weight.fill_(1.0)
        module.bias.fill_(1.0)
    _, context = _context(module)
    result = ablate_units(context, AblateUnitsArgs(module_path="", units=[0, 5]))
    assert result["units"] == [0, 5]
    assert module.weight[0, 0] == 0
    assert module.weight[1, 2] == 0
    assert module.bias[0, 0] == 0
    assert module.bias[1, 2] == 0
    assert module.weight[0, 1] == 1


@pytest.mark.parametrize(
    "module",
    [
        torch.nn.GroupNorm(2, 4),
        torch.nn.BatchNorm1d(4),
        torch.nn.BatchNorm2d(4),
        torch.nn.BatchNorm3d(4),
    ],
)
def test_norm_variants_are_supported(module: torch.nn.Module) -> None:
    with torch.no_grad():
        module.weight.fill_(1.0)
        module.bias.fill_(1.0)
    _, context = _context(module)
    ablate_units(context, AblateUnitsArgs(module_path="", units=[1]))
    assert module.weight[1] == 0
    assert module.bias[1] == 0


@pytest.mark.parametrize(
    "module",
    [
        torch.nn.Conv1d(2, 3, 2),
        torch.nn.Conv3d(2, 3, 2),
    ],
)
def test_conv_variants_output_and_input_ablation(module: torch.nn.Module) -> None:
    with torch.no_grad():
        module.weight.fill_(1.0)
        module.bias.fill_(1.0)
    _, context = _context(module)
    ablate_units(context, AblateUnitsArgs(module_path="", units=[1]))
    assert torch.all(module.weight[1] == 0)
    assert module.bias[1] == 0

    with torch.no_grad():
        module.weight.fill_(1.0)
        module.bias.fill_(1.0)
    _, context = _context(module)
    ablate_units(context, AblateUnitsArgs(module_path="", units=[0], axis="input"))
    assert torch.all(module.weight[:, 0] == 0)
    assert torch.all(module.bias == 1.0)


def test_grouped_conv_output_axis_is_supported() -> None:
    module = torch.nn.Conv2d(4, 6, 3, groups=2)
    with torch.no_grad():
        module.weight.fill_(1.0)
    _, context = _context(module)
    ablate_units(context, AblateUnitsArgs(module_path="", units=[5]))
    assert torch.all(module.weight[5] == 0)
    assert torch.all(module.weight[4] == 1.0)


def test_ablation_without_bias_and_scale_zero() -> None:
    module = torch.nn.Linear(3, 4, bias=False)
    with torch.no_grad():
        module.weight.fill_(2.0)
    _, context = _context(module)
    result = ablate_units(
        context,
        AblateUnitsArgs(module_path="", units=[0], include_bias=False),
    )
    assert torch.all(module.weight[0] == 0)
    assert len(result["edits"]) == 1

    with torch.no_grad():
        module.weight.fill_(2.0)
    _, context = _context(module)
    ablate_units(
        context,
        AblateUnitsArgs(
            module_path="", units=[0], mode="scale", factor=0.0, include_bias=False
        ),
    )
    assert torch.all(module.weight[0] == 0)


def test_norm_without_affine_weight_is_rejected() -> None:
    module = torch.nn.LayerNorm(3, elementwise_affine=False)
    _, context = _context(module)
    with pytest.raises(ToolError, match="no affine weight"):
        ablate_units(context, AblateUnitsArgs(module_path="", units=[0]))


def test_ablate_unknown_module_path_and_nested_module() -> None:
    module = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
    _, context = _context(module)
    with pytest.raises(ToolError, match="unknown module path"):
        ablate_units(context, AblateUnitsArgs(module_path="missing", units=[0]))

    result = ablate_units(context, AblateUnitsArgs(module_path="1", units=[0]))
    assert result["module"] == "1"
    assert result["module_type"] == "Linear"
    assert torch.all(module[1].weight[0] == 0)


def test_unsupported_module_error_includes_children() -> None:
    module = torch.nn.Sequential(torch.nn.ReLU())
    _, context = _context(module)
    with pytest.raises(ToolError, match="child modules: 0"):
        ablate_units(context, AblateUnitsArgs(module_path="", units=[0]))
