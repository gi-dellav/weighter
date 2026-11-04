"""Architecture-aware unit/neuron ablation."""

from __future__ import annotations

from typing import Any, Literal

import torch
from pydantic import Field

from ..errors import ToolError
from ..weights import summarize_diff
from .base import Tool, ToolArgs, ToolContext

_CONV_TYPES = (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)
_NORM_TYPES = (
    torch.nn.LayerNorm,
    torch.nn.GroupNorm,
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
)
_RMS_NORM = getattr(torch.nn, "RMSNorm", None)


class AblateUnitsArgs(ToolArgs):
    module_path: str = Field(
        default="",
        description="Path to the module; use '' for the root module, e.g. 'layers.3.mlp'",
    )
    units: list[int] = Field(
        min_length=1,
        max_length=4_096,
        description="Unit/channel/feature indices to ablate",
    )
    mode: Literal["zero", "scale"] = Field(
        default="zero", description="zero sets units to 0; scale multiplies them by factor"
    )
    factor: float = Field(default=1.0, description="Factor for mode='scale'")
    include_bias: bool = Field(
        default=True, description="Also ablate the matching bias entries when present"
    )
    axis: Literal["output", "input"] = Field(
        default="output",
        description="For Linear/Conv: ablate output units or input units",
    )
    note: str | None = Field(default=None, description="Short rationale for the revision")


def _param_name(module_path: str, name: str) -> str:
    return f"{module_path}.{name}" if module_path else name


def _normalize_units(units: list[int], size: int) -> list[int]:
    normalized: list[int] = []
    for unit in units:
        value = int(unit)
        if value < 0:
            value += size
        if value < 0 or value >= size:
            raise ToolError(f"unit {unit} is out of range for size {size}")
        if value not in normalized:
            normalized.append(value)
    return normalized


def _flat_index(shape: torch.Size, units: list[int]) -> list[Any]:
    if len(shape) == 1:
        return [units]
    coordinates = torch.unravel_index(torch.tensor(units), shape)
    return [coordinates[dim].tolist() for dim in range(len(shape))]


def _ablation_op(name: str, index: list[Any], args: AblateUnitsArgs) -> dict[str, Any]:
    if args.mode == "zero":
        return {"op": "fill", "name": name, "index": index, "value": 0.0}
    return {"op": "scale", "name": name, "index": index, "factor": args.factor}


def _linear_edits(
    module: torch.nn.Linear,
    module_path: str,
    args: AblateUnitsArgs,
) -> tuple[list[dict[str, Any]], list[int], int]:
    weight_name = _param_name(module_path, "weight")
    if args.axis == "input":
        size = int(module.in_features)
        units = _normalize_units(args.units, size)
        return [_ablation_op(weight_name, [None, units], args)], units, size

    size = int(module.out_features)
    units = _normalize_units(args.units, size)
    edits = [_ablation_op(weight_name, [units], args)]
    if args.include_bias and module.bias is not None:
        edits.append(_ablation_op(_param_name(module_path, "bias"), [units], args))
    return edits, units, size


def _conv_edits(
    module: torch.nn.Module,
    module_path: str,
    args: AblateUnitsArgs,
) -> tuple[list[dict[str, Any]], list[int], int]:
    weight = getattr(module, "weight", None)
    if weight is None:
        raise ToolError(f"{module_path!r} has no weight tensor")
    weight_name = _param_name(module_path, "weight")
    if args.axis == "input":
        groups = int(getattr(module, "groups", 1))
        if groups != 1:
            raise ToolError("input-axis ablation of grouped convolutions is not supported")
        size = int(weight.shape[1])
        units = _normalize_units(args.units, size)
        return [_ablation_op(weight_name, [None, units], args)], units, size

    size = int(weight.shape[0])
    units = _normalize_units(args.units, size)
    edits = [_ablation_op(weight_name, [units], args)]
    bias = getattr(module, "bias", None)
    if args.include_bias and bias is not None:
        edits.append(_ablation_op(_param_name(module_path, "bias"), [units], args))
    return edits, units, size


def _norm_edits(
    module: torch.nn.Module,
    module_path: str,
    args: AblateUnitsArgs,
) -> tuple[list[dict[str, Any]], list[int], int]:
    weight = getattr(module, "weight", None)
    if weight is None:
        raise ToolError(
            f"{module_path!r} has no affine weight; enable elementwise affine parameters first"
        )
    size = int(weight.numel())
    units = _normalize_units(args.units, size)
    index = _flat_index(weight.shape, units)
    edits = [_ablation_op(_param_name(module_path, "weight"), index, args)]
    bias = getattr(module, "bias", None)
    if args.include_bias and bias is not None:
        if int(bias.numel()) != size:
            raise ToolError("bias shape does not match the normalized feature count")
        edits.append(_ablation_op(_param_name(module_path, "bias"), index, args))
    return edits, units, size


def _build_edits(
    module: torch.nn.Module,
    module_path: str,
    args: AblateUnitsArgs,
) -> tuple[list[dict[str, Any]], list[int], int, str]:
    if isinstance(module, torch.nn.Linear):
        edits, units, size = _linear_edits(module, module_path, args)
    elif isinstance(module, _CONV_TYPES):
        edits, units, size = _conv_edits(module, module_path, args)
    elif isinstance(module, _NORM_TYPES) or (_RMS_NORM is not None and isinstance(module, _RMS_NORM)):
        edits, units, size = _norm_edits(module, module_path, args)
    else:
        direct = [name for name, _ in module.named_parameters(recurse=False)]
        children = [name for name, _ in module.named_children()]
        hint_parts = []
        if direct:
            hint_parts.append(f"direct parameters: {', '.join(direct)}")
        if children:
            hint_parts.append(f"child modules: {', '.join(children[:20])}")
        hint = f" ({'; '.join(hint_parts)})" if hint_parts else ""
        raise ToolError(
            f"ablate_units does not support {type(module).__name__}{hint}; "
            "use apply_edits with explicit tensor indices instead"
        )
    return edits, units, size, type(module).__name__


def ablate_units(ctx: ToolContext, args: AblateUnitsArgs) -> dict[str, Any]:
    module = ctx.store.module
    try:
        target = module.get_submodule(args.module_path)
    except AttributeError as exc:
        raise ToolError(f"unknown module path {args.module_path!r}") from exc
    edits, units, size, module_type = _build_edits(target, args.module_path, args)
    note = args.note or (
        f"Ablate {module_type} {args.module_path} units {units[:20]} "
        f"(mode={args.mode}, axis={args.axis})"
    )
    revision = ctx.store.apply(
        edits,
        note=note,
        max_edit_elements=ctx.config.max_edit_elements,
        confirm=ctx.confirm,
    )
    result: dict[str, Any] = {
        "module": args.module_path,
        "module_type": module_type,
        "units": units,
        "unit_count": len(units),
        "unit_size": size,
        "mode": args.mode,
        "axis": args.axis,
        "edits": edits,
        "revision": revision.info(),
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
        name="ablate_units",
        description=(
            "Ablate neurons/channels/features in a known layer type. Supports Linear, "
            "Conv1d/2d/3d (output units, or input units for ungrouped convs), LayerNorm, "
            "RMSNorm, GroupNorm, and BatchNorm. Modes: zero or scale. Produces one atomic, "
            "undoable revision and reports the resulting diff. For unsupported modules, use "
            "apply_edits with explicit indices."
        ),
        args_model=AblateUnitsArgs,
        handler=ablate_units,
        mutates=True,
    )
]
