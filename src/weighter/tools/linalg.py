"""Linear-algebra tools for reading structure in weight tensors."""

from __future__ import annotations

import math
import re
from typing import Any, Literal

from pydantic import Field

import torch

from ..errors import ToolError
from ..utils import safe_float, tensor_summary
from .base import Tool, ToolArgs, ToolContext


class EinsumOperand(ToolArgs):
    name: str = Field(description="Tensor name")
    index: list[Any] | None = None


class LinearAlgebraArgs(ToolArgs):
    op: Literal[
        "svd", "norm", "cosine", "topk", "matmul", "einsum", "reduce"
    ]
    name: str | None = Field(default=None, description="Tensor to analyze")
    index: list[Any] | None = None
    other: str | None = Field(
        default=None, description="Second tensor for cosine or matmul"
    )
    other_index: list[Any] | None = None
    k: int = Field(default=8, ge=1, le=256)
    dim: int | list[int] | None = Field(
        default=None, description="Axis or axes for reduce; null means all axes"
    )
    keepdim: bool = Field(default=False, description="Keep reduced axes")
    equation: str | None = Field(
        default=None, description='Einsum equation, e.g. "ij,jk->ik"'
    )
    operands: list[EinsumOperand] | None = Field(
        default=None, description="Ordered einsum operands with optional indices"
    )
    reduction: Literal[
        "sum", "mean", "max", "min", "prod", "std", "var", "l1", "l2"
    ] | None = Field(default=None, description="Reduction to apply along dim")


def _selection(ctx: ToolContext, name: str, index: list[Any] | None) -> torch.Tensor:
    selection = ctx.store.selection(name, index).detach()
    if selection.numel() > ctx.config.max_edit_elements:
        raise ToolError(
            f"selection has {selection.numel()} elements; narrow the index to at most "
            f"{ctx.config.max_edit_elements}"
        )
    return selection


def _require_name(args: LinearAlgebraArgs) -> str:
    if not args.name:
        raise ToolError(f"{args.op} requires 'name'")
    return args.name


def _meta_shape(
    fn: Any, tensors: list[torch.Tensor]
) -> tuple[int, ...] | None:
    try:
        meta = [
            torch.empty(tensor.shape, dtype=tensor.dtype, device="meta")
            for tensor in tensors
        ]
        return tuple(fn(*meta).shape)
    except Exception:
        return None


def _check_result_size(
    ctx: ToolContext, shape: tuple[int, ...] | None, operation: str
) -> None:
    if shape is None:
        return
    elements = math.prod(int(size) for size in shape)
    if elements > ctx.config.max_edit_elements:
        raise ToolError(
            f"{operation} result would have {elements} elements, above "
            f"max_edit_elements={ctx.config.max_edit_elements}; narrow the selection"
        )


def _result_summary(ctx: ToolContext, result: torch.Tensor) -> dict[str, Any]:
    if result.numel() > ctx.config.max_edit_elements:
        raise ToolError(
            f"result has {result.numel()} elements, above "
            f"max_edit_elements={ctx.config.max_edit_elements}; narrow the selection"
        )
    return tensor_summary(result.detach(), max_elements=ctx.config.max_read_elements)


def _normalize_dims(dim: int | list[int] | None, ndim: int) -> list[int] | None:
    if dim is None or ndim == 0:
        return None
    if isinstance(dim, int) and not isinstance(dim, bool):
        dims = [dim]
    elif isinstance(dim, (list, tuple)) and all(
        isinstance(item, int) and not isinstance(item, bool) for item in dim
    ):
        dims = list(dim)
    else:
        raise ToolError("reduce 'dim' must be an integer or a list of integers")
    if not dims:
        return None
    normalized: list[int] = []
    for item in dims:
        if item < -ndim or item >= ndim:
            raise ToolError(
                f"reduction axis {item} is out of range for a tensor with {ndim} dimensions"
            )
        axis = item % ndim
        if axis in normalized:
            raise ToolError("reduce 'dim' contains duplicate axes")
        normalized.append(axis)
    return normalized


def _reduce_tensor(
    tensor: torch.Tensor,
    reduction: str,
    dims: list[int] | None,
    keepdim: bool,
) -> torch.Tensor:
    axis: int | tuple[int, ...] | None
    if dims is None:
        axis = None
    elif len(dims) == 1:
        axis = dims[0]
    else:
        axis = tuple(dims)
    work = tensor if tensor.is_floating_point() else tensor.float()
    try:
        if reduction == "sum":
            return torch.sum(tensor, dim=axis, keepdim=keepdim)
        if reduction == "mean":
            return torch.mean(work, dim=axis, keepdim=keepdim)
        if reduction == "max":
            all_axes = tuple(range(tensor.ndim))
            return torch.amax(
                tensor, dim=axis if axis is not None else all_axes, keepdim=keepdim
            )
        if reduction == "min":
            all_axes = tuple(range(tensor.ndim))
            return torch.amin(
                tensor, dim=axis if axis is not None else all_axes, keepdim=keepdim
            )
        if reduction == "prod":
            if dims is None:
                result = torch.prod(tensor)
                if keepdim and tensor.ndim:
                    result = result.reshape([1] * tensor.ndim)
                return result
            result = tensor
            for item in sorted(dims):
                result = torch.prod(result, dim=item, keepdim=True)
            if not keepdim:
                for item in sorted(dims, reverse=True):
                    result = result.squeeze(item)
            return result
        if reduction == "std":
            return torch.std(work, dim=axis, keepdim=keepdim, unbiased=False)
        if reduction == "var":
            return torch.var(work, dim=axis, keepdim=keepdim, unbiased=False)
        if reduction == "l1":
            return torch.sum(tensor.abs(), dim=axis, keepdim=keepdim)
        if reduction == "l2":
            return torch.linalg.vector_norm(work, ord=2, dim=axis, keepdim=keepdim)
    except (IndexError, RuntimeError) as exc:
        raise ToolError(f"reduction {reduction!r} failed: {exc}") from exc
    raise ToolError(f"unknown reduction {reduction!r}")


def _matmul(ctx: ToolContext, args: LinearAlgebraArgs) -> dict[str, Any]:
    name = _require_name(args)
    if not args.other:
        raise ToolError("matmul requires 'other'")
    first = _selection(ctx, name, args.index)
    second = _selection(ctx, args.other, args.other_index)
    _check_result_size(
        ctx, _meta_shape(torch.matmul, [first, second]), "matmul"
    )
    try:
        result = torch.matmul(first, second)
    except (IndexError, RuntimeError) as exc:
        raise ToolError(f"matmul failed: {exc}") from exc
    return {
        "op": "matmul",
        "name": name,
        "other": args.other,
        "operand_shapes": [list(first.shape), list(second.shape)],
        "result_shape": list(result.shape),
        "result": _result_summary(ctx, result),
    }


def _einsum(ctx: ToolContext, args: LinearAlgebraArgs) -> dict[str, Any]:
    if not args.equation:
        raise ToolError("einsum requires 'equation'")
    equation = args.equation.strip()
    if not equation or len(equation) > 256:
        raise ToolError("einsum 'equation' must be between 1 and 256 characters")
    if not re.fullmatch(r"[A-Za-z0-9,.\->\s]+", equation):
        raise ToolError("einsum 'equation' contains unsupported characters")
    if not args.operands:
        raise ToolError("einsum requires a non-empty 'operands' list")
    if len(args.operands) > 32:
        raise ToolError("einsum supports at most 32 operands")
    tensors = [
        _selection(ctx, operand.name, operand.index) for operand in args.operands
    ]
    _check_result_size(
        ctx,
        _meta_shape(lambda *items: torch.einsum(equation, *items), tensors),
        "einsum",
    )
    try:
        result = torch.einsum(equation, *tensors)
    except (IndexError, RuntimeError, ValueError) as exc:
        raise ToolError(f"einsum failed: {exc}") from exc
    return {
        "op": "einsum",
        "equation": equation,
        "operands": [
            {"name": operand.name, "shape": list(tensor.shape)}
            for operand, tensor in zip(args.operands, tensors)
        ],
        "result_shape": list(result.shape),
        "result": _result_summary(ctx, result),
    }


def _reduce(
    ctx: ToolContext,
    args: LinearAlgebraArgs,
    name: str,
    selection: torch.Tensor,
) -> dict[str, Any]:
    if not args.reduction:
        raise ToolError("reduce requires 'reduction'")
    dims = _normalize_dims(args.dim, selection.ndim)
    result = _reduce_tensor(selection, args.reduction, dims, args.keepdim)
    payload: dict[str, Any] = {
        "op": "reduce",
        "name": name,
        "reduction": args.reduction,
        "dim": dims if dims is not None else "all",
        "keepdim": args.keepdim,
        "input_shape": list(selection.shape),
        "result_shape": list(result.shape),
        "result": _result_summary(ctx, result),
    }
    if result.numel() == 1:
        payload["value"] = safe_float(result.reshape(-1)[0])
    return payload


def linear_algebra(ctx: ToolContext, args: LinearAlgebraArgs) -> dict[str, Any]:
    if args.op == "matmul":
        return _matmul(ctx, args)
    if args.op == "einsum":
        return _einsum(ctx, args)
    name = _require_name(args)
    selection = _selection(ctx, name, args.index)
    if args.op == "reduce":
        return _reduce(ctx, args, name, selection)
    if args.op == "norm":
        flat = selection.float().reshape(-1)
        norm_result: dict[str, Any] = {
            "name": name,
            "shape": list(selection.shape),
            "l2": safe_float(torch.linalg.vector_norm(flat)),
            "l1": safe_float(flat.abs().sum()),
            "max_abs": safe_float(flat.abs().max()) if flat.numel() else 0.0,
            "rms": safe_float(flat.square().mean().sqrt()) if flat.numel() else 0.0,
        }
        if selection.ndim == 2:
            norm_result["frobenius"] = norm_result["l2"]
            norm_result["nuclear"] = safe_float(
                torch.linalg.matrix_norm(selection.float(), ord="nuc")
            )
        return norm_result

    if args.op == "cosine":
        if not args.other:
            raise ToolError("cosine requires 'other'")
        other = _selection(ctx, args.other, args.other_index)
        if selection.numel() != other.numel():
            raise ToolError(
                f"cosine requires equal element counts, got {selection.numel()} and {other.numel()}"
            )
        a = selection.float().reshape(-1)
        b = other.float().reshape(-1)
        denominator = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
        cosine = torch.dot(a, b) / denominator if float(denominator) != 0.0 else torch.tensor(0.0)
        return {
            "name": name,
            "other": args.other,
            "cosine": safe_float(cosine),
            "dot": safe_float(torch.dot(a, b)),
        }

    if args.op == "topk":
        flat = selection.float().reshape(-1)
        k = min(args.k, flat.numel())
        if k == 0:
            return {"name": name, "topk": [], "note": "empty selection"}
        values, indices = torch.topk(flat.abs(), k=k)
        return {
            "name": name,
            "shape": list(selection.shape),
            "topk": [
                {
                    "flat_index": int(i),
                    "value": safe_float(flat[int(i)]),
                    "abs": safe_float(v),
                }
                for v, i in zip(values, indices)
            ],
        }

    matrix = selection.float()
    if matrix.ndim == 0:
        raise ToolError("svd requires a tensor with at least one dimension")
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    elif matrix.ndim > 2:
        matrix = matrix.reshape(-1, matrix.shape[-1])
    rows, cols = matrix.shape
    if rows == 0 or cols == 0:
        raise ToolError("svd requires a non-empty matrix")
    k = min(args.k, rows, cols)
    _, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
    result: dict[str, Any] = {
        "name": name,
        "shape": list(selection.shape),
        "matrix_shape": [rows, cols],
        "rank": int(torch.linalg.matrix_rank(matrix)),
        "singular_values": [safe_float(v) for v in singular_values[:k].cpu().tolist()],
    }
    if k * cols <= ctx.config.max_read_elements:
        result["right_vectors"] = vh[:k].cpu().tolist()
    else:
        result["right_vectors"] = None
        result["hint"] = "Right singular vectors omitted; request a narrower index."
    return result


TOOLS: list[Tool] = [
    Tool(
        name="linear_algebra",
        description=(
            "Analyze tensor structure: 'svd' returns top singular values and (when small enough) "
            "right singular vectors, 'norm' returns norms, 'cosine' compares two selections, "
            "'topk' returns the largest absolute entries with flat indices. "
            "'matmul' multiplies two selections (name/other), 'einsum' evaluates an equation "
            "over 'operands' ([{name, index}, ...]), and 'reduce' applies sum/mean/max/min/"
            "prod/std/var/l1/l2 over one or more axes via 'dim' (null means all axes). "
            "Results include up to max_read_elements values."
        ),
        args_model=LinearAlgebraArgs,
        handler=linear_algebra,
    )
]
