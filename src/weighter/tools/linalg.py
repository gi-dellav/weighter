"""Linear-algebra tools for reading structure in weight tensors."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

import torch

from ..errors import ToolError
from ..utils import safe_float
from .base import Tool, ToolArgs, ToolContext


class LinearAlgebraArgs(ToolArgs):
    op: Literal["svd", "norm", "cosine", "topk"]
    name: str = Field(description="Tensor to analyze")
    index: list[Any] | None = None
    other: str | None = Field(default=None, description="Second tensor for cosine")
    other_index: list[Any] | None = None
    k: int = Field(default=8, ge=1, le=256)
    dim: int | None = None


def _selection(ctx: ToolContext, name: str, index: list[Any] | None) -> torch.Tensor:
    selection = ctx.store.selection(name, index).detach()
    if selection.numel() > ctx.config.max_edit_elements:
        raise ToolError(
            f"selection has {selection.numel()} elements; narrow the index to at most "
            f"{ctx.config.max_edit_elements}"
        )
    return selection


def linear_algebra(ctx: ToolContext, args: LinearAlgebraArgs) -> dict[str, Any]:
    selection = _selection(ctx, args.name, args.index)
    if args.op == "norm":
        flat = selection.float().reshape(-1)
        norm_result: dict[str, Any] = {
            "name": args.name,
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
            "name": args.name,
            "other": args.other,
            "cosine": safe_float(cosine),
            "dot": safe_float(torch.dot(a, b)),
        }

    if args.op == "topk":
        flat = selection.float().reshape(-1)
        k = min(args.k, flat.numel())
        if k == 0:
            return {"name": args.name, "topk": [], "note": "empty selection"}
        values, indices = torch.topk(flat.abs(), k=k)
        return {
            "name": args.name,
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
        "name": args.name,
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
            "'topk' returns the largest absolute entries with flat indices."
        ),
        args_model=LinearAlgebraArgs,
        handler=linear_algebra,
    )
]
