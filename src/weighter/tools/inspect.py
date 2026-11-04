"""Read-only architecture and tensor inspection tools."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

import torch

from ..errors import ToolError
from .base import Tool, ToolArgs, ToolContext


class DescribeModuleArgs(ToolArgs):
    prefix: str | None = Field(default=None, description="Module path to start from")
    max_depth: int = Field(default=3, ge=0, le=10, description="Recursion depth")


class ListTensorsArgs(ToolArgs):
    prefix: str | None = Field(default=None, description="Only names starting with this prefix")
    kind: Literal["all", "parameter", "buffer"] = "all"
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=100, ge=1, le=500)
    include_stats: bool = True


class ReadTensorArgs(ToolArgs):
    name: str = Field(description="Canonical tensor name from list_tensors")
    index: list[Any] | None = Field(
        default=None,
        description='Per-dimension index, e.g. [null, "0:32"] or [0, [1, 2, 5]]',
    )
    max_elements: int | None = Field(default=None, ge=1, le=100_000)


class TensorStatsArgs(ToolArgs):
    name: str
    index: list[Any] | None = None
    sample_limit: int = Field(default=100_000, ge=1, le=2_000_000)


def _module_node(module: torch.nn.Module, name: str, depth: int, max_depth: int) -> dict[str, Any]:
    node: dict[str, Any] = {
        "name": name,
        "type": type(module).__name__,
        "parameters": int(sum(p.numel() for p in module.parameters(recurse=True))),
        "direct_parameters": int(sum(p.numel() for p in module.parameters(recurse=False))),
    }
    if depth < max_depth:
        children = [
            _module_node(child, child_name, depth + 1, max_depth)
            for child_name, child in module.named_children()
        ]
        if children:
            node["children"] = children
    return node


def describe_module(ctx: ToolContext, args: DescribeModuleArgs) -> dict[str, Any]:
    module = ctx.store.module
    name = ""
    if args.prefix:
        try:
            module = module.get_submodule(args.prefix)
        except AttributeError as exc:
            raise ToolError(f"unknown module path {args.prefix!r}") from exc
        name = args.prefix
    return {
        "overview": ctx.store.overview(),
        "tree": _module_node(module, name or type(module).__name__, 0, args.max_depth),
    }


def list_tensors(ctx: ToolContext, args: ListTensorsArgs) -> dict[str, Any]:
    refs = ctx.store.list_refs(prefix=args.prefix, kind=args.kind)
    selected = refs[args.offset : args.offset + args.limit]
    tensors: list[dict[str, Any]] = []
    for ref in selected:
        item = ref.info()
        if args.include_stats:
            stats = ctx.store._stats_for(ref.tensor, sample_limit=10_000)
            item["norm"] = stats.get("norm")
            item["mean"] = stats.get("mean")
            item["std"] = stats.get("std")
            item["min"] = stats.get("min")
            item["max"] = stats.get("max")
            item["zero_fraction"] = stats.get("zero_fraction")
            item["sampled"] = stats.get("sampled")
        tensors.append(item)
    return {
        "total": len(refs),
        "offset": args.offset,
        "returned": len(tensors),
        "tensors": tensors,
        "truncated": args.offset + len(tensors) < len(refs),
    }


def read_tensor(ctx: ToolContext, args: ReadTensorArgs) -> dict[str, Any]:
    max_elements = args.max_elements or ctx.config.max_read_elements
    return ctx.store.read_values(args.name, index=args.index, max_elements=max_elements)


def tensor_stats(ctx: ToolContext, args: TensorStatsArgs) -> dict[str, Any]:
    return ctx.store.stats(args.name, index=args.index, sample_limit=args.sample_limit)


TOOLS: list[Tool] = [
    Tool(
        name="describe_module",
        description=(
            "Show the module hierarchy, top-level children, parameter counts, dtypes and devices. "
            "Use this before inspecting tensors."
        ),
        args_model=DescribeModuleArgs,
        handler=describe_module,
    ),
    Tool(
        name="list_tensors",
        description=(
            "List parameter (and optionally buffer) tensors with shapes, dtypes and basic statistics. "
            "Supports a name prefix and pagination."
        ),
        args_model=ListTensorsArgs,
        handler=list_tensors,
    ),
    Tool(
        name="read_tensor",
        description=(
            "Read numeric values from a tensor selection. The index is a JSON list with one entry "
            "per dimension; use null for the full dimension, an integer, a slice string such as "
            "'0:32' or '0:64:2', or a list of integer indices. Output is capped."
        ),
        args_model=ReadTensorArgs,
        handler=read_tensor,
    ),
    Tool(
        name="tensor_stats",
        description=(
            "Compute detailed statistics for a tensor or tensor selection: mean, std, min, max, "
            "norm, percentiles, zero fraction, NaN and Inf counts."
        ),
        args_model=TensorStatsArgs,
        handler=tensor_stats,
    ),
]
