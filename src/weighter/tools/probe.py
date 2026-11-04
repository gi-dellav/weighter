"""Budgeted forward-pass / evaluation probe."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from pydantic import Field

from ..utils import jsonable, tensor_summary
from .base import Tool, ToolArgs, ToolContext


def _default_dtype_and_device(module: torch.nn.Module) -> tuple[torch.dtype, torch.device]:
    parameter = next(module.parameters(), None)
    if parameter is not None:
        dtype = parameter.dtype if parameter.dtype.is_floating_point else torch.float32
        return dtype, parameter.device
    buffer = next(module.buffers(), None)
    if buffer is not None:
        dtype = buffer.dtype if buffer.dtype.is_floating_point else torch.float32
        return dtype, buffer.device
    return torch.float32, torch.device("cpu")


def _resolve_dtype(value: Any, default: torch.dtype) -> torch.dtype:
    if value is None:
        return default
    if isinstance(value, torch.dtype):
        return value
    aliases = {
        "float": torch.float32,
        "double": torch.float64,
        "half": torch.float16,
        "long": torch.int64,
        "int": torch.int32,
        "short": torch.int16,
        "bool": torch.bool,
        "bfloat16": torch.bfloat16,
    }
    if isinstance(value, str):
        if value in aliases:
            return aliases[value]
        candidate = getattr(torch, value, None)
        if isinstance(candidate, torch.dtype):
            return candidate
    raise ValueError(f"unknown dtype {value!r}")


def _decode_value(value: Any, dtype: torch.dtype, device: torch.device) -> Any:
    if isinstance(value, Mapping):
        if "tensor" in value or "data" in value:
            data = value.get("tensor", value.get("data"))
            return torch.as_tensor(
                data,
                dtype=_resolve_dtype(value.get("dtype"), dtype),
                device=torch.device(value.get("device", device)),
            )
        return {str(k): _decode_value(v, dtype, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        try:
            return torch.as_tensor(value, dtype=dtype, device=device)
        except (TypeError, ValueError, RuntimeError):
            return [_decode_value(item, dtype, device) for item in value]
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, float)):
        return torch.as_tensor(value, dtype=dtype, device=device)
    return value


def _summarize_output(value: Any, max_elements: int = 64, depth: int = 0) -> Any:
    if depth > 6:
        return "<max-depth>"
    if isinstance(value, torch.Tensor):
        return tensor_summary(value, max_elements=max_elements)
    if isinstance(value, Mapping):
        return {
            str(k): _summarize_output(v, max_elements=max_elements, depth=depth + 1)
            for k, v in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = list(value)
        converted = [
            _summarize_output(item, max_elements=max_elements, depth=depth + 1)
            for item in items[:max_elements]
        ]
        if len(items) > max_elements:
            converted.append(f"<{len(items) - max_elements} more items>")
        return converted
    return jsonable(value, max_items=max_elements)


def default_probe(module: torch.nn.Module, payload: dict[str, Any]) -> dict[str, Any]:
    """Default probe: decode JSON tensors, run one forward pass, summarize the output.

    Payload format::

        {"args": [<value>, ...], "kwargs": {<name>: <value>}, "training": false}

    A shorthand ``{"input": <value>}`` is passed as the single positional input.
    If neither ``args`` nor ``input`` is present, remaining payload keys are passed as
    keyword arguments. Plain nested numeric lists become tensors; use
    ``{"data": [...], "dtype": "long"}`` for explicit dtypes. The module is put
    in eval mode for the call and its previous training mode is restored.
    """

    if not isinstance(payload, Mapping):
        payload = {"input": payload}
    dtype, device = _default_dtype_and_device(module)
    if "args" in payload:
        raw_args = payload.get("args") or []
    elif "input" in payload:
        raw_args = [payload["input"]]
    else:
        raw_args = []
    if "kwargs" in payload:
        raw_kwargs = payload.get("kwargs") or {}
    elif "args" not in payload and "input" not in payload:
        raw_kwargs = {k: v for k, v in payload.items() if k != "training"}
    else:
        raw_kwargs = {}
    if not isinstance(raw_args, (list, tuple)):
        raw_args = [raw_args]
    if not isinstance(raw_kwargs, Mapping):
        raise ValueError("probe payload 'kwargs' must be an object")

    args = [_decode_value(value, dtype, device) for value in raw_args]
    kwargs = {str(k): _decode_value(v, dtype, device) for k, v in raw_kwargs.items()}
    training = bool(payload.get("training", False))
    was_training = module.training
    module.train(training)
    try:
        with torch.no_grad():
            output = module(*args, **kwargs)
    finally:
        module.train(was_training)
    return {"output": _summarize_output(output), "training": training}


class ProbeArgs(ToolArgs):
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Passed to the probe. Default probe format: "
            '{"args": [...], "kwargs": {...}, "training": false}. '
            'Use {"input": [...]} for a single positional input; if neither args nor '
            "input is present, the other payload keys become keyword arguments. "
            'Explicit tensors are {"data": [...], "dtype": "long"}.'
        ),
    )


def run_probe(ctx: ToolContext, args: ProbeArgs) -> dict[str, Any]:
    result = ctx.run_probe(args.payload)
    return {
        "result": jsonable(result, max_items=256),
        "remaining_budget": ctx.probe_budget.remaining,
    }


TOOLS: list[Tool] = [
    Tool(
        name="run_probe",
        description=(
            "Run one budgeted probe/evaluation call against the current module. The payload is "
            "forwarded to the harness probe callback. With the default probe, use "
            '{"args": [...], "kwargs": {...}}; nested lists become tensors and module outputs '
            "are summarized. Use this to get behavioral feedback after edits."
        ),
        args_model=ProbeArgs,
        handler=run_probe,
    )
]
