"""Budgeted forward-pass / evaluation probe."""

from __future__ import annotations

import fnmatch
import inspect
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

import torch
from pydantic import Field

from ..utils import jsonable, tensor_summary
from .base import Tool, ToolArgs, ToolContext


_HIDDEN_KEYS = ("hidden_states", "output_hidden_states", "return_hidden_states")
_ATTENTION_KEYS = ("attentions", "output_attentions", "return_attentions")
_OPTION_KEYS = {
    "training",
    "max_capture_tensors",
    *_HIDDEN_KEYS,
    *_ATTENTION_KEYS,
}


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


def _summarize_output(
    value: Any,
    max_elements: int = 64,
    depth: int = 0,
    skip_keys: frozenset[str] | set[str] = frozenset(),
) -> Any:
    if depth > 6:
        return "<max-depth>"
    if isinstance(value, torch.Tensor):
        return tensor_summary(value, max_elements=max_elements)
    if isinstance(value, Mapping):
        return {
            str(k): _summarize_output(
                v, max_elements=max_elements, depth=depth + 1, skip_keys=skip_keys
            )
            for k, v in value.items()
            if str(k) not in skip_keys
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _summarize_output(
                getattr(value, item.name),
                max_elements=max_elements,
                depth=depth + 1,
                skip_keys=skip_keys,
            )
            for item in fields(value)
            if item.name not in skip_keys and getattr(value, item.name) is not None
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = list(value)
        converted = [
            _summarize_output(
                item, max_elements=max_elements, depth=depth + 1, skip_keys=skip_keys
            )
            for item in items[:max_elements]
        ]
        if len(items) > max_elements:
            converted.append(f"<{len(items) - max_elements} more items>")
        return converted
    return jsonable(value, max_items=max_elements)


def _iter_tensors(value: Any, prefix: str = "", depth: int = 0) -> Iterator[tuple[str, torch.Tensor]]:
    if depth > 4:
        return
    if isinstance(value, torch.Tensor):
        yield prefix, value
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            label = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_tensors(item, label, depth + 1)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            label = f"{prefix}[{index}]" if prefix else f"[{index}]"
            yield from _iter_tensors(item, label, depth + 1)


def _is_attention_like(tensor: torch.Tensor) -> bool:
    return tensor.ndim >= 2 and tensor.shape[-1] == tensor.shape[-2]


def _looks_attention(path: str, type_name: str) -> bool:
    text = f"{path} {type_name}".lower()
    return "attention" in text or "attn" in text


def _matches_module(path: str, type_name: str, selector: str) -> bool:
    selector = selector.strip()
    if not selector:
        return False
    if (
        path == selector
        or path.startswith(selector + ".")
        or selector in path.split(".")
        or fnmatch.fnmatchcase(path, selector)
    ):
        return True
    lowered = selector.lower()
    return fnmatch.fnmatchcase(type_name.lower(), lowered) or lowered in type_name.lower()


def _selects(
    request: bool | list[str] | None,
    path: str,
    type_name: str,
    *,
    default_all: bool,
) -> bool:
    if isinstance(request, list):
        return any(_matches_module(path, type_name, selector) for selector in request)
    if request:
        return True if default_all else _looks_attention(path, type_name)
    return False


def _capture_option(
    payload: Mapping[str, Any], keys: Sequence[str]
) -> bool | list[str] | None:
    value: Any = None
    for key in keys:
        if key in payload:
            value = payload[key]
            break
    if value is None or value is False:
        return None
    if value is True:
        return True
    if isinstance(value, str):
        if value.strip().lower() in {"all", "true"}:
            return True
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        selectors = [str(item).strip() for item in value if str(item).strip()]
        return selectors or None
    raise ValueError("capture options must be true, false, or a list of module paths")


def _field_value(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _forward_accepts_kwarg(module: torch.nn.Module, name: str) -> bool:
    try:
        parameters = inspect.signature(module.forward).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == name or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


@dataclass
class _CaptureBucket:
    limit: int
    items: dict[str, Any] = field(default_factory=dict)
    truncated: bool = False

    def add(self, value: Any, prefix: str, *, attention_only: bool = False) -> None:
        candidates = list(_iter_tensors(value, prefix))
        if attention_only:
            filtered = [
                (label, tensor)
                for label, tensor in candidates
                if _is_attention_like(tensor)
            ]
            if filtered:
                candidates = filtered
        for label, tensor in candidates:
            if label in self.items:
                continue
            if len(self.items) >= self.limit:
                self.truncated = True
                break
            self.items[label] = tensor_summary(tensor, max_elements=64)


def _summarize_capture_field(
    value: Any, prefix: str, limit: int
) -> tuple[dict[str, Any], bool]:
    bucket = _CaptureBucket(limit)
    bucket.add(value, prefix)
    return bucket.items, bucket.truncated


def _make_capture_hook(
    path: str,
    hidden_bucket: _CaptureBucket | None,
    attention_bucket: _CaptureBucket | None,
):
    def hook(
        module: torch.nn.Module, inputs: tuple[Any, ...], output: Any
    ) -> None:
        try:
            if hidden_bucket is not None:
                hidden_bucket.add(output, path)
            if attention_bucket is not None:
                attention_bucket.add(output, path, attention_only=True)
        except Exception:
            pass

    return hook


def default_probe(module: torch.nn.Module, payload: dict[str, Any]) -> dict[str, Any]:
    """Default probe: decode JSON tensors, run one forward pass, summarize the output.

    Payload format::

        {"args": [<value>, ...], "kwargs": {<name>: <value>}, "training": false}

    A shorthand ``{"input": <value>}`` is passed as the single positional input.
    If neither ``args`` nor ``input`` is present, remaining payload keys are passed as
    keyword arguments. Plain nested numeric lists become tensors; use
    ``{"data": [...], "dtype": "long"}`` for explicit dtypes. The module is put
    in eval mode for the call and its previous training mode is restored.

    Hidden states and attention patterns can be requested with ``hidden_states`` and
    ``attentions`` (aliases ``output_hidden_states``/``return_hidden_states`` and
    ``output_attentions``/``return_attentions``). Both accept ``true`` or a list of
    module-path globs. Model outputs exposing ``last_hidden_state``, ``hidden_states``,
    or ``attentions`` are extracted directly; otherwise temporary forward hooks capture
    matching module outputs.
    """

    if not isinstance(payload, Mapping):
        payload = {"input": payload}

    hidden_request = _capture_option(payload, _HIDDEN_KEYS)
    attention_request = _capture_option(payload, _ATTENTION_KEYS)
    max_capture = payload.get("max_capture_tensors", 64)
    if isinstance(max_capture, bool) or not isinstance(max_capture, int):
        raise ValueError("max_capture_tensors must be an integer")
    max_capture = max(1, min(max_capture, 256))

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
        raw_kwargs = {k: v for k, v in payload.items() if k not in _OPTION_KEYS}
    else:
        raw_kwargs = {}
    if not isinstance(raw_args, (list, tuple)):
        raw_args = [raw_args]
    if not isinstance(raw_kwargs, Mapping):
        raise ValueError("probe payload 'kwargs' must be an object")

    args = [_decode_value(value, dtype, device) for value in raw_args]
    kwargs = {str(k): _decode_value(v, dtype, device) for k, v in raw_kwargs.items()}
    if (
        hidden_request is not None
        and "output_hidden_states" not in kwargs
        and _forward_accepts_kwarg(module, "output_hidden_states")
    ):
        kwargs["output_hidden_states"] = True
    if (
        attention_request is not None
        and "output_attentions" not in kwargs
        and _forward_accepts_kwarg(module, "output_attentions")
    ):
        kwargs["output_attentions"] = True

    hidden_bucket = _CaptureBucket(max_capture) if hidden_request is not None else None
    attention_bucket = (
        _CaptureBucket(max_capture) if attention_request is not None else None
    )
    handles = []
    if hidden_bucket is not None or attention_bucket is not None:
        for name, submodule in module.named_modules():
            path = name or "<root>"
            hidden_match = _selects(
                hidden_request, path, type(submodule).__name__, default_all=True
            )
            attention_match = _selects(
                attention_request, path, type(submodule).__name__, default_all=False
            )
            if path == "<root>" and hidden_request is True:
                hidden_match = False
            if not (hidden_match or attention_match):
                continue
            handles.append(
                submodule.register_forward_hook(
                    _make_capture_hook(
                        path,
                        hidden_bucket if hidden_match else None,
                        attention_bucket if attention_match else None,
                    )
                )
            )

    training = bool(payload.get("training", False))
    was_training = module.training
    module.train(training)
    try:
        with torch.no_grad():
            output = module(*args, **kwargs)
    finally:
        for handle in handles:
            handle.remove()
        module.train(was_training)

    skip_keys: set[str] = set()
    result: dict[str, Any] = {"training": training}
    hidden_truncated = False
    attention_truncated = False

    if hidden_request is not None:
        last_hidden = _field_value(output, "last_hidden_state")
        if last_hidden is not None:
            result["last_hidden_state"] = _summarize_output(last_hidden)
            skip_keys.add("last_hidden_state")
        hidden_field = _field_value(output, "hidden_states")
        hidden_items: dict[str, Any] = {}
        if hidden_field is not None:
            hidden_items, hidden_truncated = _summarize_capture_field(
                hidden_field, "hidden_states", max_capture
            )
            skip_keys.add("hidden_states")
        if hidden_bucket is not None and (
            hidden_request is not True or hidden_field is None
        ):
            hidden_items.update(hidden_bucket.items)
            hidden_truncated = hidden_truncated or hidden_bucket.truncated
        result["hidden_states"] = hidden_items

    if attention_request is not None:
        attention_field = _field_value(output, "attentions")
        attention_items: dict[str, Any] = {}
        if attention_field is not None:
            attention_items, attention_truncated = _summarize_capture_field(
                attention_field, "attentions", max_capture
            )
            skip_keys.add("attentions")
        if attention_bucket is not None and (
            attention_request is not True or attention_field is None
        ):
            attention_items.update(attention_bucket.items)
            attention_truncated = attention_truncated or attention_bucket.truncated
        result["attentions"] = attention_items

    result["output"] = _summarize_output(output, skip_keys=skip_keys)
    if hidden_request is not None or attention_request is not None:
        result["capture"] = {
            "max_capture_tensors": max_capture,
            "hidden_truncated": hidden_truncated if hidden_request is not None else None,
            "attention_truncated": (
                attention_truncated if attention_request is not None else None
            ),
        }
    return result


class ProbeArgs(ToolArgs):
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Passed to the probe. Default probe format: "
            '{"args": [...], "kwargs": {...}, "training": false}. '
            'Use {"input": [...]} for a single positional input; if neither args nor '
            "input is present, the other payload keys become keyword arguments. "
            'Explicit tensors are {"data": [...], "dtype": "long"}. '
            'Set "hidden_states": true and/or "attentions": true to capture them; '
            'either may instead be a list of module-path globs. '
            '"max_capture_tensors" caps captured tensors (default 64).'
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
            "are summarized. Add hidden_states/attentions capture flags for transformer "
            "diagnostics. Use this to get behavioral feedback after edits."
        ),
        args_model=ProbeArgs,
        handler=run_probe,
    )
]
