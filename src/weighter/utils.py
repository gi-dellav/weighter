"""Small utilities shared across the package."""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch


def get_attr(value: Any, name: str, default: Any = None) -> Any:
    """Read an attribute from either an object or a mapping."""

    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def safe_float(value: Any, digits: int = 6) -> float | str:
    """Convert a numeric value to a rounded JSON-safe float."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    return round(number, digits)


def jsonable(value: Any, *, max_items: int = 10_000, _depth: int = 0) -> Any:
    """Convert common Python/PyTorch/numpy values into JSON-safe structures."""

    if _depth > 8:
        return "<max-depth>"
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return safe_float(value)
    if isinstance(value, torch.Tensor):
        return tensor_summary(value, max_elements=min(max_items, 256))
    if isinstance(value, Mapping):
        return {str(k): jsonable(v, max_items=max_items, _depth=_depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        converted = [jsonable(v, max_items=max_items, _depth=_depth + 1) for v in items[:max_items]]
        if len(items) > max_items:
            converted.append(f"<{len(items) - max_items} more items>")
        return converted
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return jsonable(dataclasses.asdict(value), max_items=max_items, _depth=_depth + 1)
    if hasattr(value, "model_dump"):
        try:
            return jsonable(value.model_dump(), max_items=max_items, _depth=_depth + 1)
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return jsonable(value.tolist(), max_items=max_items, _depth=_depth + 1)
        except Exception:
            pass
    return repr(value)[:2_000]


def tensor_summary(tensor: torch.Tensor, *, max_elements: int = 64) -> dict[str, Any]:
    """Return a compact, JSON-safe summary of a tensor."""

    flat = tensor.detach().reshape(-1)
    total = int(flat.numel())
    shown = min(total, max(max_elements, 0))
    raw_values = flat[:shown].cpu().tolist()
    values = [safe_float(v) if isinstance(v, float) else v for v in raw_values]
    result: dict[str, Any] = {
        "type": "tensor",
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "device": str(tensor.device),
        "numel": total,
        "values": values,
        "truncated": shown < total,
    }
    if total:
        sample = flat[: min(total, 100_000)].to(torch.float32).cpu()
        result["stats"] = {
            "mean": safe_float(sample.mean()),
            "std": safe_float(sample.std(unbiased=False)),
            "min": safe_float(sample.min()),
            "max": safe_float(sample.max()),
        }
    return result


def content_to_text(content: Any) -> str:
    """Normalize provider-specific message content into plain text."""

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Mapping):
        return str(content.get("text", content.get("content", content)))
    if isinstance(content, Sequence):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                text = item.get("text")
                if text is None and isinstance(item.get("content"), str):
                    text = item["content"]
                if text:
                    parts.append(str(text))
            else:
                text = getattr(item, "text", None)
                if text:
                    parts.append(str(text))
        return "".join(parts)
    return str(content)


def truncate_text(text: str, limit: int) -> str:
    if limit < 0 or len(text) <= limit:
        return text
    if limit <= 40:
        return text[:limit]
    return text[: limit - 32] + "\n... <output truncated> ..."


def json_dumps(value: Any) -> str:
    return json.dumps(jsonable(value), ensure_ascii=False, separators=(",", ":"))
