"""Tensor indexing, inspection, and transactional weight editing."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import torch

from .errors import EditError, ToolError
from .types import EditProposal
from .utils import safe_float


@dataclass(slots=True)
class TensorRef:
    name: str
    tensor: torch.Tensor
    kind: str
    aliases: tuple[str, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)

    def info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "aliases": list(self.aliases),
            "kind": self.kind,
            "shape": list(self.tensor.shape),
            "dtype": str(self.tensor.dtype).replace("torch.", ""),
            "device": str(self.tensor.device),
            "numel": int(self.tensor.numel()),
            "requires_grad": bool(self.tensor.requires_grad),
        }


@dataclass(slots=True)
class Patch:
    name: str
    index: tuple[Any, ...]
    original: torch.Tensor


@dataclass(slots=True)
class Revision:
    id: str
    note: str | None
    ops: list[dict[str, Any]]
    tensors: list[str]
    patches: list[Patch]
    bytes: int
    timestamp: float = field(default_factory=time.time)

    def info(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "note": self.note,
            "ops": [str(op.get("op", "?")) for op in self.ops],
            "tensors": list(self.tensors),
            "bytes": self.bytes,
            "timestamp": self.timestamp,
        }


def _parse_slice_string(text: str) -> slice | int:
    text = text.strip()
    if ":" not in text:
        try:
            return int(text)
        except ValueError as exc:
            raise EditError(f"invalid index string: {text!r}") from exc
    parts = text.split(":")
    if len(parts) > 3:
        raise EditError(f"invalid slice string: {text!r}")
    values: list[int | None] = []
    for part in parts:
        part = part.strip()
        if not part:
            values.append(None)
        else:
            try:
                values.append(int(part))
            except ValueError as exc:
                raise EditError(f"invalid slice string: {text!r}") from exc
    while len(values) < 3:
        values.append(None)
    return slice(values[0], values[1], values[2])


def parse_index(index: Sequence[Any] | None, ndim: int) -> tuple[Any, ...]:
    """Parse a JSON-friendly per-dimension index into a torch index tuple."""

    if index is None:
        return tuple(slice(None) for _ in range(ndim))
    if isinstance(index, (str, int)):
        index = [index]
    if not isinstance(index, (list, tuple)):
        raise EditError("index must be a list with one entry per tensor dimension")
    if len(index) > ndim:
        raise EditError(f"index has {len(index)} dimensions but tensor has {ndim}")
    parsed: list[Any] = []
    for item in index:
        if item is None:
            parsed.append(slice(None))
        elif isinstance(item, bool):
            raise EditError("boolean indices are not supported")
        elif isinstance(item, int):
            parsed.append(item)
        elif isinstance(item, str):
            parsed.append(_parse_slice_string(item))
        elif isinstance(item, (list, tuple)):
            if not item:
                raise EditError("empty index lists are not supported")
            try:
                parsed.append([int(v) for v in item])
            except (TypeError, ValueError) as exc:
                raise EditError("index lists must contain integers") from exc
        else:
            raise EditError(f"unsupported index component: {item!r}")
    parsed.extend(slice(None) for _ in range(ndim - len(parsed)))
    return tuple(parsed)


def _index_repr(index: tuple[Any, ...]) -> list[Any]:
    result: list[Any] = []
    for item in index:
        if isinstance(item, slice):
            if item.start is None and item.stop is None and item.step is None:
                result.append(":")
            else:
                result.append(
                    f"{'' if item.start is None else item.start}:"
                    f"{'' if item.stop is None else item.stop}:"
                    f"{'' if item.step is None else item.step}"
                )
        elif isinstance(item, list):
            result.append(item)
        else:
            result.append(item)
    return result


def _selection(tensor: torch.Tensor, index: tuple[Any, ...]) -> torch.Tensor:
    try:
        return tensor[index]
    except (IndexError, RuntimeError, TypeError) as exc:
        raise EditError(f"invalid index for tensor of shape {list(tensor.shape)}: {exc}") from exc


def _dtype_name(tensor: torch.Tensor) -> str:
    return str(tensor.dtype).replace("torch.", "")


class WeightStore:
    """A view over the parameters and buffers of an ``nn.Module``.

    All edits are applied in place to the original tensor objects, so existing
    optimizers keep valid references.  Revisions are reversible unless undo was
    explicitly disabled.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        *,
        include_buffers: bool = False,
        tensor_filter: Callable[[str], bool] | None = None,
        undo: bool = True,
        max_undo_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        self.module = module
        self.include_buffers = include_buffers
        self.tensor_filter = tensor_filter
        self.undo_enabled = undo
        self.max_undo_bytes = max_undo_bytes
        self._revisions: list[Revision] = []
        self._revision_counter = 0
        self._undo_bytes = 0
        self._build_index()

    def _build_index(self) -> None:
        collected: dict[int, TensorRef] = {}
        order: list[int] = []
        sources: list[tuple[str, Iterable[tuple[str, torch.Tensor]]]] = [
            ("parameter", self.module.named_parameters(recurse=True, remove_duplicate=False))
        ]
        if self.include_buffers:
            sources.append(("buffer", self.module.named_buffers(recurse=True, remove_duplicate=False)))

        for kind, iterator in sources:
            for name, tensor in iterator:
                if self.tensor_filter is not None and not self.tensor_filter(name):
                    continue
                key = id(tensor)
                if key in collected:
                    ref = collected[key]
                    collected[key] = TensorRef(
                        name=ref.name,
                        tensor=ref.tensor,
                        kind=ref.kind,
                        aliases=(*ref.aliases, name),
                    )
                else:
                    collected[key] = TensorRef(name=name, tensor=tensor, kind=kind)
                    order.append(key)

        self._refs: dict[str, TensorRef] = {}
        self._canonical: dict[str, str] = {}
        for key in order:
            ref = collected[key]
            self._refs[ref.name] = ref
            for name in ref.names:
                self._canonical[name] = ref.name
        self._order = [self_ref.name for self_ref in self._refs.values()]

    @property
    def revisions(self) -> list[Revision]:
        return list(self._revisions)

    def resolve(self, name: str) -> TensorRef:
        canonical = self._canonical.get(name)
        if canonical is None:
            matches = [n for n in self._canonical if name in n]
            hint = f" Did you mean one of: {', '.join(matches[:8])}?" if matches else ""
            raise ToolError(f"unknown tensor {name!r}.{hint}")
        return self._refs[canonical]

    def canonical_name(self, name: str) -> str:
        return self.resolve(name).name

    def tensor(self, name: str) -> torch.Tensor:
        return self.resolve(name).tensor

    def selection(self, name: str, index: Sequence[Any] | None = None) -> torch.Tensor:
        ref = self.resolve(name)
        parsed = parse_index(index, ref.tensor.ndim)
        return _selection(ref.tensor, parsed)

    def names(self) -> list[str]:
        return list(self._order)

    def list_refs(
        self, *, prefix: str | None = None, kind: str = "all"
    ) -> list[TensorRef]:
        refs = list(self._refs.values())
        if prefix:
            refs = [ref for ref in refs if ref.name.startswith(prefix)]
        if kind != "all":
            refs = [ref for ref in refs if ref.kind == kind]
        return refs

    def overview(self, max_children: int = 32) -> dict[str, Any]:
        parameters = list(self.module.parameters())
        buffers = list(self.module.buffers())
        children = []
        for name, child in list(self.module.named_children())[:max_children]:
            count = sum(p.numel() for p in child.parameters())
            children.append({"name": name, "type": type(child).__name__, "parameters": int(count)})
        dtypes = sorted({_dtype_name(p) for p in parameters})
        devices = sorted({str(p.device) for p in parameters})
        return {
            "module_type": type(self.module).__name__,
            "parameters": int(sum(p.numel() for p in parameters)),
            "parameter_tensors": len(parameters),
            "buffers": int(sum(b.numel() for b in buffers)),
            "buffer_tensors": len(buffers),
            "dtypes": dtypes,
            "devices": devices,
            "children": children,
            "revisions": len(self._revisions),
        }

    def dirty_tensors(self) -> list[str]:
        names: list[str] = []
        seen: set[str] = set()
        for revision in self._revisions:
            for name in revision.tensors:
                if name not in seen:
                    seen.add(name)
                    names.append(name)
        return names

    def stats(
        self,
        name: str,
        *,
        index: Sequence[Any] | None = None,
        sample_limit: int = 100_000,
    ) -> dict[str, Any]:
        ref = self.resolve(name)
        tensor = ref.tensor
        parsed = parse_index(index, tensor.ndim)
        selection = _selection(tensor, parsed)
        info = ref.info()
        info["index"] = _index_repr(parsed)
        info["selection_shape"] = list(selection.shape)
        info["selection_numel"] = int(selection.numel())
        info.update(self._stats_for(selection, sample_limit=sample_limit))
        return info

    @staticmethod
    def _stats_for(tensor: torch.Tensor, *, sample_limit: int = 100_000) -> dict[str, Any]:
        flat = tensor.detach().reshape(-1)
        total = int(flat.numel())
        if total == 0:
            return {
                "count": 0,
                "sampled": False,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "norm": 0.0,
                "zero_fraction": None,
                "nan_count": 0,
                "inf_count": 0,
            }
        sampled = total > sample_limit
        if sampled:
            positions = torch.linspace(0, total - 1, sample_limit, device=flat.device).long()
            sample = flat[positions]
        else:
            sample = flat
        if sample.is_complex():
            sample = sample.abs()
        sample = sample.detach().to(torch.float32).cpu()
        finite_mask = torch.isfinite(sample)
        finite = sample[finite_mask]
        nan_count = int(torch.isnan(sample).sum()) if sample.is_floating_point() else 0
        inf_count = int(torch.isinf(sample).sum()) if sample.is_floating_point() else 0
        result: dict[str, Any] = {
            "count": total,
            "sampled": sampled,
            "mean": safe_float(finite.mean()) if finite.numel() else None,
            "std": safe_float(finite.std(unbiased=False)) if finite.numel() else None,
            "min": safe_float(finite.min()) if finite.numel() else None,
            "max": safe_float(finite.max()) if finite.numel() else None,
            "norm": safe_float(torch.linalg.vector_norm(finite)) if finite.numel() else 0.0,
            "zero_fraction": safe_float((finite == 0).float().mean()) if finite.numel() else None,
            "nan_count": nan_count,
            "inf_count": inf_count,
        }
        if finite.numel():
            quantiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
            try:
                values = torch.quantile(finite, torch.tensor(quantiles)).tolist()
                result["percentiles"] = {
                    str(q): safe_float(v) for q, v in zip(quantiles, values)
                }
            except RuntimeError:
                pass
        return result

    def read_values(
        self,
        name: str,
        *,
        index: Sequence[Any] | None = None,
        max_elements: int = 2_048,
    ) -> dict[str, Any]:
        ref = self.resolve(name)
        tensor = ref.tensor
        parsed = parse_index(index, tensor.ndim)
        selection = _selection(tensor, parsed).detach()
        flat = selection.reshape(-1)
        total = int(flat.numel())
        shown = min(total, max_elements)
        raw_values = flat[:shown].cpu().tolist()
        values = [safe_float(v) if isinstance(v, float) else v for v in raw_values]
        return {
            "name": ref.name,
            "aliases": list(ref.aliases),
            "index": _index_repr(parsed),
            "shape": list(selection.shape),
            "dtype": _dtype_name(selection),
            "device": str(selection.device),
            "values": values,
            "returned": shown,
            "total": total,
            "truncated": shown < total,
            "hint": (
                "Use a narrower index to read the remaining values."
                if shown < total
                else None
            ),
        }

    def propose(
        self,
        ops: Sequence[dict[str, Any]],
        *,
        note: str | None = None,
        max_edit_elements: int = 1_000_000,
    ) -> EditProposal:
        targets: list[str] = []
        total = 0
        for op in ops:
            canonical, parsed, selection_numel = self._validate_op(op, max_edit_elements)
            if canonical not in targets:
                targets.append(canonical)
            total += selection_numel
        return EditProposal(
            ops=[dict(op) for op in ops],
            targets=targets,
            selection_elements=total,
            note=note,
        )

    def _validate_op(
        self, op: dict[str, Any], max_edit_elements: int
    ) -> tuple[str, tuple[Any, ...], int]:
        if not isinstance(op, dict):
            raise EditError("each edit must be an object")
        op_name = op.get("op")
        allowed = {"set", "add", "scale", "fill", "copy", "project_out", "random", "clamp"}
        if op_name not in allowed:
            raise EditError(f"unknown edit op {op_name!r}; expected one of {sorted(allowed)}")
        name = op.get("name")
        if not isinstance(name, str) or not name:
            raise EditError("each edit needs a tensor 'name'")
        required = {
            "set": ("values",),
            "add": ("values",),
            "scale": ("factor",),
            "fill": ("value",),
            "copy": ("src",),
            "project_out": ("direction",),
            "random": (),
            "clamp": (),
        }[op_name]
        for key in required:
            if key not in op:
                raise EditError(f"edit op {op_name!r} requires {key!r}")
        ref = self.resolve(name)
        parsed = parse_index(op.get("index"), ref.tensor.ndim)
        selection = _selection(ref.tensor, parsed)
        selection_numel = int(selection.numel())
        if selection_numel > max_edit_elements:
            raise EditError(
                f"selection for {name!r} has {selection_numel} elements, "
                f"above max_edit_elements={max_edit_elements}"
            )
        if op_name == "copy":
            src_ref = self.resolve(str(op["src"]))
            src_index = parse_index(op.get("src_index"), src_ref.tensor.ndim)
            src_selection = _selection(src_ref.tensor, src_index)
            if src_selection.numel() > max_edit_elements:
                raise EditError("copy source selection is too large")
        if op_name == "project_out":
            try:
                direction = torch.as_tensor(op["direction"])
            except (RuntimeError, ValueError, TypeError) as exc:
                raise EditError(f"invalid project_out direction: {exc}") from exc
            if direction.ndim != 1:
                raise EditError("project_out direction must be a 1D vector")
            axis = int(op.get("axis", -1))
            if axis < 0:
                axis += selection.ndim
            if axis < 0 or axis >= selection.ndim:
                raise EditError("project_out axis is out of range")
            if direction.numel() != selection.shape[axis]:
                raise EditError(
                    f"project_out direction has length {direction.numel()}, "
                    f"expected {selection.shape[axis]} for axis {axis}"
                )
        if op_name == "clamp" and "min" not in op and "max" not in op:
            raise EditError("clamp requires at least one of 'min' or 'max'")
        return ref.name, parsed, selection_numel

    def apply(
        self,
        ops: Sequence[dict[str, Any]],
        *,
        note: str | None = None,
        max_edit_elements: int = 1_000_000,
        confirm: Callable[[EditProposal], bool] | None = None,
    ) -> Revision:
        if not ops:
            raise EditError("no edits supplied")
        proposal = self.propose(ops, note=note, max_edit_elements=max_edit_elements)
        if confirm is not None and not confirm(proposal):
            raise EditError("weight edit was rejected by the confirmation callback")

        planned: list[tuple[dict[str, Any], TensorRef, tuple[Any, ...]]] = []
        for op in ops:
            ref = self.resolve(str(op["name"]))
            planned.append((op, ref, parse_index(op.get("index"), ref.tensor.ndim)))

        patches: list[Patch] = []
        temp_bytes = 0
        touched: list[str] = []
        try:
            for op, ref, parsed in planned:
                tensor = ref.tensor
                selection = _selection(tensor, parsed)
                patch_bytes = int(selection.numel() * selection.element_size())
                if (
                    self.undo_enabled
                    and self._undo_bytes + temp_bytes + patch_bytes > self.max_undo_bytes
                ):
                    raise EditError(
                        "edit would exceed max_undo_bytes; narrow the edit or raise the limit"
                    )
                if not self.undo_enabled and temp_bytes + patch_bytes > self.max_undo_bytes:
                    raise EditError(
                        "edit transaction is too large for atomic rollback; narrow the edit "
                        "or raise max_undo_bytes"
                    )
                original = selection.detach().to("cpu", copy=True)
                patches.append(Patch(name=ref.name, index=parsed, original=original))
                temp_bytes += patch_bytes
                self._apply_op(op, tensor, parsed, ref, max_edit_elements)
                if ref.name not in touched:
                    touched.append(ref.name)
        except Exception:
            self._restore_patches(reversed(patches))
            raise

        self._revision_counter += 1
        revision_id = f"r{self._revision_counter}"
        stored_patches = patches if self.undo_enabled else []
        if self.undo_enabled:
            self._undo_bytes += temp_bytes
        revision = Revision(
            id=revision_id,
            note=note,
            ops=[dict(op) for op in ops],
            tensors=touched,
            patches=stored_patches,
            bytes=temp_bytes if self.undo_enabled else 0,
        )
        self._revisions.append(revision)
        return revision

    def _apply_op(
        self,
        op: dict[str, Any],
        tensor: torch.Tensor,
        index: tuple[Any, ...],
        ref: TensorRef,
        max_edit_elements: int,
    ) -> None:
        op_name = str(op["op"])
        try:
            with torch.no_grad():
                selection = _selection(tensor, index)
                if op_name == "set":
                    tensor[index] = self._values(op["values"], ref, selection, max_edit_elements)
                elif op_name == "add":
                    values = self._values(op["values"], ref, selection, max_edit_elements)
                    scale = float(op.get("scale", 1.0))
                    tensor[index] = selection + values * scale
                elif op_name == "scale":
                    tensor[index] = selection * float(op["factor"])
                elif op_name == "fill":
                    tensor[index] = op["value"]
                elif op_name == "copy":
                    src_ref = self.resolve(str(op["src"]))
                    src_index = parse_index(op.get("src_index"), src_ref.tensor.ndim)
                    source = _selection(src_ref.tensor, src_index).detach().clone()
                    tensor[index] = source.to(device=tensor.device, dtype=tensor.dtype)
                elif op_name == "project_out":
                    self._project_out(tensor, index, selection, op)
                elif op_name == "random":
                    self._randomize(tensor, index, selection, op)
                elif op_name == "clamp":
                    minimum = op.get("min")
                    maximum = op.get("max")
                    tensor[index] = torch.clamp(selection, min=minimum, max=maximum)
                else:  # pragma: no cover - validated earlier
                    raise EditError(f"unsupported edit op {op_name!r}")
        except EditError:
            raise
        except (RuntimeError, ValueError, TypeError) as exc:
            raise EditError(f"edit {op_name!r} on {ref.name!r} failed: {exc}") from exc

    def _values(
        self,
        values: Any,
        ref: TensorRef,
        selection: torch.Tensor,
        max_edit_elements: int,
    ) -> torch.Tensor:
        try:
            value_tensor = torch.as_tensor(values)
        except (RuntimeError, ValueError, TypeError) as exc:
            raise EditError(f"could not convert edit values to a tensor: {exc}") from exc
        if value_tensor.numel() > max_edit_elements:
            raise EditError("edit values are too large")
        if value_tensor.is_complex() and not ref.tensor.is_complex():
            raise EditError("cannot write complex values into a real tensor")
        try:
            torch.broadcast_shapes(tuple(value_tensor.shape), tuple(selection.shape))
        except RuntimeError as exc:
            raise EditError(
                f"values shape {list(value_tensor.shape)} does not broadcast to "
                f"selection shape {list(selection.shape)}"
            ) from exc
        return value_tensor.to(device=ref.tensor.device, dtype=ref.tensor.dtype)

    def _project_out(
        self,
        tensor: torch.Tensor,
        index: tuple[Any, ...],
        selection: torch.Tensor,
        op: dict[str, Any],
    ) -> None:
        if not selection.is_floating_point():
            raise EditError("project_out requires a floating-point tensor")
        direction = torch.as_tensor(
            op["direction"], dtype=selection.dtype, device=selection.device
        )
        axis = int(op.get("axis", -1))
        if axis < 0:
            axis += selection.ndim
        shape = [1] * selection.ndim
        shape[axis] = direction.numel()
        direction = direction.reshape(shape)
        denominator = (direction * direction).sum()
        if float(denominator) == 0.0:
            raise EditError("project_out direction must not be the zero vector")
        scale = float(op.get("scale", 1.0))
        projection = (selection * direction).sum(dim=axis, keepdim=True) / denominator
        tensor[index] = selection - scale * projection * direction

    def _randomize(
        self,
        tensor: torch.Tensor,
        index: tuple[Any, ...],
        selection: torch.Tensor,
        op: dict[str, Any],
    ) -> None:
        if not selection.is_floating_point():
            raise EditError("random edits require a floating-point tensor")
        distribution = str(op.get("distribution", "normal"))
        generator = torch.Generator(device=selection.device)
        seed = op.get("seed")
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(int(seed))
        if distribution == "normal":
            values = torch.randn(
                selection.shape,
                generator=generator,
                device=selection.device,
                dtype=selection.dtype,
            )
            values = values * float(op.get("std", 0.02))
        elif distribution == "uniform":
            low = float(op.get("low", -0.1))
            high = float(op.get("high", 0.1))
            if high < low:
                raise EditError("uniform high must be >= low")
            values = torch.rand(
                selection.shape,
                generator=generator,
                device=selection.device,
                dtype=selection.dtype,
            )
            values = values * (high - low) + low
        else:
            raise EditError("random distribution must be 'normal' or 'uniform'")
        tensor[index] = values

    def _restore_patches(self, patches: Iterable[Patch]) -> None:
        with torch.no_grad():
            for patch in patches:
                tensor = self.tensor(patch.name)
                tensor[patch.index] = patch.original.to(
                    device=tensor.device, dtype=tensor.dtype
                )

    def undo(self, n: int = 1) -> list[Revision]:
        if not self.undo_enabled:
            raise ToolError("undo is disabled for this harness")
        if n < 1:
            raise ToolError("n must be >= 1")
        if not self._revisions:
            raise ToolError("there are no revisions to undo")
        n = min(n, len(self._revisions))
        undone: list[Revision] = []
        for _ in range(n):
            revision = self._revisions[-1]
            self._restore_patches(reversed(revision.patches))
            self._revisions.pop()
            self._undo_bytes = max(0, self._undo_bytes - revision.bytes)
            undone.append(revision)
        return undone

    def revert_all(self) -> list[Revision]:
        return self.undo(len(self._revisions)) if self._revisions else []

    def revision_infos(self) -> list[dict[str, Any]]:
        return [revision.info() for revision in self._revisions]

    def total_elements(self) -> int:
        return sum(int(ref.tensor.numel()) for ref in self._refs.values())

    def _revisions_for_diff(self, since: str | None) -> list[Revision]:
        if not self.undo_enabled:
            raise ToolError("diff is unavailable when undo is disabled")
        if not self._revisions:
            return []
        if since is None or since == "" or since in {"initial", "all"}:
            return list(self._revisions)
        if since == "last":
            return [self._revisions[-1]]
        ids = [revision.id for revision in self._revisions]
        if since not in ids:
            available = ", ".join(ids[-20:]) or "(none)"
            raise ToolError(
                f"unknown revision {since!r}; use 'initial', 'last', or one of: {available}"
            )
        index = ids.index(since)
        return self._revisions[index + 1 :]

    def diff(
        self,
        since: str | None = None,
        *,
        tensor_prefix: str | None = None,
    ) -> list[dict[str, Any]]:
        reverse_revisions = self._revisions_for_diff(since)
        if not reverse_revisions:
            return []
        names: list[str] = []
        seen: set[str] = set()
        for revision in reverse_revisions:
            for name in revision.tensors:
                if tensor_prefix and not name.startswith(tensor_prefix):
                    continue
                if name not in seen:
                    seen.add(name)
                    names.append(name)
        results: list[dict[str, Any]] = []
        for name in names:
            current = self.tensor(name).detach().cpu()
            original = current.clone()
            for revision in reversed(reverse_revisions):
                for patch in reversed(revision.patches):
                    if patch.name == name:
                        original[patch.index] = patch.original
            delta = (current.float() - original.float()).reshape(-1)
            finite = torch.isfinite(delta)
            changed = int((delta[finite] != 0).sum())
            original_norm = torch.linalg.vector_norm(original.float().reshape(-1))
            delta_norm = torch.linalg.vector_norm(delta[finite]) if finite.any() else torch.tensor(0.0)
            results.append(
                {
                    "name": name,
                    "shape": list(current.shape),
                    "changed_elements": changed,
                    "total_elements": int(current.numel()),
                    "delta_norm": safe_float(delta_norm),
                    "original_norm": safe_float(original_norm),
                    "relative_delta": safe_float(
                        delta_norm / original_norm if float(original_norm) != 0.0 else 0.0
                    ),
                    "max_abs_delta": safe_float(
                        delta[finite].abs().max() if finite.any() else 0.0
                    ),
                }
            )
        return results


def _as_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def summarize_diff(
    diffs: list[dict[str, Any]],
    *,
    module_elements: int | None = None,
    top_n: int = 10,
) -> dict[str, Any]:
    """Summarize a list of per-tensor diffs, including global deltas."""

    total_changed = sum(int(item.get("changed_elements", 0)) for item in diffs)
    dirty_elements = sum(int(item.get("total_elements", 0)) for item in diffs)
    delta_squares = sum(_as_float(item.get("delta_norm", 0.0)) ** 2 for item in diffs)
    original_squares = sum(_as_float(item.get("original_norm", 0.0)) ** 2 for item in diffs)
    global_delta = math.sqrt(delta_squares)
    global_original = math.sqrt(original_squares)
    max_abs = max((_as_float(item.get("max_abs_delta", 0.0)) for item in diffs), default=0.0)
    top = sorted(
        diffs,
        key=lambda item: (
            int(item.get("changed_elements", 0)),
            _as_float(item.get("delta_norm", 0.0)),
        ),
        reverse=True,
    )[:top_n]
    summary: dict[str, Any] = {
        "changed_tensors": len(diffs),
        "total_changed_elements": total_changed,
        "dirty_tensor_elements": dirty_elements,
        "global_delta_norm": safe_float(global_delta),
        "global_original_norm": safe_float(global_original),
        "global_relative_delta": safe_float(
            global_delta / global_original if global_original else 0.0
        ),
        "max_abs_delta": safe_float(max_abs),
        "top_changed": [
            {
                "name": item.get("name"),
                "changed_elements": item.get("changed_elements"),
                "total_elements": item.get("total_elements"),
                "delta_norm": item.get("delta_norm"),
                "relative_delta": item.get("relative_delta"),
            }
            for item in top
        ],
    }
    if module_elements is not None:
        summary["module_elements"] = module_elements
        summary["changed_fraction_of_module"] = safe_float(
            total_changed / module_elements if module_elements else 0.0
        )
        summary["dirty_fraction_of_module"] = safe_float(
            dirty_elements / module_elements if module_elements else 0.0
        )
    return summary
