from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from weighter.utils import (
    content_to_text,
    get_attr,
    json_dumps,
    jsonable,
    safe_float,
    tensor_summary,
    truncate_text,
)


class Obj:
    def __init__(self) -> None:
        self.value = 3


def test_get_attr_from_object_and_mapping() -> None:
    assert get_attr(Obj(), "value") == 3
    assert get_attr(Obj(), "missing", "fallback") == "fallback"
    assert get_attr({"value": 4}, "value") == 4
    assert get_attr({"value": 4}, "missing", 5) == 5


def test_safe_float_handles_special_and_complex_values() -> None:
    assert safe_float(1.23456789) == 1.234568
    assert safe_float(None) == "None"
    assert safe_float("abc") == "abc"
    assert safe_float(float("nan")) == "nan"
    assert safe_float(float("inf")) == "inf"
    assert safe_float(float("-inf")) == "-inf"
    assert isinstance(safe_float(1 + 2j), str)
    assert isinstance(safe_float(torch.tensor(1 + 2j)), str)


def test_tensor_summary_includes_values_and_stats() -> None:
    summary = tensor_summary(torch.tensor([1, 2, 3]), max_elements=2)
    assert summary["type"] == "tensor"
    assert summary["shape"] == [3]
    assert summary["dtype"] == "int64"
    assert summary["values"] == [1, 2]
    assert summary["truncated"] is True
    assert summary["stats"]["mean"] == 2.0
    assert summary["stats"]["min"] == 1.0
    assert summary["stats"]["max"] == 3.0


def test_tensor_summary_empty_and_zero_limit() -> None:
    empty = tensor_summary(torch.tensor([]))
    assert empty["numel"] == 0
    assert "stats" not in empty
    zero = tensor_summary(torch.tensor([1.0, 2.0]), max_elements=0)
    assert zero["values"] == []
    assert zero["truncated"] is True


def test_tensor_summary_complex_uses_magnitudes_for_stats() -> None:
    summary = tensor_summary(torch.tensor([1 + 2j, 3 + 4j]))
    assert summary["values"] == ["(1+2j)", "(3+4j)"]
    assert summary["stats"]["max"] == pytest.approx(5.0)
    assert summary["stats"]["min"] == pytest.approx(2.236068, abs=1e-6)


def test_jsonable_common_python_values() -> None:
    assert jsonable(None) is None
    assert jsonable(True) is True
    assert jsonable("text") == "text"
    assert jsonable(1.23456789) == 1.234568
    assert jsonable(float("nan")) == "nan"
    assert jsonable({1: "a"}) == {"1": "a"}
    assert jsonable((1, 2)) == [1, 2]
    assert jsonable({1, 2, 3}, max_items=2) == [1, 2, "<1 more items>"]


def test_jsonable_dataclass_model_dump_and_tolist() -> None:
    @dataclass
    class Point:
        x: int
        y: int

    assert jsonable(Point(1, 2)) == {"x": 1, "y": 2}

    class WithDump:
        def model_dump(self) -> dict:
            return {"value": 5}

    assert jsonable(WithDump()) == {"value": 5}

    class WithToList:
        def tolist(self) -> list:
            return [1, 2]

    assert jsonable(WithToList()) == [1, 2]


def test_jsonable_max_depth_and_unknown_repr() -> None:
    value = "leaf"
    for _ in range(10):
        value = [value]
    assert "<max-depth>" in json_dumps(value)

    class BrokenDump:
        def model_dump(self) -> dict:
            raise ValueError("nope")

        def __repr__(self) -> str:
            return "x" * 3_000

    rendered = jsonable(BrokenDump())
    assert isinstance(rendered, str)
    assert len(rendered) == 2_000


def test_content_to_text_normalizes_provider_content() -> None:
    assert content_to_text(None) == ""
    assert content_to_text("hello") == "hello"
    assert content_to_text({"text": "a"}) == "a"
    assert content_to_text({"content": "b"}) == "b"
    assert content_to_text([{"text": "a"}, {"content": "b"}, "c"]) == "abc"

    class Block:
        text = "d"

    assert content_to_text([Block()]) == "d"
    assert content_to_text(123) == "123"


def test_truncate_text_boundaries() -> None:
    text = "x" * 100
    assert truncate_text(text, -1) == text
    assert truncate_text(text, 200) == text
    assert truncate_text(text, 40) == "x" * 40
    truncated = truncate_text(text, 41)
    assert truncated.endswith("<output truncated> ...")
    assert 0 < len(truncated) <= 41


def test_json_dumps_is_compact_and_json_safe() -> None:
    dumped = json_dumps({"name": "città", "values": [1.23456789]})
    assert dumped == '{"name":"città","values":[1.234568]}'
