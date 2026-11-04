from __future__ import annotations

import pytest
import torch

from weighter.config import HarnessConfig
from weighter.errors import ToolError
from weighter.tools.base import ProbeBudget, ToolContext
from weighter.tools.linalg import LinearAlgebraArgs, linear_algebra
from weighter.weights import WeightStore


class TensorModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = torch.nn.Parameter(torch.arange(6.0).reshape(2, 3))
        self.b = torch.nn.Parameter(torch.arange(12.0).reshape(3, 4))
        self.c = torch.nn.Parameter(torch.arange(24.0).reshape(2, 3, 4))
        self.d = torch.nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        self.u = torch.nn.Parameter(torch.arange(4.0).reshape(4, 1))
        self.v = torch.nn.Parameter(torch.arange(4.0).reshape(1, 4))


def _context(**config_kwargs) -> ToolContext:
    config = HarnessConfig(llm="fake/model", **config_kwargs)
    return ToolContext(
        store=WeightStore(TensorModule()),
        config=config,
        probe=None,
        probe_budget=ProbeBudget(),
    )


def _assert_values(values: list[float], expected: torch.Tensor) -> None:
    assert values == pytest.approx(expected.reshape(-1).tolist())


def test_linear_algebra_matmul() -> None:
    result = linear_algebra(
        _context(), LinearAlgebraArgs(op="matmul", name="a", other="b")
    )
    expected = torch.matmul(
        torch.arange(6.0).reshape(2, 3), torch.arange(12.0).reshape(3, 4)
    )
    assert result["result_shape"] == [2, 4]
    assert result["operand_shapes"] == [[2, 3], [3, 4]]
    assert result["result"]["dtype"] == "float32"
    _assert_values(result["result"]["values"], expected)


def test_linear_algebra_matmul_requires_other() -> None:
    with pytest.raises(ToolError, match="requires 'other'"):
        linear_algebra(_context(), LinearAlgebraArgs(op="matmul", name="a"))


def test_linear_algebra_einsum() -> None:
    result = linear_algebra(
        _context(),
        LinearAlgebraArgs(
            op="einsum",
            equation="ij,jk->ik",
            operands=[{"name": "a"}, {"name": "b"}],
        ),
    )
    expected = torch.matmul(
        torch.arange(6.0).reshape(2, 3), torch.arange(12.0).reshape(3, 4)
    )
    assert result["result_shape"] == [2, 4]
    assert result["operands"][0] == {"name": "a", "shape": [2, 3]}
    _assert_values(result["result"]["values"], expected)


def test_linear_algebra_einsum_invalid_equation() -> None:
    with pytest.raises(ToolError, match="unsupported characters"):
        linear_algebra(
            _context(),
            LinearAlgebraArgs(
                op="einsum",
                equation="ij,jk->ik!",
                operands=[{"name": "a"}, {"name": "b"}],
            ),
        )

    with pytest.raises(ToolError, match="einsum failed"):
        linear_algebra(
            _context(),
            LinearAlgebraArgs(
                op="einsum",
                equation="ij,jk->ik",
                operands=[{"name": "a"}],
            ),
        )


def test_linear_algebra_axis_reductions() -> None:
    ctx = _context()
    total = linear_algebra(
        ctx,
        LinearAlgebraArgs(op="reduce", name="a", reduction="sum", dim=1),
    )
    assert total["result_shape"] == [2]
    assert total["result"]["values"] == [3.0, 12.0]

    mean = linear_algebra(
        ctx,
        LinearAlgebraArgs(
            op="reduce",
            name="a",
            reduction="mean",
            dim=[0, 1],
            keepdim=True,
        ),
    )
    assert mean["result_shape"] == [1, 1]
    assert mean["value"] == 2.5

    l2 = linear_algebra(
        ctx,
        LinearAlgebraArgs(op="reduce", name="a", reduction="l2", dim=0),
    )
    assert l2["result"]["values"] == pytest.approx(
        [3.0, 4.123106, 5.385165], abs=1e-6
    )

    prod = linear_algebra(
        ctx,
        LinearAlgebraArgs(op="reduce", name="d", reduction="prod", dim=[0, 1]),
    )
    assert prod["value"] == 24.0

    scalar = linear_algebra(
        ctx,
        LinearAlgebraArgs(op="reduce", name="a", reduction="sum"),
    )
    assert scalar["result_shape"] == []
    assert scalar["value"] == 15.0
    assert scalar["dim"] == "all"


def test_linear_algebra_reduce_rejects_bad_axis() -> None:
    with pytest.raises(ToolError, match="out of range"):
        linear_algebra(
            _context(),
            LinearAlgebraArgs(op="reduce", name="a", reduction="sum", dim=3),
        )


def test_linear_algebra_result_size_guard() -> None:
    with pytest.raises(ToolError, match="max_edit_elements"):
        linear_algebra(
            _context(max_edit_elements=10),
            LinearAlgebraArgs(op="matmul", name="u", other="v"),
        )


def test_linear_algebra_result_values_are_truncated() -> None:
    result = linear_algebra(
        _context(max_read_elements=3),
        LinearAlgebraArgs(op="matmul", name="a", other="b"),
    )
    assert result["result"]["truncated"] is True
    assert len(result["result"]["values"]) == 3
    assert result["result"]["numel"] == 8
