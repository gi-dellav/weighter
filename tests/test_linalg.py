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
        self.z = torch.nn.Parameter(torch.zeros(2))
        self.one_d = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
        self.empty = torch.nn.Parameter(torch.empty(0))
        self.complex = torch.nn.Parameter(torch.tensor([1 + 2j, 3 + 4j]))


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


def test_linear_algebra_norm_and_cosine() -> None:
    ctx = _context()
    norm = linear_algebra(ctx, LinearAlgebraArgs(op="norm", name="a"))
    assert norm["l2"] == pytest.approx(7.416198, abs=1e-6)
    assert norm["l1"] == 15.0
    assert norm["max_abs"] == 5.0
    assert norm["rms"] == pytest.approx(3.02765, abs=1e-5)
    assert norm["frobenius"] == norm["l2"]
    assert "nuclear" in norm

    cosine = linear_algebra(
        ctx, LinearAlgebraArgs(op="cosine", name="a", other="a")
    )
    assert cosine["cosine"] == pytest.approx(1.0)
    assert cosine["dot"] == 55.0

    zero = linear_algebra(
        ctx, LinearAlgebraArgs(op="cosine", name="z", other="z")
    )
    assert zero["cosine"] == 0.0

    with pytest.raises(ToolError, match="equal element counts"):
        linear_algebra(ctx, LinearAlgebraArgs(op="cosine", name="a", other="z"))
    with pytest.raises(ToolError, match="requires 'other'"):
        linear_algebra(ctx, LinearAlgebraArgs(op="cosine", name="a"))


def test_linear_algebra_topk_and_empty_selection() -> None:
    ctx = _context()
    topk = linear_algebra(ctx, LinearAlgebraArgs(op="topk", name="a", k=2))
    assert [item["flat_index"] for item in topk["topk"]] == [5, 4]
    assert [item["value"] for item in topk["topk"]] == [5.0, 4.0]

    empty = linear_algebra(ctx, LinearAlgebraArgs(op="topk", name="empty"))
    assert empty["topk"] == []
    assert "empty selection" in empty["note"]

    norm = linear_algebra(ctx, LinearAlgebraArgs(op="norm", name="empty"))
    assert norm["l2"] == 0.0
    assert norm["max_abs"] == 0.0
    assert norm["rms"] == 0.0


def test_linear_algebra_svd_shapes_and_truncation() -> None:
    ctx = _context()
    svd = linear_algebra(ctx, LinearAlgebraArgs(op="svd", name="b", k=2))
    assert svd["matrix_shape"] == [3, 4]
    assert svd["rank"] == 2
    assert svd["singular_values"] == pytest.approx([22.409296, 1.95534], abs=1e-5)
    assert len(svd["right_vectors"]) == 2

    one_d = linear_algebra(ctx, LinearAlgebraArgs(op="svd", name="one_d"))
    assert one_d["matrix_shape"] == [1, 3]
    assert one_d["singular_values"] == pytest.approx([3.741657], abs=1e-6)

    high_dim = linear_algebra(ctx, LinearAlgebraArgs(op="svd", name="c"))
    assert high_dim["matrix_shape"] == [6, 4]

    omitted = linear_algebra(
        _context(max_read_elements=2),
        LinearAlgebraArgs(op="svd", name="b", k=2),
    )
    assert omitted["right_vectors"] is None
    assert "hint" in omitted

    with pytest.raises(ToolError, match="non-empty matrix"):
        linear_algebra(ctx, LinearAlgebraArgs(op="svd", name="empty"))


def test_linear_algebra_reductions() -> None:
    ctx = _context()
    max_result = linear_algebra(
        ctx, LinearAlgebraArgs(op="reduce", name="a", reduction="max", dim=1)
    )
    assert max_result["result"]["values"] == [2.0, 5.0]

    min_result = linear_algebra(
        ctx, LinearAlgebraArgs(op="reduce", name="a", reduction="min", dim=0)
    )
    assert min_result["result"]["values"] == [0.0, 1.0, 2.0]

    std_result = linear_algebra(
        ctx, LinearAlgebraArgs(op="reduce", name="a", reduction="std", dim=1)
    )
    assert std_result["result"]["values"] == pytest.approx(
        [0.816497, 0.816497], abs=1e-6
    )

    var_result = linear_algebra(
        ctx, LinearAlgebraArgs(op="reduce", name="a", reduction="var", dim=1)
    )
    assert var_result["result"]["values"] == pytest.approx(
        [0.666667, 0.666667], abs=1e-6
    )

    l1_result = linear_algebra(
        ctx, LinearAlgebraArgs(op="reduce", name="a", reduction="l1", dim=0)
    )
    assert l1_result["result"]["values"] == [3.0, 5.0, 7.0]

    l2_result = linear_algebra(
        ctx, LinearAlgebraArgs(op="reduce", name="a", reduction="l2", dim=-1)
    )
    assert l2_result["result"]["values"] == pytest.approx(
        [2.236068, 7.071068], abs=1e-6
    )

    prod_keepdim = linear_algebra(
        ctx,
        LinearAlgebraArgs(
            op="reduce",
            name="d",
            reduction="prod",
            dim=[0, 1],
            keepdim=True,
        ),
    )
    assert prod_keepdim["result_shape"] == [1, 1]
    assert prod_keepdim["value"] == 24.0


def test_linear_algebra_reduce_validation() -> None:
    ctx = _context()
    with pytest.raises(ToolError, match="requires 'reduction'"):
        linear_algebra(ctx, LinearAlgebraArgs(op="reduce", name="a"))
    with pytest.raises(ToolError, match="duplicate axes"):
        linear_algebra(
            ctx,
            LinearAlgebraArgs(op="reduce", name="a", reduction="sum", dim=[0, 0]),
        )
    with pytest.raises(ToolError, match="out of range"):
        linear_algebra(
            ctx,
            LinearAlgebraArgs(op="reduce", name="a", reduction="sum", dim=[2]),
        )
    string_dim = LinearAlgebraArgs.model_construct(
        op="reduce", name="a", reduction="sum", dim="0"
    )
    with pytest.raises(ToolError, match="integer or a list"):
        linear_algebra(ctx, string_dim)
    bool_dim = LinearAlgebraArgs.model_construct(
        op="reduce", name="a", reduction="sum", dim=[True]
    )
    with pytest.raises(ToolError, match="integer or a list"):
        linear_algebra(ctx, bool_dim)


def test_linear_algebra_matmul_shape_error() -> None:
    with pytest.raises(ToolError, match="matmul failed"):
        linear_algebra(
            _context(), LinearAlgebraArgs(op="matmul", name="a", other="d")
        )


def test_linear_algebra_einsum_validation_limits() -> None:
    ctx = _context()
    with pytest.raises(ToolError, match="requires 'equation'"):
        linear_algebra(ctx, LinearAlgebraArgs(op="einsum", operands=[{"name": "a"}]))
    with pytest.raises(ToolError, match="between 1 and 256"):
        linear_algebra(
            ctx,
            LinearAlgebraArgs(
                op="einsum",
                equation="i" * 300,
                operands=[{"name": "a"}],
            ),
        )
    with pytest.raises(ToolError, match="non-empty 'operands'"):
        linear_algebra(
            ctx, LinearAlgebraArgs(op="einsum", equation="ij->ji")
        )
    with pytest.raises(ToolError, match="at most 32"):
        linear_algebra(
            ctx,
            LinearAlgebraArgs(
                op="einsum",
                equation=",".join(["i"] * 33) + "->i",
                operands=[{"name": "one_d"}] * 33,
            ),
        )


def test_linear_algebra_requires_name_and_rejects_complex() -> None:
    ctx = _context()
    with pytest.raises(ToolError, match="requires 'name'"):
        linear_algebra(ctx, LinearAlgebraArgs(op="norm"))
    with pytest.raises(ToolError, match="complex"):
        linear_algebra(ctx, LinearAlgebraArgs(op="norm", name="complex"))
    with pytest.raises(ToolError, match="complex"):
        linear_algebra(
            ctx, LinearAlgebraArgs(op="matmul", name="complex", other="complex")
        )
