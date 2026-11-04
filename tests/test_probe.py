from __future__ import annotations

import torch

from weighter.tools.probe import default_probe


def test_default_probe_positional_and_kwargs() -> None:
    module = torch.nn.Sequential(torch.nn.Linear(3, 2))
    module.train()
    result = default_probe(module, {"args": [[[1.0, 2.0, 3.0]]]})
    assert result["output"]["shape"] == [1, 2]
    assert result["training"] is False
    assert module.training is True

    linear = torch.nn.Linear(3, 2)
    result = default_probe(linear, {"kwargs": {"input": [[1.0, 2.0, 3.0]]}})
    assert result["output"]["shape"] == [1, 2]


def test_default_probe_explicit_dtype() -> None:
    embedding = torch.nn.Embedding(10, 4)
    result = default_probe(
        embedding,
        {"args": [{"data": [1, 3], "dtype": "long"}]},
    )
    assert result["output"]["shape"] == [2, 4]


def test_default_probe_input_shorthand() -> None:
    module = torch.nn.Linear(2, 1)
    result = default_probe(module, {"input": [1.0, 2.0]})
    assert result["output"]["shape"] == [1]


def test_default_probe_summarizes_large_output() -> None:
    module = torch.nn.Identity()
    result = default_probe(module, {"args": [[[float(i) for i in range(200)]]]})
    assert result["output"]["truncated"] is True
    assert len(result["output"]["values"]) == 64


def test_default_probe_direct_kwargs() -> None:
    module = torch.nn.Linear(3, 2)
    result = default_probe(module, {"input": [[1.0, 2.0, 3.0]]})
    assert result["output"]["shape"] == [1, 2]

    class TwoInputs(torch.nn.Module):
        def forward(self, x, scale):
            return x * scale

    result = default_probe(TwoInputs(), {"x": [1.0, 2.0], "scale": 2.0})
    assert result["output"]["values"] == [2.0, 4.0]
