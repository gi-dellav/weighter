from __future__ import annotations

from dataclasses import dataclass

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


@dataclass
class FakeModelOutput:
    last_hidden_state: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None


class FakeTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(
        self,
        x: torch.Tensor,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> FakeModelOutput:
        hidden = self.linear(x)
        hidden_states = (x, hidden) if output_hidden_states else None
        attentions = None
        if output_attentions:
            weights = torch.softmax(
                torch.zeros(x.shape[0], 2, x.shape[1], x.shape[1]), dim=-1
            )
            attentions = (weights,)
        return FakeModelOutput(hidden, hidden_states, attentions)


def test_default_probe_hf_style_hidden_states_and_attentions() -> None:
    module = FakeTransformer()
    result = default_probe(
        module,
        {
            "input": [[1.0, 2.0, 3.0, 4.0]],
            "hidden_states": True,
            "attentions": True,
        },
    )
    assert result["last_hidden_state"]["shape"] == [1, 4]
    assert set(result["hidden_states"]) == {"hidden_states[0]", "hidden_states[1]"}
    assert result["attentions"]["attentions[0]"]["shape"] == [1, 2, 4, 4]
    assert result["output"] == {}
    assert result["capture"]["hidden_truncated"] is False
    assert result["capture"]["attention_truncated"] is False


def test_default_probe_hidden_state_aliases_and_selectors() -> None:
    module = torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.ReLU(),
        torch.nn.Linear(4, 2),
    )
    result = default_probe(
        module,
        {
            "input": [[1.0, 2.0, 3.0]],
            "return_hidden_states": ["0", "2"],
        },
    )
    assert set(result["hidden_states"]) == {"0", "2"}
    assert result["hidden_states"]["0"]["shape"] == [1, 4]
    assert result["hidden_states"]["2"]["shape"] == [1, 2]


def test_default_probe_attention_hooks() -> None:
    class SelfAttention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = torch.nn.MultiheadAttention(4, 2, batch_first=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            output, _ = self.self_attn(x, x, x)
            return output

    result = default_probe(
        SelfAttention(),
        {
            "input": [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]],
            "attentions": True,
        },
    )
    attention = next(
        summary
        for name, summary in result["attentions"].items()
        if "self_attn" in name
    )
    assert attention["shape"][-2] == attention["shape"][-1] == 2
