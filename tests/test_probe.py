from __future__ import annotations

from dataclasses import dataclass

import pytest
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


def test_default_probe_preserves_falsey_args() -> None:
    result = default_probe(torch.nn.Identity(), {"args": 0})
    assert result["output"]["values"] == [0.0]

    false_result = default_probe(torch.nn.Identity(), {"args": False})
    assert false_result["output"] is False


def test_default_probe_accepts_non_mapping_payloads() -> None:
    result = default_probe(torch.nn.Identity(), [1, 2])
    assert result["output"]["values"] == [1.0, 2.0]


def test_default_probe_validates_payload_options() -> None:
    module = torch.nn.Identity()
    with pytest.raises(ValueError, match="kwargs"):
        default_probe(module, {"input": [1], "kwargs": 5})
    with pytest.raises(ValueError, match="max_capture_tensors"):
        default_probe(module, {"input": [1], "max_capture_tensors": "2"})
    with pytest.raises(ValueError, match="capture options"):
        default_probe(module, {"input": [1], "hidden_states": 5})
    with pytest.raises(ValueError, match="unknown dtype"):
        default_probe(
            module, {"args": [{"data": [1], "dtype": "definitely-not-a-dtype"}]}
        )


def test_default_probe_explicit_device_and_nested_mapping() -> None:
    result = default_probe(
        torch.nn.Identity(),
        {
            "args": [
                {"data": [[1.0, 2.0]], "dtype": "float", "device": "cpu"}
            ]
        },
    )
    assert result["output"]["shape"] == [1, 2]
    assert result["output"]["dtype"] == "float32"


def test_default_probe_training_flag_is_temporary() -> None:
    module = torch.nn.Identity()
    module.eval()
    result = default_probe(module, {"args": [1.0], "training": True})
    assert result["training"] is True
    assert module.training is False


def test_default_probe_restores_state_after_forward_error() -> None:
    class Exploding(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("boom")

    module = Exploding()
    module.train()
    with pytest.raises(RuntimeError, match="boom"):
        default_probe(module, {"input": [[1.0, 2.0]], "hidden_states": True})
    assert module.training is True
    assert len(module._forward_hooks) == 0
    assert len(module.linear._forward_hooks) == 0


def test_default_probe_summarizes_mapping_dataclass_and_sequence_outputs() -> None:
    class MappingOutput(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> dict:
            return {"value": value, "label": "mapped"}

    mapped = default_probe(MappingOutput(), {"input": [1.0, 2.0]})
    assert mapped["output"]["value"]["shape"] == [2]
    assert mapped["output"]["label"] == "mapped"

    @dataclass
    class Output:
        value: torch.Tensor
        missing: torch.Tensor | None = None

    class DataclassOutput(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> Output:
            return Output(value)

    dataclass_result = default_probe(DataclassOutput(), {"input": [1.0]})
    assert dataclass_result["output"]["value"]["shape"] == [1]
    assert "missing" not in dataclass_result["output"]

    class SequenceOutput(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> list:
            return [value, value * 2]

    sequence_result = default_probe(SequenceOutput(), {"args": [1.0]})
    assert sequence_result["output"][0]["values"] == [1.0]
    assert sequence_result["output"][1]["values"] == [2.0]


def test_default_probe_hidden_capture_limit_marks_truncation() -> None:
    module = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU())
    result = default_probe(
        module,
        {
            "input": [[1.0, 2.0, 3.0]],
            "hidden_states": True,
            "max_capture_tensors": 1,
        },
    )
    assert len(result["hidden_states"]) == 1
    assert result["capture"]["hidden_truncated"] is True


def test_default_probe_hidden_state_alias_kwargs() -> None:
    result = default_probe(
        FakeTransformer(),
        {
            "input": [[1.0, 2.0, 3.0, 4.0]],
            "output_hidden_states": True,
            "output_attentions": True,
        },
    )
    assert result["last_hidden_state"]["shape"] == [1, 4]
    assert result["attentions"]["attentions[0]"]["shape"] == [1, 2, 4, 4]


def test_default_probe_option_keys_are_not_forwarded_as_kwargs() -> None:
    class EchoKwargs(torch.nn.Module):
        def forward(self, **kwargs) -> dict:
            return {key: torch.as_tensor(value, dtype=torch.float32) for key, value in kwargs.items()}

    result = default_probe(
        EchoKwargs(),
        {"alpha": 1.0, "training": False, "hidden_states": False},
    )
    assert set(result["output"]) == {"alpha"}
    assert result["output"]["alpha"]["values"] == [1.0]
