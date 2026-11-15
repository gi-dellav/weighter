from __future__ import annotations

import re

import pytest
import torch

from weighter.config import HarnessConfig, _compile_filter
from weighter.errors import ConfigurationError
from weighter.prompts import build_system_prompt
from weighter.types import EditProposal, PromptResult, ToolCallRecord, Usage
from weighter.weights import WeightStore


def test_config_uses_environment_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEIGHTER_LLM", "env/model")
    config = HarnessConfig()
    assert config.llm == "env/model"


def test_config_validates_options() -> None:
    invalid = [
        {"edit_policy": "sometimes"},
        {"probe_budget": -1},
        {"max_steps": 0},
        {"max_read_elements": 0},
        {"max_edit_elements": 0},
        {"max_undo_bytes": -1},
        {"max_history_messages": 0},
    ]
    for kwargs in invalid:
        with pytest.raises(ConfigurationError):
            HarnessConfig(llm="fake", **kwargs)


def test_config_copies_completion_kwargs() -> None:
    source = {"temperature": 0.1}
    config = HarnessConfig(llm="fake", completion_kwargs=source)
    source["temperature"] = 0.9
    assert config.completion_kwargs == {"temperature": 0.1}
    config.completion_kwargs["top_p"] = 0.5
    assert "top_p" not in source


def test_tensor_filter_compilation() -> None:
    assert _compile_filter(None) is None

    callable_filter = _compile_filter(lambda name: name.endswith("weight"))
    assert callable_filter is not None
    assert callable_filter("layer.weight")
    assert not callable_filter("layer.bias")

    string_filter = _compile_filter(r"layers\.\d+")
    assert string_filter is not None
    assert string_filter("layers.3.weight")
    assert not string_filter("encoder.weight")

    pattern = re.compile("bias")
    pattern_filter = _compile_filter(pattern)
    assert pattern_filter is not None
    assert pattern_filter("bias")
    assert not pattern_filter("weight")

    with pytest.raises(ConfigurationError):
        _compile_filter(5)  # type: ignore[arg-type]


def test_usage_from_none_dict_and_object() -> None:
    assert Usage.from_object(None) == Usage()

    from_dict = Usage.from_object(
        {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
    )
    assert from_dict == Usage(prompt_tokens=2, completion_tokens=3, total_tokens=5)

    class RawUsage:
        prompt_tokens = 4
        completion_tokens = 1

    from_object = Usage.from_object(RawUsage())
    assert from_object.prompt_tokens == 4
    assert from_object.completion_tokens == 1
    assert from_object.total_tokens == 5


def test_usage_add_accumulates() -> None:
    first = Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3)
    first.add(Usage(prompt_tokens=4, completion_tokens=5, total_tokens=9))
    assert first == Usage(prompt_tokens=5, completion_tokens=7, total_tokens=12)


def test_prompt_result_and_records() -> None:
    record = ToolCallRecord(
        id="c1",
        name="read_tensor",
        arguments={"name": "weight"},
        ok=True,
        result={"ok": True},
    )
    result = PromptResult(text="done", tool_calls=[record])
    assert str(result) == "done"
    assert result.messages == []
    assert result.stop_reason == "complete"
    proposal = EditProposal(ops=[], targets=["weight"], selection_elements=0)
    assert proposal.note is None


def test_system_prompt_states() -> None:
    store = WeightStore(torch.nn.Linear(2, 2))

    disabled = build_system_prompt(
        store,
        HarnessConfig(llm="fake", probe=None),
        tool_names=["read_tensor"],
        probe_remaining=None,
    )
    assert '"probe": "disabled"' in disabled
    assert "read_tensor" in disabled

    unlimited = build_system_prompt(
        store,
        HarnessConfig(llm="fake", probe=lambda module, payload: {}),
        tool_names=["run_probe"],
        probe_remaining=None,
    )
    assert "enabled, unlimited budget" in unlimited

    budgeted = build_system_prompt(
        store,
        HarnessConfig(
            llm="fake",
            probe=lambda module, payload: {},
            system_prompt="Use care.",
        ),
        tool_names=["run_probe"],
        probe_remaining=3,
    )
    assert "enabled, 3 calls remaining" in budgeted
    assert "USER SYSTEM PROMPT\nUse care." in budgeted
    assert '"revisions": 0' in budgeted
