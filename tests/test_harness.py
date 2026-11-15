from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from weighter import ConfigurationError, Harness, LLMError, Usage


def tool_call(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def response(content: str | None = None, tool_calls: list[dict] | None = None) -> dict:
    message: dict = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "choices": [{"message": message}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


class ScriptedCompletion:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("no scripted completion left")
        return self.responses.pop(0)


@pytest.fixture()
def module() -> torch.nn.Linear:
    return torch.nn.Linear(3, 2)


def test_prompt_tool_loop_and_edit(module: torch.nn.Linear) -> None:
    fake = ScriptedCompletion(
        [
            response(tool_calls=[tool_call("c1", "list_tensors", {"prefix": "weight"})]),
            response(
                tool_calls=[
                    tool_call(
                        "c2",
                        "apply_edits",
                        {
                            "edits": [
                                {
                                    "op": "set",
                                    "name": "weight",
                                    "index": [0, 0],
                                    "values": 3.0,
                                }
                            ],
                            "note": "test edit",
                        },
                    )
                ]
            ),
            response("Edited the weight."),
        ]
    )
    harness = Harness(module, llm="fake/model", completion_fn=fake)
    result = harness.prompt("change one weight")
    assert result.text == "Edited the weight."
    assert result.stop_reason == "complete"
    assert [record.name for record in result.tool_calls] == ["list_tensors", "apply_edits"]
    assert all(record.ok for record in result.tool_calls)
    assert module.weight[0, 0] == 3.0
    assert result.revisions == ["r1"]
    assert result.usage.total_tokens == 45
    assert fake.calls[0]["tools"]
    assert harness.messages[0]["role"] == "user"
    assert harness.messages[-1]["role"] == "assistant"
    assert harness.revisions()[0]["note"] == "test edit"


def test_tool_errors_are_returned_to_model(module: torch.nn.Linear) -> None:
    fake = ScriptedCompletion(
        [
            response(tool_calls=[tool_call("c1", "read_tensor", {"name": "nope"})]),
            response("I could not find it."),
        ]
    )
    harness = Harness(module, llm="fake/model", completion_fn=fake)
    result = harness.prompt("read a missing tensor")
    assert result.tool_calls[0].ok is False
    tool_message = [m for m in harness.messages if m["role"] == "tool"][0]
    assert "unknown tensor" in tool_message["content"]


def test_read_only_mode_hides_edit_tools(module: torch.nn.Linear) -> None:
    fake = ScriptedCompletion(
        [
            response(
                tool_calls=[
                    tool_call(
                        "c1",
                        "apply_edits",
                        {"edits": [{"op": "fill", "name": "weight", "value": 0.0}]},
                    )
                ]
            ),
            response("No edit."),
        ]
    )
    harness = Harness(module, llm="fake/model", edit_policy="read_only", completion_fn=fake)
    assert "apply_edits" not in harness._tools
    assert "undo_edits" not in harness._tools
    result = harness.prompt("try to edit")
    assert result.tool_calls[0].ok is False
    assert "unknown tool" in result.tool_calls[0].result["error"]


def test_max_steps(module: torch.nn.Linear) -> None:
    fake = ScriptedCompletion(
        [
            response(tool_calls=[tool_call("c1", "list_tensors", {})]),
            response(tool_calls=[tool_call("c2", "list_tensors", {})]),
        ]
    )
    harness = Harness(module, llm="fake/model", max_steps=2, completion_fn=fake)
    result = harness.prompt("loop")
    assert result.stop_reason == "max_steps"
    assert "max_steps=2" in result.text


def test_custom_probe_and_budget(module: torch.nn.Linear) -> None:
    seen: list[dict] = []

    def my_probe(mod, payload):
        seen.append(payload)
        return {"echo": payload, "training": mod.training}

    fake = ScriptedCompletion(
        [
            response(
                tool_calls=[
                    tool_call("c1", "run_probe", {"payload": {"input": [[1.0, 2.0, 3.0]]}})
                ]
            ),
            response("Probe done."),
            response(
                tool_calls=[
                    tool_call("c2", "run_probe", {"payload": {"input": [[1.0, 2.0, 3.0]]}})
                ]
            ),
            response("Second probe failed."),
        ]
    )
    harness = Harness(
        module,
        llm="fake/model",
        probe=my_probe,
        probe_budget=1,
        completion_fn=fake,
    )
    result = harness.prompt("probe")
    assert result.tool_calls[0].ok is True
    assert result.tool_calls[0].result["data"]["remaining_budget"] == 0
    assert seen[0]["input"] == [[1.0, 2.0, 3.0]]
    second = harness.prompt("probe again")
    assert second.tool_calls[0].ok is False
    assert "budget exhausted" in second.tool_calls[0].result["error"]


def test_default_probe_is_available(module: torch.nn.Linear) -> None:
    fake = ScriptedCompletion(
        [
            response(
                tool_calls=[
                    tool_call("c1", "run_probe", {"payload": {"input": [[1.0, 2.0, 3.0]]}})
                ]
            ),
            response("Done."),
        ]
    )
    harness = Harness(module, llm="fake/model", completion_fn=fake)
    result = harness.prompt("probe")
    assert result.tool_calls[0].ok is True
    output = result.tool_calls[0].result["data"]["result"]["output"]
    assert output["shape"] == [1, 2]


def test_probe_can_be_disabled(module: torch.nn.Linear) -> None:
    harness = Harness(
        module,
        llm="fake/model",
        probe=False,
        completion_fn=ScriptedCompletion([]),
    )
    assert "run_probe" not in harness._tools


def test_confirm_rejects_edit(module: torch.nn.Linear) -> None:
    original = module.weight.detach().clone()
    fake = ScriptedCompletion(
        [
            response(
                tool_calls=[
                    tool_call(
                        "c1",
                        "apply_edits",
                        {"edits": [{"op": "fill", "name": "weight", "value": 0.0}]},
                    )
                ]
            ),
            response("Rejected."),
        ]
    )
    harness = Harness(
        module,
        llm="fake/model",
        edit_policy="confirm",
        confirm_callback=lambda proposal: False,
        completion_fn=fake,
    )
    result = harness.prompt("edit")
    assert result.tool_calls[0].ok is False
    assert "rejected" in result.tool_calls[0].result["error"]
    assert torch.equal(module.weight, original)


def test_streaming_text_events(module: torch.nn.Linear) -> None:
    chunks = [
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
        {"choices": [{"delta": {"content": "!"}}]},
    ]
    fake = ScriptedCompletion([iter(chunks)])
    harness = Harness(module, llm="fake/model", completion_fn=fake)
    events = []
    result = harness.prompt("say hi", stream=True, on_event=events.append)
    assert result.text == "Hello!"
    deltas = "".join(e["text"] for e in events if e["type"] == "text_delta")
    assert deltas == "Hello!"
    assert any(e["type"] == "assistant_message" for e in events)


def test_streaming_tool_call_assembly(module: torch.nn.Linear) -> None:
    streamed = [
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "function": {"name": "read_tensor", "arguments": '{"name":'},
                            }
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": '"weight"}'},
                            }
                        ]
                    }
                }
            ]
        },
    ]
    fake = ScriptedCompletion([iter(streamed), response("Done.")])
    harness = Harness(module, llm="fake/model", completion_fn=fake)
    result = harness.prompt("inspect", stream=True)
    assert result.tool_calls[0].name == "read_tensor"
    assert result.tool_calls[0].ok is True
    assert result.tool_calls[0].arguments == {"name": "weight"}


def test_new_tools_are_registered(module: torch.nn.Linear) -> None:
    harness = Harness(module, llm="fake/model", completion_fn=ScriptedCompletion([]))
    assert {"calculate", "weight_diff", "ablate_units"} <= set(harness._tools)

    read_only = Harness(
        module,
        llm="fake/model",
        edit_policy="read_only",
        completion_fn=ScriptedCompletion([]),
    )
    assert {"calculate", "weight_diff"} <= set(read_only._tools)
    assert "ablate_units" not in read_only._tools


def test_calculate_tool_call(module: torch.nn.Linear) -> None:
    fake = ScriptedCompletion(
        [
            response(
                tool_calls=[
                    tool_call("c1", "calculate", {"expression": "167 * 28 + 1.9"})
                ]
            ),
            response("Calculated."),
        ]
    )
    harness = Harness(module, llm="fake/model", completion_fn=fake)
    result = harness.prompt("calculate")
    assert result.tool_calls[0].ok is True
    assert result.tool_calls[0].result["data"]["result"] == 4677.9


def test_ablate_tool_call(module: torch.nn.Linear) -> None:
    fake = ScriptedCompletion(
        [
            response(
                tool_calls=[
                    tool_call("c1", "ablate_units", {"module_path": "", "units": [0]})
                ]
            ),
            response("Ablated."),
        ]
    )
    harness = Harness(module, llm="fake/model", completion_fn=fake)
    result = harness.prompt("ablate unit 0")
    assert result.tool_calls[0].ok is True
    assert torch.all(module.weight[0] == 0)
    assert result.tool_calls[0].result["data"]["diff"]["total_changed_elements"] > 0


def test_status_reset_and_revert(module: torch.nn.Linear) -> None:
    fake = ScriptedCompletion([])
    harness = Harness(module, llm="fake/model", completion_fn=fake)
    status = harness.status()
    assert status["llm"] == "fake/model"
    assert status["module_type"] == "Linear"
    assert status["parameters"] == 8
    assert status["edit_policy"] == "auto"
    assert status["tools"] == list(harness.tools)
    assert status["probe_enabled"] is True

    original = module.weight.detach().clone()
    harness.store.apply([{"op": "fill", "name": "weight", "value": 0.0}])
    harness.reset()
    assert harness.messages == []
    assert not torch.equal(module.weight, original)
    harness.reset(revert_edits=True)
    assert torch.equal(module.weight, original)
    assert harness.revisions() == []


def test_undo_revert_and_diff_summary(module: torch.nn.Linear) -> None:
    harness = Harness(
        module, llm="fake/model", completion_fn=ScriptedCompletion([])
    )
    harness.store.apply(
        [{"op": "set", "name": "weight", "index": [0, 0], "values": 1.0}]
    )
    harness.store.apply(
        [{"op": "set", "name": "bias", "index": [0], "values": 1.0}]
    )
    summary = harness.diff_summary()
    assert summary["total_changed_elements"] == 2

    undone = harness.undo()
    assert [item["id"] for item in undone] == ["r2"]
    assert len(harness.revisions()) == 1
    rest = harness.revert()
    assert [item["id"] for item in rest] == ["r1"]
    assert harness.diff() == []


def test_save_module(tmp_path) -> None:
    module = torch.nn.Linear(2, 2)
    harness = Harness(
        module,
        llm="fake/model",
        completion_fn=ScriptedCompletion([]),
    )
    target = tmp_path / "nested" / "state.pt"
    saved = harness.save_module(target)
    assert saved == str(target)
    state = torch.load(target, weights_only=True)
    assert set(state) == set(module.state_dict())


def test_constructor_validation(module: torch.nn.Linear) -> None:
    with pytest.raises(TypeError, match="torch.nn.Module"):
        Harness(object(), llm="fake/model")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="LLM is required"):
        Harness(module)
    with pytest.raises(ConfigurationError, match="probe must be"):
        Harness(
            module,
            llm="fake/model",
            probe="not-callable",  # type: ignore[arg-type]
            completion_fn=ScriptedCompletion([]),
        )
    with pytest.raises(ConfigurationError, match="tool_extras"):
        Harness(
            module,
            llm="fake/model",
            tool_extras=[1],  # type: ignore[arg-type]
            completion_fn=ScriptedCompletion([]),
        )


def test_completion_kwargs_are_forwarded(module: torch.nn.Linear) -> None:
    fake = ScriptedCompletion([response("done")])
    harness = Harness(
        module,
        llm="fake/model",
        api_key="key",
        api_base="https://example.invalid",
        temperature=0.2,
        request_timeout=3.0,
        completion_kwargs={"top_p": 0.5, "model": "ignored"},
        completion_fn=fake,
    )
    harness.prompt("hi")
    call = fake.calls[0]
    assert call["model"] == "fake/model"
    assert call["top_p"] == 0.5
    assert call["temperature"] == 0.2
    assert call["api_key"] == "key"
    assert call["base_url"] == "https://example.invalid"
    assert call["timeout"] == 3.0
    assert call["tool_choice"] == "auto"
    assert call["tools"]


def test_prompt_rejects_empty_messages(module: torch.nn.Linear) -> None:
    harness = Harness(
        module, llm="fake/model", completion_fn=ScriptedCompletion([])
    )
    for message in ["", "   ", None]:
        with pytest.raises(ValueError, match="non-empty"):
            harness.prompt(message)  # type: ignore[arg-type]


def test_unknown_tools_and_invalid_arguments(module: torch.nn.Linear) -> None:
    calls = [
        tool_call("c1", "unknown_tool", {}),
        {
            "id": "c2",
            "type": "function",
            "function": {"name": "read_tensor", "arguments": "not-json"},
        },
        {
            "id": "c3",
            "type": "function",
            "function": {"name": "read_tensor", "arguments": "[]"},
        },
        tool_call("c4", "list_tensors", {"limit": 1, "bogus": True}),
    ]
    fake = ScriptedCompletion([response(tool_calls=calls), response("done")])
    harness = Harness(module, llm="fake/model", completion_fn=fake)
    result = harness.prompt("call tools")
    assert [record.ok for record in result.tool_calls] == [False, False, False, False]
    assert "unknown tool" in result.tool_calls[0].result["error"]
    assert result.tool_calls[1].result["error_type"] == "JSONDecodeError"
    assert "JSON object" in result.tool_calls[2].result["error"]
    assert "Extra inputs" in result.tool_calls[3].result["error"]


def test_events_are_emitted_and_sink_errors_are_swallowed(
    module: torch.nn.Linear,
) -> None:
    fake = ScriptedCompletion(
        [
            response(tool_calls=[tool_call("c1", "list_tensors", {})]),
            response("done"),
        ]
    )
    harness = Harness(module, llm="fake/model", completion_fn=fake)
    events: list[dict] = []

    def sink(event: dict) -> None:
        events.append(event)
        raise RuntimeError("sink failed")

    result = harness.prompt("hi", on_event=sink)
    assert result.text == "done"
    assert [event["type"] for event in events] == [
        "assistant_message",
        "tool_start",
        "tool_result",
        "assistant_message",
    ]


def test_streaming_usage_reasoning_and_multiple_tool_calls(
    module: torch.nn.Linear,
) -> None:
    chunks = [
        {
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            "choices": [
                {
                    "delta": {
                        "reasoning_content": "thinking ",
                        "tool_calls": [
                            {
                                "index": 1,
                                "function": {
                                    "name": "list_",
                                    "arguments": '{"lim',
                                },
                            },
                            {
                                "index": 0,
                                "id": "c0",
                                "function": {
                                    "name": "list_tensors",
                                    "arguments": {"prefix": "weight"},
                                },
                            },
                        ],
                    }
                }
            ],
        },
        {
            "choices": [
                {
                    "delta": {
                        "content": [{"type": "text", "text": "hello"}],
                        "tool_calls": [
                            {
                                "index": 1,
                                "function": {"name": "tensors", "arguments": 'it":1}'},
                            }
                        ],
                    }
                }
            ]
        },
    ]
    fake = ScriptedCompletion(
        [iter(chunks), {"choices": [{"message": {"content": "done"}}]}]
    )
    harness = Harness(module, llm="fake/model", completion_fn=fake)
    result = harness.prompt("inspect", stream=True)
    assert result.text == "done"
    assert result.usage == Usage(
        prompt_tokens=1, completion_tokens=2, total_tokens=3
    )
    assert harness.messages[1]["reasoning_content"] == "thinking "
    assert [record.id for record in result.tool_calls] == ["c0", "call_1"]
    assert [record.name for record in result.tool_calls] == [
        "list_tensors",
        "list_tensors",
    ]
    assert result.tool_calls[0].arguments == {"prefix": "weight"}
    assert result.tool_calls[1].arguments == {"limit": 1}


def test_non_stream_missing_choices_raises_llm_error(
    module: torch.nn.Linear,
) -> None:
    harness = Harness(
        module, llm="fake/model", completion_fn=lambda **kwargs: {}
    )
    with pytest.raises(LLMError, match="no choices"):
        harness.prompt("hi")


def test_retryable_errors_are_retried(module: torch.nn.Linear, monkeypatch) -> None:
    class RateLimitError(Exception):
        pass

    attempts: list[int] = []
    sleeps: list[float] = []

    def fake(**kwargs):
        attempts.append(1)
        if len(attempts) < 3:
            raise RateLimitError("slow down")
        return response("done")

    monkeypatch.setattr("weighter.harness.time.sleep", sleeps.append)
    harness = Harness(module, llm="fake/model", completion_fn=fake)
    assert harness.prompt("hi").text == "done"
    assert len(attempts) == 3
    assert sleeps == [1, 2]


def test_non_retryable_errors_are_wrapped(module: torch.nn.Linear) -> None:
    def fake(**kwargs):
        raise ValueError("bad request")

    harness = Harness(module, llm="fake/model", completion_fn=fake)
    with pytest.raises(LLMError, match="ValueError: bad request"):
        harness.prompt("hi")


def test_normalizes_object_style_responses(module: torch.nn.Linear) -> None:
    class Message:
        role = "assistant"
        content = "working"
        tool_calls = [
            SimpleNamespace(
                id="c1",
                function=SimpleNamespace(
                    name="read_tensor", arguments='{"name": "weight"}'
                ),
            )
        ]

    class Response:
        choices = [SimpleNamespace(message=Message())]
        usage = SimpleNamespace(prompt_tokens=3, completion_tokens=4)

    fake = ScriptedCompletion([Response(), response("done")])
    harness = Harness(module, llm="fake/model", completion_fn=fake)
    result = harness.prompt("hi")
    assert result.tool_calls[0].arguments == {"name": "weight"}
    assert result.usage.total_tokens == 22
    assert harness.messages[1]["content"] == "working"


def test_tool_output_truncation(module: torch.nn.Linear) -> None:
    harness = Harness(
        module,
        llm="fake/model",
        max_tool_output_chars=300,
        completion_fn=ScriptedCompletion([]),
    )
    formatted = harness._format_tool_result({"ok": True, "data": "x" * 500})
    parsed = json.loads(formatted)
    assert parsed["ok"] is True
    assert parsed["truncated"] is True
    assert 0 < len(parsed["preview"]) <= 100


def test_max_history_messages_and_leading_tool_trimming(
    module: torch.nn.Linear,
) -> None:
    fake = ScriptedCompletion(
        [
            response(tool_calls=[tool_call("c1", "list_tensors", {})]),
            response(tool_calls=[tool_call("c2", "list_tensors", {})]),
            response("done"),
        ]
    )
    harness = Harness(
        module,
        llm="fake/model",
        max_history_messages=2,
        completion_fn=fake,
    )
    harness.prompt("hi")
    assert [len(call["messages"]) for call in fake.calls] == [2, 3, 3]
    for call in fake.calls[1:]:
        assert call["messages"][0]["role"] == "system"
        assert call["messages"][1]["role"] != "tool"


def test_include_buffers_and_tensor_filter() -> None:
    module = torch.nn.BatchNorm1d(2)
    harness = Harness(
        module,
        llm="fake/model",
        include_buffers=True,
        completion_fn=ScriptedCompletion([]),
    )
    assert "running_mean" in harness.store.names()

    filtered = Harness(
        module,
        llm="fake/model",
        include_buffers=True,
        tensor_filter=lambda name: name == "weight",
        completion_fn=ScriptedCompletion([]),
    )
    assert filtered.store.names() == ["weight"]
