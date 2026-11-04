from __future__ import annotations

import json

import pytest
import torch

from weighter import Harness


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
