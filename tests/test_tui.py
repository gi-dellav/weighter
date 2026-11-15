from __future__ import annotations

import threading
import time

import torch
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from weighter import Harness
from weighter.tui import HarnessTUI


def _harness() -> Harness:
    return Harness(
        torch.nn.Linear(2, 2),
        llm="fake/model",
        completion_fn=lambda **kwargs: None,
    )


def test_tui_renders_events_and_commands() -> None:
    tui = HarnessTUI(_harness())
    tui._handle_event({"type": "assistant_message", "content": "hello"})
    tui._handle_command("/help")
    tui._handle_command("/tensors weight")
    tui._handle_command("/diff")
    tui._handle_command("/bogus")
    rendered = "".join(text for _, text in tui._fragments)
    assert "hello" in rendered
    assert "Commands:" in rendered
    assert "weight" in rendered
    assert "unknown command" in rendered


def test_tui_runs_and_quits() -> None:
    tui = HarnessTUI(_harness())
    with create_pipe_input() as pipe_input:
        def feed() -> None:
            time.sleep(0.1)
            pipe_input.send_text("/help\r")
            time.sleep(0.1)
            pipe_input.send_text("/quit\r")

        thread = threading.Thread(target=feed, daemon=True)
        thread.start()
        tui.run(input=pipe_input, output=DummyOutput())


def test_tui_model_and_run_commands() -> None:
    harness = _harness()
    tui = HarnessTUI(harness)

    tui._handle_command("/model fake/other")
    assert harness.config.llm == "fake/other"
    tui._handle_command("/model")
    tui._handle_command("/run 1 + 1")
    tui._handle_command("/run value = 41")
    tui._handle_command("/run value + 1")
    tui._handle_command("/run raise ValueError('boom')")
    tui._handle_command("/run")

    rendered = "".join(text for _, text in tui._fragments)
    assert "Model set to fake/other" in rendered
    assert "model: fake/other" in rendered
    assert "\n2\n" in rendered
    assert "42" in rendered
    assert "ValueError: boom" in rendered
    assert "usage: /run <python>" in rendered


def test_tui_transcript_follow_and_trimming() -> None:
    tui = HarnessTUI(_harness())
    tui.max_transcript_lines = 20
    tui.trim_chunk = 5

    tui._add("system", "first\n")
    assert tui._follow_bottom
    assert tui._cursor_line == tui._total_lines

    tui._page_up()
    assert not tui._follow_bottom
    anchor = tui._cursor_line
    tui._add("system", "while scrolled\n")
    assert tui._cursor_line == anchor

    tui._scroll_to_bottom()
    assert tui._follow_bottom
    assert tui._cursor_line == tui._total_lines

    for index in range(100):
        tui._add("system", f"line-{index}\n")

    rendered = "".join(text for _, text in tui._fragments)
    assert "older output trimmed" in rendered
    assert "line-99" in rendered
    assert "line-0\n" not in rendered
    assert tui._total_lines <= tui.max_transcript_lines
    assert tui._blocks[0].meta.get("trim_notice") is True
    assert tui._fragments == [
        fragment for block in tui._blocks for fragment in block.cached_fragments
    ]


def test_tui_format_tool_result_variants() -> None:
    tui = HarnessTUI(_harness())
    revision = tui._format_tool_result(
        {
            "type": "tool_result",
            "name": "apply_edits",
            "duration": 0.25,
            "result": {
                "ok": True,
                "data": {
                    "revision": {"id": "r2", "tensors": ["weight", "bias"]},
                    "diff": {"total_changed_elements": 3, "dirty_tensor_elements": 4},
                },
            },
        }
    )
    assert "revision=r2" in revision
    assert "tensors=weight, bias" in revision
    assert "changed=3/4" in revision

    scalar_revision = tui._format_tool_result(
        {
            "name": "apply_edits",
            "result": {"ok": True, "data": {"revision": "r1"}},
        }
    )
    assert "revision=r1" in scalar_revision

    tensors = tui._format_tool_result(
        {"name": "list_tensors", "result": {"ok": True, "data": {"tensors": [1, 2]}}}
    )
    assert "2 tensors" in tensors

    values = tui._format_tool_result(
        {
            "name": "read_tensor",
            "result": {"ok": True, "data": {"values": [1], "returned": 1}},
        }
    )
    assert "1 values" in values

    calculated = tui._format_tool_result(
        {"name": "calculate", "result": {"ok": True, "data": {"result": 3, "repr": "3"}}}
    )
    assert "= 3" in calculated

    probed = tui._format_tool_result(
        {"name": "run_probe", "result": {"ok": True, "data": {"result": {}}}}
    )
    assert "probe result" in probed

    summary = tui._format_tool_result(
        {
            "name": "weight_diff",
            "result": {
                "ok": True,
                "data": {
                    "summary": {
                        "total_changed_elements": 2,
                        "changed_tensors": 1,
                    }
                },
            },
        }
    )
    assert "changed=2 tensors=1" in summary

    compact = tui._format_tool_result(
        {"name": "custom", "result": {"ok": True, "data": {"value": "x" * 400}}}
    )
    assert "..." in compact

    failure = tui._format_tool_result(
        {"name": "read_tensor", "result": {"ok": False, "error": "boom"}}
    )
    assert "failed" in failure and "boom" in failure


def test_tui_streaming_events_and_error() -> None:
    tui = HarnessTUI(_harness())
    tui._busy = True
    tui._handle_event({"type": "text_delta", "text": "hello "})
    tui._handle_event({"type": "text_delta", "text": "world"})
    tui._handle_event({"type": "assistant_message", "content": "hello world"})
    tui._handle_event(
        {"type": "tool_start", "name": "read_tensor", "arguments": "x" * 400}
    )
    tui._handle_event(
        {
            "type": "tool_result",
            "name": "read_tensor",
            "duration": 0.1,
            "result": {"ok": True, "data": {"values": [1], "returned": 1}},
        }
    )
    tui._handle_event({"type": "error", "message": "boom"})
    tui._handle_event({"type": "done"})
    rendered = "".join(text for _, text in tui._fragments)
    assert "assistant> hello world" in rendered
    assert "tool> read_tensor xxx" in rendered
    assert "error> boom" in rendered
    assert tui._busy is False
    assert tui._streaming_active is False


def test_tui_execute_python_variants() -> None:
    tui = HarnessTUI(_harness())
    tui._execute_python("print('hello')")
    assert "hello" in "".join(text for _, text in tui._fragments)
    tui._execute_python("1 + 1")
    assert "\n2\n" in "".join(text for _, text in tui._fragments)
    tui._execute_python("value = 41")
    tui._execute_python("value + 1")
    assert "42" in "".join(text for _, text in tui._fragments)
    tui._execute_python("raise ValueError('boom')")
    assert "ValueError: boom" in "".join(text for _, text in tui._fragments)
    tui._execute_python("raise SystemExit(3)")
    assert "SystemExit: 3" in "".join(text for _, text in tui._fragments)
    tui._execute_python("")
    assert "usage: /run <python>" in "".join(text for _, text in tui._fragments)


def test_tui_status_revisions_undo_save_and_reset(tmp_path) -> None:
    harness = _harness()
    tui = HarnessTUI(harness)
    tui._handle_command("/status")
    tui._handle_command("/revisions")
    tui._handle_command("/diff")
    tui._handle_command("/undo")

    harness.store.apply([{"op": "fill", "name": "weight", "value": 0.0}])
    tui._handle_command("/revisions")
    tui._handle_command("/diff")
    tui._handle_command("/undo")
    tui._handle_command("/reset")
    target = tmp_path / "state.pt"
    tui._handle_command(f"/save {target}")

    rendered = "".join(text for _, text in tui._fragments)
    assert "fake/model" in rendered
    assert "(no revisions)" in rendered
    assert "(no edits)" in rendered
    assert "Undid revisions: r1" in rendered
    assert "Conversation history cleared." in rendered
    assert f"Saved state_dict to {target}" in rendered
    assert target.exists()


def test_tui_post_event_without_application_and_status_text() -> None:
    tui = HarnessTUI(_harness())
    tui._post_event({"type": "error", "message": "direct event"})
    rendered = "".join(text for _, text in tui._fragments)
    assert "direct event" in rendered
    status = tui._status_text()[0][1]
    assert "fake/model" in status
    assert "Linear" in status
    assert "ready" in status


def test_tui_page_down_and_scroll_to_bottom() -> None:
    tui = HarnessTUI(_harness())
    for index in range(30):
        tui._add("system", f"line-{index}\n")
    tui._page_up()
    assert tui._follow_bottom is False
    tui._page_down()
    assert tui._cursor_line > 0
    tui._scroll_to_bottom()
    assert tui._follow_bottom is True
    assert tui._cursor_line == tui._total_lines
