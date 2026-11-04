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
