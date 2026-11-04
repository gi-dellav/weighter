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
