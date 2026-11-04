"""Full-screen prompt_toolkit interface for the harness."""

from __future__ import annotations

import json
import shlex
import threading
from typing import Any

from prompt_toolkit import Application
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.data_structures import Point
from prompt_toolkit.filters import has_completions
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Dimension, HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.margins import ScrollbarMargin
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import TextArea

from .utils import json_dumps

HELP_TEXT = """\
Commands:
  /help                 show this help
  /status               harness, model, revision, probe and token status
  /tensors [prefix]     list visible tensors
  /revisions            list weight revisions
  /diff                 compare current weights with their original values
  /undo [n]             roll back the last n revisions (default 1)
  /reset                clear conversation history
  /clear                clear the transcript
  /save <path>          save the module state_dict to a file
  /quit                 exit

Editing is automatic and undoable. Ask the agent to inspect, explain,
edit, or probe the module in natural language.
"""

STYLE = Style.from_dict(
    {
        "transcript": "",
        "user": "bold #5fafff",
        "assistant": "#d7d7d7",
        "tool": "#87d787",
        "error": "bold #ff5f5f",
        "system": "#af87ff",
        "dim": "#808080",
        "status": "reverse",
        "prompt": "bold #5fafff",
    }
)


class HarnessTUI:
    """Interactive full-screen terminal UI."""

    def __init__(self, harness: Any) -> None:
        self.harness = harness
        self._fragments: list[tuple[str, str]] = []
        self._last_line = 0
        self._busy = False
        self._streaming_active = False
        self._app: Application[None] | None = None

        self._input = TextArea(
            multiline=True,
            prompt=FormattedText([("class:prompt", "you> ")]),
            height=Dimension(min=1, max=8),
            history=InMemoryHistory(),
            completer=WordCompleter(
                [
                    "/help",
                    "/status",
                    "/tensors",
                    "/revisions",
                    "/diff",
                    "/undo",
                    "/reset",
                    "/clear",
                    "/save",
                    "/quit",
                ],
                ignore_case=True,
                sentence=True,
            ),
            complete_while_typing=False,
            style="class:input",
        )
        self._transcript_control = FormattedTextControl(
            text=lambda: self._fragments,
            focusable=False,
            show_cursor=False,
            get_cursor_position=lambda: Point(self._last_line, 0),
        )
        self._transcript_window = Window(
            content=self._transcript_control,
            wrap_lines=True,
            right_margins=[ScrollbarMargin(display_arrows=True)],
            style="class:transcript",
        )
        self._status_control = FormattedTextControl(text=self._status_text)
        status_window = Window(
            content=self._status_control,
            height=1,
            style="class:status",
        )
        self._layout = Layout(
            HSplit([self._transcript_window, status_window, self._input]),
            focused_element=self._input,
        )
        self._key_bindings = self._make_key_bindings()

    def _make_key_bindings(self) -> KeyBindings:
        bindings = KeyBindings()

        @bindings.add("enter", filter=~has_completions)
        def _submit(event: Any) -> None:
            if self._busy:
                self._add("class:dim", "A run is already in progress.")
                return
            text = self._input.text.strip()
            if not text:
                return
            self._input.text = ""
            self._submit_text(text)

        @bindings.add("c-j")
        @bindings.add("escape", "enter")
        def _newline(event: Any) -> None:
            event.current_buffer.insert_text("\n")

        @bindings.add("c-c")
        def _cancel_or_clear(event: Any) -> None:
            if self._busy:
                self._add("class:dim", "Cannot cancel an in-flight request.")
            elif self._input.text:
                self._input.text = ""
            else:
                self._add("class:dim", "Press Ctrl+D or /quit to exit.")

        @bindings.add("c-d")
        def _quit(event: Any) -> None:
            if self._app is not None:
                self._app.exit()

        @bindings.add("c-l")
        def _clear(event: Any) -> None:
            self._clear_transcript()

        @bindings.add("pageup")
        def _scroll_up(event: Any) -> None:
            self._transcript_window.vertical_scroll = max(
                0, self._transcript_window.vertical_scroll - 10
            )

        @bindings.add("pagedown")
        def _scroll_down(event: Any) -> None:
            self._transcript_window.vertical_scroll += 10

        return bindings

    def run(self, *, input: Any = None, output: Any = None) -> None:
        self._app = Application(
            layout=self._layout,
            key_bindings=self._key_bindings,
            style=STYLE,
            full_screen=True,
            mouse_support=False,
            input=input,
            output=output,
        )
        self._add(
            "class:system",
            "Weighter — weight-space programming harness. Type /help for commands.",
        )
        self._app.run()

    def _add(self, style: str, text: str, *, newline: bool = True) -> None:
        text = str(text).replace("\r\n", "\n")
        if newline and not text.endswith("\n"):
            text += "\n"
        if text:
            self._fragments.append((style, text))
            self._last_line += text.count("\n")
        self._invalidate()

    def _clear_transcript(self) -> None:
        self._fragments.clear()
        self._last_line = 0
        self._add("class:system", "Transcript cleared. Type /help for commands.")

    def _invalidate(self) -> None:
        if self._app is not None:
            self._app.invalidate()

    def _status_text(self) -> FormattedText:
        status = self.harness.status()
        budget = status["probe_budget"]
        if not status.get("probe_enabled", True):
            budget_text = "off"
        else:
            budget_text = "unlimited" if budget is None else str(budget)
        state = "running" if self._busy else "ready"
        tokens = status["usage"]["total_tokens"]
        text = (
            f" {status['llm']} | {status['module_type']} | "
            f"{status['parameters']:,} params | rev {status['revisions']} | "
            f"probe {budget_text} | tokens {tokens} | {state} "
        )
        return FormattedText([("class:status", text)])

    def _submit_text(self, text: str) -> None:
        self._add("class:user", f"you> {text}")
        if text.startswith("/"):
            self._handle_command(text)
            return
        self._busy = True
        self._streaming_active = False
        self._invalidate()
        thread = threading.Thread(target=self._run_prompt, args=(text,), daemon=True)
        thread.start()

    def _run_prompt(self, text: str) -> None:
        try:
            self.harness.prompt(text, on_event=self._post_event, stream=True)
        except Exception as exc:  # pragma: no cover - exercised through TUI
            self._post_event(
                {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
            )
        finally:
            self._post_event({"type": "done"})

    def _post_event(self, event: dict[str, Any]) -> None:
        loop = self._app.loop if self._app is not None else None
        if loop is None or not loop.is_running():
            self._handle_event(event)
        else:
            try:
                loop.call_soon_threadsafe(self._handle_event, event)
            except RuntimeError:
                pass

    def _handle_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "text_delta":
            if not self._streaming_active:
                self._add("class:assistant", "assistant> ", newline=False)
                self._streaming_active = True
            self._add("class:assistant", event.get("text", ""), newline=False)
        elif event_type == "assistant_message":
            if self._streaming_active:
                self._add("class:assistant", "", newline=True)
                self._streaming_active = False
            else:
                content = event.get("content", "")
                if content:
                    self._add("class:assistant", f"assistant> {content}")
        elif event_type == "tool_start":
            arguments = str(event.get("arguments", ""))
            if len(arguments) > 300:
                arguments = arguments[:297] + "..."
            self._add("class:tool", f"tool> {event.get('name')} {arguments}")
        elif event_type == "tool_result":
            self._add("class:tool", self._format_tool_result(event))
        elif event_type == "error":
            self._add("class:error", f"error> {event.get('message', 'unknown error')}")
        elif event_type == "done":
            self._busy = False
            self._streaming_active = False
            self._invalidate()
            self._update_status()
        self._update_status()

    def _format_tool_result(self, event: dict[str, Any]) -> str:
        name = event.get("name", "tool")
        result = event.get("result") or {}
        duration = float(event.get("duration", 0.0))
        if not result.get("ok", False):
            error = result.get("error", "unknown error")
            return f"tool< {name} failed in {duration:.2f}s: {error}"
        data = result.get("data") or {}
        detail = ""
        if isinstance(data, dict):
            revision = data.get("revision")
            if isinstance(revision, dict):
                detail = (
                    f" revision={revision.get('id')} "
                    f"tensors={', '.join(revision.get('tensors', []))}"
                )
            elif "revision" in data:
                detail = f" revision={data.get('revision')}"
            elif "tensors" in data and isinstance(data["tensors"], list):
                detail = f" {len(data['tensors'])} tensors"
            elif "values" in data:
                detail = f" {data.get('returned', '?')} values"
            elif "result" in data:
                detail = " probe result"
            else:
                compact = json_dumps(data)
                if len(compact) > 260:
                    compact = compact[:257] + "..."
                detail = f" {compact}"
        return f"tool< {name} ok in {duration:.2f}s{detail}"

    def _handle_command(self, text: str) -> None:
        try:
            parts = shlex.split(text)
        except ValueError as exc:
            self._add("class:error", f"error> {exc}")
            return
        if not parts:
            return
        command = parts[0].lower()
        try:
            if command in {"/help", "/h"}:
                self._add("class:system", HELP_TEXT)
            elif command == "/status":
                self._add("class:system", json.dumps(self.harness.status(), indent=2))
            elif command == "/tensors":
                prefix = parts[1] if len(parts) > 1 else None
                refs = self.harness.store.list_refs(prefix=prefix)[:200]
                lines = [
                    f"{ref.name}  shape={list(ref.tensor.shape)}  "
                    f"dtype={str(ref.tensor.dtype).replace('torch.', '')}"
                    for ref in refs
                ]
                self._add("class:system", "\n".join(lines) if lines else "(no tensors)")
            elif command == "/revisions":
                revisions = self.harness.revisions()
                self._add(
                    "class:system",
                    json.dumps(revisions, indent=2) if revisions else "(no revisions)",
                )
            elif command == "/diff":
                diff = self.harness.diff()
                self._add(
                    "class:system",
                    json.dumps(diff, indent=2) if diff else "(no edits)",
                )
            elif command == "/undo":
                if self._busy:
                    self._add("class:dim", "Wait for the current run to finish.")
                    return
                n = int(parts[1]) if len(parts) > 1 else 1
                undone = self.harness.undo(n)
                self._add(
                    "class:system",
                    f"Undid revisions: {', '.join(item['id'] for item in undone)}",
                )
            elif command == "/reset":
                self.harness.reset()
                self._add("class:system", "Conversation history cleared.")
            elif command == "/clear":
                self._clear_transcript()
            elif command == "/save":
                if self._busy:
                    self._add("class:dim", "Wait for the current run to finish.")
                    return
                if len(parts) < 2:
                    self._add("class:error", "usage: /save <path>")
                else:
                    path = self.harness.save_module(" ".join(parts[1:]))
                    self._add("class:system", f"Saved state_dict to {path}")
            elif command in {"/quit", "/exit"}:
                if self._app is not None:
                    self._app.exit()
            else:
                self._add("class:error", f"unknown command {command!r}; try /help")
        except Exception as exc:
            self._add("class:error", f"error> {type(exc).__name__}: {exc}")
        self._update_status()

    def _update_status(self) -> None:
        self._invalidate()


def run_tui(harness: Any) -> None:
    HarnessTUI(harness).run()
