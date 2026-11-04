"""Full-screen prompt_toolkit interface for the harness."""

from __future__ import annotations

import ast
import contextlib
import io
import json
import shlex
import threading
import traceback
from dataclasses import dataclass, field
from typing import Any

import torch
from prompt_toolkit import Application
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.data_structures import Point
from prompt_toolkit.filters import Condition, has_completions
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
  /model [model]        show or change the LiteLLM model id
  /tensors [prefix]     list visible tensors
  /revisions            list weight revisions
  /diff [baseline]      compare weights to initial, last, or a revision id
  /undo [n]             roll back the last n revisions (default 1)
  /reset                clear conversation history
  /clear                clear the transcript
  /run <python>         run Python locally in the TUI process
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
        "diff": "#ffd75f",
        "status": "reverse",
        "prompt": "bold #5fafff",
    }
)

STYLE_FOR_KIND = {
    "user": "class:user",
    "assistant": "class:assistant",
    "tool": "class:tool",
    "error": "class:error",
    "system": "class:system",
    "dim": "class:dim",
    "diff": "class:diff",
}
KIND_FOR_STYLE = {style: kind for kind, style in STYLE_FOR_KIND.items()}


@dataclass(slots=True)
class TranscriptBlock:
    kind: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)
    cached_fragments: list[tuple[str, str]] = field(default_factory=list)
    line_count: int = 0


class HarnessTUI:
    """Interactive full-screen terminal UI."""

    def __init__(self, harness: Any) -> None:
        self.harness = harness
        self.max_transcript_lines = 10_000
        self.trim_chunk = 1_000
        self.scroll_step = 10
        self._blocks: list[TranscriptBlock] = []
        self._fragments: list[tuple[str, str]] = []
        self._total_lines = 0
        self._cursor_line = 0
        self._follow_bottom = True
        self._busy = False
        self._streaming_active = False
        self._stream_block: TranscriptBlock | None = None
        self._app: Application[None] | None = None
        self._run_namespace: dict[str, Any] = {
            "harness": harness,
            "module": harness.module,
            "store": harness.store,
            "torch": torch,
        }

        self._input = TextArea(
            multiline=True,
            prompt=FormattedText([("class:prompt", "you> ")]),
            height=Dimension(min=1, max=8),
            history=InMemoryHistory(),
            completer=WordCompleter(
                [
                    "/help",
                    "/status",
                    "/model",
                    "/tensors",
                    "/revisions",
                    "/diff",
                    "/undo",
                    "/reset",
                    "/clear",
                    "/run",
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
            get_cursor_position=lambda: Point(self._cursor_line, 0),
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
                self._add("dim", "A run is already in progress.")
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
                self._add("dim", "Cannot cancel an in-flight request.")
            elif self._input.text:
                self._input.text = ""
            else:
                self._add("dim", "Press Ctrl+D or /quit to exit.")

        @bindings.add("c-d")
        def _quit(event: Any) -> None:
            if self._app is not None:
                self._app.exit()

        @bindings.add("c-l")
        def _clear(event: Any) -> None:
            self._clear_transcript()

        @bindings.add("pageup")
        def _scroll_up(event: Any) -> None:
            self._page_up()

        @bindings.add("pagedown")
        def _scroll_down(event: Any) -> None:
            self._page_down()

        @bindings.add("end", filter=Condition(lambda: not self._input.text))
        def _scroll_to_end(event: Any) -> None:
            self._scroll_to_bottom()

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
            "system",
            "Weighter — weight-space programming harness. Type /help for commands.",
        )
        self._app.run()

    def _add(
        self,
        kind: str,
        text: str,
        *,
        newline: bool = True,
        meta: dict[str, Any] | None = None,
    ) -> None:
        if kind in KIND_FOR_STYLE:
            kind = KIND_FOR_STYLE[kind]
        style = STYLE_FOR_KIND.get(kind, "")
        text = str(text).replace("\r\n", "\n")
        if newline and not text.endswith("\n"):
            text += "\n"
        if not text:
            return
        block = TranscriptBlock(kind=kind, text=text, meta=meta or {})
        block.cached_fragments = [(style, text)]
        block.line_count = text.count("\n")
        self._blocks.append(block)
        self._fragments.extend(block.cached_fragments)
        self._total_lines += block.line_count
        self._trim_transcript()
        if self._follow_bottom:
            self._cursor_line = self._total_lines
        self._invalidate()

    def _append_stream(self, text: str) -> None:
        text = str(text).replace("\r\n", "\n")
        if not text:
            return
        if (
            self._stream_block is None
            or not self._blocks
            or self._blocks[-1] is not self._stream_block
        ):
            self._add("assistant", "assistant> ", newline=False)
            self._stream_block = self._blocks[-1]
        block = self._stream_block
        style = STYLE_FOR_KIND["assistant"]
        block.text += text
        block.cached_fragments.append((style, text))
        block.line_count += text.count("\n")
        self._fragments.append((style, text))
        self._total_lines += text.count("\n")
        self._trim_transcript()
        if self._follow_bottom:
            self._cursor_line = self._total_lines
        self._invalidate()

    def _finish_stream(self) -> None:
        if self._stream_block is None:
            return
        if not self._stream_block.text.endswith("\n"):
            self._append_stream("\n")
        self._streaming_active = False
        self._stream_block = None

    def _trim_transcript(self) -> None:
        if self._total_lines <= self.max_transcript_lines:
            return
        has_notice = bool(self._blocks) and bool(
            self._blocks[0].meta.get("trim_notice")
        )
        remove_from = 1 if has_notice else 0
        target = max(0, self.max_transcript_lines - self.trim_chunk)
        removed_lines = 0
        while (
            self._total_lines > target
            and len(self._blocks) - remove_from > 1
        ):
            block = self._blocks.pop(remove_from)
            removed_lines += block.line_count
            self._total_lines -= block.line_count
        if not removed_lines:
            return
        added_notice = 0
        if not has_notice:
            notice_text = "… older output trimmed …\n"
            notice = TranscriptBlock(
                kind="dim",
                text=notice_text,
                meta={"trim_notice": True},
            )
            notice.cached_fragments = [(STYLE_FOR_KIND["dim"], notice_text)]
            notice.line_count = notice_text.count("\n")
            self._blocks.insert(0, notice)
            self._total_lines += notice.line_count
            added_notice = notice.line_count
        self._rebuild_fragments()
        if self._follow_bottom:
            self._cursor_line = self._total_lines
        else:
            shift = removed_lines - added_notice
            self._cursor_line = max(0, self._cursor_line - shift)

    def _rebuild_fragments(self) -> None:
        self._fragments = []
        self._total_lines = 0
        for block in self._blocks:
            self._fragments.extend(block.cached_fragments)
            self._total_lines += block.line_count

    def _clear_transcript(self) -> None:
        self._blocks.clear()
        self._fragments.clear()
        self._total_lines = 0
        self._cursor_line = 0
        self._follow_bottom = True
        self._streaming_active = False
        self._stream_block = None
        self._add("system", "Transcript cleared. Type /help for commands.")

    def _page_up(self) -> None:
        info = self._transcript_window.render_info
        if self._follow_bottom:
            current = self._total_lines
        elif info is not None:
            current = info.vertical_scroll
        else:
            current = self._cursor_line
        self._follow_bottom = False
        target = max(0, current - self.scroll_step)
        self._cursor_line = target
        self._transcript_window.vertical_scroll = target
        self._invalidate()

    def _page_down(self) -> None:
        info = self._transcript_window.render_info
        current = info.vertical_scroll if info is not None else self._cursor_line
        target = current + self.scroll_step
        if target >= self._total_lines:
            target = self._total_lines
            self._follow_bottom = True
        else:
            self._follow_bottom = False
        self._cursor_line = target
        self._transcript_window.vertical_scroll = target
        self._invalidate()

    def _scroll_to_bottom(self) -> None:
        self._follow_bottom = True
        self._cursor_line = self._total_lines
        self._transcript_window.vertical_scroll = self._total_lines
        self._invalidate()

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
        self._scroll_to_bottom()
        self._add("user", f"you> {text}")
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
                self._add("assistant", "assistant> ", newline=False)
                self._stream_block = self._blocks[-1]
                self._streaming_active = True
            self._append_stream(event.get("text", ""))
        elif event_type == "assistant_message":
            if self._streaming_active:
                self._finish_stream()
            else:
                content = event.get("content", "")
                if content:
                    self._add("assistant", f"assistant> {content}")
        elif event_type == "tool_start":
            arguments = str(event.get("arguments", ""))
            if len(arguments) > 300:
                arguments = arguments[:297] + "..."
            self._add("tool", f"tool> {event.get('name')} {arguments}")
        elif event_type == "tool_result":
            self._add("tool", self._format_tool_result(event))
        elif event_type == "error":
            self._add("error", f"error> {event.get('message', 'unknown error')}")
        elif event_type == "done":
            self._finish_stream()
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
            elif name == "calculate" and "result" in data:
                detail = f" = {data.get('repr', data.get('result'))}"
            elif name == "run_probe":
                detail = " probe result"
            elif "summary" in data and isinstance(data["summary"], dict):
                summary = data["summary"]
                detail = (
                    f" changed={summary.get('total_changed_elements')}"
                    f" tensors={summary.get('changed_tensors')}"
                )
            else:
                compact = json_dumps(data)
                if len(compact) > 260:
                    compact = compact[:257] + "..."
                detail = f" {compact}"
            diff = data.get("diff")
            if isinstance(diff, dict):
                detail += (
                    f" changed={diff.get('total_changed_elements')}"
                    f"/{diff.get('dirty_tensor_elements')}"
                )
        return f"tool< {name} ok in {duration:.2f}s{detail}"

    def _execute_python(self, source: str) -> None:
        if not source:
            self._add("error", "usage: /run <python>")
            return
        output = io.StringIO()
        failed = False
        try:
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(
                output
            ):
                try:
                    tree = ast.parse(source, filename="<weighter-tui>", mode="eval")
                except SyntaxError:
                    tree = None
                if tree is not None:
                    value = eval(
                        compile(tree, "<weighter-tui>", "eval"),
                        self._run_namespace,
                    )
                    if value is not None:
                        print(repr(value))
                else:
                    code = compile(source, "<weighter-tui>", "exec")
                    exec(code, self._run_namespace)
        except SystemExit as exc:
            failed = True
            output.write(f"SystemExit: {exc.code}\n")
        except BaseException:
            failed = True
            output.write(traceback.format_exc())
        result = output.getvalue().rstrip("\n")
        if result:
            self._add("error" if failed else "system", result)
        elif failed:
            self._add("error", "Python execution failed.")
        else:
            self._add("dim", "(no output)")

    def _handle_command(self, text: str) -> None:
        stripped = text.strip()
        if stripped == "/run" or stripped.startswith("/run "):
            if self._busy:
                self._add("dim", "Wait for the current run to finish.")
            else:
                self._execute_python(stripped[4:].strip())
            self._update_status()
            return
        try:
            parts = shlex.split(text)
        except ValueError as exc:
            self._add("error", f"error> {exc}")
            return
        if not parts:
            return
        command = parts[0].lower()
        try:
            if command in {"/help", "/h"}:
                self._add("system", HELP_TEXT)
            elif command == "/status":
                self._add("system", json.dumps(self.harness.status(), indent=2))
            elif command == "/model":
                if len(parts) == 1:
                    self._add("system", f"model: {self.harness.config.llm}")
                elif self._busy:
                    self._add("dim", "Wait for the current run to finish.")
                else:
                    model = " ".join(parts[1:]).strip()
                    self.harness.config.llm = model
                    self._add("system", f"Model set to {model}")
            elif command == "/tensors":
                prefix = parts[1] if len(parts) > 1 else None
                refs = self.harness.store.list_refs(prefix=prefix)[:200]
                lines = [
                    f"{ref.name}  shape={list(ref.tensor.shape)}  "
                    f"dtype={str(ref.tensor.dtype).replace('torch.', '')}"
                    for ref in refs
                ]
                self._add("system", "\n".join(lines) if lines else "(no tensors)")
            elif command == "/revisions":
                revisions = self.harness.revisions()
                self._add(
                    "system",
                    json.dumps(revisions, indent=2) if revisions else "(no revisions)",
                )
            elif command == "/diff":
                since = parts[1] if len(parts) > 1 else None
                diff = self.harness.diff(since=since)
                if not diff:
                    self._add("system", "(no edits)")
                else:
                    summary = self.harness.diff_summary(since=since)
                    self._add(
                        "system",
                        json.dumps(
                            {
                                "since": since or "initial",
                                "summary": summary,
                                "tensors": diff[:50],
                                "truncated": len(diff) > 50,
                            },
                            indent=2,
                        ),
                    )
            elif command == "/undo":
                if self._busy:
                    self._add("dim", "Wait for the current run to finish.")
                    return
                n = int(parts[1]) if len(parts) > 1 else 1
                undone = self.harness.undo(n)
                self._add(
                    "system",
                    f"Undid revisions: {', '.join(item['id'] for item in undone)}",
                )
            elif command == "/reset":
                self.harness.reset()
                self._add("system", "Conversation history cleared.")
            elif command == "/clear":
                self._clear_transcript()
            elif command == "/save":
                if self._busy:
                    self._add("dim", "Wait for the current run to finish.")
                    return
                if len(parts) < 2:
                    self._add("error", "usage: /save <path>")
                else:
                    path = self.harness.save_module(" ".join(parts[1:]))
                    self._add("system", f"Saved state_dict to {path}")
            elif command in {"/quit", "/exit"}:
                if self._app is not None:
                    self._app.exit()
            else:
                self._add("error", f"unknown command {command!r}; try /help")
        except Exception as exc:
            self._add("error", f"error> {type(exc).__name__}: {exc}")
        self._update_status()

    def _update_status(self) -> None:
        self._invalidate()


def run_tui(harness: Any) -> None:
    HarnessTUI(harness).run()
