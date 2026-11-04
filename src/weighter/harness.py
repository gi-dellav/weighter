"""The public Harness class."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping, cast

import torch

from .config import HarnessConfig
from .errors import ConfigurationError, LLMError, ToolError
from .prompts import build_system_prompt
from .tools import ToolContext, build_tools, default_probe
from .tools.base import ProbeBudget
from .types import (
    ConfirmCallback,
    EditPolicy,
    HarnessEvent,
    ProbeFn,
    PromptResult,
    ToolCallRecord,
    Usage,
)
from .utils import content_to_text, get_attr, json_dumps, truncate_text
from .weights import WeightStore, summarize_diff

CompletionFn = Callable[..., Any]
EventSink = Callable[[HarnessEvent], None]


class Harness:
    """An AI agent that reads and writes the weights of a PyTorch module.

    Parameters
    ----------
    module:
        The ``torch.nn.Module`` to operate on.  Edits are applied in place.
    llm:
        LiteLLM model id, e.g. ``"openai/gpt-4.1-mini"``.  Falls back to the
        ``WEIGHTER_LLM`` environment variable.
    probe:
        Optional ``(module, payload) -> result`` callback used by the
        ``run_probe`` tool.  If omitted, :func:`default_probe` is used.  Pass
        ``False`` to disable probing entirely.
    probe_budget:
        Maximum number of probe calls, or ``None`` for unlimited.
    edit_policy:
        ``"auto"`` applies edits immediately, ``"confirm"`` requires a
        confirmation callback, and ``"read_only"`` hides editing tools.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        *,
        llm: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float | None = None,
        request_timeout: float | None = None,
        completion_kwargs: Mapping[str, Any] | None = None,
        system_prompt: str | None = None,
        probe: ProbeFn | bool | None = None,
        probe_budget: int | None = None,
        edit_policy: EditPolicy = "auto",
        confirm_callback: ConfirmCallback | None = None,
        include_buffers: bool = False,
        tensor_filter: Callable[[str], bool] | str | None = None,
        undo: bool = True,
        max_undo_bytes: int = 512 * 1024 * 1024,
        max_edit_elements: int = 1_000_000,
        max_read_elements: int = 2_048,
        max_tool_output_chars: int = 50_000,
        max_steps: int = 16,
        max_history_messages: int | None = None,
        stream: bool = False,
        completion_fn: CompletionFn | None = None,
    ) -> None:
        if not isinstance(module, torch.nn.Module):
            raise TypeError("module must be a torch.nn.Module")

        if probe is False:
            resolved_probe: ProbeFn | None = None
        elif probe is None:
            resolved_probe = default_probe
        elif callable(probe):
            resolved_probe = probe
        else:
            raise ConfigurationError("probe must be a callable, False, or None")

        self.config = HarnessConfig(
            llm=llm,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            request_timeout=request_timeout,
            completion_kwargs=completion_kwargs or {},
            system_prompt=system_prompt,
            probe=resolved_probe,
            probe_budget=probe_budget,
            edit_policy=edit_policy,
            confirm_callback=confirm_callback,
            include_buffers=include_buffers,
            tensor_filter=tensor_filter,
            undo=undo,
            max_undo_bytes=max_undo_bytes,
            max_edit_elements=max_edit_elements,
            max_read_elements=max_read_elements,
            max_tool_output_chars=max_tool_output_chars,
            max_steps=max_steps,
            max_history_messages=max_history_messages,
            stream=stream,
        )

        if self.config.llm is None:
            if completion_fn is None:
                raise ConfigurationError(
                    "an LLM is required; pass llm='provider/model' or set WEIGHTER_LLM"
                )
            self.config.llm = "injected"

        self._store = WeightStore(
            module,
            include_buffers=self.config.include_buffers,
            tensor_filter=cast(Callable[[str], bool] | None, self.config.tensor_filter),
            undo=self.config.undo,
            max_undo_bytes=self.config.max_undo_bytes,
        )
        self._probe_budget = ProbeBudget(remaining=self.config.probe_budget)
        self._messages: list[dict[str, Any]] = []
        self._usage = Usage()

        allow_writes = self.config.edit_policy != "read_only"
        self._tools = build_tools(
            allow_writes=allow_writes,
            undo_available=self.config.undo and allow_writes,
            probe_available=self.config.probe is not None,
        )
        self._tool_schemas = [tool.schema() for tool in self._tools.values()]

        self._litellm: Any | None = None
        if completion_fn is None:
            import litellm

            self._litellm = litellm
        self._completion_fn = completion_fn

    @property
    def module(self) -> torch.nn.Module:
        return self._store.module

    @property
    def store(self) -> WeightStore:
        return self._store

    @property
    def messages(self) -> list[dict[str, Any]]:
        return list(self._messages)

    @property
    def usage(self) -> Usage:
        return self._usage

    def status(self) -> dict[str, Any]:
        overview = self._store.overview()
        return {
            "llm": self.config.llm,
            "module_type": overview["module_type"],
            "parameters": overview["parameters"],
            "parameter_tensors": overview["parameter_tensors"],
            "revisions": len(self._store.revisions),
            "dirty_tensors": self._store.dirty_tensors(),
            "probe_budget": self._probe_budget.remaining,
            "probe_enabled": self.config.probe is not None,
            "messages": len(self._messages),
            "usage": {
                "prompt_tokens": self._usage.prompt_tokens,
                "completion_tokens": self._usage.completion_tokens,
                "total_tokens": self._usage.total_tokens,
            },
            "edit_policy": self.config.edit_policy,
        }

    def prompt(
        self,
        message: str,
        *,
        on_event: EventSink | None = None,
        stream: bool | None = None,
    ) -> PromptResult:
        """Send a user message and run the tool-use loop until the model answers."""

        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string")
        use_stream = self.config.stream if stream is None else stream
        before_revision_ids = {revision.id for revision in self._store.revisions}
        self._messages.append({"role": "user", "content": message})
        records: list[ToolCallRecord] = []
        prompt_usage = Usage()
        final_text = ""
        stop_reason = "max_steps"
        last_content = ""
        completed = False

        for _step in range(self.config.max_steps):
            messages = self._build_messages()
            assistant_message, step_usage = self._complete(
                messages, stream=use_stream, on_event=on_event
            )
            prompt_usage.add(step_usage)
            self._usage.add(step_usage)

            content = content_to_text(assistant_message.get("content"))
            last_content = content
            tool_calls = assistant_message.get("tool_calls") or []
            assistant_message["content"] = content
            assistant_message.setdefault("role", "assistant")
            if tool_calls:
                assistant_message["tool_calls"] = tool_calls
            else:
                assistant_message.pop("tool_calls", None)
            self._messages.append(assistant_message)
            self._emit(
                on_event,
                {
                    "type": "assistant_message",
                    "content": content,
                    "tool_calls": len(tool_calls),
                    "streamed": bool(use_stream),
                },
            )

            if not tool_calls:
                final_text = content
                stop_reason = "complete"
                completed = True
                break

            for tool_call in tool_calls:
                records.append(self._run_tool_call(tool_call, on_event))

        if not completed:
            final_text = last_content or (
                f"Stopped after reaching max_steps={self.config.max_steps} "
                "without a final answer."
            )
            stop_reason = "max_steps"

        revisions = [
            revision.id
            for revision in self._store.revisions
            if revision.id not in before_revision_ids
        ]
        return PromptResult(
            text=final_text,
            messages=list(self._messages),
            tool_calls=records,
            usage=prompt_usage,
            revisions=revisions,
            stop_reason=stop_reason,
        )

    def revisions(self) -> list[dict[str, Any]]:
        return self._store.revision_infos()

    def diff(
        self,
        since: str | None = None,
        *,
        tensor_prefix: str | None = None,
    ) -> list[dict[str, Any]]:
        return self._store.diff(since=since, tensor_prefix=tensor_prefix)

    def diff_summary(
        self,
        since: str | None = None,
        *,
        tensor_prefix: str | None = None,
    ) -> dict[str, Any]:
        return summarize_diff(
            self._store.diff(since=since, tensor_prefix=tensor_prefix),
            module_elements=self._store.total_elements(),
        )

    def undo(self, n: int = 1) -> list[dict[str, Any]]:
        return [revision.info() for revision in self._store.undo(n)]

    def revert(self) -> list[dict[str, Any]]:
        return [revision.info() for revision in self._store.revert_all()]

    def reset(self, *, clear_history: bool = True, revert_edits: bool = False) -> None:
        if clear_history:
            self._messages.clear()
        if revert_edits:
            self._store.revert_all()

    def save_module(self, path: str | os.PathLike[str]) -> str:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.module.state_dict(), target)
        return str(target)

    def tui(self) -> None:
        """Launch the full-screen interactive TUI."""

        from .tui import HarnessTUI

        HarnessTUI(self).run()

    def _build_messages(self) -> list[dict[str, Any]]:
        history = self._messages
        if self.config.max_history_messages is not None:
            history = history[-self.config.max_history_messages :]
            while history and history[0].get("role") == "tool":
                history = history[1:]
        system = build_system_prompt(
            self._store,
            self.config,
            tool_names=list(self._tools),
            probe_remaining=self._probe_budget.remaining,
        )
        return [{"role": "system", "content": system}, *history]

    def _tool_context(self) -> ToolContext:
        confirm = (
            self.config.confirm_callback
            if self.config.edit_policy == "confirm"
            else None
        )
        return ToolContext(
            store=self._store,
            config=self.config,
            probe=self.config.probe,
            probe_budget=self._probe_budget,
            confirm=confirm,
        )

    def _run_tool_call(
        self, tool_call: dict[str, Any], on_event: EventSink | None
    ) -> ToolCallRecord:
        call_id = str(tool_call.get("id") or "call_0")
        function = tool_call.get("function") or {}
        name = str(function.get("name") or "")
        raw_arguments = function.get("arguments", "{}")
        arguments: dict[str, Any] = {}
        started = time.perf_counter()
        self._emit(
            on_event,
            {
                "type": "tool_start",
                "id": call_id,
                "name": name,
                "arguments": raw_arguments if isinstance(raw_arguments, str) else json_dumps(raw_arguments),
            },
        )
        try:
            if isinstance(raw_arguments, str):
                arguments = json.loads(raw_arguments or "{}")
            elif isinstance(raw_arguments, Mapping):
                arguments = dict(raw_arguments)
            else:
                raise ToolError("tool arguments must be a JSON object")
            if not isinstance(arguments, dict):
                raise ToolError("tool arguments must be a JSON object")
            tool = self._tools.get(name)
            if tool is None:
                raise ToolError(f"unknown tool {name!r}")
            if (
                tool.mutates
                and self.config.edit_policy == "confirm"
                and self.config.confirm_callback is None
            ):
                raise ToolError(
                    "edit_policy='confirm' requires a confirm_callback on the Harness"
                )
            parsed_args = tool.args_model.model_validate(arguments)
            data = tool.handler(self._tool_context(), parsed_args)
            result: dict[str, Any] = {"ok": True, "data": data}
            ok = True
        except Exception as exc:
            result = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            ok = False

        content = self._format_tool_result(result)
        self._messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": content,
            }
        )
        duration = time.perf_counter() - started
        record = ToolCallRecord(
            id=call_id,
            name=name,
            arguments=arguments,
            ok=ok,
            result=result,
            duration=duration,
        )
        self._emit(
            on_event,
            {
                "type": "tool_result",
                "id": call_id,
                "name": name,
                "ok": ok,
                "result": result,
                "duration": duration,
            },
        )
        return record

    def _format_tool_result(self, result: dict[str, Any]) -> str:
        text = json_dumps(result)
        limit = self.config.max_tool_output_chars
        if len(text) <= limit:
            return text
        preview = truncate_text(text, max(0, limit - 200))
        return json_dumps(
            {
                "ok": result.get("ok", False),
                "truncated": True,
                "preview": preview,
            }
        )

    def _complete(
        self,
        messages: list[dict[str, Any]],
        *,
        stream: bool,
        on_event: EventSink | None,
    ) -> tuple[dict[str, Any], Usage]:
        kwargs: dict[str, Any] = dict(self.config.completion_kwargs)
        kwargs.update(
            {
                "model": self.config.llm,
                "messages": messages,
            }
        )
        if self._tool_schemas:
            kwargs.setdefault("tools", self._tool_schemas)
            kwargs.setdefault("tool_choice", "auto")
        if self.config.temperature is not None:
            kwargs.setdefault("temperature", self.config.temperature)
        if self.config.api_key is not None:
            kwargs.setdefault("api_key", self.config.api_key)
        if self.config.api_base is not None:
            kwargs.setdefault("base_url", self.config.api_base)
        if self.config.request_timeout is not None:
            kwargs.setdefault("timeout", self.config.request_timeout)
        if stream:
            kwargs["stream"] = True

        if stream:
            last_error: Exception | None = None
            for attempt in range(3):
                emitted = {"value": False}
                try:
                    response = self._call_completion(kwargs)
                    if hasattr(response, "choices") or (
                        isinstance(response, Mapping) and "choices" in response
                    ):
                        return self._normalize_response(response)
                    return self._consume_stream(response, on_event, emitted)
                except Exception as exc:
                    last_error = exc
                    if emitted["value"] or attempt == 2 or not self._is_retryable(exc):
                        raise LLMError(f"LLM request failed: {type(exc).__name__}: {exc}") from exc
                    time.sleep(min(2**attempt, 4))
            raise LLMError(f"LLM request failed: {last_error}")

        last_error = None
        for attempt in range(3):
            try:
                response = self._call_completion(kwargs)
                if hasattr(response, "choices") or isinstance(response, Mapping):
                    return self._normalize_response(response)
                return self._consume_stream(response, on_event, {"value": False})
            except Exception as exc:
                last_error = exc
                if attempt == 2 or not self._is_retryable(exc):
                    raise LLMError(f"LLM request failed: {type(exc).__name__}: {exc}") from exc
                time.sleep(min(2**attempt, 4))
        raise LLMError(f"LLM request failed: {last_error}")

    def _call_completion(self, kwargs: dict[str, Any]) -> Any:
        if self._completion_fn is not None:
            return self._completion_fn(**kwargs)
        if self._litellm is None:  # pragma: no cover - defensive
            raise LLMError("no completion function configured")
        return self._litellm.completion(**kwargs)

    def _consume_stream(
        self,
        response: Any,
        on_event: EventSink | None,
        emitted: dict[str, bool],
    ) -> tuple[dict[str, Any], Usage]:
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_parts: dict[int, dict[str, str]] = {}
        usage = Usage()
        try:
            for chunk in response:
                chunk_usage = get_attr(chunk, "usage", None)
                if chunk_usage is not None:
                    parsed_usage = Usage.from_object(chunk_usage)
                    if parsed_usage.total_tokens or parsed_usage.prompt_tokens or parsed_usage.completion_tokens:
                        usage = parsed_usage
                choices = get_attr(chunk, "choices", None) or []
                if not choices:
                    continue
                choice = choices[0]
                delta = get_attr(choice, "delta", None) or get_attr(choice, "message", None)
                if delta is None:
                    continue
                reasoning = get_attr(delta, "reasoning_content", None)
                if reasoning:
                    reasoning_parts.append(str(reasoning))
                text = content_to_text(get_attr(delta, "content", None))
                if text:
                    content_parts.append(text)
                    emitted["value"] = True
                    self._emit(on_event, {"type": "text_delta", "text": text})
                for tool_delta in get_attr(delta, "tool_calls", None) or []:
                    index = int(get_attr(tool_delta, "index", 0) or 0)
                    entry = tool_parts.setdefault(
                        index, {"id": "", "name": "", "arguments": ""}
                    )
                    call_id = get_attr(tool_delta, "id", None)
                    if call_id:
                        entry["id"] = str(call_id)
                    function = get_attr(tool_delta, "function", None)
                    if function is not None:
                        function_name = get_attr(function, "name", None)
                        if function_name:
                            function_name = str(function_name)
                            if not entry["name"]:
                                entry["name"] = function_name
                            elif function_name != entry["name"]:
                                entry["name"] += function_name
                        arguments = get_attr(function, "arguments", None)
                        if arguments:
                            if isinstance(arguments, Mapping):
                                entry["arguments"] += json.dumps(arguments)
                            else:
                                entry["arguments"] += str(arguments)
        except Exception as exc:
            if emitted["value"]:
                raise LLMError(f"LLM stream failed: {type(exc).__name__}: {exc}") from exc
            raise

        tool_calls = []
        for index in sorted(tool_parts):
            entry = tool_parts[index]
            tool_calls.append(
                {
                    "id": entry["id"] or f"call_{index}",
                    "type": "function",
                    "function": {
                        "name": entry["name"],
                        "arguments": entry["arguments"] or "{}",
                    },
                }
            )
        message: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(content_parts),
        }
        if reasoning_parts:
            message["reasoning_content"] = "".join(reasoning_parts)
        if tool_calls:
            message["tool_calls"] = tool_calls
        return message, usage

    def _normalize_response(self, response: Any) -> tuple[dict[str, Any], Usage]:
        choices = get_attr(response, "choices", None)
        if not choices:
            raise LLMError("LLM response contained no choices")
        choice = choices[0]
        message = get_attr(choice, "message", None)
        if message is None:
            message = choice
        if hasattr(message, "model_dump"):
            message = message.model_dump()
        elif not isinstance(message, dict):
            message = {
                "role": get_attr(message, "role", "assistant"),
                "content": get_attr(message, "content", None),
                "tool_calls": get_attr(message, "tool_calls", None),
            }
        content = content_to_text(get_attr(message, "content", None))
        raw_tool_calls = get_attr(message, "tool_calls", None) or []
        tool_calls: list[dict[str, Any]] = []
        for index, tool_call in enumerate(raw_tool_calls):
            call_id = get_attr(tool_call, "id", None) or f"call_{index}"
            function = get_attr(tool_call, "function", None)
            name = get_attr(function, "name", "") if function is not None else ""
            arguments = get_attr(function, "arguments", "{}") if function is not None else "{}"
            if isinstance(arguments, Mapping):
                arguments = json.dumps(arguments)
            tool_calls.append(
                {
                    "id": str(call_id),
                    "type": "function",
                    "function": {"name": str(name), "arguments": str(arguments or "{}")},
                }
            )
        normalized: dict[str, Any] = {"role": "assistant", "content": content}
        reasoning = get_attr(message, "reasoning_content", None)
        if reasoning:
            normalized["reasoning_content"] = str(reasoning)
        if tool_calls:
            normalized["tool_calls"] = tool_calls
        return normalized, Usage.from_object(get_attr(response, "usage", None))

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        name = type(exc).__name__
        return name in {
            "APIConnectionError",
            "RateLimitError",
            "Timeout",
            "APITimeoutError",
            "InternalServerError",
            "ServiceUnavailableError",
            "OverloadedError",
        }

    def _emit(self, on_event: EventSink | None, event: HarnessEvent) -> None:
        if on_event is None:
            return
        try:
            on_event(event)
        except Exception:
            pass
