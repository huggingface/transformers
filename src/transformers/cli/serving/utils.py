# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Shared types, constants, and utilities for the serving layer.
"""

import asyncio
import copy
import enum
import json
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from queue import Queue
from typing import TYPE_CHECKING

from transformers.utils import logging
from transformers.utils.chat_parsing import ResponseParser


if TYPE_CHECKING:
    import pydantic
    import tokenizers
    import torch

    from transformers import (
        ContinuousBatchingConfig,
        GenerationConfig,
        PreTrainedModel,
        PreTrainedTokenizerFast,
        ProcessorMixin,
    )
    from transformers.generation.continuous_batching.continuous_api import ContinuousBatchingManager
    from transformers.generation.continuous_batching.requests import GenerationOutput
    from transformers.generation.continuous_batching.scheduler import Scheduler

    from .model_manager import ModelManager


logger = logging.get_logger(__name__)


X_REQUEST_ID = "x-request-id"


class Modality(enum.Enum):
    LLM = "LLM"
    VLM = "VLM"
    MULTIMODAL = "MULTIMODAL"  # supports text, image, video, and audio
    STT = "STT"
    TTS = "TTS"


class _StreamError:
    """Sentinel to signal an error from the generate thread."""

    def __init__(self, msg: str):
        self.msg = msg


class _GenerationCancelled(Exception):
    """Raised inside ``DirectStreamer.put()`` to abort ``model.generate()``."""


class ReasoningText(str):
    """Tagged str subclass: text chunk belonging to a thinking/reasoning block.

    Streamers wrap reasoning text with this so handlers can route it to
    ``reasoning_content`` deltas instead of ``content``.
    """


class CBWorkerDeadError(RuntimeError):
    """Raised when a request is submitted to a CB worker that has died.

    Surfaced as 503 by the FastAPI exception handler. Carries the original error message
    that killed the worker so the client knows why the server is in this state.
    """


# Fallback response_template per model_type for tokenizers that don't ship one.
_RESPONSE_TEMPLATE_FALLBACKS = {
    (
        "qwen2",
        "qwen2_moe",
        "qwen2_vl",
        "qwen2_5_vl",
        "qwen3",
        "qwen3_moe",
        "qwen3_next",
        "qwen3_vl",
        "qwen3_vl_moe",
    ): {
        "defaults": {"role": "assistant"},
        "start_anchor": "<|im_start|>assistant\n",
        "fields": {
            "thinking": {
                "open": "<think>",
                "close": "</think>",
                "content": "text",
            },
            "tool_calls": {
                "open": "<tool_call>",
                "close": "</tool_call>",
                "repeats": True,
                "content": "json",
                "transform": {"type": "function", "function": "{content}"},
            },
            "content": {
                "close": ["<|im_end|>", "<|endoftext|>", "<|eot_id|>"],
                "content": "text",
            },
        },
    },
    ("gemma4",): {
        "defaults": {"role": "assistant"},
        "start_anchor": ["<|turn>model\n", "<tool_response|>"],
        "fields": {
            "thinking": {
                "open": "<|channel>thought\n",
                "close": "<channel|>",
                "content": "text",
            },
            "tool_calls": {
                "open_pattern": r"<\|tool_call>call:(?P<name>\w+)",
                "close": "<tool_call|>",
                "repeats": True,
                "content": "json",
                "content_args": {
                    "unquoted_keys": True,
                    "string_delims": [['<|"|>', '<|"|>']],
                },
                "transform": {"type": "function", "function": {"name": "{name}", "arguments": "{content}"}},
            },
            "content": {
                "close": ["<turn|>", "<|tool_response>", "<eos>"],
                "content": "text",
            },
        },
    },
}


@dataclass
class ToolCall:
    """A parsed tool call surfaced through the stream queue when the parser
    closes a `tool_calls` region. ``arguments`` is always a JSON string."""

    name: str
    arguments: str


def get_response_template(processor, model: "PreTrainedModel") -> dict | None:
    """Return the response template (new declarative format) for this model.

    Resolution order:
        1. ``tokenizer.response_template`` if set on the tokenizer.
        2. Fallback in :data:`_RESPONSE_TEMPLATE_FALLBACKS` matched by ``model.config.model_type``.
        3. ``None`` (no parsing performed; raw text is streamed through).
    """
    tokenizer = getattr(processor, "tokenizer", processor)
    tmpl = getattr(tokenizer, "response_template", None)
    if tmpl is not None:
        return tmpl
    model_type = model.config.model_type
    return next((v for types, v in _RESPONSE_TEMPLATE_FALLBACKS.items() if model_type in types), None)


def build_response_parser(
    processor,
    model: "PreTrainedModel",
    input_ids,
) -> "ResponseParser | None":
    """Build a streaming :class:`ResponseParser` for the current generation.

    Returns ``None`` when no template is available for the model (then text
    streams through unparsed).

    `input_ids` is decoded and passed to the parser as `prefix=` so templates
    that declare a `start_anchor` get prefill-aware parsing (e.g. for a future
    prefilled `<think>\\n` opener). Templates without `start_anchor` fail
    closed and ignore the prefix — safe for chat templates where the anchor
    isn't at the actual generation point (e.g. Gemma 4 tool-result followups).
    """
    template = get_response_template(processor, model)
    if template is None:
        return None
    tokenizer = getattr(processor, "tokenizer", processor)
    return ResponseParser(template, prefix=_decode_prefix(tokenizer, input_ids))


def _normalize_tool_call(value: dict) -> ToolCall:
    """Convert one parsed ``tool_calls`` entry into a :class:`ToolCall`.

    Every template in this codebase shapes a tool call as
    ``{"type": "function", "function": {"name": ..., "arguments": ...}}`` —
    the parser already extracted ``name`` and ``arguments`` for us. The only
    work left is re-serializing ``arguments`` to a JSON string, since the
    field uses ``"content": "json"`` so the parser hands us a parsed dict
    while the OpenAI API expects a string.
    """
    fn = value["function"]
    args = fn["arguments"]
    if not isinstance(args, str):
        args = json.dumps(args)
    return ToolCall(name=fn["name"], arguments=args)


def _decode_prefix(tokenizer, input_ids) -> str:
    """Decode ``input_ids`` to the chat prompt string the parser expects as ``prefix=``.

    Serve always handles one request at a time, but the upstream shape varies:
    ``apply_chat_template(return_tensors="pt")`` hands us a 2D ``[1, N]`` tensor
    while continuous batching hands us a flat ``list[int]``. Both forms hold a
    single sequence; this helper unwraps it before decoding."""
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return tokenizer.decode(input_ids)


def parse_assistant_message(
    processor,
    model: "PreTrainedModel",
    generated_ids,
    input_ids,
    cleaned_content: str = "",
) -> tuple[str, str | None, list[ToolCall] | None]:
    """Parse a full assistant message into ``(content, reasoning, tool_calls)``.

    Uses :meth:`tokenizer.parse_response` with the resolved response template
    (see :func:`get_response_template`). When no template is available, returns
    ``(cleaned_content, None, None)`` — i.e., surface the raw model decode.

    ``cleaned_content`` is the ``skip_special_tokens=True`` decode of
    ``generated_ids``. The parser itself runs on the raw decode (keeping
    special tokens so it can match delimiters like Gemma 4's ``<|tool_call>``);
    ``cleaned_content`` is only used on the no-template path."""
    template = get_response_template(processor, model)
    if template is None:
        return cleaned_content, None, None
    tokenizer = getattr(processor, "tokenizer", processor)
    parsed = tokenizer.parse_response(generated_ids, template, prefix=_decode_prefix(tokenizer, input_ids))
    # Absent ``content`` key means the model emitted only tool calls / thinking;
    # surface that as an empty string rather than swapping in the raw decode.
    content = parsed.get("content", "")
    reasoning = parsed.get("thinking")
    raw = parsed.get("tool_calls") or []
    tool_calls = [_normalize_tool_call(v) for v in raw] or None
    return content, reasoning, tool_calls


def response_events_to_chunks(events: list[dict]) -> list:
    """Translate :class:`ResponseParser` events into stream-queue items.

    Output items are one of:
      - ``str``                 — plain content delta (also raw text when no parser is active)
      - :class:`ReasoningText`  — reasoning / thinking delta
      - :class:`ToolCall`       — completed tool call on region close

    Other events (``region_open``, raw dirty chunks of structured fields like
    ``json`` tool calls, ``region_close`` for non-tool fields) carry no
    user-visible payload and are dropped.
    """
    out: list = []
    for ev in events:
        kind = ev["type"]
        field = ev.get("field")
        if kind == "region_chunk":
            if ev.get("dirty"):
                # Structured fields (json/xml-inline) stream raw bytes; the
                # parsed value lands in `region_close`.
                continue
            text = ev.get("text") or ""
            if not text:
                continue
            if field == "thinking":
                out.append(ReasoningText(text))
            elif field == "content":
                out.append(text)
            # Unknown fields are ignored — handlers can extend this if needed.
        elif kind == "region_close" and field == "tool_calls":
            tc = _normalize_tool_call(ev.get("value"))
            if tc is not None:
                out.append(tc)
    return out


class DownloadAggregator:
    """Aggregates byte-progress across multiple concurrent download tqdm bars.

    huggingface_hub opens one tqdm bar per file shard. This class tracks them all and emits
    a single aggregate ``{"stage": "download", "progress": {...}}`` event whenever any updates.
    """

    def __init__(self, enqueue: Callable, model_id: str):
        self.enqueue = enqueue
        self.model = model_id
        self.bars: dict[int, tuple[int, int | None]] = {}
        self.last_emitted_current: int | None = None

    def register(self, bar_id: int, total: int | None) -> None:
        """Register a new download bar with its total byte count."""
        self.bars[bar_id] = (0, total)
        self._emit()

    def update(self, bar_id: int, current: int, total: int | None) -> None:
        """Update a bar's current byte count and emit aggregate progress."""
        self.bars[bar_id] = (current, total)
        self._emit()

    def close(self, bar_id: int) -> None:
        pass  # keep the bar so totals remain correct

    def _emit(self) -> None:
        agg_current = sum(c for c, _ in self.bars.values())
        if agg_current == self.last_emitted_current:
            return
        self.last_emitted_current = agg_current
        totals = [t for _, t in self.bars.values() if t is not None]
        agg_total = sum(totals) if totals else None
        self.enqueue(
            {
                "status": "loading",
                "model": self.model,
                "stage": "download",
                "progress": {"current": agg_current, "total": agg_total},
            }
        )


def make_progress_tqdm_class(callback: Callable, model_id: str) -> type:
    """Create a tqdm subclass that routes progress to a callback.

    Bars with ``unit="B"`` are download bars — aggregated via ``DownloadAggregator``.
    Other bars (e.g. "Loading weights") emit ``weights`` stage events.

    Args:
        callback (`callable`): Called with a dict payload
            ``{"status": "loading", "model": ..., "stage": ..., "progress": ...}``.
        model_id (`str`): The model ID (included in progress payloads).

    Returns:
        A tqdm subclass that forwards progress to *callback*.
    """
    from tqdm.auto import tqdm as base_tqdm

    download_aggregator = DownloadAggregator(callback, model_id)

    class ProgressTqdm(base_tqdm):  # type: ignore[misc]
        def __init__(self, *args, **kwargs):
            self.sse_unit = kwargs.get("unit") or "it"
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)
            self.n = 0
            self.last_emitted = -1
            if self.sse_unit == "B":
                self._bar_id = id(self)
                download_aggregator.register(self._bar_id, self.total)

        def update(self, n=1):
            if n is None:
                n = 1
            self.n += n
            if self.sse_unit == "B":
                download_aggregator.update(self._bar_id, self.n, self.total)
            elif self.n != self.last_emitted:
                self.last_emitted = self.n
                callback(
                    {
                        "status": "loading",
                        "model": model_id,
                        "stage": "weights",
                        "progress": {"current": self.n, "total": self.total},
                    }
                )

        def __iter__(self):
            for item in self.iterable:
                self.n += 1
                if self.sse_unit == "B":
                    download_aggregator.update(self._bar_id, self.n, self.total)
                elif self.n != self.last_emitted:
                    self.last_emitted = self.n
                    callback(
                        {
                            "status": "loading",
                            "model": model_id,
                            "stage": "weights",
                            "progress": {"current": self.n, "total": self.total},
                        }
                    )
                yield item

        def close(self):
            if self.sse_unit == "B":
                download_aggregator.close(self._bar_id)
            super().close()

    return ProgressTqdm


class DirectStreamer:
    """Streamer for ``model.generate()`` (used by :class:`GenerateManager`).

    Implements the ``put``/``end`` protocol that ``model.generate()`` expects:
    generate calls ``put(token_tensor)`` after each decode step, and ``end()``
    when generation is complete. Tokens are decoded incrementally via the Rust
    ``DecodeStream`` (O(1) per token) and pushed as text to an asyncio.Queue.
    """

    def __init__(
        self,
        tokenizer: "tokenizers.Tokenizer",
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue,
        response_parser: "ResponseParser | None" = None,
    ):
        """
        Args:
            tokenizer: The Rust tokenizer (``tokenizer._tokenizer``).
            loop (`asyncio.AbstractEventLoop`): The event loop to push decoded text to.
            queue (`asyncio.Queue`): The queue that receives decoded text chunks.
            response_parser (`ResponseParser`, *optional*): When provided, every
                decoded chunk is fed through this parser; the resulting events
                are translated into typed queue items (plain ``str`` for content,
                :class:`ReasoningText` for thinking, :class:`ToolCall` on tool
                call close). Without a parser, raw decoded text is streamed.
        """
        from tokenizers.decoders import DecodeStream

        self._tokenizer = tokenizer
        self._loop = loop
        self._queue = queue
        skip_special_tokens = response_parser is None
        self._decode_stream = DecodeStream([], skip_special_tokens=skip_special_tokens)
        self._parser = response_parser
        self._first = True
        self._cancelled = threading.Event()
        self.total_tokens = 0
        if self._parser is not None:
            for item in response_events_to_chunks(self._parser.initial_events):
                self._loop.call_soon_threadsafe(self._queue.put_nowait, item)

    def _enqueue(self, item) -> None:
        self._loop.call_soon_threadsafe(self._queue.put_nowait, item)

    def put(self, value: "torch.Tensor") -> None:
        """Called by ``model.generate()`` after each decode step with new token(s)."""
        if self._cancelled.is_set():
            raise _GenerationCancelled()
        # The first put() contains the prompt tokens — skip since we only stream generated tokens.
        if self._first:
            self._first = False
            return
        for token_id in value.tolist():
            self.total_tokens += 1
            text = self._decode_stream.step(self._tokenizer, token_id)
            if text is None:
                continue
            if self._parser is None:
                self._enqueue(text)
            else:
                for item in response_events_to_chunks(self._parser.feed(text)):
                    self._enqueue(item)

    def end(self) -> None:
        """Flush any final parser events, then push the EOS sentinel."""
        if self._parser is not None:
            try:
                _msg, final_events = self._parser.finalize()
            except (ValueError, RuntimeError) as e:
                # Generation stopped mid-region (likely max_new_tokens hit
                # inside a structured field like a tool call).
                logger.warning(f"ResponseParser.finalize() failed; dropping final tail events: {e}")
                final_events = []
            for item in response_events_to_chunks(final_events):
                self._enqueue(item)
        self._enqueue(None)

    def cancel(self) -> None:
        """Signal cancellation. The next ``put()`` call will raise and abort ``model.generate()``."""
        self._cancelled.set()


class CBStreamer:
    """Streamer for continuous batching (used by :class:`CBGenerateManager`).

    Same ``put``/``end`` protocol as :class:`DirectStreamer`, but called manually
    by :class:`CBGenerateManager` instead of by ``model.generate()``:
    ``put(output)`` receives a CB ``GenerationOutput``, decodes new tokens, and
    pushes text to the asyncio.Queue. ``end()`` signals the stream is complete.
    """

    def __init__(
        self,
        cb_manager: "ContinuousBatchingManager",
        request_id: str,
        tokenizer: "tokenizers.Tokenizer",
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue,
        response_parser: "ResponseParser | None" = None,
    ):
        """
        Args:
            cb_manager (`ContinuousBatchingManager`): The CB manager instance.
            request_id (`str`): The request ID to track in the CB scheduler.
            tokenizer: The Rust tokenizer (``tokenizer._tokenizer``).
            loop (`asyncio.AbstractEventLoop`): The event loop to push decoded text to.
            queue (`asyncio.Queue`): The queue that receives decoded text chunks.
            response_parser (`ResponseParser`, *optional*): See :class:`DirectStreamer`.
        """
        from tokenizers.decoders import DecodeStream

        self._cb = cb_manager
        self._request_id = request_id
        self._loop = loop
        self._queue = queue
        self._tokenizer = tokenizer
        # See note in DirectStreamer: keep special tokens when a parser is active.
        skip_special_tokens = response_parser is None
        self._decode_stream = DecodeStream([], skip_special_tokens=skip_special_tokens)
        self._parser = response_parser
        self._prev_len = 0
        self.total_tokens = 0
        if self._parser is not None:
            for item in response_events_to_chunks(self._parser.initial_events):
                self._queue.put_nowait(item)

    def put(self, output: "GenerationOutput") -> None:
        """Decode new tokens from a CB ``GenerationOutput`` and push text to the queue."""
        new_tokens = output.generated_tokens[self._prev_len :]
        self._prev_len = len(output.generated_tokens)
        for token_id in new_tokens:
            self.total_tokens += 1
            text = self._decode_stream.step(self._tokenizer, token_id)
            if text is None:
                continue
            if self._parser is None:
                self._queue.put_nowait(text)
            else:
                for item in response_events_to_chunks(self._parser.feed(text)):
                    self._queue.put_nowait(item)

    def end(self) -> None:
        """Flush any final parser events, then push the EOS sentinel."""
        if self._parser is not None:
            try:
                _msg, final_events = self._parser.finalize()
            except (ValueError, RuntimeError) as e:
                # Generation stopped mid-region (likely max_new_tokens hit
                # inside a structured field like a tool call).
                logger.warning(f"ResponseParser.finalize() failed; dropping final tail events: {e}")
                final_events = []
            for item in response_events_to_chunks(final_events):
                self._queue.put_nowait(item)
        self._queue.put_nowait(None)

    def cancel(self) -> None:
        """Cancel the CB request."""
        self._cb.cancel_request(self._request_id)


def set_torch_seed(seed: int) -> None:
    """Set the PyTorch random seed for reproducible generation."""
    import torch

    torch.manual_seed(seed)


def reset_torch_cache() -> None:
    """Empty the CUDA cache if a GPU is available."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class InferenceThread:
    """Persistent thread for ``model.generate()`` calls.

    ``torch.compile`` with CUDA graphs stores state in thread-local storage.
    All inference must run on the same thread to avoid corrupted graph state.
    """

    def __init__(self):
        self._queue: Queue = Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            fn, args, kwargs, future, loop = self._queue.get()
            try:
                result = fn(*args, **kwargs)
                if loop is not None:
                    loop.call_soon_threadsafe(future.set_result, result)
                else:
                    future.set_result(result)
            except Exception as e:
                if loop is not None:
                    loop.call_soon_threadsafe(future.set_exception, e)
                else:
                    future.set_exception(e)

    def submit(self, fn, *args, **kwargs) -> Future:
        """Submit a callable to the inference thread. Returns a blocking Future."""
        future: Future = Future()
        self._queue.put((fn, args, kwargs, future, None))
        return future

    def async_submit(self, fn, *args, **kwargs) -> asyncio.Future:
        """Submit a callable to the inference thread. Returns an awaitable asyncio.Future."""
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._queue.put((fn, args, kwargs, future, loop))
        return future


class BaseGenerateManager(ABC):
    """Base class for generation managers.

    Subclasses:
    - :class:`GenerateManager` — sequential ``model.generate()`` on a persistent thread.
    - :class:`CBGenerateManager` — continuous batching with paged attention.
    """

    def init_cb(self, model: "PreTrainedModel", gen_config: "GenerationConfig") -> None:
        """Initialize continuous batching. No-op for non-CB managers."""

    @abstractmethod
    def generate_streaming(
        self,
        model: "PreTrainedModel",
        processor: "ProcessorMixin | PreTrainedTokenizerFast",
        inputs: dict,
        gen_config: "GenerationConfig",
        request_id: str,
        response_parser: "ResponseParser | None" = None,
    ) -> tuple[asyncio.Queue, "DirectStreamer | CBStreamer"]:
        """Start streaming generation.

        Args:
            model (`PreTrainedModel`): The loaded model.
            processor: The processor or tokenizer for decoding.
            inputs (`dict`): Tokenized inputs (tensors for sequential, lists for CB).
            gen_config (`GenerationConfig`): Generation parameters.
            request_id (`str`): Unique request identifier.
            response_parser (`ResponseParser`, *optional*): When provided, decoded
                text is fed through this parser and typed items (``str`` for content,
                :class:`ReasoningText`, :class:`ToolCall`) are pushed onto the queue
                instead of raw text.

        Returns:
            `tuple[asyncio.Queue, DirectStreamer | CBStreamer]`: A ``(queue, streamer)`` pair
            where *queue* yields ``str | ReasoningText | ToolCall | _StreamError | None``
            and *streamer* exposes ``.total_tokens`` and ``.cancel()``.
        """

    @abstractmethod
    async def generate_non_streaming(
        self,
        model: "PreTrainedModel",
        processor: "ProcessorMixin | PreTrainedTokenizerFast",
        inputs: dict,
        gen_config: "GenerationConfig",
        request_id: str,
    ) -> tuple[str, int, list[int]]:
        """Run generation to completion.

        Args:
            model (`PreTrainedModel`): The loaded model.
            processor: The processor or tokenizer for decoding.
            inputs (`dict`): Tokenized inputs (tensors for sequential, lists for CB).
            gen_config (`GenerationConfig`): Generation parameters.
            request_id (`str`): Unique request identifier.

        Returns:
            `tuple[str, int, list[int]]`: ``(text, input_len, generated_ids)``.
        """

    @abstractmethod
    def stop(self) -> None:
        """Stop the generation manager and free resources."""


class GenerateManager(BaseGenerateManager):
    """Sequential generation via ``model.generate()`` on a persistent thread."""

    def __init__(self):
        self._thread = InferenceThread()

    def generate_streaming(
        self,
        model: "PreTrainedModel",
        processor: "ProcessorMixin | PreTrainedTokenizerFast",
        inputs: dict,
        gen_config: "GenerationConfig",
        request_id: str,
        response_parser: "ResponseParser | None" = None,
    ) -> tuple[asyncio.Queue, DirectStreamer]:
        """Start streaming generation via ``model.generate()`` on the inference thread."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        # ProcessorMixin exposes the fast tokenizer as .tokenizer; PreTrainedTokenizerFast is already one.
        rust_tokenizer = getattr(processor, "tokenizer", processor)._tokenizer  # type: ignore[union-attr]
        streamer = DirectStreamer(rust_tokenizer, loop, queue, response_parser=response_parser)
        gen_kwargs = {**inputs, "streamer": streamer, "generation_config": gen_config, "tokenizer": processor}
        if hasattr(model, "has_talker"):
            gen_kwargs["generation_mode"] = "text"

        def _run() -> None:
            try:
                model.generate(**gen_kwargs)
            except _GenerationCancelled:
                loop.call_soon_threadsafe(queue.put_nowait, None)
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, _StreamError(str(e)))

        self.submit(_run)
        return queue, streamer

    async def generate_non_streaming(
        self,
        model: "PreTrainedModel",
        processor: "ProcessorMixin | PreTrainedTokenizerFast",
        inputs: dict,
        gen_config: "GenerationConfig",
        request_id: str,
    ) -> tuple[str, int, "torch.Tensor"]:
        """Run generation to completion via ``model.generate()`` on the inference thread."""
        # Multimodal models (e.g. Qwen2.5-Omni) may generate audio alongside text by default;
        # force text-only output since the serve layer only handles text
        generate_kwargs = {**inputs, "generation_config": gen_config, "tokenizer": processor}
        if hasattr(model, "has_talker"):
            generate_kwargs["generation_mode"] = "text"
        sequences = await self.async_submit(model.generate, **generate_kwargs)
        input_len = inputs["input_ids"].shape[-1]
        generated_ids = sequences[0, input_len:]
        text = processor.decode(generated_ids, skip_special_tokens=True)
        return text, input_len, generated_ids

    def submit(self, fn: Callable, *args, **kwargs) -> Future:
        """Submit a callable to the inference thread. Returns a blocking Future."""
        return self._thread.submit(fn, *args, **kwargs)

    def async_submit(self, fn: Callable, *args, **kwargs) -> asyncio.Future:
        """Submit a callable to the inference thread. Returns an awaitable asyncio.Future."""
        return self._thread.async_submit(fn, *args, **kwargs)

    def stop(self) -> None:
        pass  # inference thread is a daemon


class CBGenerateManager(BaseGenerateManager):
    """Continuous batching generation via paged attention.

    Translates between the handler's text-level asyncio.Queue and CB's
    token-level interface. Per-request: ``max_new_tokens``, ``eos_token_id``.

    The CB manager is initialized lazily on the first request via
    :meth:`ensure_initialized`, using that request's ``gen_config`` for shared
    sampling params (temperature, top_p, do_sample).

    .. todo:: Remove :meth:`init_cb` when CB supports per-request
       generation config. At that point, ``gen_config`` can be passed directly
       to ``add_request`` and the CB manager no longer needs a shared config.
    """

    def __init__(self, cb_config: "ContinuousBatchingConfig | None" = None):
        self._cb: ContinuousBatchingManager | None = None
        self._cb_config = cb_config

    def init_cb(self, model: "PreTrainedModel", gen_config: "GenerationConfig") -> None:
        """Initialize the CB manager on first call with the request's generation config.

        .. todo:: Remove when CB supports per-request generation config.

        Args:
            model (`PreTrainedModel`): The loaded model (must support ``init_continuous_batching``).
            gen_config (`GenerationConfig`): Generation config used for shared sampling params.
        """
        if self._cb is not None:
            return

        self._cb = model.init_continuous_batching(
            generation_config=gen_config, continuous_batching_config=self._cb_config
        )
        self._cb.start()

    def is_alive(self) -> bool:
        """Whether the CB worker is healthy. ``True`` before ``init_cb()`` is called."""
        return self._cb is None or self._cb.fatal_error is None

    def _check_alive(self, request_id: str) -> None:
        """Raise :class:`CBWorkerDeadError` if the CB worker has died.

        Called at request entry to fail fast — submitting to a dead worker would otherwise
        enqueue the request into a void where it never gets processed.
        """
        if self._cb is not None and self._cb.fatal_error is not None:
            raise CBWorkerDeadError(
                f"CB worker is dead and cannot accept request {request_id}: {self._cb.fatal_error}"
            )

    def generate_streaming(
        self,
        model: "PreTrainedModel",
        processor: "ProcessorMixin | PreTrainedTokenizerFast",
        inputs: dict,
        gen_config: "GenerationConfig",
        request_id: str,
        response_parser: "ResponseParser | None" = None,
    ) -> tuple[asyncio.Queue, CBStreamer]:
        """Start streaming CB generation. Registers a per-request output handler."""
        cb = self._cb
        if cb is None:
            raise RuntimeError("CB manager not initialized. Call `init_cb()` first.")
        self._check_alive(request_id)

        loop = asyncio.get_running_loop()
        text_queue: asyncio.Queue = asyncio.Queue()

        input_ids = inputs["input_ids"]
        request_id = cb.add_request(
            input_ids,
            request_id=request_id,
            streaming=True,
            max_new_tokens=gen_config.max_new_tokens,
            eos_token_id=gen_config.eos_token_id,
        )
        # ProcessorMixin exposes the fast tokenizer as .tokenizer; PreTrainedTokenizerFast is already one.
        rust_tokenizer = getattr(processor, "tokenizer", processor)._tokenizer  # type: ignore[union-attr]
        streamer = CBStreamer(
            self._cb,
            request_id,
            rust_tokenizer,
            loop,
            text_queue,
            response_parser=response_parser,
        )

        # Register a direct callback: the dispatcher calls this on the event loop with each GenerationOutput.
        # This decodes tokens and pushes text straight to the SSE text_queue
        def _on_output(output):
            try:
                streamer.put(output)
                # ``error`` is set together with ``status = FAILED`` in CB's _handle_request_error.
                # Surface it as an end-of-stream error so the SSE handler can emit it and close,
                # instead of leaving the client hanging on a stream that will never end.
                if output.error is not None:
                    text_queue.put_nowait(_StreamError(output.error))
                    streamer.end()
                elif output.is_finished():
                    streamer.end()
            except Exception as e:
                text_queue.put_nowait(_StreamError(str(e)))

        cb.register_result_handler(request_id, _on_output)
        return text_queue, streamer

    async def generate_non_streaming(
        self,
        model: "PreTrainedModel",
        processor: "ProcessorMixin | PreTrainedTokenizerFast",
        inputs: dict,
        gen_config: "GenerationConfig",
        request_id: str,
    ) -> tuple[str, int, list[int]]:
        """Run non-streaming CB generation. Registers a handler that resolves an asyncio.Future on completion."""
        cb = self._cb
        if cb is None:
            raise RuntimeError("CB manager not initialized. Call `init_cb()` first.")
        self._check_alive(request_id)

        input_ids = inputs["input_ids"]
        input_len = len(input_ids)

        # Register future BEFORE add_request to avoid race with fast completion
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        def _on_result(result):
            if not future.done():
                future.set_result(result)

        cb.register_result_handler(request_id, _on_result)

        cb.add_request(
            input_ids,
            request_id=request_id,
            max_new_tokens=gen_config.max_new_tokens,
            streaming=False,
            eos_token_id=gen_config.eos_token_id,
        )
        result = await future
        # CB signals a failed request by setting ``error`` (and ``status = FAILED``) on the
        # delivered GenerationOutput, often with empty ``generated_tokens``. Surface it instead
        # of returning an empty success that downstream parsing/decoding would silently mask.
        # If the worker itself died, route to CBWorkerDeadError so the client gets the same 503
        # as requests submitted post-crash; otherwise it's a per-request failure (e.g. unsupported
        # logit-processor kwarg) and a plain RuntimeError -> 500 is appropriate.
        if result.error is not None:
            if cb.fatal_error is not None:
                raise CBWorkerDeadError(f"CB worker died during request {request_id}: {result.error}")
            raise RuntimeError(f"CB generation failed for {request_id}: {result.error}")
        generated_ids = result.generated_tokens
        text = processor.decode(generated_ids, skip_special_tokens=True)
        return text, input_len, generated_ids

    @property
    def scheduler(self) -> "Scheduler":
        """The CB scheduler (for testing/monitoring)."""
        if self._cb is None or self._cb.batch_processor is None:
            raise RuntimeError("Continuous batching processor not initialized.")
        return self._cb.batch_processor.scheduler

    def stop(self) -> None:
        if self._cb is not None:
            self._cb.stop(block=True, timeout=2)


class GenerationState:
    """Shared generation state across all handlers.

    Manages per-model :class:`GenerateManager` instances (each with its own
    :class:`InferenceThread` so different models can run concurrently while
    ``torch.compile`` / CUDA graphs require same-model-same-thread) and a
    single :class:`CBGenerateManager` for continuous batching.

    Args:
        continuous_batching (`bool`, *optional*, defaults to `False`):
            Whether to use continuous batching with paged attention instead of
            sequential ``model.generate()`` calls.
    """

    def __init__(
        self,
        continuous_batching: bool = False,
        compile: bool = False,
        cb_config: "ContinuousBatchingConfig | None" = None,
    ):
        self._continuous_batching = continuous_batching
        self._compile = compile
        self._cb_config = cb_config
        self._generate_managers: dict[str, GenerateManager] = {}
        self._cb_manager: CBGenerateManager | None = None
        self._cb_model_id: str | None = None

    def use_continuous_batching(self, model: "PreTrainedModel", modality: Modality) -> bool:
        """Check if continuous batching can be used for this model and modality.

        Args:
            model (`PreTrainedModel`): The loaded model.
            modality (`Modality`): The detected model modality (LLM, VLM, etc.).

        Returns:
            `bool`: ``True`` if CB is enabled and the model supports it, ``False`` otherwise.
        """
        if not self._continuous_batching:
            return False
        can = hasattr(model, "init_continuous_batching") and modality == Modality.LLM
        if not can:
            logger.warning_once(
                f"{model.__class__.__name__} does not support continuous batching. "
                "Falling back to sequential generation."
            )
        return can

    def get_manager(self, model_id: str, use_cb: bool = False) -> BaseGenerateManager:
        """Return a per-model generation manager, lazily created on first request.

        Args:
            model_id (`str`): The model ID in ``'model_id@revision'`` format.
            use_cb (`bool`): Whether to return a CB manager or a sequential one.

        Returns:
            `BaseGenerateManager`: Either a `GenerateManager` or `CBGenerateManager`.
        """
        if use_cb:
            if self._cb_model_id != model_id:
                if self._cb_manager is not None:
                    self._cb_manager.stop()
                    self._cb_manager = None
            if self._cb_manager is None:
                self._cb_manager = CBGenerateManager(cb_config=self._cb_config)
                self._cb_model_id = model_id
            return self._cb_manager
        if model_id not in self._generate_managers:
            self._generate_managers[model_id] = GenerateManager()
        return self._generate_managers[model_id]

    def shutdown(self) -> None:
        """Stop any active generation managers."""
        if self._cb_manager is not None:
            self._cb_manager.stop()
            self._cb_manager = None

    def is_cb_alive(self) -> bool:
        """Whether the CB worker is healthy. ``True`` if CB is disabled or not yet initialized."""
        return self._cb_manager is None or self._cb_manager.is_alive()


class BaseHandler:
    """Shared logic for chat completion and responses handlers.

    Provides model resolution, generation config building, and SSE formatting.
    Generation is delegated to the shared :class:`GenerationState`.

    Args:
        model_manager (`ModelManager`):
            Handles model loading, caching, and lifecycle.
        generation_state (`GenerationState`):
            Shared state managing per-model generation managers.
    """

    _valid_params_class: type | None = None
    _unused_fields: set[str] = set()

    def __init__(
        self,
        model_manager: "ModelManager",
        generation_state: GenerationState,
        chat_template_kwargs: dict | None = None,
    ):
        self.model_manager = model_manager
        self.generation_state = generation_state
        self.chat_template_kwargs = chat_template_kwargs or {}

    def _validate_request(self, body: dict) -> None:
        """Validate request fields against the handler's params class and unused fields."""
        from fastapi import HTTPException

        input_keys = set(body.keys())
        if self._valid_params_class is not None:
            unexpected = input_keys - getattr(self._valid_params_class, "__mutable_keys__", set())
            if unexpected:
                raise HTTPException(status_code=422, detail=f"Unexpected fields in the request: {unexpected}")
        unused = input_keys & self._unused_fields
        if unused:
            logger.warning_once(f"Ignoring unsupported fields in the request: {unused}")

    @staticmethod
    def chunk_to_sse(chunk: "pydantic.BaseModel") -> str:
        """Format a pydantic model as an SSE ``data:`` line."""
        return f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

    def _resolve_model(self, body: dict) -> tuple[str, "PreTrainedModel", "ProcessorMixin | PreTrainedTokenizerFast"]:
        """Apply force_model, load model + processor.

        Returns ``(model_id, model, processor)``.
        """
        from fastapi import HTTPException

        if self.model_manager.force_model is not None:
            requested = body.get("model")
            if requested is not None and requested != self.model_manager.force_model:
                raise HTTPException(
                    status_code=400,
                    detail=(f"Server is pinned to '{self.model_manager.force_model}'; requested '{requested}'."),
                )
            body["model"] = self.model_manager.force_model

        model_id = self.model_manager.process_model_name(body["model"])
        model, processor = self.model_manager.load_model_and_processor(model_id)

        return model_id, model, processor

    def _build_generation_config(
        self, body: dict, model_generation_config: "GenerationConfig", use_cb: bool = False
    ) -> "GenerationConfig":
        """Build a GenerationConfig from shared params (temperature, top_p, seed, generation_config JSON).

        Subclasses should call ``super()._build_generation_config(...)`` then apply
        endpoint-specific params (``max_tokens``, ``max_output_tokens``, etc.).

        Args:
            body (`dict`):
                The raw request body.
            model_generation_config (`GenerationConfig`):
                The model's default generation config (will be deep-copied).
            use_cb (`bool`, *optional*, defaults to `False`):
                Whether continuous batching is active. If ``True``, disables the model's
                internal KV cache (CB manages its own paged cache).

        Returns:
            `GenerationConfig`: A new config with request-specific overrides applied.
        """
        from transformers import GenerationConfig

        if body.get("generation_config") is not None:
            generation_config = GenerationConfig(**json.loads(body["generation_config"]))
        else:
            generation_config = copy.deepcopy(model_generation_config)
            if generation_config.max_new_tokens is None or generation_config.max_new_tokens < 1024:
                generation_config.max_new_tokens = 1024

        if body.get("temperature") is not None:
            generation_config.temperature = float(body["temperature"])
            if float(body["temperature"]) == 0.0:
                generation_config.do_sample = False
        if body.get("top_p") is not None:
            generation_config.top_p = float(body["top_p"])
        if body.get("seed") is not None:
            set_torch_seed(body["seed"])

        # --compile flag: use static cache + torch.compile for faster decode
        if self.generation_state._compile and generation_config.cache_implementation is None:
            generation_config.cache_implementation = "static"

        # CB manages its own paged KV cache
        if use_cb:
            generation_config.use_cache = False

        # TODO: add prefix caching for the non-CB path (reuse KV cache across multi-turn conversations)

        return generation_config

    @staticmethod
    def get_processor_inputs_from_messages(messages: list[dict], modality: Modality) -> list[dict]:
        """Convert OpenAI-format messages to the format expected by HF processors.

        All modalities extract text. VLM additionally handles ``image_url`` and ``video_url``.
        MULTIMODAL handles all of the above plus ``input_audio`` and ``audio_url``.
        For LLMs, the content parts are collapsed into a plain text string.

        Args:
            messages (`list[dict]`): OpenAI-format chat messages.
            modality (`Modality`): The model modality (LLM, VLM, or MULTIMODAL).

        Returns:
            `list[dict]`: Processor-compatible messages.
        """
        processor_inputs = []

        for message in messages:
            parsed = {"role": message["role"], "content": []}

            # Parse function.arguments back to a dict — chat templates iterate it as a mapping.
            if "tool_calls" in message:
                tool_calls = []
                for tc in message["tool_calls"]:
                    tc = copy.deepcopy(tc)
                    if isinstance(tc["function"]["arguments"], str):
                        tc["function"]["arguments"] = json.loads(tc["function"]["arguments"])
                    tool_calls.append(tc)
                parsed["tool_calls"] = tool_calls
            if "tool_call_id" in message:
                parsed["tool_call_id"] = message["tool_call_id"]

            # When tool_calls are present, ignore content — it's either empty or contains
            # raw tool call markup that would confuse the chat template if rendered.
            raw_content = [] if "tool_calls" in message else (message.get("content") or [])
            if isinstance(raw_content, str):
                raw_content = [{"type": "text", "text": raw_content}]

            for content in raw_content:
                content_type = content["type"]
                # Text: chat completions ("text") and Responses API ("input_text")
                if content_type in ("text", "input_text", "output_text"):
                    parsed["content"].append({"type": "text", "text": content["text"]})
                # Image: chat completions ("image_url") and Responses API ("input_image")
                elif content_type in ("image_url", "input_image") and modality in (Modality.VLM, Modality.MULTIMODAL):
                    # chat completions: {"image_url": {"url": "..."}}, Responses API: {"image_url": "..."}
                    url = content["image_url"]
                    if isinstance(url, dict):
                        url = url["url"]
                    parsed["content"].append({"type": "image", "url": url})
                # Audio: unlike images, load_audio doesn't accept raw base64 — wrap as a data URI
                elif content_type == "input_audio" and modality == Modality.MULTIMODAL:
                    input_audio = content["input_audio"]
                    fmt = input_audio.get("format", "wav") if isinstance(input_audio, dict) else "wav"
                    audio_b64 = input_audio["data"]
                    parsed["content"].append({"type": "audio", "url": f"data:audio/{fmt};base64,{audio_b64}"})
                # Extensions (not part of the OpenAI API standard)
                elif content_type == "video_url" and modality in (Modality.VLM, Modality.MULTIMODAL):
                    parsed["content"].append({"type": "video", "url": content["video_url"]["url"]})
                elif content_type == "audio_url" and modality == Modality.MULTIMODAL:
                    parsed["content"].append({"type": "audio", "url": content["audio_url"]["url"]})

            # LLMs expect plain text, not a list of content parts
            if modality == Modality.LLM:
                parsed["content"] = " ".join(c["text"] for c in parsed["content"])

            processor_inputs.append(parsed)
        return processor_inputs
