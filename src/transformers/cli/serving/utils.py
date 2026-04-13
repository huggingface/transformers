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
import base64
import copy
import enum
import json
import re
import tempfile
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import Future
from io import BytesIO
from queue import Queue
from typing import TYPE_CHECKING

from transformers.utils import logging


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
    STT = "STT"
    TTS = "TTS"


class _StreamError:
    """Sentinel to signal an error from the generate thread."""

    def __init__(self, msg: str):
        self.msg = msg


class _GenerationCancelled(Exception):
    """Raised inside ``DirectStreamer.put()`` to abort ``model.generate()``."""


# Model-specific tokens that mark the start/end of a tool call block.
# TODO: extract these from the chat template at runtime instead of hardcoding.
# Qwen/Hermes use <tool_call>/<tool_call>, Mistral uses [TOOL_CALLS], etc.
# The markers are defined in each model's Jinja chat template.
_TOOL_CALL_TOKENS = {
    "qwen": {
        "start": "<tool_call>",
        "end": "</tool_call>",
    },
}


def detect_tool_format(model: "PreTrainedModel") -> dict | None:
    """Return the tool call token format for a model, if supported.

    Args:
        model (`PreTrainedModel`): The loaded model.

    Returns:
        `dict | None`: A dict ``{"start": str, "end": str}`` with the model's tool call
        delimiters, or ``None`` if the model family is not recognized.
    """
    architecture = model.config.architectures[0].lower()
    for family in _TOOL_CALL_TOKENS:
        if family in architecture:
            return _TOOL_CALL_TOKENS[family]
    return None


class ToolCallParser:
    """Parses tool calls from model output.

    The model emits tool calls as structured text between start/end tokens
    (e.g. ``<tool_call>{"name": "fn", "arguments": {...}}</tool_call>``).

    **Streaming** (``feed``): buffers tokens between start/end markers, parses
    the complete block when the end marker is seen, returns a ``ChoiceDeltaToolCall``.

    **Non-streaming** (``parse``): extracts all tool call blocks from complete text.

    Usage::

        parser = ToolCallParser(tool_format={"start": ..., "end": ...})
        for text_chunk in streamer:
            result = parser.feed(text_chunk)
            if result is None:
                # Normal text — emit as content
            elif result is ToolCallParser.CONSUMED:
                # Buffering — skip
            else:
                # result is a ChoiceDeltaToolCall — emit it
    """

    def __init__(self, tool_format: dict):
        self._tokens = tool_format
        self._inside = False
        self._buffer = ""

    # Sentinel: token was consumed by the parser but produced no output.
    CONSUMED = object()

    def feed(self, text: str) -> object | dict | None:
        """Feed a text chunk (streaming).

        Returns:
        - ``None`` — normal text, not a tool token. Emit as content.
        - ``CONSUMED`` — token consumed internally (buffering/markers). Skip.
        - A ``ChoiceDeltaToolCall`` — emit as a tool call delta.
        """
        if text.strip() == self._tokens["start"]:
            self._inside = True
            self._buffer = ""
            return self.CONSUMED

        if text.strip() == self._tokens["end"]:
            self._inside = False
            block = self._buffer.strip()
            self._buffer = ""
            return self._parse_block(block) or self.CONSUMED

        if self._inside:
            self._buffer += text
            return self.CONSUMED

        return None

    @staticmethod
    def _extract_name_and_args(block: str) -> tuple[str, str] | None:
        """Extract (name, arguments_json) from a tool call block, or None if invalid."""
        if not block:
            return None
        parsed = json.loads(block)
        name = parsed.get("name")
        if name is None:
            return None
        arguments = parsed.get("arguments", {})
        return name, json.dumps(arguments)

    @staticmethod
    def parse(text: str, tool_format: dict) -> list[dict] | None:
        """Parse tool calls from complete text.

        Returns a list of ``{"name": str, "arguments": str}`` dicts, or ``None`` if none found.
        """
        start, end = tool_format["start"], tool_format["end"]
        tool_calls = []
        pos = 0
        while True:
            s = text.find(start, pos)
            if s < 0:
                break
            e = text.find(end, s + len(start))
            if e < 0:
                break
            result = ToolCallParser._extract_name_and_args(text[s + len(start) : e].strip())
            if result is not None:
                tool_calls.append({"name": result[0], "arguments": result[1]})
            pos = e + len(end)
        return tool_calls if tool_calls else None

    def _parse_block(self, block: str) -> dict | None:
        """Parse a buffered tool call block. Returns ``{"name": str, "arguments": str}`` or None."""
        result = self._extract_name_and_args(block)
        if result is None:
            return None
        return {"name": result[0], "arguments": result[1]}


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

    class ProgressTqdm(base_tqdm):
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
        skip_special_tokens: bool = True,
    ):
        """
        Args:
            tokenizer: The Rust tokenizer (``tokenizer._tokenizer``).
            loop (`asyncio.AbstractEventLoop`): The event loop to push decoded text to.
            queue (`asyncio.Queue`): The queue that receives decoded text chunks.
            skip_special_tokens (`bool`, *optional*, defaults to `True`):
                Whether to strip special tokens during decoding.
        """
        from tokenizers.decoders import DecodeStream

        self._tokenizer = tokenizer
        self._loop = loop
        self._queue = queue
        self._decode_stream = DecodeStream([], skip_special_tokens)
        self._first = True
        self._cancelled = threading.Event()
        self.total_tokens = 0

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
            if text is not None:
                self._loop.call_soon_threadsafe(self._queue.put_nowait, text)

    def end(self) -> None:
        """Called by ``model.generate()`` when generation is complete."""
        self._loop.call_soon_threadsafe(self._queue.put_nowait, None)

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
    ):
        """
        Args:
            cb_manager (`ContinuousBatchingManager`): The CB manager instance.
            request_id (`str`): The request ID to track in the CB scheduler.
            tokenizer: The Rust tokenizer (``tokenizer._tokenizer``).
            loop (`asyncio.AbstractEventLoop`): The event loop to push decoded text to.
            queue (`asyncio.Queue`): The queue that receives decoded text chunks.
        """
        from tokenizers.decoders import DecodeStream

        self._cb = cb_manager
        self._request_id = request_id
        self._loop = loop
        self._queue = queue
        self._tokenizer = tokenizer
        self._decode_stream = DecodeStream([], True)
        self._prev_len = 0
        self.total_tokens = 0

    def put(self, output: "GenerationOutput") -> None:
        """Decode new tokens from a CB ``GenerationOutput`` and push text to the queue."""
        new_tokens = output.generated_tokens[self._prev_len :]
        self._prev_len = len(output.generated_tokens)
        for token_id in new_tokens:
            self.total_tokens += 1
            text = self._decode_stream.step(self._tokenizer, token_id)
            if text is not None:
                self._queue.put_nowait(text)

    def end(self) -> None:
        """Signal end of stream."""
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

    @abstractmethod
    def generate_streaming(
        self,
        model: "PreTrainedModel",
        processor: "ProcessorMixin | PreTrainedTokenizerFast",
        inputs: dict,
        gen_config: "GenerationConfig",
        request_id: str,
    ) -> tuple[asyncio.Queue, "DirectStreamer | CBStreamer"]:
        """Start streaming generation.

        Args:
            model (`PreTrainedModel`): The loaded model.
            processor: The processor or tokenizer for decoding.
            inputs (`dict`): Tokenized inputs (tensors for sequential, lists for CB).
            gen_config (`GenerationConfig`): Generation parameters.
            request_id (`str`): Unique request identifier.

        Returns:
            `tuple[asyncio.Queue, DirectStreamer | CBStreamer]`: A ``(queue, streamer)`` pair
            where *queue* yields ``str | _StreamError | None`` and *streamer* exposes
            ``.total_tokens`` and ``.cancel()``.
        """

    @abstractmethod
    def generate_non_streaming(
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
    ) -> tuple[asyncio.Queue, DirectStreamer]:
        """Start streaming generation via ``model.generate()`` on the inference thread."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        # ProcessorMixin exposes the fast tokenizer as .tokenizer; PreTrainedTokenizerFast is already one.
        rust_tokenizer = getattr(processor, "tokenizer", processor)._tokenizer
        streamer = DirectStreamer(rust_tokenizer, loop, queue, skip_special_tokens=True)
        gen_kwargs = {**inputs, "streamer": streamer, "generation_config": gen_config, "tokenizer": processor}

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
        sequences = await self.async_submit(
            model.generate, **inputs, generation_config=gen_config, tokenizer=processor
        )
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
        self._cb = None
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

    def generate_streaming(
        self,
        model: "PreTrainedModel",
        processor: "ProcessorMixin | PreTrainedTokenizerFast",
        inputs: dict,
        gen_config: "GenerationConfig",
        request_id: str,
    ) -> tuple[asyncio.Queue, CBStreamer]:
        """Start streaming CB generation. Registers a per-request output handler."""
        loop = asyncio.get_running_loop()
        text_queue: asyncio.Queue = asyncio.Queue()

        input_ids = inputs["input_ids"]
        request_id = self._cb.add_request(
            input_ids,
            request_id=request_id,
            streaming=True,
            max_new_tokens=gen_config.max_new_tokens,
            eos_token_id=gen_config.eos_token_id,
        )
        # ProcessorMixin exposes the fast tokenizer as .tokenizer; PreTrainedTokenizerFast is already one.
        rust_tokenizer = getattr(processor, "tokenizer", processor)._tokenizer
        streamer = CBStreamer(self._cb, request_id, rust_tokenizer, loop, text_queue)

        # Register a direct callback: the dispatcher calls this on the event loop with each GenerationOutput.
        # This decodes tokens and pushes text straight to the SSE text_queue
        def _on_output(output):
            try:
                streamer.put(output)
                if output.is_finished():
                    streamer.end()
            except Exception as e:
                text_queue.put_nowait(_StreamError(str(e)))

        self._cb.register_result_handler(request_id, _on_output)
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
        input_ids = inputs["input_ids"]
        input_len = len(input_ids)

        # Register future BEFORE add_request to avoid race with fast completion
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        def _on_result(result):
            if not future.done():
                future.set_result(result)

        self._cb.register_result_handler(request_id, _on_result)

        self._cb.add_request(
            input_ids,
            request_id=request_id,
            max_new_tokens=gen_config.max_new_tokens,
            streaming=False,
            eos_token_id=gen_config.eos_token_id,
        )
        result = await future
        if result is None:
            raise RuntimeError(f"CB manager stopped before producing a result for {request_id}")
        generated_ids = result.generated_tokens
        text = processor.decode(generated_ids, skip_special_tokens=True)
        return text, input_len, generated_ids

    @property
    def scheduler(self) -> "Scheduler":
        """The CB scheduler (for testing/monitoring)."""
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
    ):
        self.model_manager = model_manager
        self.generation_state = generation_state

    def _validate_request(self, body: dict) -> None:
        """Validate request fields against the handler's params class and unused fields."""
        from fastapi import HTTPException

        input_keys = set(body.keys())
        if self._valid_params_class is not None:
            unexpected = input_keys - self._valid_params_class.__mutable_keys__
            if unexpected:
                raise HTTPException(status_code=422, detail=f"Unexpected fields in the request: {unexpected}")
        unused = input_keys & self._unused_fields
        if unused:
            logger.warning_once(f"Ignoring unsupported fields in the request: {unused}")

    @staticmethod
    def chunk_to_sse(chunk: "str | pydantic.BaseModel") -> str:
        """Format a pydantic model or string as an SSE ``data:`` line."""
        if isinstance(chunk, str):
            return chunk if chunk.startswith("data: ") else f"data: {chunk}\n\n"
        return f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

    def _resolve_model(self, body: dict) -> tuple[str, "PreTrainedModel", "ProcessorMixin | PreTrainedTokenizerFast"]:
        """Apply force_model, load model + processor.

        Returns ``(model_id, model, processor)``.
        """
        if self.model_manager.force_model is not None:
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

        For LLMs, collapses list content blocks into plain text. For VLMs, converts
        ``image_url`` content parts (including base64) into ``{"type": "image", "url": ...}``
        entries that HF processors understand.

        Args:
            messages (`list[dict]`): OpenAI-format chat messages.
            modality (`Modality`): Whether the model is an LLM or VLM.

        Returns:
            `list[dict]`: Processor-compatible messages.
        """
        processor_inputs = []

        for message in messages:
            parsed = {"role": message["role"], "content": []}

            content = message.get("content")
            if modality == Modality.LLM:
                if isinstance(content, str):
                    parsed["content"] = content
                elif isinstance(content, list):
                    texts = [c["text"] for c in content if c["type"] == "text"]
                    parsed["content"] = " ".join(texts)

            elif modality == Modality.VLM:
                if isinstance(content, str):
                    parsed["content"].append({"type": "text", "text": content})
                elif isinstance(content, list):
                    for content_block in content:
                        if content_block["type"] == "text":
                            parsed["content"].append(content_block)
                        elif content_block["type"] == "image_url":
                            from PIL import Image

                            url = content_block["image_url"]["url"]
                            if "base64" in url:
                                image_data = re.sub("^data:image/.+;base64,", "", url)
                                image = Image.open(BytesIO(base64.b64decode(image_data)))
                                file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                                file.close()  # close handle immediately after creation
                                image.save(file.name)
                                url = file.name
                            # We don't delete the file as tne caller need it (via the `url` key).
                            # TODO: Better approach to avoid file accumulation.
                            parsed["content"].append({"type": "image", "url": url})

            processor_inputs.append(parsed)
        return processor_inputs
