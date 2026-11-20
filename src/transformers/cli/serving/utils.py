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
import gc
import json
import multiprocessing
import os
import queue
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import TYPE_CHECKING, Any

from huggingface_hub import scan_cache_dir
from tqdm import tqdm

from transformers.utils import logging
from transformers.utils.chat_parsing import ResponseParser
from transformers.utils.chat_template_utils import _REASONING_FIELDS
from transformers.utils.import_utils import is_serve_available


if is_serve_available():
    from starlette.datastructures import Headers, MutableHeaders
    from starlette.responses import StreamingResponse


if TYPE_CHECKING:
    import pydantic
    import tokenizers

    from transformers import (
        ContinuousBatchingConfig,
        GenerationConfig,
        PretrainedConfig,
        PreTrainedModel,
        PreTrainedTokenizerFast,
        ProcessorMixin,
    )
    from transformers.decision import DecisionConfig
    from transformers.generation.continuous_batching import ContinuousBatchingManager
    from transformers.pipelines import DecisionPipeline

    from .model_manager import EngineClient, ModelManager


logger = logging.get_logger(__name__)
# A line per request (its body at debug): off unless `--enable-log-requests`, see `set_serve_log_level`
request_logger = logging.get_logger("transformers.cli.serving.requests")


# Model ids
X_REQUEST_ID = "x-request-id"


def parse_model_id(model_id: str) -> tuple[str, str | None, str | None]:
    """Split a model name, `repo[@revision][:file.gguf]`, into its repo, revision and GGUF file:
    `"org/model@dev:weights.gguf"` -> `("org/model", "dev", "weights.gguf")`.

    Only a `.gguf` file after the last colon counts as one. An existing local path is the repo as it is: it has no
    revision, and an `@` or a colon in it is part of the path.
    """
    repo, colon, gguf_file = model_id.rpartition(":")
    if not colon or not gguf_file.endswith(".gguf"):
        repo, gguf_file = model_id, None
    if os.path.exists(repo):
        return repo, None, gguf_file
    repo, _, revision = repo.partition("@")
    return repo, revision or None, gguf_file


def canonical_model_id(model_id: str) -> str:
    """The one name a model goes by: `repo@revision` for a Hub repo (`@main` by default), an existing local path as
    it is, then `:file.gguf` if it names a GGUF file."""
    repo, revision, gguf_file = parse_model_id(model_id)
    if not os.path.exists(repo):
        repo = f"{repo}@{revision or 'main'}"
    return repo if gguf_file is None else f"{repo}:{gguf_file}"


def model_owner(model_id: str) -> str:
    """Owner of a model for `/v1/models`: `"local"` for a local path, the organization of a Hub repo."""
    if os.path.exists(model_id):
        return "local"
    return model_id.split("/")[0] if "/" in model_id else ""


@lru_cache
def get_gen_models(cache_dir: str | None = None) -> list[dict]:
    """List generative models (LLMs and VLMs) available in the HuggingFace cache.

    Args:
        cache_dir (`str`, *optional*): Path to the HuggingFace cache directory.
            Defaults to the standard cache location.

    Returns:
        `list[dict]`: OpenAI-compatible model list entries with ``id``, ``object``, etc.
    """
    from transformers.models.auto.modeling_auto import (
        MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
        MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
        MODEL_FOR_MULTIMODAL_LM_MAPPING_NAMES,
    )

    generative_architectures = {
        *MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values(),
        *MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.values(),
        *MODEL_FOR_MULTIMODAL_LM_MAPPING_NAMES.values(),
    }

    generative_models = []
    for repo in tqdm(scan_cache_dir(cache_dir).repos):
        if repo.repo_type != "model":
            continue
        owned_by = repo.repo_id.split("/")[0] if "/" in repo.repo_id else ""

        for ref, revision_info in repo.refs.items():
            # One quantization per file, each a separate model, so list files rather than the repo.
            gguf_files = sorted(f.file_name for f in revision_info.files if f.file_name.endswith(".gguf"))
            model_ids = [f"{repo.repo_id}:{gguf_file}" for gguf_file in gguf_files]

            # The repo itself is listed when its config names a generative architecture.
            config_path = next((f.file_path for f in revision_info.files if f.file_name == "config.json"), None)
            config = json.loads(config_path.read_text()) if config_path is not None else None
            architectures = (config.get("architectures") or []) if isinstance(config, dict) else []
            if any(architecture in generative_architectures for architecture in architectures):
                model_ids.append(repo.repo_id if ref == "main" else f"{repo.repo_id}@{ref}")

            for model_id in model_ids:
                generative_models.append(
                    {"owned_by": owned_by, "id": model_id, "object": "model", "created": int(repo.last_modified)}
                )

    return generative_models


class Modality(enum.Enum):
    LLM = "LLM"
    VLM = "VLM"
    MULTIMODAL = "MULTIMODAL"  # supports text, image, video, and audio
    STT = "STT"
    TTS = "TTS"


# Small helpers
def prompt_length(inputs: dict) -> int:
    """The number of tokens of a prompt: lists for continuous batching, tensors otherwise."""
    input_ids = inputs["input_ids"]
    return len(input_ids) if isinstance(input_ids, list) else input_ids.shape[-1]


def set_serve_log_level(level: int, log_requests: bool = False) -> None:
    """Serve's messages (its `transformers.cli` loggers) at `level`, with a line per request only if `log_requests`.
    The library's other messages, loading a model included, keep their own level (warning by default, see
    `TRANSFORMERS_VERBOSITY`), so that info stays readable."""
    logging.get_logger("transformers.cli").setLevel(level)
    request_logger.setLevel(level if log_requests else logging.WARNING)  # it only writes info and debug lines


def set_torch_seed(seed: int) -> None:
    """Set the PyTorch random seed for reproducible generation."""
    import torch

    torch.manual_seed(seed)


def reset_torch_cache() -> None:
    """Empty the CUDA cache if a GPU is available."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def exit_with_parent() -> None:
    """In a process the server spawned: exits once its parent does, even one killed without cleanup (`kill -9`), which
    would leave it running (a model's process holding its GPU memory, an API server its port)."""
    parent = multiprocessing.parent_process()
    if parent is None:  # not spawned
        return

    def wait_and_exit() -> None:
        parent.join()  # returns once the parent is gone
        os._exit(1)

    threading.Thread(target=wait_and_exit, name="exit-with-parent", daemon=True).start()


def freeze_garbage_collector() -> None:
    """Runs a full collection, then freezes all the objects the garbage collector tracks. To call once a process is
    initialized (model loaded, server built), to reduce the collector's overhead from static objects while serving."""
    gc.collect()
    gc.freeze()


class RequestIdMiddleware:
    """
    Gets or sets the request ID in the header. A plain ASGI middleware: `@app.middleware("http")` would pass every
    chunk of a streamed response through an extra task.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request_id = Headers(scope=scope).get(X_REQUEST_ID) or str(uuid.uuid4())
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_with_request_id(message) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)[X_REQUEST_ID] = request_id
            await send(message)

        await self.app(scope, receive, send_with_request_id)


class CancelOnDisconnect:
    """
    Cancels the handling of a request when its client leaves before the answer is complete: nobody is there to read
    it. Generation stops with it (see `EngineClient._send_request`).
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        handling = asyncio.current_task()
        messages: asyncio.Queue = asyncio.Queue()
        answered = False

        async def send_and_note_the_end(message) -> None:
            nonlocal answered
            await send(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                answered = True

        async def pass_messages_on() -> None:
            while True:
                message = await receive()
                messages.put_nowait(message)
                if message["type"] == "http.disconnect":
                    if not answered:
                        handling.cancel()
                    return

        watcher = asyncio.create_task(pass_messages_on())
        try:
            await self.app(scope, messages.get, send_and_note_the_end)
        except asyncio.CancelledError:
            if not watcher.done():  # the server stopping, not the client leaving
                raise
            handling.uncancel()
        finally:
            watcher.cancel()


if is_serve_available():

    class EventStream(StreamingResponse):
        """A `StreamingResponse` without the task group Starlette uses before ASGI 2.4 to notice the client leaving:
        `CancelOnDisconnect` does it."""

        async def __call__(self, scope, receive, send) -> None:
            spec_version = tuple(map(int, scope.get("asgi", {}).get("spec_version", "2.0").split(".")))
            if spec_version >= (2, 4):  # Starlette streams without the task group then
                return await super().__call__(scope, receive, send)
            await self.stream_response(send)
            if self.background is not None:
                await self.background()


# Errors
class ModelNotServedError(ValueError):
    """A request named another model than the only one the server serves: a 400 for its client."""


class EngineDeadError(RuntimeError):
    """A model's engine died, for good, with the cause in its message: its process crashed, or its continuous batching
    did. Its requests fail with this error, a 503 for their clients, and the server stops (see `EngineClient.is_dead`).
    """


class _GenerationCancelled(Exception):
    """Raised inside ``DirectStreamer.put()`` to abort ``model.generate()``."""


class _StreamError:
    """Sentinel to signal an error from the generate thread."""

    def __init__(self, msg: str):
        self.msg = msg


# Loading progress, for `/load_model`
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


# Parsing answers: reasoning and tool calls
class ReasoningText(str):
    """Tagged str subclass: text chunk belonging to a thinking/reasoning block.

    Streamers wrap reasoning text with this so handlers can route it to
    ``reasoning_content`` deltas instead of ``content``.
    """


@dataclass
class ToolCall:
    """A parsed tool call surfaced through the stream queue when the parser
    closes a `tool_calls` region. ``arguments`` is always a JSON string."""

    name: str
    arguments: str


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
            "reasoning_content": {
                "open": "<think>",
                "close": "</think>",
                "content": "text",
            },
            "tool_calls": {
                # Leading \s* swallows the separator newline between consecutive tool calls (and the
                # blank line after content) so it can't open an empty implicit `content` region,
                # whose close would overwrite the real content.
                "open_pattern": r"\s*<tool_call>",
                "close": "</tool_call>",
                "repeats": True,
                "content": "json",
                "transform": {"type": "function", "function": "{content}"},
            },
            "content": {
                # Leading \s* keeps the whitespace between two end markers (e.g. Qwen 3.5's
                # "<|im_end|>\n<|endoftext|>" tail) out of the implicit content region — an
                # empty region close would overwrite the real content with "".
                "close_pattern": r"\s*(?:<\|im_end\|>|<\|endoftext\|>|<\|eot_id\|>)",
                "content": "text",
            },
        },
    },
    # Qwen 3.5 family wraps tool calls in <tool_call>...</tool_call> around an inner
    # <function=NAME><parameter=KEY>VALUE</parameter></function> markup that holds the call data.
    # The chat template prefills the assistant turn with either "<think>\n" (thinking on) or
    # "<think>\n\n</think>\n\n" (default), which prefix-aware parsing picks up via start_anchor.
    ("qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"): {
        "defaults": {"role": "assistant"},
        "start_anchor": "<|im_start|>assistant\n",
        "fields": {
            "reasoning_content": {
                "open": "<think>",
                "close": "</think>",
                "content": "text",
            },
            "tool_calls": {
                # Leading \s* swallows the separator newline between consecutive tool calls (and the
                # blank line after content) so it can't open an empty implicit `content` region,
                # whose close would overwrite the real content.
                "open_pattern": r"\s*<tool_call>\s*<function=(?P<name>[^>\n]+)>",
                "close_pattern": r"</function>\s*</tool_call>",
                "repeats": True,
                "content": "xml-inline",
                "content_args": {
                    "tag_pattern": r"<parameter=(?P<key>[^>\n]+)>\s*(?P<value>.*?)\s*</parameter>",
                },
                "transform": {"type": "function", "function": {"name": "{name}", "arguments": "{content}"}},
            },
            "content": {
                # See the qwen2 entry: Qwen 3.5's eos is <|endoftext|>, so generations end with
                # "<|im_end|>\n<|endoftext|>"; the \s* keeps that "\n" from clobbering content.
                "close_pattern": r"\s*(?:<\|im_end\|>|<\|endoftext\|>)",
                "content": "text",
            },
        },
    },
    ("gemma4",): {
        "defaults": {"role": "assistant"},
        "start_anchor": ["<|turn>model\n", "<tool_response|>"],
        "fields": {
            "reasoning_content": {
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


def get_response_template(processor, config: "PretrainedConfig") -> dict | None:
    """Return the response template (new declarative format) for this model.

    Resolution order:
        1. ``tokenizer.response_template`` if set on the tokenizer.
        2. Fallback in :data:`_RESPONSE_TEMPLATE_FALLBACKS` matched by ``config.model_type``.
        3. ``None`` (no parsing performed; raw text is streamed through).
    """
    tokenizer = getattr(processor, "tokenizer", processor)
    tmpl = getattr(tokenizer, "response_template", None)
    if tmpl is not None:
        return tmpl
    model_type = config.model_type
    return next((v for types, v in _RESPONSE_TEMPLATE_FALLBACKS.items() if model_type in types), None)


def build_response_parser(
    processor,
    config: "PretrainedConfig",
    input_ids,
    tools=None,
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
    template = get_response_template(processor, config)
    if template is None:
        return None
    tokenizer = getattr(processor, "tokenizer", processor)
    return ResponseParser(template, prefix=_decode_prefix(tokenizer, input_ids), tools=tools)


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
    config: "PretrainedConfig",
    generated_ids,
    input_ids,
    cleaned_content: str = "",
    tools=None,
) -> tuple[str, str | None, list[ToolCall] | None]:
    """Parse a full assistant message into ``(content, reasoning, tool_calls)``.

    Uses :meth:`tokenizer.parse_response` with the resolved response template
    (see :func:`get_response_template`). When no template is available, returns
    ``(cleaned_content, None, None)`` — i.e., surface the raw model decode.

    ``cleaned_content`` is the ``skip_special_tokens=True`` decode of
    ``generated_ids``. The parser itself runs on the raw decode (keeping
    special tokens so it can match delimiters like Gemma 4's ``<|tool_call>``);
    ``cleaned_content`` is only used on the no-template path."""
    template = get_response_template(processor, config)
    if template is None:
        return cleaned_content, None, None
    tokenizer = getattr(processor, "tokenizer", processor)
    parsed = tokenizer.parse_response(
        generated_ids, template, prefix=_decode_prefix(tokenizer, input_ids), tools=tools
    )
    # Absent ``content`` key means the model emitted only tool calls / thinking;
    # surface that as an empty string rather than swapping in the raw decode.
    content = parsed.get("content", "")
    # Response templates should name the field `reasoning_content`, but some on the Hub still use `thinking`
    reasoning = next((parsed[field] for field in _REASONING_FIELDS if parsed.get(field) is not None), None)
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
            if field in _REASONING_FIELDS:
                out.append(ReasoningText(text))
            elif field == "content":
                out.append(text)
            # Unknown fields are ignored — handlers can extend this if needed.
        elif kind == "region_close" and field == "tool_calls":
            tc = _normalize_tool_call(ev.get("value"))
            if tc is not None:
                out.append(tc)
    return out


# In the API server: decoding a request's streamed tokens
class Detokenizer:
    """
    Turns one request's tokens into text, as they come from the model's process: decodes them incrementally (the Rust
    tokenizer's ``DecodeStream``), and with a response parser, turns the text into typed items: plain ``str`` content,
    :class:`ReasoningText`, :class:`ToolCall`. The items go to `queue`, which ends with ``None`` (see `end`), or a
    `_StreamError`. `cancel` cancels the request (see `EngineClient.generate_streaming`).

    Args:
        tokenizer (`tokenizers.Tokenizer`): The Rust tokenizer (``tokenizer._tokenizer``).
        queue (`asyncio.Queue`): Receives the text, on the event loop.
        response_parser (`ResponseParser`, *optional*): Parses the text into content, reasoning and tool calls.
            Without it, the text streams as is, without special tokens.
    """

    def __init__(
        self, tokenizer: "tokenizers.Tokenizer", queue: asyncio.Queue, response_parser: ResponseParser | None = None
    ):
        from tokenizers.decoders import DecodeStream

        self._tokenizer, self._queue, self._parser = tokenizer, queue, response_parser
        self._decode_stream = DecodeStream([], skip_special_tokens=response_parser is None)
        self.total_tokens = 0
        self.task: asyncio.Task | None = None
        if response_parser is not None:
            for item in response_events_to_chunks(response_parser.initial_events):
                queue.put_nowait(item)

    def put_token(self, token_id: int) -> None:
        """A token from the model's process: decoded, and its text queued."""
        self.total_tokens += 1
        text = self._decode_stream.step(self._tokenizer, token_id)
        if text is None:
            return
        if self._parser is None:
            self._queue.put_nowait(text)
        else:
            for item in response_events_to_chunks(self._parser.feed(text)):
                self._queue.put_nowait(item)

    def end(self) -> None:
        """The request ended: flushes the parser's last events, then queues ``None``."""
        if self._parser is not None:
            try:
                _msg, final_events = self._parser.finalize()
            except (ValueError, RuntimeError) as e:
                # Generation stopped mid-region (likely max_new_tokens hit inside a structured field like a tool call)
                logger.warning(f"ResponseParser.finalize() failed; dropping final tail events: {e}")
                final_events = []
            for item in response_events_to_chunks(final_events):
                self._queue.put_nowait(item)
        self._queue.put_nowait(None)

    def cancel(self) -> None:
        """Cancels the request, e.g. when the client left."""
        if self.task is not None:
            self.task.cancel()


# In a model's process: running its requests
# More than the requests an API server ever sends: each has a range of ids of its own
IDS_PER_API_SERVER = 1 << 40


def make_request_id(api_server: int, request_number: int) -> int:
    """The id of the `request_number`-th request of an API server: unique among all the API servers' requests."""
    return api_server * IDS_PER_API_SERVER + request_number


def api_server_of(request_id: int) -> int:
    """The API server that sent a request, which its results go to."""
    return request_id // IDS_PER_API_SERVER


class _TokenStreamer:
    """The streamer of a request's `model.generate()`: sends each new token (not the prompt's), and stops the
    generation once the request is cancelled."""

    def __init__(self, request_id: int, runner: "GenerateRunner"):
        self.request_id = request_id
        self.runner = runner
        self.prompt = True

    def put(self, value) -> None:
        if self.request_id in self.runner.cancelled_request_ids:
            raise _GenerationCancelled()
        if self.prompt:
            self.prompt = False
            return
        tokens = value.tolist()
        self.runner.count_tokens(generated=len(tokens))
        self.runner.send_results([self.request_id] * len(tokens), tokens, [])

    def end(self) -> None:
        pass


class GenerateRunner:
    """Runs one request at a time, on a thread of its own (``torch.compile``'s CUDA graphs keep their state per
    thread): text generation with `model.generate()`, or a submitted function.

    `send_results(ids, tokens, finished)` sends its results: streamed tokens as two lists, each token's request id
    and the token, and the requests that ended, as ``(request id, result)`` pairs, the result being what the request
    returned (its generated tokens, or what the function returned) or the error that failed it."""

    def __init__(self, model: "PreTrainedModel", processor: Any, send_results: Callable):
        self.model, self.processor, self.send_results = model, processor, send_results
        self.queue: queue.Queue = queue.Queue()
        self.cancelled_request_ids: set[int] = set()
        self.running = False
        self.lock = threading.Lock()
        self.prompt_tokens = self.generated_tokens = 0  # since the last stats line
        threading.Thread(target=self._process_requests, name="generate", daemon=True).start()

    def generate(self, request_id: int, inputs: dict, generation_kwargs: dict, stream: bool) -> None:
        """Generates with the model's generation config and the request's `generation_kwargs` on top (see
        `BaseHandler._build_generation_kwargs`)."""

        def generate_tokens(model: "PreTrainedModel", processor: Any, streamer: Any = None) -> list[int]:
            seed = generation_kwargs.pop("seed", None)
            if seed is not None:
                set_torch_seed(seed)
            model_inputs = {
                key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()
            }
            kwargs = {
                **model_inputs,
                **generation_kwargs,
                "tokenizer": processor,
                "streamer": streamer,
            }
            # Multimodal models (e.g. Qwen2.5-Omni) may generate audio alongside text by default: text only
            if hasattr(model, "has_talker"):
                kwargs["generation_mode"] = "text"
            sequences = model.generate(**kwargs)
            return sequences[0, model_inputs["input_ids"].shape[-1] :].tolist()

        self.count_tokens(prompt=prompt_length(inputs))
        self.queue.put((request_id, generate_tokens, (), stream))

    def submit(self, request_id: int, function: Callable, args: tuple, stream: bool) -> None:
        self.queue.put((request_id, function, args, stream))

    def cancel(self, request_id: int) -> None:
        self.cancelled_request_ids.add(request_id)

    def count_tokens(self, prompt: int = 0, generated: int = 0) -> None:
        """Counts prompt tokens received and tokens generated, for the stats line (see `log_stats`)."""
        with self.lock:
            self.prompt_tokens += prompt
            self.generated_tokens += generated

    def log_stats(self, interval: float = 10.0) -> None:
        """Every `interval` seconds unless idle, a line at info: the prompt tokens received and the tokens generated
        per second, the requests running and waiting (see `state`)."""
        start = time.perf_counter()
        while True:
            time.sleep(interval)
            now = time.perf_counter()
            elapsed, start = now - start, now
            with self.lock:
                prompt_tokens, generated_tokens = self.prompt_tokens, self.generated_tokens
                self.prompt_tokens = self.generated_tokens = 0
            running, waiting, details = self.state()
            if prompt_tokens == generated_tokens == running == waiting == 0:
                continue
            logger.info(
                "Prompts: %.1f tokens/s, generation: %.1f tokens/s, requests: %d running, %d waiting%s",
                prompt_tokens / elapsed,
                generated_tokens / elapsed,
                running,
                waiting,
                details,
            )

    def state(self) -> tuple[int, int, str]:
        """The requests running and waiting, and what more the stats line says."""
        return int(self.running), self.queue.qsize(), ""

    def _process_requests(self) -> None:
        while True:
            request_id, function, args, stream = self.queue.get()
            if request_id in self.cancelled_request_ids:
                self.cancelled_request_ids.discard(request_id)
                continue
            kwargs = {}
            if stream:
                kwargs["streamer"] = _TokenStreamer(request_id, self)
            self.running = True
            try:
                result = function(self.model, self.processor, *args, **kwargs)
                if not stream and isinstance(result, list):  # generated tokens, counted as they stream otherwise
                    self.count_tokens(generated=len(result))
            except _GenerationCancelled:
                result = None
            except ValueError as error:  # the request's fault: crosses as one, for a 400
                result = ValueError(str(error))
            except Exception as error:
                logger.exception("A request failed")
                result = RuntimeError(f"{type(error).__name__}: {error}")  # as text: any error crosses to the server
            self.running = False
            self.cancelled_request_ids.discard(request_id)
            self.send_results([], [], [(request_id, result)])


class CBGenerateRunner(GenerateRunner):
    """
    A `GenerateRunner` that generates text with continuous batching: each request joins the running batch as it
    arrives, and leaves it once it ends. The batching manager starts with the runner, with the model's generation
    config. Each request has its own ``max_new_tokens`` and end of sequence tokens, and its own temperature, top-k and
    top-p when the model's config sets them. The other settings are the model's.
    """

    def __init__(
        self,
        model: "PreTrainedModel",
        processor: Any,
        send_results: Callable,
        cb_config: "ContinuousBatchingConfig | None",
    ):
        super().__init__(model, processor, send_results)
        self.active_requests: dict[int, list] = {}
        self.manager = self._start_continuous_batching(model, cb_config)

    def generate(self, request_id: int, inputs: dict, generation_kwargs: dict, stream: bool) -> None:
        seed = generation_kwargs.pop("seed", None)  # the request's: continuous batching shares the random numbers
        if seed is not None:
            set_torch_seed(seed)
        if self._fail_if_worker_died(request_id):
            return
        with self.lock:
            self.active_requests[request_id] = [stream, 0]
        self.count_tokens(prompt=prompt_length(inputs))
        batching_kwargs = self._build_continuous_batching_kwargs(generation_kwargs)
        # The batching manager's ids are strings
        self.manager.add_request(inputs["input_ids"], request_id=str(request_id), streaming=stream, **batching_kwargs)

    def cancel(self, request_id: int) -> None:
        with self.lock:
            batched = self.active_requests.pop(request_id, None) is not None
        if batched:
            self.manager.cancel_request(str(request_id))
        else:
            super().cancel(request_id)

    def _build_continuous_batching_kwargs(self, generation_kwargs: dict) -> dict:
        """What the batching manager takes from a request's `generation_kwargs`."""
        if generation_kwargs["do_sample"] and generation_kwargs["temperature"] != 0:
            sampling = {name: generation_kwargs[name] for name in ("temperature", "top_k", "top_p")}
        else:
            sampling = {"top_k": 1}
        # Only the sampling settings the manager can set per request (those of the model's config), with its types
        supported = self.manager.logit_processor.supported_keys
        ignored = [name for name, value in sampling.items() if name not in supported and value is not None]
        sampling = {
            name: supported[name](value) for name, value in sampling.items() if name in supported and value is not None
        }
        batching_kwargs = {
            "max_new_tokens": generation_kwargs["max_new_tokens"],
            "eos_token_id": generation_kwargs["eos_token_id"],
            **sampling,
        }
        logger.debug(f"Continuous batching settings of the request: {batching_kwargs}, ignored: {ignored}")
        return batching_kwargs

    def state(self) -> tuple[int, int, str]:
        """The requests the batching manager runs and holds, and how full its KV cache is."""
        batch_processor = self.manager.batch_processor
        if batch_processor is None:  # its worker hasn't started yet
            return 0, 0, ""
        used = 100 * (1 - batch_processor.cache.compute_free_capacity())
        scheduler = batch_processor.scheduler
        return len(scheduler.active_requests), len(scheduler.waiting_requests), f", KV cache: {used:.1f}% used"

    def _fail_if_worker_died(self, request_id: int) -> bool:
        """Fails the request if the batching manager's worker died, as it can't take requests anymore."""
        fatal_error = self.manager.background_thread_status.fatal_error
        if fatal_error is None:
            return False
        error = EngineDeadError(f"The continuous batching worker is dead and can't take requests: {fatal_error}")
        self.send_results([], [], [(request_id, error)])
        return True

    def _start_continuous_batching(
        self, model: "PreTrainedModel", cb_config: "ContinuousBatchingConfig | None"
    ) -> "ContinuousBatchingManager":
        """Starts the batching manager, and the thread sending its outputs."""
        from transformers import ContinuousBatchingConfig

        if cb_config is None:
            cb_config = ContinuousBatchingConfig()
        cb_config.per_request_processors = True  # each request sets its own temperature, top-k and top-p
        manager = model.init_continuous_batching(continuous_batching_config=cb_config)
        manager.start()

        def send_outputs_to_api_servers() -> None:
            while True:
                output = manager.get_result(timeout=1)
                if output is None:
                    fatal_error = manager.background_thread_status.fatal_error
                    if fatal_error is not None:
                        with self.lock:
                            failed = list(self.active_requests)
                            self.active_requests.clear()
                        error = EngineDeadError(f"The continuous batching worker died: {fatal_error}")
                        if failed:
                            self.send_results([], [], [(request_id, error) for request_id in failed])
                    continue
                # All the outputs that are ready go in one message: a message per token would cost much more
                ids, tokens, finished = [], [], []
                while output is not None:
                    self._collect_output(output, ids, tokens, finished)
                    output = manager.get_result(timeout=0)
                if ids or finished:
                    self.send_results(ids, tokens, finished)

        threading.Thread(target=send_outputs_to_api_servers, name="continuous-batching-outputs", daemon=True).start()
        return manager

    def _collect_output(self, output, ids: list[int], tokens: list[int], finished: list[tuple[int, Any]]) -> None:
        """Collects what to send for an output of the manager: its request's new tokens if it streams, and its end."""
        request_id = int(output.request_id)
        with self.lock:
            entry = self.active_requests.get(request_id)
            if entry is None:  # cancelled
                return
            stream, sent = entry
            if output.error is not None:
                del self.active_requests[request_id]
                error = RuntimeError(f"Continuous batching failed for the request: {output.error}")
                finished.append((request_id, error))
                return
            generated = output.generated_tokens
            self.generated_tokens += len(generated) - sent
            if stream:
                ids.extend([request_id] * (len(generated) - sent))
                tokens.extend(generated[sent:])
            entry[1] = len(generated)
            if output.is_finished():
                del self.active_requests[request_id]
                finished.append((request_id, list(generated)))


def transcribe(model: "PreTrainedModel", processor: Any, inputs: dict, streamer=None) -> list[int]:
    """Runs in the model's process (see `EngineClient.submit`): the audio's transcription, as token ids."""
    inputs = {name: value.to(model.device) for name, value in inputs.items()}
    inputs["input_features"] = inputs["input_features"].to(model.dtype)
    return model.generate(**inputs, streamer=streamer)[0].tolist()


def decision_model_class(model_id: str) -> "type[PreTrainedModel] | None":
    """The decision model of a checkpoint shipping its head next to its weights, as in `pipeline("decision")`, read
    from its config files. `None` (the class its config names) for any other model."""
    from transformers.models.auto.modeling_auto import MODEL_FOR_DECISION_MAPPING

    repo, revision, _ = parse_model_id(model_id)
    model_classes = list(dict.fromkeys(MODEL_FOR_DECISION_MAPPING.values()))
    for model_class in model_classes:
        if model_class.checkpoint_config(repo, revision=revision)[0] is not None:
            return model_class
    return None


def decision_pipeline(
    model: "PreTrainedModel",
    processor: "ProcessorMixin | PreTrainedTokenizerFast",
    decision_config: "DecisionConfig | None",
) -> "DecisionPipeline":
    """The decision pipeline of the model, built once and kept on it."""
    from transformers import ProcessorMixin
    from transformers.pipelines import DecisionPipeline

    if not hasattr(model, "decision_pipeline"):
        model.decision_pipeline = DecisionPipeline(
            model=model,
            tokenizer=getattr(processor, "tokenizer", processor),
            processor=processor if isinstance(processor, ProcessorMixin) else None,
            decision_config=decision_config,
        )
    return model.decision_pipeline


def answer_questions(
    model: "PreTrainedModel",
    processor: "ProcessorMixin | PreTrainedTokenizerFast",
    request: dict[str, Any],
    decision_config: "DecisionConfig | None",
) -> tuple[dict[str, Any], int]:
    """Runs in the model's process (see `EngineClient.submit`): the pipeline's answers to the request's questions, and
    the number of input tokens."""
    pipe = decision_pipeline(model, processor, decision_config)
    model_inputs = pipe.preprocess(request)
    answers = pipe.postprocess(pipe.forward(model_inputs))
    return answers, sum(inputs["input_ids"].shape[-1] for inputs, _ in model_inputs["prompts"])


# The server's generation settings
class GenerationState:
    """
    How the models generate, the same for all of them: with continuous batching or sequential ``model.generate()``
    calls, the continuous batching settings, and ``torch.compile``. Handlers ask it whether a model generates with
    continuous batching (`use_continuous_batching`), and each model's process reads its settings (see
    `EngineClient.run_engine`).

    Args:
        continuous_batching (`bool`, *optional*, defaults to `False`):
            Whether to use continuous batching with paged attention instead of
            sequential ``model.generate()`` calls.
        compile (`bool`, *optional*, defaults to `False`):
            Whether to generate with a static cache and ``torch.compile``.
        cb_config (`ContinuousBatchingConfig`, *optional*):
            The continuous batching settings.
        seed (`int`, *optional*):
            The torch seed each model's process sets as it starts.
    """

    def __init__(
        self,
        continuous_batching: bool = False,
        compile: bool = False,
        cb_config: "ContinuousBatchingConfig | None" = None,
        seed: int | None = None,
    ):
        self._continuous_batching = continuous_batching
        self._compile = compile
        self.cb_config = cb_config
        self.seed = seed

    def use_continuous_batching(self, config: "PretrainedConfig", processor: Any) -> bool:
        """Whether a model generates text with continuous batching: if it's enabled, and the model is an LLM."""
        from .model_manager import ModelLoader

        if not self._continuous_batching:
            return False
        modality = ModelLoader.get_model_modality(config, processor)
        if modality == Modality.LLM:
            return True
        logger.warning_once("Only LLMs support continuous batching. Falling back to sequential generation.")
        return False


# What the handlers share
class BaseHandler:
    """Shared logic for chat completion and responses handlers.

    Provides model resolution, generation config building, and SSE formatting. Generation happens in the model's
    process (see `ModelManager.get_engine`).

    Args:
        model_manager (`ModelManager`):
            The models the server runs.
        generation_state (`GenerationState`):
            How the models generate.
    """

    _valid_params_class: type | None = None
    _unused_fields: set[str] = set()
    # One thread, shared by all the handlers, tokenizes the prompts (see `run_in_thread`)
    _tokenizer_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tokenizer")

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
    def chunk_to_sse(chunk: "str | pydantic.BaseModel") -> str:
        """Format a pydantic model or string as an SSE ``data:`` line."""
        if isinstance(chunk, str):
            return chunk if chunk.startswith("data: ") else f"data: {chunk}\n\n"
        return f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

    @staticmethod
    def sse_template(sse: str, **fields: Any) -> str:
        """
        Turns an SSE event into a `str.format` template, with a hole in place of the value of each of `fields` (the
        JSON fields that change from one event to the next, given with the value they have in `sse`).

        Streaming sends an event per token, all alike but for their text: the event is built once, as any other, and
        each token's is then ``template.format(<field>=json.dumps(text, ensure_ascii=False))``.
        """
        template = sse.replace("{", "{{").replace("}", "}}")
        for name, value in fields.items():
            field = f"{json.dumps(name)}:{json.dumps(value)}"
            if template.count(field) != 1:
                raise ValueError(f"Expected {field} once in the event {sse!r}")
            template = template.replace(field, f"{json.dumps(name)}:{{{name}}}")
        return template

    def _get_engine(self, body: dict) -> "EngineClient":
        """The request's model's engine (see `ModelManager.get_engine`). Sets the body's model to the one served."""
        engine = self.model_manager.get_engine(body.get("model"))
        # `<repo>:<file>.gguf` names both the repository and the weights to read out of it: the model's id keeps both
        repo, revision, _ = parse_model_id(body.get("model") or self.model_manager.force_model)
        body["model"] = repo if revision is None else f"{repo}@{revision}"
        return engine

    def _chat_template_kwargs(self, body: dict) -> dict:
        """The chat template's arguments: the server's defaults, overridden by the request's."""
        return {**self.chat_template_kwargs, **(body.get("chat_template_kwargs") or {})}

    @staticmethod
    def _log_request(
        request_id: str, engine: "EngineClient", inputs: dict, generation_kwargs: dict, stream: bool, body: dict
    ) -> None:
        """A line per request at info: its id, model, prompt length and generation settings; its body at debug."""
        if not request_logger.isEnabledFor(logging.INFO):
            return
        settings = {name: value for name, value in generation_kwargs.items() if name != "eos_token_id"}
        request_logger.info(
            "Request %s: model %s, %d prompt tokens, %s, stream %s, %s",
            request_id,
            engine.model_id,
            prompt_length(inputs),
            "continuous batching" if engine.use_cb else "generate",
            bool(stream),
            settings,
        )
        if request_logger.isEnabledFor(logging.DEBUG):
            request_logger.debug("Request %s body: %s", request_id, json.dumps(body, default=str)[:4000])

    def _build_generation_kwargs(self, body: dict, default: "GenerationConfig") -> dict:
        """
        What the request generates with, on top of `default`, the model's generation config (see
        `GenerateRunner.generate`). It always has ``max_new_tokens``, ``eos_token_id``, ``do_sample``,
        ``temperature``, ``top_k`` and ``top_p``, the model's unless the request sets them. Here, the params the
        endpoints share (temperature, top_p, seed, and `generation_config`, a JSON of generation settings).

        Subclasses call ``super()._build_generation_kwargs(...)`` then add theirs (``max_tokens``, ``stop``, etc.).
        """
        from fastapi import HTTPException

        required = ("max_new_tokens", "eos_token_id", "do_sample", "temperature", "top_k", "top_p")
        generation_kwargs = {name: getattr(default, name) for name in required}
        if default.max_new_tokens is None or default.max_new_tokens < 1024:
            generation_kwargs["max_new_tokens"] = 1024
        if body.get("generation_config") is not None:
            requested = json.loads(body["generation_config"])
            # They become arguments of `generate`, which runs a repository's code with this one
            if "trust_remote_code" in requested:
                raise HTTPException(status_code=422, detail="`trust_remote_code` can't be set by a request.")
            generation_kwargs.update(requested)

        if body.get("temperature") is not None:
            generation_kwargs["temperature"] = float(body["temperature"])
            if generation_kwargs["temperature"] == 0.0:
                generation_kwargs["do_sample"] = False
        if body.get("top_p") is not None:
            generation_kwargs["top_p"] = float(body["top_p"])
        if body.get("seed") is not None:
            generation_kwargs["seed"] = int(body["seed"])  # set in the model's process, which generates

        # --compile flag: use static cache + torch.compile for faster decode
        if self.generation_state._compile and default.cache_implementation is None:
            generation_kwargs["cache_implementation"] = "static"

        return generation_kwargs

    async def run_in_thread(self, function: Callable, *args, **kwargs) -> Any:
        """Runs `function` in the tokenizer thread, so the event loop isn't blocked."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._tokenizer_thread, partial(function, *args, **kwargs))

    def render_messages(self, engine: "EngineClient", messages: list[dict], body: dict) -> dict:
        """OpenAI messages → the model's inputs: the processor's messages, with the request's tools, through the chat
        template, tokenized. Token ids as lists for continuous batching, as tensors otherwise."""
        processor_inputs = self.get_processor_inputs_from_messages(messages, engine.modality)
        has_video = any(
            part.get("type") == "video"
            for message in processor_inputs
            for part in (message.get("content") if isinstance(message.get("content"), list) else [])
        )
        # 32 video frames (Gemma 4's default) unless the template's arguments say otherwise: some processors load
        # them all otherwise
        kwargs: dict = {"num_frames": 32} if has_video else {}
        kwargs.update(self._chat_template_kwargs(body))
        return engine.processor.apply_chat_template(
            processor_inputs,
            add_generation_prompt=True,
            tools=self._normalize_tools(body.get("tools")),
            return_tensors=None if engine.use_cb else "pt",
            return_dict=True,
            tokenize=True,
            load_audio_from_video=engine.modality == Modality.MULTIMODAL and has_video,
            **kwargs,
        )

    @staticmethod
    def _normalize_tools(tools: list[dict] | None) -> list[dict] | None:
        """Tools as ``apply_chat_template`` expects them: nested, as chat completions send them,
        ``{"type": "function", "function": {"name": ..., "parameters": ...}}``. The Responses API's flat ones,
        ``{"type": "function", "name": ..., "parameters": ...}``, are nested; nested ones pass through unchanged.
        """
        if not tools:
            return tools
        return [
            {"type": "function", "function": {k: v for k, v in t.items() if k != "type"}} if "function" not in t else t
            for t in tools
        ]

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
            if message.get("tool_calls"):
                tool_calls = []
                for tc in message["tool_calls"]:
                    tc = copy.deepcopy(tc)
                    fn = tc.get("function") or tc
                    if isinstance(fn["arguments"], str):
                        fn["arguments"] = json.loads(fn["arguments"])
                    tool_calls.append(tc)
                parsed["tool_calls"] = tool_calls
            if "tool_call_id" in message:
                parsed["tool_call_id"] = message["tool_call_id"]

            # When tool_calls are present, ignore content — it's either empty or contains
            # raw tool call markup that would confuse the chat template if rendered.
            raw_content = [] if message.get("tool_calls") else (message.get("content") or [])
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
                # Audio: OpenAI's input_audio is {"data": <base64>, "format": "wav"|"mp3"}, enabling URI for load_audio
                # If format is missing, we can just hand over raw base64 and let load_audio sniff the format from the bytes.
                elif content_type == "input_audio" and modality == Modality.MULTIMODAL:
                    input_audio = content["input_audio"]
                    if isinstance(input_audio, dict):
                        audio_b64 = input_audio["data"]
                        fmt = input_audio.get("format")
                        url = f"data:audio/{fmt};base64,{audio_b64}" if fmt else audio_b64
                    else:
                        url = input_audio
                    parsed["content"].append({"type": "audio", "url": url})
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
