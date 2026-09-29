# Copyright 2023 The HuggingFace Inc. team.
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

from __future__ import annotations

import asyncio
import sys
import time
from queue import Empty, Queue
from typing import TYPE_CHECKING, Any, cast

import numpy as np


if TYPE_CHECKING:
    from ..tokenization_utils_base import PreTrainedTokenizerBase


class BaseStreamer:
    """
    Base class from which `.generate()` streamers should inherit.
    """

    def put(self, value):
        """Function that is called by `.generate()` to push new tokens"""
        raise NotImplementedError()

    def end(self):
        """Function that is called by `.generate()` to signal the end of generation"""
        raise NotImplementedError()


class TextStreamer(BaseStreamer):
    """
    Simple text streamer that prints the token(s) to stdout as soon as entire words are formed.

    Parameters:
        tokenizer (`AutoTokenizer`):
            The tokenizer used to decode the tokens.
        skip_prompt (`bool`, *optional*, defaults to `False`):
            Whether to skip the prompt to `.generate()` or not. Useful e.g. for chatbots.
        decode_kwargs (`dict`, *optional*):
            Additional keyword arguments to pass to the tokenizer's `decode` method.

    Examples:

        ```python
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

        >>> tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        >>> inputs = tok(["An increasing sequence: one,"], return_tensors="pt")
        >>> streamer = TextStreamer(tok)

        >>> # Despite returning the usual output, the streamer will also print the generated text to stdout.
        >>> _ = model.generate(**inputs, streamer=streamer, max_new_tokens=20)
        An increasing sequence: one, two, three, four, five, six, seven, eight, nine, ten, eleven,
        ```
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, skip_prompt: bool = False, **decode_kwargs: Any):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.decode_kwargs = decode_kwargs

        # variables used in the streaming process
        self.token_cache: list[int] = []
        self.print_len = 0
        self.next_tokens_are_prompt = True

    def put(self, value):
        """
        Receives tokens, decodes them, and prints them to stdout as soon as they form entire words.
        """
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        # Add the new token to the cache and decodes the entire thing.
        self.token_cache.extend(value.tolist())
        text = cast(str, self.tokenizer.decode(self.token_cache, **self.decode_kwargs))

        # After the symbol for a new line, we flush the cache.
        if text.endswith("\n"):
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        # If the last token is a CJK character, we print the characters.
        elif len(text) > 0 and self._is_chinese_char(ord(text[-1])):
            printable_text = text[self.print_len :]
            self.print_len += len(printable_text)
        # Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
        # which may change with the subsequent token -- there are probably smarter ways to do this!)
        else:
            printable_text = text[self.print_len : text.rfind(" ") + 1]
            self.print_len += len(printable_text)

        self.on_finalized_text(printable_text)

    def end(self):
        """Flushes any remaining cache and prints a newline to stdout."""
        # Flush the cache, if it exists
        if len(self.token_cache) > 0:
            text = cast(str, self.tokenizer.decode(self.token_cache, **self.decode_kwargs))
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        else:
            printable_text = ""

        self.next_tokens_are_prompt = True
        self.on_finalized_text(printable_text, stream_end=True)

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Prints the new text to stdout. If the stream is ending, also prints a newline."""
        print(text, flush=True, end="" if not stream_end else None)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)
            or (cp >= 0x20000 and cp <= 0x2A6DF)
            or (cp >= 0x2A700 and cp <= 0x2B73F)
            or (cp >= 0x2B740 and cp <= 0x2B81F)
            or (cp >= 0x2B820 and cp <= 0x2CEAF)
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)
        ):
            return True

        return False


class TextIteratorStreamer(TextStreamer):
    """
    Streamer that stores print-ready text in a queue, to be used by a downstream application as an iterator. This is
    useful for applications that benefit from accessing the generated text in a non-blocking way (e.g. in an interactive
    Gradio demo).

    Parameters:
        tokenizer (`AutoTokenizer`):
            The tokenizer used to decode the tokens.
        skip_prompt (`bool`, *optional*, defaults to `False`):
            Whether to skip the prompt to `.generate()` or not. Useful e.g. for chatbots.
        timeout (`float`, *optional*):
            The timeout for the text queue. If `None`, the queue will block indefinitely. Useful to handle exceptions
            in `.generate()`, when it is called in a separate thread.
        decode_kwargs (`dict`, *optional*):
            Additional keyword arguments to pass to the tokenizer's `decode` method.

    Examples:

        ```python
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
        >>> from threading import Thread

        >>> tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        >>> inputs = tok(["An increasing sequence: one,"], return_tensors="pt")
        >>> streamer = TextIteratorStreamer(tok)

        >>> # Run the generation in a separate thread, so that we can fetch the generated text in a non-blocking way.
        >>> generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=20)
        >>> thread = Thread(target=model.generate, kwargs=generation_kwargs)
        >>> thread.start()
        >>> generated_text = ""
        >>> for new_text in streamer:
        ...     generated_text += new_text
        >>> generated_text
        'An increasing sequence: one, two, three, four, five, six, seven, eight, nine, ten, eleven,'
        ```
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        skip_prompt: bool = False,
        timeout: float | None = None,
        **decode_kwargs: Any,
    ):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.text_queue = Queue()
        self.stop_signal = None
        self.timeout = timeout

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Put the new text in the queue. If the stream is ending, also put a stop signal in the queue."""
        self.text_queue.put(text, timeout=self.timeout)
        if stream_end:
            self.text_queue.put(self.stop_signal, timeout=self.timeout)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.text_queue.get(timeout=self.timeout)
        if value == self.stop_signal:
            raise StopIteration()
        else:
            return value


class AsyncTextIteratorStreamer(TextStreamer):
    """
    Streamer that stores print-ready text in a queue, to be used by a downstream application as an async iterator.
    This is useful for applications that benefit from accessing the generated text asynchronously (e.g. in an
    interactive Gradio demo).

    Parameters:
        tokenizer (`AutoTokenizer`):
            The tokenizer used to decode the tokens.
        skip_prompt (`bool`, *optional*, defaults to `False`):
            Whether to skip the prompt to `.generate()` or not. Useful e.g. for chatbots.
        timeout (`float`, *optional*):
            The timeout for the text queue. If `None`, the queue will block indefinitely. Useful to handle exceptions
            in `.generate()`, when it is called in a separate thread.
        decode_kwargs (`dict`, *optional*):
            Additional keyword arguments to pass to the tokenizer's `decode` method.

    Raises:
        TimeoutError: If token generation time exceeds timeout value.

    Examples:

        ```python
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer, AsyncTextIteratorStreamer
        >>> from threading import Thread
        >>> import asyncio

        >>> tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        >>> inputs = tok(["An increasing sequence: one,"], return_tensors="pt")

        >>> # Run the generation in a separate thread, so that we can fetch the generated text in a non-blocking way.
        >>> async def main():
        ...     # Important: AsyncTextIteratorStreamer must be initialized inside a coroutine!
        ...     streamer = AsyncTextIteratorStreamer(tok)
        ...     generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=20)
        ...     thread = Thread(target=model.generate, kwargs=generation_kwargs)
        ...     thread.start()
        ...     generated_text = ""
        ...     async for new_text in streamer:
        ...         generated_text += new_text
        >>>     print(generated_text)
        >>> asyncio.run(main())
        An increasing sequence: one, two, three, four, five, six, seven, eight, nine, ten, eleven,
        ```
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        skip_prompt: bool = False,
        timeout: float | None = None,
        **decode_kwargs: Any,
    ):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.text_queue = asyncio.Queue()
        self.stop_signal = None
        self.timeout = timeout
        self.loop = asyncio.get_running_loop()
        timeout_context = getattr(asyncio, "timeout", None)
        self.has_asyncio_timeout = sys.version_info >= (3, 11) and callable(timeout_context)
        self.asyncio_timeout = timeout_context if self.has_asyncio_timeout else None

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Put the new text in the queue. If the stream is ending, also put a stop signal in the queue."""
        self.loop.call_soon_threadsafe(self.text_queue.put_nowait, text)
        if stream_end:
            self.loop.call_soon_threadsafe(self.text_queue.put_nowait, self.stop_signal)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            if self.has_asyncio_timeout and self.asyncio_timeout is not None:
                async with self.asyncio_timeout(self.timeout):
                    value = await self.text_queue.get()
            else:
                value = await asyncio.wait_for(self.text_queue.get(), timeout=self.timeout)
        except asyncio.TimeoutError:
            raise TimeoutError()
        else:
            if value == self.stop_signal:
                raise StopAsyncIteration()
            else:
                return value


class TextDiffusionStreamer(TextStreamer):
    """
    Streamer that prints text diffusion outputs. Intermediate diffusion steps (drafts) are temporary
    and overwritten by subsequent drafts, and removed when confirmed text is printed.

    <Tip warning={true}>

    If you're running on an environment like tmux, the draft text may fail to overwrite itself.

    </Tip>


    Parameters:
        tokenizer (`AutoTokenizer`):
            The tokenized used to decode the tokens.
        skip_prompt (`bool`, *optional*, defaults to `False`):
            Whether to skip the prompt to `.generate()` or not. Useful e.g. for chatbots.
        sleep_time (`float`, *optional*):
            Time to sleep between diffusion drafts, which may be helpful to visualize intermediate outputs.
        decode_kwargs (`dict`, *optional*):
            Additional keyword arguments to pass to the tokenizer's `decode` method.

    Examples:

        ```python
        >>> from transformers import DiffusionGemmaForBlockDiffusion, AutoProcessor, TextDiffusionStreamer

        >>> model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        ...     "google/diffusiongemma-26B-A4B-it", device_map="auto",
        ... )
        >>> processor = AutoProcessor.from_pretrained("google/diffusiongemma-26B-A4B-it")

        >>> chat = [{"role": "user", "content": "Why is the sky blue?"},]
        >>> input_ids = processor.apply_chat_template(
        ...     chat, tokenize=True, return_tensors="pt", add_generation_prompt=True
        ... )
        >>> streamer = TextDiffusionStreamer(tokenizer=processor.tokenizer)
        >>> model.generate(input_ids.to(model.device), max_new_tokens=512, streamer=streamer)
        ```
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        skip_prompt: bool = False,
        sleep_time: float | None = None,
        **decode_kwargs: Any,
    ):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self._has_draft = False
        # `_takes_logits`: Overwrite this attribute if you want your new Streamer class to take the draft
        # logits as an input to `put_draft`. On diffusion models, `logits` can be a very large tensor, so
        # we recommend setting it to `False` by default.
        self._takes_logits = False
        self.sleep_time = sleep_time

    def _clear_draft(self):
        if self._has_draft:
            # Restore cursor and clear to end of screen
            print("\0338\033[J", end="", flush=True)
            self._has_draft = False

    def put_draft(self, value, **kwargs):
        """
        Receives the full sequence of draft tokens, decodes them, and prints them in yellow.
        Overwrites previous draft.
        """
        self._clear_draft()

        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextDiffusionStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        text = self.tokenizer.decode(value, **self.decode_kwargs)

        # Save cursor position
        print("\0337", end="", flush=True)
        # Print draft in yellow
        print(f"\033[33m{text}\033[0m", end="", flush=True)
        self._has_draft = True
        if self.sleep_time is not None:
            time.sleep(self.sleep_time)

    def put(self, value):
        """Receives confirmed tokens, clears draft, and prints them permanently."""
        self._clear_draft()
        super().put(value)

    def end(self):
        """Flushes any remaining cache and prints a newline."""
        self._clear_draft()
        super().end()


class BaseInputStreamer:
    """
    Base class from which `.generate()` *input* streamers should inherit.

    This is the input-side counterpart to [`BaseStreamer`]. A producer feeds data with `put()` and signals
    completion with `end()`, while `.generate()` consumes the stream by pulling model inputs with `get()`.
    The class is also a Python iterator (`__next__` calls `get()` and `__iter__` returns `self`), so a
    streaming-capable model can detect a `BaseInputStreamer` passed as `input_features` and drive it with
    either `get()` or `next()`.
    """

    def put(self, value):
        """Push new input data into the stream (producer side)."""
        raise NotImplementedError()

    def end(self):
        """Signal that no more input will be pushed (producer side)."""
        raise NotImplementedError()

    def get(self):
        """Return the next model input (consumer side). Raises `StopIteration` when the stream is exhausted."""
        raise NotImplementedError()

    def __iter__(self):
        return self

    def __next__(self):
        return self.get()

    @property
    def model_kwargs(self):
        """Auxiliary keyword arguments to forward to `.generate()` alongside `input_features`."""
        raise NotImplementedError()


class AudioIteratorStreamer(BaseInputStreamer):
    """
    Synchronous, thread-safe audio input streamer for streaming-ASR `.generate()`.

    A producer thread pushes raw mono audio samples with [`~AudioIteratorStreamer.put`] (one or many calls, e.g.
    a real-time microphone feed) and calls [`~AudioIteratorStreamer.end`] when the stream is over. A consumer
    (typically `model.generate` running in a worker thread) iterates over the streamer to receive the
    `input_features` chunks the model expects. All chunking geometry is derived from the processor, so the
    same class works for any processor exposing `num_samples_first_audio_chunk` and
    `num_samples_per_audio_chunk`, plus a feature extractor with `hop_length`, `win_length`, `n_fft`,
    `feature_size`.

    Parameters:
        processor:
            A streaming-ASR processor (e.g. `NemotronAsrStreamingProcessor`, `VoxtralRealtimeProcessor`).
        device (*optional*):
            Device to move yielded feature chunks (and tensor `model_kwargs`) to.
        dtype (*optional*):
            Dtype to cast yielded feature chunks to.
        timeout (`float`, *optional*):
            Seconds to wait for the next audio in the blocking get. `None` blocks indefinitely. Useful to
            avoid hanging `.generate()` forever if the producer dies.

    Examples:

        ```python
        >>> from threading import Thread
        >>> from transformers import AudioIteratorStreamer, TextIteratorStreamer

        >>> input_streamer = AudioIteratorStreamer(processor, device=model.device, dtype=model.dtype)
        >>> output_streamer = TextIteratorStreamer(processor.tokenizer, skip_special_tokens=True)
        >>> generate_kwargs = {
        ...     "input_features": input_streamer,
        ...     "streamer": output_streamer,
        ...     **input_streamer.model_kwargs,
        ... }
        >>> thread = Thread(target=model.generate, kwargs=generate_kwargs)
        >>> thread.start()
        >>> input_streamer.put(audio)
        >>> input_streamer.end()
        >>> for text_chunk in output_streamer:
        ...     print(text_chunk, end="", flush=True)
        >>> thread.join()
        ```
    """

    _SENTINEL = object()

    def __init__(self, processor, device=None, dtype=None, timeout: float | None = None):
        self._setup(processor, device, dtype, timeout)
        self._queue = Queue()

    def _setup(self, processor, device, dtype, timeout):
        self.processor = processor
        self.device = device
        self.dtype = dtype
        self.timeout = timeout

        fe = processor.feature_extractor
        self._hop = fe.hop_length
        self._win = fe.win_length
        self._n_fft = fe.n_fft
        self._feature_size = fe.feature_size
        self._first_chunk_samples = processor.num_samples_first_audio_chunk
        self._chunk_samples = processor.num_samples_per_audio_chunk
        # Subsequent windows overlap by `win_length` (the STFT window context that `center=False` needs), so
        # they advance by `num_samples_per_audio_chunk - win_length`. The first subsequent window starts at
        # the raw-sample offset that makes its `center=False` frames continue seamlessly after the first
        # (`center=True`) chunk.
        self._advance = self._chunk_samples - self._win
        self._second_chunk_start = self._first_chunk_samples + self._hop - self._n_fft // 2 - self._win // 2

        # consumer-private rolling buffer (only the consuming thread touches it)
        self._buffer = np.zeros(0, dtype=np.float32)
        self._base = 0  # absolute index of self._buffer[0]
        self._total = 0  # absolute number of samples received so far
        self._ended = False

        # consumer iteration state
        self._started = False  # whether the first chunk has been emitted
        self._start = 0  # absolute start of the next subsequent window
        self._exhausted = False  # whether `get()` has raised StopIteration

        # `drops_last`: whether the feature extractor itself drops the trailing padded STFT frame
        # (`stft[..., :-1]`). If it does, we never trim; if not, we must drop the trailing padded frame of
        # the first (`center=True`) chunk ourselves. Detected empirically, so no model-specific config or
        # processor edits are needed.
        self._drops_last = self._detect_drops_last()

        self._model_kwargs = self._compute_model_kwargs()

    def _compute_model_kwargs(self):
        # input_ids (voxtral) depends only on the *length* of the first chunk, not its content, so a zero
        # placeholder of the right length yields the correct auxiliary kwargs. Feature output is discarded.
        placeholder = np.zeros(self._first_chunk_samples, dtype=np.float32)
        batch = self.processor(placeholder, is_streaming=True, is_first_audio_chunk=True, return_tensors="pt")
        kwargs = {}
        for key, value in batch.items():
            if key in ("input_features", "attention_mask"):
                continue
            # Move aux tensors (e.g. `input_ids`) to device but never cast their dtype: they are integer
            # tensors, unlike the float feature chunks handled in `_extract`. Duck-typed on `.to` to keep
            # this module backend-agnostic; scalar aux values (e.g. `num_delay_tokens`) have no `.to`.
            if self.device is not None and hasattr(value, "to"):
                value = value.to(self.device)
            kwargs[key] = value
        return kwargs

    @property
    def model_kwargs(self):
        return self._model_kwargs

    # ---- producer side ----
    def put(self, value):
        """Push raw mono audio samples (numpy array, list, or any array-like that exposes `__array__`)."""
        self._queue.put(self._to_numpy(value))

    def end(self):
        """Signal end of the audio stream."""
        self._queue.put(self._SENTINEL)

    @staticmethod
    def _to_numpy(value):
        # `np.asarray` converts lists, numpy arrays, and any array-like exposing `__array__` (e.g. CPU
        # tensors) to a 1-D float32 buffer, without this module depending on a specific array backend.
        return np.asarray(value, dtype=np.float32).reshape(-1)

    def _get(self):
        try:
            return self._queue.get(timeout=self.timeout)
        except Empty as e:
            raise TimeoutError("AudioIteratorStreamer timed out waiting for audio input.") from e

    # ---- consumer side ----
    def _ensure(self, abs_index):
        """Pull from the queue until `self._total >= abs_index` or the stream ends. Returns `self._total`."""
        while self._total < abs_index and not self._ended:
            item = self._get()
            if item is self._SENTINEL:
                self._ended = True
            else:
                self._buffer = np.concatenate([self._buffer, item])
                self._total += item.shape[0]
        return self._total

    def _slice(self, start, length):
        """Return `length` samples starting at absolute `start`, zero-padded if the buffer is short."""
        local_start = start - self._base
        end = local_start + length
        if local_start >= 0 and end <= self._buffer.shape[0]:
            return self._buffer[local_start:end].copy()
        out = np.zeros(length, dtype=np.float32)
        valid = self._buffer.shape[0] - max(local_start, 0)
        if valid > 0:
            src = max(local_start, 0)
            out[: min(valid, length)] = self._buffer[src : src + min(valid, length)]
        return out

    def _discard_before(self, abs_index):
        """Drop buffered samples before absolute `abs_index` to bound memory."""
        drop = min(abs_index - self._base, self._buffer.shape[0])
        if drop > 0:
            self._buffer = self._buffer[drop:]
            self._base += drop

    def _time_axis(self, features):
        # Time axis = the non-batch axis whose size differs from `feature_size`. Falls back to the last axis
        # when the layout can't be told apart (e.g. the frame count coincides with `feature_size`).
        if features.dim() == 3 and features.shape[2] == self._feature_size and features.shape[1] != self._feature_size:
            return 1
        return 2

    def _detect_drops_last(self):
        # Feed a subsequent (`center=False`) chunk: it should yield `1 + (L - n_fft) // hop` frames. A feature
        # extractor that drops the trailing padded frame (`stft[..., :-1]`) returns one fewer.
        probe = np.zeros(self._chunk_samples, dtype=np.float32)
        features = self.processor(
            probe, is_streaming=True, is_first_audio_chunk=False, return_tensors="pt"
        ).input_features
        produced = features.shape[self._time_axis(features)]
        expected = 1 + (self._chunk_samples - self._n_fft) // self._hop
        return produced == expected - 1

    def _extract(self, samples, is_first):
        batch = self.processor(samples, is_streaming=True, is_first_audio_chunk=is_first, return_tensors="pt")
        features = batch.input_features
        # The first chunk is the only `center=True` chunk, so it is the only one carrying a trailing padded
        # frame. If the feature extractor does not already drop it (`drops_last`), drop it here.
        if is_first and not self._drops_last:
            features = self._drop_last_frame(features)
        return features.to(device=self.device, dtype=self.dtype)

    def _drop_last_frame(self, features):
        if self._time_axis(features) == 1:
            return features[:, :-1]
        return features[:, :, :-1]

    def get(self):
        """Return the next `input_features` chunk. Raises `StopIteration` when the stream is exhausted."""
        if self._exhausted:
            raise StopIteration

        # First chunk.
        if not self._started:
            self._started = True
            total = self._ensure(self._first_chunk_samples)
            if total == 0 and self._ended:
                self._exhausted = True
                raise StopIteration
            first = self._slice(0, self._first_chunk_samples)
            features = self._extract(first, is_first=True)
            self._discard_before(self._second_chunk_start)
            self._start = self._second_chunk_start
            return features

        # Subsequent chunks.
        start = self._start
        total = self._ensure(start + self._chunk_samples + 1)
        if total <= start + self._chunk_samples:
            # No full window lies strictly inside the (now-ended) stream. This matches the reference
            # generators' `while end_idx < len` guard: a final window ending exactly at the last sample is
            # dropped. For live feeds the next `put` resolves it; pre-recorded callers who need the exact
            # tail should pad the audio (as voxtral's `num_right_pad_tokens` padding does).
            self._exhausted = True
            raise StopIteration
        window = self._slice(start, self._chunk_samples)
        features = self._extract(window, is_first=False)
        self._start = start + self._advance
        self._discard_before(self._start)
        return features


class AsyncAudioIteratorStreamer(AudioIteratorStreamer):
    """
    Asyncio variant of [`AudioIteratorStreamer`] for async producers (e.g. audio arriving over a websocket).

    `put()` and `end()` are coroutines that enqueue onto an `asyncio.Queue`. The consumer side
    (`get()`) is still synchronous — `model.generate` runs in a worker thread and pulls chunks
    synchronously — and bridges to the event loop via `asyncio.run_coroutine_threadsafe`. This mirrors how
    [`AsyncTextIteratorStreamer`] bridges the output side with `loop.call_soon_threadsafe`.

    Must be constructed inside a running event loop. The producer (`put`/`end`) runs on the event loop,
    but consumption (by `.generate()`) must happen on a *different* thread — e.g. pass the streamer as
    `input_features` to `Thread(target=model.generate)`. Consuming on the loop's own thread deadlocks,
    because the synchronous `_get` blocks that thread via `run_coroutine_threadsafe(...).result()` and the
    queue can never be served.

    Parameters: identical to [`AudioIteratorStreamer`].

    Examples:

        ```python
        >>> # Inside an async context, with model.generate running in a worker thread:
        >>> input_streamer = AsyncAudioIteratorStreamer(processor, device=model.device, dtype=model.dtype)
        >>> # ... start the generate thread with input_features=input_streamer and **input_streamer.model_kwargs ...
        >>> async for audio_chunk in mic_source():
        ...     await input_streamer.put(audio_chunk)
        >>> await input_streamer.end()
        ```
    """

    def __init__(self, processor, device=None, dtype=None, timeout: float | None = None):
        self._loop = asyncio.get_running_loop()
        self._setup(processor, device, dtype, timeout)
        self._queue = asyncio.Queue()

    async def put(self, value):
        """Push raw mono audio samples onto the async queue."""
        await self._queue.put(self._to_numpy(value))

    async def end(self):
        """Signal end of the audio stream on the async queue."""
        await self._queue.put(self._SENTINEL)

    def _get(self):
        import concurrent.futures

        future = asyncio.run_coroutine_threadsafe(self._queue.get(), self._loop)
        try:
            return future.result(timeout=self.timeout)
        except concurrent.futures.TimeoutError as e:
            raise TimeoutError("AsyncAudioIteratorStreamer timed out waiting for audio input.") from e
