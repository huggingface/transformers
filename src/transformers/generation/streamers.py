# coding=utf-8
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
import asyncio
from queue import Queue
from typing import TYPE_CHECKING, Callable, Dict, List, Optional


if TYPE_CHECKING:
    from ..models.auto import AutoTokenizer


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


class MultiBeamBaseStreamer(BaseStreamer):
    """
    Base class from which all multi-beam streamers should inherit.
    Extends the BaseStreamer class with functionality specific to handling multiple beams.
    """

    def __init__(self, num_beams: int):
        super().__init__()
        if not isinstance(num_beams, int) or num_beams <= 0:
            raise ValueError(f"num_beams must be a positive integer, got {num_beams}")
        self.num_beams = num_beams
        self.current_beam = 0

    def beam_finished(self, beam_idx: int):
        """
        Called when a specific beam has finished generating.
        Must be implemented by the derived class.

        Args:
            beam_idx (`int`):
                Index of the beam that finished generating.
        """
        raise NotImplementedError()


class TextStreamer(BaseStreamer):
    """
    Simple text streamer that prints the token(s) to stdout as soon as entire words are formed.

    <Tip warning={true}>

    The API for the streamer classes is still under development and may change in the future.

    </Tip>

    Parameters:
        tokenizer (`AutoTokenizer`):
            The tokenized used to decode the tokens.
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

    def __init__(self, tokenizer: "AutoTokenizer", skip_prompt: bool = False, **decode_kwargs):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.decode_kwargs = decode_kwargs

        # variables used in the streaming process
        self.token_cache = []
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
        text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)

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
            text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)
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
            or (cp >= 0x3400 and cp <= 0x4DBF)  #
            or (cp >= 0x20000 and cp <= 0x2A6DF)  #
            or (cp >= 0x2A700 and cp <= 0x2B73F)  #
            or (cp >= 0x2B740 and cp <= 0x2B81F)  #
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):  #
            return True

        return False


class TextIteratorStreamer(TextStreamer):
    """
    Streamer that stores print-ready text in a queue, to be used by a downstream application as an iterator. This is
    useful for applications that benefit from acessing the generated text in a non-blocking way (e.g. in an interactive
    Gradio demo).

    <Tip warning={true}>

    The API for the streamer classes is still under development and may change in the future.

    </Tip>

    Parameters:
        tokenizer (`AutoTokenizer`):
            The tokenized used to decode the tokens.
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
        self, tokenizer: "AutoTokenizer", skip_prompt: bool = False, timeout: Optional[float] = None, **decode_kwargs
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


class MultiBeamTextStreamer(MultiBeamBaseStreamer):
    """
    A streamer that handles beam search generation, allowing real-time tracking and processing of multiple beam outputs.
    This is useful for applications that need to monitor or display multiple candidate sequences during beam search
    generation, such as interactive applications showing alternative generations in real-time.

    <Tip warning={true}>

    The API for the streamer classes is still under development and may change in the future.

    </Tip>

    Parameters:
        tokenizer (`AutoTokenizer`):
            The tokenizer used to decode the tokens.
        num_beams (`int`):
            The number of beams to handle during generation.
        on_beam_update (`Callable[[int, str], None]`):
            A callback function that gets called whenever a beam's text is updated.
            The function receives two arguments:
                - beam_idx (`int`): The index of the updated beam
                - text (`str`): The current complete text for this beam
        on_beam_finished (`Callable[[str], None]`, *optional*):
            A callback function that gets called when a beam reaches the EOS token.
            The function receives one argument:
                - text (`str`): The final text of the finished beam
        skip_prompt (`bool`, *optional*, defaults to `True`):
            Whether to skip the prompt tokens in the generation output.
        decode_kwargs (`dict`, *optional*):
            Additional keyword arguments to pass to the tokenizer's `decode` method.

    Examples:

        ```python
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer, MultiBeamTextStreamer

        >>> # Create a dictionary to store beam outputs
        >>> beam_outputs = {}

        >>> # Define callback functions that store outputs in the dictionary
        >>> def on_beam_update(beam_idx: int, text: str):
        ...     beam_outputs[f"beam_{beam_idx}"] = text

        >>> def on_beam_finished(text: str):
        ...     beam_outputs["completed"] = text

        >>> # Initialize model, tokenizer and streamer
        >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        >>> inputs = tokenizer(["An increasing sequence: one,"], return_tensors="pt")

        >>> # Create streamer with 2 beams
        >>> streamer = MultiBeamTextStreamer(
        ...     tokenizer=tokenizer,
        ...     num_beams=2,
        ...     on_beam_update=on_beam_update,
        ...     on_beam_finished=on_beam_finished
        ... )

        >>> # Generate with beam search
        >>> _ = model.generate(
        ...     **inputs,
        ...     streamer=streamer,
        ...     num_beams=2,
        ...     max_new_tokens=10
        ... )

        >>> # Access the final outputs from the dictionary
        >>> print(beam_outputs)
        {
            'beam_0': 'An increasing sequence: one, two, three, four,',
            'beam_1': 'An increasing sequence: one, two, three, five,',
            'completed': 'An increasing sequence: one, two, three, four,'
        }
        ```

    The streamer maintains internal state for each beam and provides real-time updates through the callback functions.
    It handles beam switching during beam search and ensures proper tracking of beam histories. The streamer is particularly
    useful for:

    - Interactive applications showing multiple generation alternatives
    - Debugging beam search behavior
    - Creating UIs that display beam search progress
    - Analyzing beam search decision patterns

    Note that this streamer requires more memory than single-sequence streamers as it needs to maintain state for all beams.
    For applications that only need the final best sequence, consider using `TextStreamer` instead.
    """

    def __init__(
        self,
        tokenizer: "AutoTokenizer",
        num_beams: int,
        on_beam_update: Callable[[int, str], None],
        on_beam_finished: Callable[[str], None] = None,
        skip_prompt: bool = True,
        **decode_kwargs,
    ):
        super().__init__(num_beams)
        self.tokenizer = tokenizer
        self.num_beams = num_beams
        self.skip_prompt = skip_prompt
        self.decode_kwargs = decode_kwargs
        self.on_beam_update = on_beam_update
        self.on_beam_finished = on_beam_finished

        # Initialize storage for each beam
        self.beam_tokens: Dict[int, List[int]] = {i: [] for i in range(num_beams)}
        self.beam_texts: Dict[int, str] = {i: "" for i in range(num_beams)}
        self.beam_print_lens: Dict[int, int] = {i: 0 for i in range(num_beams)}

        # Track beam states at each position
        self.beam_history: Dict[int, Dict[int, List[int]]] = {}  # position -> beam_idx -> tokens
        self.current_position = 0

        # Track current state
        self.next_tokens_are_prompt = True

        # Store finished beams
        self.finished_beams: List[str] = []

    def _switch_beam_content(self, position: int, previous_beam_idx: int, new_beam_idx: int):
        """
        Internal helper to handle beam content switching with position tracking.
        """
        if new_beam_idx >= self.num_beams:
            raise ValueError(f"Beam index {new_beam_idx} is out of range (num_beams={self.num_beams})")

        if previous_beam_idx != new_beam_idx:
            # Get the correct historical state for the previous beam at this position
            if position > 0 and position in self.beam_history:
                source_tokens = self.beam_history[position][previous_beam_idx].copy()
            else:
                source_tokens = self.beam_tokens[previous_beam_idx].copy()

            # Update tokens for the new beam
            self.beam_tokens[new_beam_idx] = source_tokens

            # Update text and calculate new state
            text = self.tokenizer.decode(source_tokens, **self.decode_kwargs)
            self.beam_texts[new_beam_idx] = text
            self.beam_print_lens[new_beam_idx] = len(text)

            # Notify handler of the beam update
            self.on_beam_update(new_beam_idx, text)

    def put(self, values, beam_indices=None):
        """
        Handle new tokens for all beams at once.
        Args:
            values: List or array-like of shape (num_beams, 1) containing the next token for each beam
            beam_indices: Optional list/array/tensor containing the previous beam indices for each current beam
        """
        # Convert values to list if it's a tensor or array
        if hasattr(values, "tolist"):
            values = values.tolist()

        # Validate input shape
        if len(values) == 1 and isinstance(values[0], list) and len(values[0]) > 1:
            values = [[token] for token in values[0]]
        else:
            if not isinstance(values, list) or not all(isinstance(row, list) and len(row) == 1 for row in values):
                raise ValueError("Expected values to be a list of lists, each inner list having length 1")

        if len(values) > self.num_beams:
            raise ValueError(
                f"Number of beams in values ({len(values)}) exceeds initialized num_beams ({self.num_beams})"
            )

        # Handle beam_indices
        if beam_indices is None:
            # Create a simple list of indices from 0 to num_beams-1
            beam_indices = list(range(len(values)))
        else:
            # Convert beam_indices to list if it's a tensor or array
            if hasattr(beam_indices, "tolist"):
                beam_indices = beam_indices.tolist()
            elif not isinstance(beam_indices, list):
                beam_indices = list(beam_indices)

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        # Save current state before modifications
        current_state = {beam_idx: self.beam_tokens[beam_idx].copy() for beam_idx in range(self.num_beams)}
        self.beam_history[self.current_position] = current_state

        # Handle beam switching
        for i in range(len(beam_indices)):
            self._switch_beam_content(self.current_position, beam_indices[i], i)

        # Iterate through each beam
        for beam_idx in range(len(values)):
            # Get token for current beam
            value = values[beam_idx]

            # Add new tokens to current beam
            self.beam_tokens[beam_idx].extend(value)

            # Decode the entire sequence for current beam
            text = self.tokenizer.decode(self.beam_tokens[beam_idx], **self.decode_kwargs)

            # Update beam text and calculate printable portion
            self.beam_texts[beam_idx] = text
            self.beam_print_lens[beam_idx] = len(text)

            # Notify handler of the beam update with new text
            self.on_beam_update(beam_idx, text)

        self.current_position += 1

    def beam_finished(self, beam_idx: int):
        """Mark a beam as finished and notify the handler."""
        if beam_idx in self.beam_texts:
            self.finished_beams.append(self.beam_texts[beam_idx])

            # Notify handler that the beam is finished
            if self.on_beam_finished:
                self.on_beam_finished(self.finished_beams[-1])

    def end(self):
        """Finish streaming and handle any remaining beams."""
        try:
            # Clean up all beam-related storage
            self.beam_tokens.clear()
            self.beam_texts.clear()
            self.beam_print_lens.clear()
            self.finished_beams.clear()

            # Clean up position tracking
            self.beam_history.clear()
            self.current_position = 0

            # Reset state variables
            self.next_tokens_are_prompt = True

            # Reinitialize storage for potential reuse
            self.beam_tokens = {i: [] for i in range(self.num_beams)}
            self.beam_texts = {i: "" for i in range(self.num_beams)}
            self.beam_print_lens = {i: 0 for i in range(self.num_beams)}
            self.finished_beams = {}
            self.beam_history = {}

        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
            raise


class AsyncTextIteratorStreamer(TextStreamer):
    """
    Streamer that stores print-ready text in a queue, to be used by a downstream application as an async iterator.
    This is useful for applications that benefit from acessing the generated text asynchronously (e.g. in an
    interactive Gradio demo).

    <Tip warning={true}>

    The API for the streamer classes is still under development and may change in the future.

    </Tip>

    Parameters:
        tokenizer (`AutoTokenizer`):
            The tokenized used to decode the tokens.
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
        self, tokenizer: "AutoTokenizer", skip_prompt: bool = False, timeout: float | None = None, **decode_kwargs
    ):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.text_queue = asyncio.Queue()
        self.stop_signal = None
        self.timeout = timeout
        self.loop = asyncio.get_running_loop()
        self.has_asyncio_timeout = hasattr(asyncio, "timeout")

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Put the new text in the queue. If the stream is ending, also put a stop signal in the queue."""
        self.loop.call_soon_threadsafe(self.text_queue.put_nowait, text)
        if stream_end:
            self.loop.call_soon_threadsafe(self.text_queue.put_nowait, self.stop_signal)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            if self.has_asyncio_timeout:
                async with asyncio.timeout(self.timeout):
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
