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

from queue import Queue
from typing import TYPE_CHECKING


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


class TextStreamer(BaseStreamer):
    """
    Simple text streamer that prints the token(s) to stdout as soon as entire words are formed.

    Parameters:
        tokenizer (`AutoTokenizer`):
            The tokenized used to decode the tokens.

    Examples:

        ```python
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

        >>> tok = AutoTokenizer.from_pretrained("gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> inputs = tok(["An increasing sequence: one,"], return_tensors="pt")
        >>> streamer = TextStreamer(tok)

        >>> # Despite returning the usual output, the streamer will also print the generated text to stdout.
        >>> _ = model.generate(**inputs, streamer=streamer, max_new_tokens=20)
        An increasing sequence: one, two, three, four, five, six, seven, eight, nine, ten, eleven,
        ```
    """

    def __init__(self, tokenizer: "AutoTokenizer"):
        self.tokenizer = tokenizer
        self.token_cache = []
        self.print_len = 0

    def put(self, value):
        """
        Recives tokens, decodes them, and prints them to stdout as soon as they form entire words.
        """
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        # Add the new token to the cache and decodes the entire thing.
        self.token_cache.extend(value.tolist())
        text = self.tokenizer.decode(self.token_cache)

        # After symbol for a new line, we flush the cache.
        if text.endswith("\n"):
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        # Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
        # which may change with the subsequent token -- there are probably smarter ways to do this!)
        else:
            printable_text = text[self.print_len : text.rfind(" ") + 1]
            self.print_len += len(printable_text)

        print(printable_text, flush=True, end="")

    def end(self):
        """Flushes any remaining cache and prints a newline to stdout."""
        # Flush the cache, if it exists
        if len(self.token_cache) > 0:
            text = self.tokenizer.decode(self.token_cache)
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        else:
            printable_text = ""

        # Print a newline (and the remaining text, if any)
        print(printable_text, flush=True)


class TextIteratorStreamer(BaseStreamer):
    """
    Streamer that stores print-ready text in a queue, to be used by a downstream application as an iterator. This is
    useful for applications that want to use the generated text in a non-blocking way (e.g. in an interactive Gradio
    demo).

    Parameters:
        tokenizer (`AutoTokenizer`):
            The tokenized used to decode the tokens.

    Examples:

        ```python
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
        >>> from threading import Thread

        >>> tok = AutoTokenizer.from_pretrained("gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
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

    def __init__(self, tokenizer: "AutoTokenizer"):
        self.tokenizer = tokenizer
        self.token_cache = []
        self.print_len = 0
        self.queue = Queue()
        self.stop_signal = None

    def __iter__(self):
        return self

    def __next__(self):
        value = self.queue.get()
        if value == self.stop_signal:
            raise StopIteration()
        else:
            return value

    def put(self, value):
        """
        Recives tokens, decodes them, and pushes text to the queue as soon as it form entire words.
        """
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        # Add the new token to the cache and decodes the entire thing.
        self.token_cache.extend(value.tolist())
        text = self.tokenizer.decode(self.token_cache)

        # After symbol for a new line, we flush the cache.
        if text.endswith("\n"):
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        # Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
        # which may change with the subsequent token -- there are probably smarter ways to do this!)
        else:
            printable_text = text[self.print_len : text.rfind(" ") + 1]
            self.print_len += len(printable_text)
        self.queue.put(printable_text)

    def end(self):
        """Flushes any remaining cache and puts the stop signal in the queue."""
        # Flush the cache, if it exists
        if len(self.token_cache) > 0:
            text = self.tokenizer.decode(self.token_cache)
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        else:
            printable_text = ""

        self.queue.put(printable_text)
        self.queue.put(self.stop_signal)  # Put the stop signal
