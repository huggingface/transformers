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
    Simple text streamer that prints a token as soon as it gets them.

    Parameters:
        tokenizer (`AutoTokenizer`):
            The tokenized used to decode the tokens.

    Examples:

        ```python
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

        >>> tok = AutoTokenizer.from_pretrained("distilgpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        >>> inputs = tok(["This cat is"], return_tensors="pt")
        >>> streamer = TextStreamer(tok)
        >>> model.generate(**inputs, streamer=streamer)
        ```
    """

    def __init__(self, tokenizer: "AutoTokenizer"):
        self.tokenizer = tokenizer

    def put(self, value):
        """Prints the token(s) to stdout"""
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]
        text = self.tokenizer.decode(value)
        print(text, flush=True, end="")

    def end(self):
        """Prints a newline to stdout"""
        print("", flush=True)
