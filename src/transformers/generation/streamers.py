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

import multiprocessing

from transformers import AutoTokenizer


class BaseStreamer:
    """
    Base class from which `.generate()` streamers should inherit. In a nutshell, a `put()` method needs to be
    implemented, as it is called by `.generate()` to send tokens to the streamer.
    """

    def put(self, value):
        raise NotImplementedError()


class TextStreamer(BaseStreamer):
    """
    Simple text streamer that uses a queue to receive tokens and print them to stdout in a separate process. It is
    meant to be used as a context manager wrapping `.generate()`. Since it relies on spawning a new process, the `if
    __name__ == '__main__':` guard is required when using this class.

    Parameters:
        tokenizer (`AutoTokenizer`):
            The tokenized used to decode the tokens.

    Examples:

        ```python
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

        >>> if __name__ == "__main__":
        ...     tok = AutoTokenizer.from_pretrained("distilgpt2")
        ...     model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        ...     inputs = tok(["This cat is"], return_tensors="pt")
        ...     with TextStreamer(tok) as streamer:
        ...         model.generate(**inputs, streamer=streamer)
        ```
    """

    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self.ctx = multiprocessing.get_context("spawn")  # CUDA complains otherwise :)
        self.queue = self.ctx.Queue()
        self.end_signal = -1

    def put(self, value):
        self.queue.put(value)

    def _decode_text(self):
        while True:
            queue_value = self.queue.get()

            if isinstance(queue_value, int) and queue_value == self.end_signal:
                print("", flush=True)
                break
            else:
                if len(queue_value.shape) > 1 and queue_value.shape[0] > 1:
                    raise ValueError("TextStreamer only supports batch size 1")
                elif len(queue_value.shape) > 1:
                    queue_value = queue_value[0]
                text = self.tokenizer.decode(queue_value)
                print(text, flush=True, end="")

    def __enter__(self):
        self.process = self.ctx.Process(target=self._decode_text)
        self.process.start()
        return self

    def __exit__(self, *args):
        self.queue.put(self.end_signal)
        self.process.join()
