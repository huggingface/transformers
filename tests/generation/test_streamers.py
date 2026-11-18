# Copyright 2023 The HuggingFace Team Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a clone of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import unittest
from queue import Empty
from threading import Event, Thread
from unittest.mock import patch

import pytest

from transformers import (
    AsyncTextIteratorStreamer,
    AutoTokenizer,
    TextIteratorStreamer,
    TextStreamer,
    is_torch_available,
)
from transformers.testing_utils import CaptureStdout, require_torch, torch_device
from transformers.utils.logging import _get_library_root_logger

from ..test_modeling_common import ids_tensor


if is_torch_available():
    import torch

    from transformers import AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList

    class BlockingLogitsProcessor(LogitsProcessor):
        """Holds generation on the first decoding step until `release` is set."""

        def __init__(self, release):
            self.release = release

        def __call__(self, input_ids, scores):
            self.release.wait()
            return scores


@require_torch
class StreamerTester(unittest.TestCase):
    def test_text_streamer_matches_non_streaming(self):
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)
        model.config.eos_token_id = -1

        input_ids = ids_tensor((1, 5), vocab_size=model.config.vocab_size).to(torch_device)
        greedy_ids = model.generate(input_ids, max_new_tokens=10, do_sample=False)
        greedy_text = tokenizer.decode(greedy_ids[0])

        with CaptureStdout() as cs:
            streamer = TextStreamer(tokenizer)
            model.generate(input_ids, max_new_tokens=10, do_sample=False, streamer=streamer)
        # The greedy text should be printed to stdout, except for the final "\n" in the streamer
        streamer_text = cs.out[:-1]

        self.assertEqual(streamer_text, greedy_text)

    def test_iterator_streamer_matches_non_streaming(self):
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)
        model.config.eos_token_id = -1

        input_ids = ids_tensor((1, 5), vocab_size=model.config.vocab_size).to(torch_device)
        greedy_ids = model.generate(input_ids, max_new_tokens=10, do_sample=False)
        greedy_text = tokenizer.decode(greedy_ids[0])

        streamer = TextIteratorStreamer(tokenizer)
        generation_kwargs = {"input_ids": input_ids, "max_new_tokens": 10, "do_sample": False, "streamer": streamer}
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        streamer_text = ""
        for new_text in streamer:
            streamer_text += new_text

        self.assertEqual(streamer_text, greedy_text)

    def test_text_streamer_skip_prompt(self):
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)
        model.config.eos_token_id = -1

        input_ids = ids_tensor((1, 5), vocab_size=model.config.vocab_size).to(torch_device)
        greedy_ids = model.generate(input_ids, max_new_tokens=10, do_sample=False)
        new_greedy_ids = greedy_ids[:, input_ids.shape[1] :]
        new_greedy_text = tokenizer.decode(new_greedy_ids[0])

        with CaptureStdout() as cs:
            streamer = TextStreamer(tokenizer, skip_prompt=True)
            model.generate(input_ids, max_new_tokens=10, do_sample=False, streamer=streamer)
        # The greedy text should be printed to stdout, except for the final "\n" in the streamer
        streamer_text = cs.out[:-1]

        self.assertEqual(streamer_text, new_greedy_text)

    def test_text_streamer_decode_kwargs(self):
        # Tests that we can pass `decode_kwargs` to the streamer to control how the tokens are decoded. Must be tested
        # with actual models -- the dummy models' tokenizers are not aligned with their models, and
        # `skip_special_tokens=True` has no effect on them
        tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
        model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2").to(torch_device)
        model.config.eos_token_id = -1

        input_ids = torch.ones((1, 5), device=torch_device).long() * model.config.bos_token_id

        root = _get_library_root_logger()
        with patch.object(root, "propagate", False):
            with CaptureStdout() as cs:
                streamer = TextStreamer(tokenizer, skip_special_tokens=True)
                model.generate(input_ids, max_new_tokens=1, do_sample=False, streamer=streamer)

        # The prompt contains a special token, so the streamer should not print it. As such, the output text, when
        # re-tokenized, must only contain one token
        streamer_text = cs.out[:-1]  # Remove the final "\n"
        streamer_text_tokenized = tokenizer(streamer_text, return_tensors="pt")
        self.assertEqual(streamer_text_tokenized.input_ids.shape, (1, 1))

    def test_text_streamer_long_line_decodes_bounded_window(self):
        # Without newlines, the streamer only decodes the most recent tokens instead of the whole line
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        token_ids = tokenizer.encode("Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 100)

        with CaptureStdout() as cs:
            streamer = TextStreamer(tokenizer)
            with patch.object(tokenizer, "decode", wraps=tokenizer.decode) as mock_decode:
                for token_id in token_ids:
                    streamer.put(torch.tensor([token_id]))
                streamer.end()

        self.assertEqual(cs.out[:-1], tokenizer.decode(token_ids))
        longest_decode = max(len(call.args[0]) for call in mock_decode.call_args_list)
        self.assertLessEqual(longest_decode, 2 * TextStreamer._decode_window_size + 1)

    def test_text_streamer_invalid_byte_before_window(self):
        # With byte fallback, an invalid byte turns the whole run of byte tokens into replacement characters, including
        # the tokens that are already before the decode window
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
        token_ids = tokenizer.encode("Look: " + "😀🎉🦄🧬" * 10, add_special_tokens=False)
        token_ids += [tokenizer.convert_tokens_to_ids("<0xF0>")] + tokenizer.encode(" next", add_special_tokens=False)

        with CaptureStdout() as cs:
            streamer = TextStreamer(tokenizer)
            for token_id in token_ids:
                streamer.put(torch.tensor([token_id]))
            streamer.end()

        self.assertEqual(cs.out[:-1], tokenizer.decode(token_ids))

    def test_text_streamer_matches_full_decode(self):
        # Every printed chunk must be the same as when the whole cache is decoded at each step, which is what the
        # streamer does when its decode window never moves
        class RecordingStreamer(TextStreamer):
            def on_finalized_text(self, text, stream_end=False):
                self.chunks.append((text, stream_end))

        class FullDecodeStreamer(RecordingStreamer):
            _decode_window_size = 10**9

        rng = random.Random(0)
        emojis = "😀🎉🦄🧬𠀀 " * 30
        text = "Streaming 😀 émojis, 我们今天 and long lines , isn't it ? It ' s fine !\n" * 8
        for checkpoint in (
            "openai-community/gpt2",
            "hf-internal-testing/llama-tokenizer",
            "google-bert/bert-base-uncased",
        ):
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            text_ids = [tokenizer.encode(t, add_special_tokens=False) for t in (text, text.replace("\n", " "), emojis)]
            for _ in range(15):
                token_ids = list(rng.choice(text_ids))
                # Random tokens, including the byte tokens at the start of the vocabulary
                for _ in range(10):
                    token_id = rng.randrange(rng.choice([300, len(tokenizer)]))
                    token_ids.insert(rng.randrange(len(token_ids)), token_id)
                decode_kwargs = rng.choice([{}, {"skip_special_tokens": True}, {"clean_up_tokenization_spaces": True}])
                streamers = [
                    RecordingStreamer(tokenizer, **decode_kwargs),
                    FullDecodeStreamer(tokenizer, **decode_kwargs),
                ]
                streamers[0]._decode_window_size = rng.choice([1, 4, 32])
                for streamer in streamers:
                    streamer.chunks = []
                    # A streamer can be used for several generations
                    for _ in range(2):
                        for token_id in token_ids:
                            streamer.put(torch.tensor([token_id]))
                        streamer.end()
                self.assertEqual(streamers[0].chunks, streamers[1].chunks)

    def test_iterator_streamer_timeout(self):
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)
        model.config.eos_token_id = -1

        input_ids = ids_tensor((1, 5), vocab_size=model.config.vocab_size).to(torch_device)
        # `skip_prompt` keeps the prompt out of the queue, and generation is held until the timeout has fired, so
        # the queue is guaranteed to be empty: otherwise the test races generation against the timeout
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, timeout=0.001)
        release = Event()
        generation_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": 10,
            "do_sample": False,
            "streamer": streamer,
            "logits_processor": LogitsProcessorList([BlockingLogitsProcessor(release)]),
        }
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        # The streamer will timeout after 0.001 seconds, so an exception will be raised
        try:
            with self.assertRaises(Empty):
                for _ in streamer:
                    pass
        finally:
            release.set()
            thread.join()


@require_torch
@pytest.mark.asyncio(loop_scope="class")
class AsyncStreamerTester(unittest.IsolatedAsyncioTestCase):
    async def test_async_iterator_streamer_matches_non_streaming(self):
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)
        model.config.eos_token_id = -1

        input_ids = ids_tensor((1, 5), vocab_size=model.config.vocab_size).to(torch_device)
        greedy_ids = model.generate(input_ids, max_new_tokens=10, do_sample=False)
        greedy_text = tokenizer.decode(greedy_ids[0])

        streamer = AsyncTextIteratorStreamer(tokenizer)
        generation_kwargs = {"input_ids": input_ids, "max_new_tokens": 10, "do_sample": False, "streamer": streamer}
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        streamer_text = ""
        async for new_text in streamer:
            streamer_text += new_text

        self.assertEqual(streamer_text, greedy_text)

    async def test_async_iterator_streamer_timeout(self):
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)
        model.config.eos_token_id = -1

        input_ids = ids_tensor((1, 5), vocab_size=model.config.vocab_size).to(torch_device)
        # Same as the sync test: no prompt in the queue and generation held, so the timeout must fire
        streamer = AsyncTextIteratorStreamer(tokenizer, skip_prompt=True, timeout=0.001)
        release = Event()
        generation_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": 10,
            "do_sample": False,
            "streamer": streamer,
            "logits_processor": LogitsProcessorList([BlockingLogitsProcessor(release)]),
        }
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        # The streamer will timeout after 0.001 seconds, so TimeoutError will be raised
        try:
            with self.assertRaises(TimeoutError):
                async for _ in streamer:
                    pass
        finally:
            release.set()
            thread.join()
