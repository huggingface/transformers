# coding=utf-8
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

import unittest
from queue import Empty
from threading import Thread

import pytest

from transformers import (
    AsyncTextIteratorStreamer,
    AutoTokenizer,
    MultiBeamTextStreamer,
    TextIteratorStreamer,
    TextStreamer,
    is_torch_available,
)
from transformers.testing_utils import CaptureStdout, require_torch, torch_device

from ..test_modeling_common import ids_tensor


if is_torch_available():
    import torch

    from transformers import AutoModelForCausalLM


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
        with CaptureStdout() as cs:
            streamer = TextStreamer(tokenizer, skip_special_tokens=True)
            model.generate(input_ids, max_new_tokens=1, do_sample=False, streamer=streamer)

        # The prompt contains a special token, so the streamer should not print it. As such, the output text, when
        # re-tokenized, must only contain one token
        streamer_text = cs.out[:-1]  # Remove the final "\n"
        streamer_text_tokenized = tokenizer(streamer_text, return_tensors="pt")
        self.assertEqual(streamer_text_tokenized.input_ids.shape, (1, 1))

    def test_iterator_streamer_timeout(self):
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)
        model.config.eos_token_id = -1

        input_ids = ids_tensor((1, 5), vocab_size=model.config.vocab_size).to(torch_device)
        streamer = TextIteratorStreamer(tokenizer, timeout=0.001)
        generation_kwargs = {"input_ids": input_ids, "max_new_tokens": 10, "do_sample": False, "streamer": streamer}
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        # The streamer will timeout after 0.001 seconds, so an exception will be raised
        with self.assertRaises(Empty):
            streamer_text = ""
            for new_text in streamer:
                streamer_text += new_text

    def test_multibeam_text_streamer_matches_non_streaming(self):
        """Test that the MultiBeamTextStreamer produces the same output as non-streaming beam search."""
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)
        model.config.eos_token_id = -1

        # Generate input and get baseline output
        input_ids = ids_tensor((1, 2), vocab_size=model.config.vocab_size).to(torch_device)
        beam_outputs = model.generate(
            input_ids,
            max_new_tokens=10,
            num_beams=2,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            num_return_sequences=2,
        )
        beam_texts = [tokenizer.decode(output) for output in beam_outputs.sequences]

        # Store for comparing streaming output
        beam_texts_from_streamer = {i: "" for i in range(2)}
        beams_from_streamer = {i: "" for i in range(2)}
        idx = 0

        def on_beam_update(beam_idx: int, new_text: str):
            """Handler for beam updates"""
            beam_texts_from_streamer[beam_idx] = new_text

        def on_beam_finished(text: str):
            global idx
            beams_from_streamer[idx] = text

        # Create streamer with handlers
        streamer = MultiBeamTextStreamer(
            tokenizer=tokenizer, num_beams=2, on_beam_update=on_beam_update, on_beam_finished=on_beam_finished, skip_prompt=False
        )

        # Generate with streamer
        model.generate(input_ids, max_new_tokens=10, num_beams=2, do_sample=False, streamer=streamer)

        # Compare streamed outputs
        for i, beam in enumerate(beams_from_streamer.values()):
            self.assertTrue(beam, beam_texts[i])

    def test_multibeam_text_streamer_beam_history(self):
        """Test that the MultiBeamTextStreamer correctly maintains beam history and handles beam switching."""
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)
        model.config.eos_token_id = -1

        input_ids = ids_tensor((1, 2), vocab_size=model.config.vocab_size).to(torch_device)

        # Track beam history
        beam_histories = {i: [] for i in range(2)}
        beam_switch_count = 0

        def on_beam_update(beam_idx: int, new_text: str):
            beam_histories[beam_idx].append(new_text)

        def on_beam_finished(text: str):
            nonlocal beam_switch_count
            beam_switch_count += 1

        streamer = MultiBeamTextStreamer(
            tokenizer=tokenizer,
            num_beams=2,
            on_beam_update=on_beam_update,
            on_beam_finished=on_beam_finished,
            skip_prompt=False,
        )

        # Generate with beam search
        model.generate(input_ids, max_new_tokens=10, num_beams=2, do_sample=False, streamer=streamer)

        # Verify that both beams received updates
        self.assertTrue(len(beam_histories[0]) > 0)
        self.assertTrue(len(beam_histories[1]) > 0)

        # Verify that beam histories are different
        self.assertNotEqual(beam_histories[0], beam_histories[1])

    def test_multibeam_text_streamer_error_handling(self):
        """Test that the MultiBeamTextStreamer properly handles error conditions."""
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")

        # Test invalid num_beams
        with self.assertRaises(ValueError):
            MultiBeamTextStreamer(tokenizer, num_beams=0, on_beam_update=lambda x, y: None)

        with self.assertRaises(ValueError):
            MultiBeamTextStreamer(tokenizer, num_beams=-1, on_beam_update=lambda x, y: None)

        # Initialize valid streamer for further tests
        streamer = MultiBeamTextStreamer(tokenizer=tokenizer, num_beams=2, on_beam_update=lambda x, y: None)

        # Test putting values with wrong shape
        with self.assertRaises(ValueError):
            streamer.put(torch.tensor([1, 2, 3]))  # Wrong shape (1D instead of 2D)

        # Test putting values with too many beams
        with self.assertRaises(ValueError):
            streamer.put(torch.tensor([[1], [2], [3]]))  # 3 beams when initialized with 2

        # Test invalid beam switching
        with self.assertRaises(ValueError):
            streamer._switch_beam_content(0, 0, 5)  # Invalid new beam index


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
        streamer = AsyncTextIteratorStreamer(tokenizer, timeout=0.001)
        generation_kwargs = {"input_ids": input_ids, "max_new_tokens": 10, "do_sample": False, "streamer": streamer}
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        # The streamer will timeout after 0.001 seconds, so TimeoutError will be raised
        with self.assertRaises(TimeoutError):
            streamer_text = ""
            async for new_text in streamer:
                streamer_text += new_text
