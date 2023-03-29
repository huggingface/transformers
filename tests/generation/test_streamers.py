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

import pytest
import unittest

from transformers import AutoTokenizer, TextStreamer, is_torch_available
from transformers.testing_utils import require_torch, torch_device

from ..test_modeling_common import ids_tensor

if is_torch_available():
    from transformers import AutoModelForCausalLM


@require_torch
class StreamerTester(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def capfd(self, capfd):
        self.capfd = capfd

    def test_text_streamer_stdout(self):
        # Note: this test relies on pytest's capfd fixture to capture the stdout from a subprocess.
        # testing_utils.CaptureStdout cannot capture the output from a subprocess.
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)
        model.config.eos_token_id = -1

        input_ids = ids_tensor((1, 5), vocab_size=model.config.vocab_size).to(torch_device)
        greedy_ids = model.generate(input_ids, max_new_tokens=10, do_sample=False)
        greedy_text = tokenizer.decode(greedy_ids[0])
        print(greedy_text)
        captured_print = self.capfd.readouterr().out

        with TextStreamer(tokenizer) as streamer:
            model.generate(input_ids, max_new_tokens=10, do_sample=False, streamer=streamer)
        captured_stream = self.capfd.readouterr().out

        # The greedy text should be printed to stdout
        self.assertEqual(captured_print, captured_stream)
