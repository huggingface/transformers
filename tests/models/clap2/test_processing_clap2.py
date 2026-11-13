# Copyright 2026 the HuggingFace Team. All rights reserved.
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
"""Testing suite for the CLAP2 processor."""

import unittest

import numpy as np

from transformers import Clap2Processor, GPT2Tokenizer
from transformers.testing_utils import require_torch

from ...test_processing_common import ProcessorTesterMixin


@require_torch
class Clap2ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Clap2Processor

    @classmethod
    def _setup_tokenizer(cls):
        # the conversion script's tokenizer: GPT-2 BPE wrapped as <|bos|> text <|eos|>, padded with <|pad|>
        tokenizer = GPT2Tokenizer.from_pretrained(
            "openai-community/gpt2", model_max_length=256, split_special_tokens=True, truncation=True
        )
        for name, token in [("eos_token", "<|eos|>"), ("pad_token", "<|pad|>"), ("bos_token", "<|bos|>")]:
            tokenizer.add_special_tokens({name: token})
        tokenizer.add_bos_token = tokenizer.add_eos_token = True
        tokenizer.update_post_processor()
        return tokenizer

    def test_tokenizer_wraps_text(self):
        input_ids = self.get_processor()(text="a dog barking")["input_ids"]
        self.assertEqual(input_ids, [50259, 64, 3290, 48961, 50257])  # tiktoken r50k_base, between <|bos|> and <|eos|>
        # special-token strings inside a text are ordinary text, as with tiktoken's `encode_ordinary`
        input_ids = self.get_processor()(text="<|bos|> <|endoftext|> <|pad|>")["input_ids"]
        self.assertTrue(set(input_ids[1:-1]).isdisjoint({50256, 50257, 50258, 50259}))

    def test_truncation(self):
        input_ids = self.get_processor()(text="a " * 300, padding="max_length")["input_ids"]
        self.assertEqual(len(input_ids), 256)
        self.assertEqual(input_ids[-1], 50257)

    def test_text_and_audio(self):
        processor = self.get_processor()
        audio = np.random.default_rng(0).standard_normal(32000).astype(np.float32) * 0.1
        inputs = processor(
            text=["a dog barking", "rain"], audio=audio, sampling_rate=32000, padding=True, return_tensors="pt"
        )
        self.assertSetEqual(set(inputs), {"input_ids", "attention_mask", "input_features", "input_features_mask"})
        self.assertEqual(inputs["input_ids"][1, -1].item(), 50258)
        features = processor.feature_extractor(audio, sampling_rate=32000, return_tensors="pt")
        self.assertTrue(inputs["input_features"].equal(features["input_features"]))

    def test_max_num_patches(self):
        audio = np.random.default_rng(0).standard_normal(5 * 32000).astype(np.float32) * 0.1
        inputs = self.get_processor()(audio=audio, sampling_rate=32000, max_num_patches=16, return_tensors="pt")
        self.assertEqual(inputs["input_features"].shape[1], 16)
