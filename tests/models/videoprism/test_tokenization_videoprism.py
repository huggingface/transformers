# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import unittest

import torch

from transformers import VideoPrismTokenizer
from transformers.testing_utils import require_tokenizers

from ...test_tokenization_common import TokenizerTesterMixin


# Golden token ids from convert_videoprism_weights_to_hf.py
VIDEO_PRISM_DEMO_SENTENCES = [
    [262, 266, 768, 267, 1376, 14293, 259],
    [262, 266, 768, 267, 2865, 259],
    [262, 266, 768, 267, 1376, 20682, 259],
    [262, 266, 768, 267, 1376, 289, 10691, 259],
    [262, 266, 768, 267, 4605, 259],
]
TEXT_QUERY_CSV = "playing drums,sitting,playing flute,playing at playground,concert"
PROMPT_TEMPLATE = "a video of {}."
VIDEO_PRISM_DEMO_TEXT = [PROMPT_TEMPLATE.format(t) for t in TEXT_QUERY_CSV.split(",")]


@require_tokenizers
class VideoPrismTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = VideoPrismTokenizer
    from_pretrained_id = "MHRDYN7/videoprism-lvt-base-f16r288"

    integration_expected_tokens = ['▁This', '▁is', '▁', 'a', '▁test', '▁', '😊', '▁I', '▁was', '▁born', '▁in', '▁9', '2000', ',', '▁and', '▁this', '▁is', '▁fal', 's', 'é', '.', '▁', '生活的真谛是', '▁Hi', '▁Hello', '▁Hi', '▁Hello', '▁Hello', '<s>', '▁hi', '<s>', '▁there', '▁The', '▁following', '▁string', '▁should', '▁be', '▁properly', '▁', 'encoded', ':', '▁Hello', '.', '▁But', '▁', 'i', 'r', 'd', '▁and', '▁', 'ปี', '▁', 'i', 'r', 'd', '▁', 'ด', '▁Hey', '▁how', '▁are', '▁you', '▁doing']  # fmt: skip
    integration_expected_token_ids = [330, 269, 262, 266, 937, 262, 2, 274, 292, 1700, 268, 914, 12125, 261, 263, 291, 269, 16201, 264, 2083, 259, 262, 2, 2038, 5930, 2038, 5930, 5930, 32100, 7808, 32100, 354, 281, 840, 4652, 412, 282, 2366, 262, 25966, 304, 5930, 259, 464, 262, 302, 331, 303, 263, 262, 2, 262, 302, 331, 303, 262, 2, 6919, 364, 280, 273, 742]  # fmt: skip
    expected_tokens_from_ids = ['▁This', '▁is', '▁', 'a', '▁test', '▁', '<unk>', '▁I', '▁was', '▁born', '▁in', '▁9', '2000', ',', '▁and', '▁this', '▁is', '▁fal', 's', 'é', '.', '▁', '<unk>', '▁Hi', '▁Hello', '▁Hi', '▁Hello', '▁Hello', '<s>', '▁hi', '<s>', '▁there', '▁The', '▁following', '▁string', '▁should', '▁be', '▁properly', '▁', 'encoded', ':', '▁Hello', '.', '▁But', '▁', 'i', 'r', 'd', '▁and', '▁', '<unk>', '▁', 'i', 'r', 'd', '▁', '<unk>', '▁Hey', '▁how', '▁are', '▁you', '▁doing']  # fmt: skip
    integration_expected_decoded_text = "This is a test <unk> I was born in 92000, and this is falsé. <unk> Hi Hello Hi Hello Hello<s> hi<s> there The following string should be properly encoded: Hello. But ird and <unk> ird <unk> Hey how are you doing"

    def test_pad_token_id(self):
        tokenizer = self.get_tokenizer()
        self.assertEqual(tokenizer.pad_token_id, 0)

    def test_vocab_size(self):
        tokenizer = self.get_tokenizer()
        self.assertGreaterEqual(tokenizer.vocab_size, 32000)

    def test_no_eos_appended(self):
        tokenizer = self.get_tokenizer()
        encoded = tokenizer("a video of playing drums", add_special_tokens=False)
        self.assertNotEqual(encoded[-1], tokenizer.eos_token_id)

    def test_demo_sentence_tokenization(self):
        tokenizer = self.get_tokenizer()
        encoded = tokenizer(
            VIDEO_PRISM_DEMO_TEXT,
            max_length=64,
            padding="max_length",
            return_tensors="pt",
        )["input_ids"]
        expected = torch.tensor(
            [sentence + [0] * (64 - len(sentence)) for sentence in VIDEO_PRISM_DEMO_SENTENCES],
            dtype=torch.long,
        )
        self.assertTrue(torch.equal(encoded, expected))
