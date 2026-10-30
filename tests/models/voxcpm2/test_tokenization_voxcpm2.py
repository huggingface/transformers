# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from transformers.models.llama.tokenization_llama import LlamaTokenizer
from transformers.models.voxcpm2.tokenization_voxcpm2 import VoxCPM2Tokenizer
from transformers.testing_utils import require_tokenizers

from ...test_tokenization_common import TokenizerTesterMixin


def get_tiny_voxcpm2_tokenizer() -> VoxCPM2Tokenizer:
    vocab = {
        "<unk>": 0,
        "<s>": 1,
        "</s>": 2,
        "<pad>": 3,
        "▁": 4,
        "你": 5,
        "好": 6,
        "你好": 7,
        "▁你好": 8,
        "A": 9,
        "B": 10,
        "AB": 11,
        "▁AB": 12,
    }
    for character in map(chr, range(33, 127)):
        if character not in vocab:
            vocab[character] = len(vocab)
    for byte in range(256):
        vocab[f"<0x{byte:02X}>"] = len(vocab)
    merges = [("你", "好"), ("▁", "你好"), ("A", "B"), ("▁", "AB")]
    return VoxCPM2Tokenizer(vocab=vocab, merges=merges, pad_token="<pad>")


@require_tokenizers
class VoxCPM2TokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = VoxCPM2Tokenizer

    @classmethod
    def setUpClass(cls):
        cls.from_pretrained_id = []
        super().setUpClass()
        tokenizer = get_tiny_voxcpm2_tokenizer()
        tokenizer.save_pretrained(cls.tmpdirname)
        cls.from_pretrained_id = [cls.tmpdirname]
        cls.tokenizers_list = [(cls.tokenizer_class, cls.tmpdirname, {})]

    def test_han_characters_are_split_before_bpe_merges(self):
        tokenizer = self.get_tokenizer()
        base_tokenizer = LlamaTokenizer(
            vocab=tokenizer._vocab,
            merges=tokenizer._merges,
            pad_token="<pad>",
            add_prefix_space=True,
        )

        self.assertEqual(base_tokenizer.encode("你好", add_special_tokens=False), [8])
        self.assertEqual(tokenizer.encode("你好", add_special_tokens=False), [4, 5, 6])
        self.assertEqual(tokenizer.tokenize("你好"), ["▁", "你", "好"])

        self.assertEqual(
            tokenizer.encode("AB", add_special_tokens=False),
            base_tokenizer.encode("AB", add_special_tokens=False),
        )
