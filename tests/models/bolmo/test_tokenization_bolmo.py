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

import json
import os
import unittest

from transformers import BolmoTokenizer
from transformers.convert_slow_tokenizer import bytes_to_unicode

from ...test_tokenization_common import TokenizerTesterMixin
from ..byt5.test_tokenization_byt5 import ByT5TokenizationTest


class BolmoTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = BolmoTokenizer
    # TODO: "allenai/Bolmo-7B" once its `vocab.json` is on the Hub
    from_pretrained_id = []

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # every single byte, a few merges, and the renamed eos/pad of the subword tokenizer
        byte_chars = list(bytes_to_unicode().values())
        tokens = byte_chars + ["He", "Hell", "Hello", "Ġwor", "Ġworld", "<bos>", "<pad>"]
        vocab_file = os.path.join(cls.tmpdirname, "vocab.json")
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump({token: i for i, token in enumerate(tokens)}, f)
        BolmoTokenizer(vocab_file).save_pretrained(cls.tmpdirname)

    @classmethod
    def get_tokenizer(cls, pretrained_name=None, **kwargs) -> BolmoTokenizer:
        return cls.tokenizer_class.from_pretrained(pretrained_name or cls.tmpdirname, **kwargs)

    get_clean_sequence = ByT5TokenizationTest.get_clean_sequence

    @unittest.skip("Bytes have no word boundaries to pre-tokenize on")
    def test_pretokenized_inputs(self):
        pass

    def test_bytes_and_expansion(self):
        tokenizer = self.get_tokenizer()
        vocab = tokenizer.subword_vocab
        encoding = tokenizer("Hello world")
        # <bos>, then UTF-8 bytes after the 4 special tokens
        self.assertEqual(encoding["input_ids"], [1] + [byte + 4 for byte in b"Hello world"])
        # longest subword ending at each byte
        expected = ["<bos>", "H", "He", "l", "Hell", "Hello", "Ġ", "w", "o", "Ġwor", "l", "Ġworld"]
        self.assertEqual(encoding["expanded_input_ids"], [vocab[token] for token in expected])

    def test_decode_strips_boundaries(self):
        tokenizer = self.get_tokenizer()
        input_ids = tokenizer("Hello")["input_ids"]
        # bytes shifted by 260 close a patch, `<bpe_token_end>` (3) is a wait marker
        fused = [token + 260 if i % 2 else token for i, token in enumerate(input_ids)] + [3]
        self.assertEqual(tokenizer.decode(fused, skip_special_tokens=True), "Hello")
        self.assertEqual(tokenizer.expand_byte_ids(fused, n_last=2), tokenizer.expand_byte_ids(fused)[-2:])
