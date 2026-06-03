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

import tempfile
import unittest

from transformers import AutoTokenizer, ESMCTokenizer
from transformers.testing_utils import require_tokenizers, slow


@require_tokenizers
class ESMCTokenizationTest(unittest.TestCase):
    tokenizer_class = ESMCTokenizer

    def get_tokenizer(self, **kwargs) -> ESMCTokenizer:
        # ESMC is a fast-only tokenizer; the vocab is built in __init__ (no vocab file needed).
        return ESMCTokenizer(**kwargs)

    def test_documented_example(self):
        tokenizer = self.get_tokenizer()
        # 20-residue sequence -> 20 residues wrapped in <cls> ... <eos> = 22 ids.
        ids = tokenizer("ACDEFGHIKLMNPQRSTVWY")["input_ids"]
        self.assertListEqual(
            ids,
            [0, 5, 23, 13, 9, 18, 6, 21, 12, 15, 4, 20, 17, 14, 16, 10, 8, 11, 7, 22, 19, 2],
        )

    def test_tokenize_is_character_level(self):
        tokenizer = self.get_tokenizer()
        self.assertListEqual(tokenizer.tokenize("LAGVS"), ["L", "A", "G", "V", "S"])
        self.assertListEqual(tokenizer.convert_tokens_to_ids(["L", "A", "G", "V", "S"]), [4, 5, 6, 7, 8])

    def test_encode_wraps_cls_eos(self):
        tokenizer = self.get_tokenizer()
        self.assertListEqual(tokenizer.encode("LAGVS"), [0, 4, 5, 6, 7, 8, 2])

    def test_special_token_ids(self):
        tokenizer = self.get_tokenizer()
        self.assertEqual(tokenizer.cls_token_id, 0)
        self.assertEqual(tokenizer.pad_token_id, 1)
        self.assertEqual(tokenizer.eos_token_id, 2)
        self.assertEqual(tokenizer.unk_token_id, 3)
        self.assertEqual(tokenizer.mask_token_id, 32)
        # ESMC uses <cls> as the sequence-start token; it is aliased to bos.
        self.assertEqual(tokenizer.bos_token_id, tokenizer.cls_token_id)
        self.assertEqual(tokenizer.vocab_size, 33)

    def test_batch_padding(self):
        tokenizer = self.get_tokenizer()
        batch = tokenizer(["LAGVS", "WC"], padding=True)
        self.assertListEqual(batch["input_ids"][0], [0, 4, 5, 6, 7, 8, 2])
        self.assertListEqual(batch["input_ids"][1], [0, 22, 23, 2, 1, 1, 1])
        self.assertListEqual(batch["attention_mask"][1], [1, 1, 1, 1, 0, 0, 0])

    def test_chain_break_token(self):
        tokenizer = self.get_tokenizer()
        self.assertEqual(tokenizer.chain_break_token, "|")
        ids = tokenizer("MK|AY")["input_ids"]
        self.assertIn(tokenizer.chain_break_token_id, ids)
        self.assertEqual(tokenizer.chain_break_token_id, 31)

    def test_mask_token(self):
        tokenizer = self.get_tokenizer()
        self.assertIn(tokenizer.mask_token_id, tokenizer("MK<mask>T")["input_ids"])

    def test_unknown_residue_maps_to_unk(self):
        tokenizer = self.get_tokenizer()
        # "J" is not a valid amino-acid token in the ESMC vocabulary.
        self.assertIn(tokenizer.unk_token_id, tokenizer("MKJT")["input_ids"])

    def test_decode_round_trip(self):
        tokenizer = self.get_tokenizer()
        seq = "MKTAYIAKQR"
        decoded = tokenizer.decode(tokenizer(seq)["input_ids"], skip_special_tokens=True).replace(" ", "")
        self.assertEqual(decoded, seq)

    def test_save_and_load(self):
        tokenizer = self.get_tokenizer()
        seq = "MKTAYIAKQR"
        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer.save_pretrained(tmpdir)
            reloaded = ESMCTokenizer.from_pretrained(tmpdir)
        self.assertListEqual(tokenizer(seq)["input_ids"], reloaded(seq)["input_ids"])

    @slow
    def test_tokenizer_integration(self):
        # The published checkpoint's tokenizer.json must match the code-built tokenizer,
        # and AutoTokenizer must resolve to ESMCTokenizer.
        seq = "ACDEFGHIKLMNPQRSTVWY"
        built = self.get_tokenizer()
        auto = AutoTokenizer.from_pretrained("biohub/ESMC-6B")
        self.assertIsInstance(auto, ESMCTokenizer)
        self.assertListEqual(built(seq)["input_ids"], auto(seq)["input_ids"])
