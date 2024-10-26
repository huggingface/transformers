# Copyright 2021 The HuggingFace Team. All rights reserved.
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
from pathlib import Path
from shutil import copyfile

from transformers import PrismTokenizer, is_torch_available
from transformers.testing_utils import (
    get_tests_dir,
    nested_simplify,
    require_sentencepiece,
    require_tokenizers,
    require_torch,
    slow,
)
from transformers.utils import is_sentencepiece_available


if is_sentencepiece_available():
    from transformers.models.prism.tokenization_prism import VOCAB_FILES_NAMES, save_json

from ...test_tokenization_common import TokenizerTesterMixin


if is_sentencepiece_available():
    SAMPLE_SP = get_tests_dir("fixtures/test_sentencepiece.model")


if is_torch_available():
    from transformers.models.m2m_100.modeling_m2m_100 import shift_tokens_right

EN_CODE = 37
FR_CODE = 71


@require_sentencepiece
class PrismTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "facebook/prism"
    tokenizer_class = PrismTokenizer
    test_rust_tokenizer = False
    test_seq2seq = False
    test_sentencepiece = True

    def setUp(self):
        super().setUp()

        vocab = ["</s>", "<unk>", "▁This", "▁is", "▁a", "▁t", "est", "\u0120", "<pad>", "<en>"]
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        save_dir = Path(self.tmpdirname)
        save_json(vocab_tokens, save_dir / VOCAB_FILES_NAMES["vocab_file"])
        if not (save_dir / VOCAB_FILES_NAMES["spm_file"]).exists():
            copyfile(SAMPLE_SP, save_dir / VOCAB_FILES_NAMES["spm_file"])

        tokenizer = PrismTokenizer.from_pretrained(self.tmpdirname)
        tokenizer.save_pretrained(self.tmpdirname)

    def get_tokenizer(self, **kwargs):
        return PrismTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def test_convert_token_and_id(self):
        """Test ``_convert_token_to_id`` and ``_convert_id_to_token``."""
        token = "</s>"
        token_id = 0

        self.assertEqual(self.get_tokenizer()._convert_token_to_id(token), token_id)
        self.assertEqual(self.get_tokenizer()._convert_id_to_token(token_id), token)

    def test_get_vocab(self):
        tokenizer = self.get_tokenizer()
        vocab_keys = list(tokenizer.get_vocab().keys())

        self.assertEqual(vocab_keys[0], "</s>")
        self.assertEqual(vocab_keys[1], "<unk>")
        self.assertEqual(vocab_keys[-1], "<s>")


    def test_full_tokenizer(self):
        tokenizer = self.get_tokenizer()

        tokens = tokenizer.tokenize("This is a test")
        self.assertListEqual(tokens, ["▁This", "▁is", "▁a", "▁t", "est"])

        self.assertListEqual(
            tokenizer.convert_tokens_to_ids(tokens),
            [2, 3, 4, 5, 6],
        )

        back_tokens = tokenizer.convert_ids_to_tokens([2, 3, 4, 5, 6])
        self.assertListEqual(back_tokens, ["▁This", "▁is", "▁a", "▁t", "est"])

        text = tokenizer.convert_tokens_to_string(tokens)
        self.assertEqual(text, "This is a test")


@require_torch
@require_sentencepiece
@require_tokenizers
class PrismTokenizerIntegrationTest(unittest.TestCase):
    checkpoint_name = "facebook/prism"
    src_text = ["Hi world.", 
            "This is a Test.",
            "Some of my Best Friends are Linguists."]
    tgt_text = ['<fr> Hé, monde!', "<fr> C'est un test.", '<fr> Certains de mes meilleurs amis sont linguistes.']

    expected_src_tokens = [37, 5050, 21, 1951, 13934, 33789, 7, 269, 11348, 983, 9393, 6, 2]
    
    @classmethod
    def setUpClass(cls):
        cls.tokenizer: PrismTokenizer = PrismTokenizer.from_pretrained(
            cls.checkpoint_name, src_lang="en", tgt_lang="fr"
        )
        cls.pad_token_id = 1
        return cls

    def test_language_codes(self):
        self.assertEqual(self.tokenizer.get_lang_id("bg"), 327)
        self.assertEqual(self.tokenizer.get_lang_id("en"), 37)
        self.assertEqual(self.tokenizer.get_lang_id("ro"), 299)
        self.assertEqual(self.tokenizer.get_lang_id("uk"), 401)

    def test_get_vocab(self):
        vocab_keys = list(self.tokenizer.get_vocab().keys())

        self.assertEqual(vocab_keys[2], "</s>")
        self.assertEqual(vocab_keys[3], "<unk>")
        self.assertEqual(vocab_keys[0], "<s>")
        self.assertEqual(vocab_keys[1], "<pad>")

    def test_tokenizer_batch_encode_plus(self):
        self.tokenizer.src_lang = "en"
        ids = self.tokenizer.batch_encode_plus(self.src_text).input_ids[2]
        self.assertListEqual(self.expected_src_tokens, ids)

    def test_encoding(self):
        text = "Hello, world!"
        encoded = self.tokenizer.encode(text)
        self.assertIsInstance(encoded, list)
        self.assertGreater(len(encoded), 0)
        self.assertEqual(encoded[0], self.tokenizer.get_lang_id("en"))

    def test_special_tokens_unaffacted_by_save_load(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            original_special_tokens = self.tokenizer.lang_token_to_id
            self.tokenizer.save_pretrained(tmpdirname)
            new_tok = PrismTokenizer.from_pretrained(tmpdirname)
            self.assertDictEqual(new_tok.lang_token_to_id, original_special_tokens)
            
    def test_decoding(self):
        text = "Hello, world!"
        encoded = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(encoded)
        self.assertIsInstance(decoded, str)
        self.assertGreater(len(decoded), 0)

    @require_torch 
    def test_src_lang_setter(self):
        self.tokenizer.src_lang = "uk"
        self.assertListEqual(self.tokenizer.prefix_tokens, [self.tokenizer.get_lang_id("uk")])
        self.assertListEqual(self.tokenizer.suffix_tokens, [self.tokenizer.eos_token_id])

        self.tokenizer.src_lang = "zh"
        self.assertListEqual(self.tokenizer.prefix_tokens, [self.tokenizer.get_lang_id("zh")])
        self.assertListEqual(self.tokenizer.suffix_tokens, [self.tokenizer.eos_token_id])

    @require_torch
    def test_tokenizer_target_mode(self):
        self.tokenizer.tgt_lang = "ro"
        self.tokenizer._switch_to_target_mode()
        self.assertListEqual(self.tokenizer.prefix_tokens, [self.tokenizer.get_lang_id("ro")])
        self.assertListEqual(self.tokenizer.suffix_tokens, [self.tokenizer.eos_token_id])
        self.tokenizer._switch_to_input_mode()
        self.assertListEqual(self.tokenizer.prefix_tokens, [self.tokenizer.get_lang_id(self.tokenizer.src_lang)])

        self.tokenizer.tgt_lang = "zh"
        self.tokenizer._switch_to_target_mode()
        self.assertListEqual(self.tokenizer.prefix_tokens, [self.tokenizer.get_lang_id("zh")])
        self.assertListEqual(self.tokenizer.suffix_tokens, [self.tokenizer.eos_token_id])
        self.tokenizer._switch_to_input_mode()
        self.assertListEqual(self.tokenizer.prefix_tokens, [self.tokenizer.get_lang_id(self.tokenizer.src_lang)])

    @require_torch
    def test_tokenizer_translation(self):
        inputs = self.tokenizer._build_translation_inputs("A test", return_tensors="pt", src_lang="en", tgt_lang="ja")

        self.assertEqual(
            nested_simplify(inputs),
            {
                # XX, _A, _test, EOS
                "input_ids": [[37, 77, 3204, 2]],
                "attention_mask": [[1, 1, 1, 1]],
                # ar_AR
                "forced_bos_token_id": 806,
            },
        )
