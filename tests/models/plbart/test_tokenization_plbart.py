# Copyright 2022 The HuggingFace Team. All rights reserved.
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

from transformers import SPIECE_UNDERLINE, BatchEncoding, PLBartTokenizer, is_torch_available
from transformers.testing_utils import (
    get_tests_dir,
    nested_simplify,
    require_sentencepiece,
    require_tokenizers,
    require_torch,
)

from ...test_tokenization_common import TokenizerTesterMixin


SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")


if is_torch_available():
    from transformers.models.plbart.modeling_plbart import shift_tokens_right

EN_CODE = 50003
PYTHON_CODE = 50002


@require_sentencepiece
@require_tokenizers
class PLBartTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = PLBartTokenizer
    rust_tokenizer_class = None
    test_rust_tokenizer = False

    def setUp(self):
        super().setUp()

        # We have a SentencePiece fixture for testing
        tokenizer = PLBartTokenizer(SAMPLE_VOCAB, language_codes="base", keep_accents=True)
        tokenizer.save_pretrained(self.tmpdirname)

    def test_full_base_tokenizer(self):
        tokenizer = PLBartTokenizer(SAMPLE_VOCAB, language_codes="base", keep_accents=True)

        tokens = tokenizer.tokenize("This is a test")
        self.assertListEqual(tokens, ["▁This", "▁is", "▁a", "▁t", "est"])

        self.assertListEqual(
            tokenizer.convert_tokens_to_ids(tokens),
            [value + tokenizer.fairseq_offset for value in [285, 46, 10, 170, 382]],
        )

        tokens = tokenizer.tokenize("I was born in 92000, and this is falsé.")
        self.assertListEqual(
            tokens,
            [
                SPIECE_UNDERLINE + "I",
                SPIECE_UNDERLINE + "was",
                SPIECE_UNDERLINE + "b",
                "or",
                "n",
                SPIECE_UNDERLINE + "in",
                SPIECE_UNDERLINE + "",
                "9",
                "2",
                "0",
                "0",
                "0",
                ",",
                SPIECE_UNDERLINE + "and",
                SPIECE_UNDERLINE + "this",
                SPIECE_UNDERLINE + "is",
                SPIECE_UNDERLINE + "f",
                "al",
                "s",
                "é",
                ".",
            ],
        )
        ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertListEqual(
            ids,
            [
                value + tokenizer.fairseq_offset
                for value in [8, 21, 84, 55, 24, 19, 7, 2, 602, 347, 347, 347, 3, 12, 66, 46, 72, 80, 6, 2, 4]
            ],
        )

        back_tokens = tokenizer.convert_ids_to_tokens(ids)
        self.assertListEqual(
            back_tokens,
            [
                SPIECE_UNDERLINE + "I",
                SPIECE_UNDERLINE + "was",
                SPIECE_UNDERLINE + "b",
                "or",
                "n",
                SPIECE_UNDERLINE + "in",
                SPIECE_UNDERLINE + "",
                "<unk>",
                "2",
                "0",
                "0",
                "0",
                ",",
                SPIECE_UNDERLINE + "and",
                SPIECE_UNDERLINE + "this",
                SPIECE_UNDERLINE + "is",
                SPIECE_UNDERLINE + "f",
                "al",
                "s",
                "<unk>",
                ".",
            ],
        )

        end = tokenizer.vocab_size
        language_tokens = [tokenizer.convert_ids_to_tokens(x) for x in range(end - 4, end)]

        self.assertListEqual(language_tokens, ["__java__", "__python__", "__en_XX__", "<mask>"])

        code = "java.lang.Exception, python.lang.Exception, javascript, php, ruby, go"
        input_ids = tokenizer(code).input_ids
        self.assertEqual(
            tokenizer.decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False),
            code,
        )

    def test_full_multi_tokenizer(self):
        tokenizer = PLBartTokenizer(SAMPLE_VOCAB, language_codes="multi", keep_accents=True)

        tokens = tokenizer.tokenize("This is a test")
        self.assertListEqual(tokens, ["▁This", "▁is", "▁a", "▁t", "est"])

        self.assertListEqual(
            tokenizer.convert_tokens_to_ids(tokens),
            [value + tokenizer.fairseq_offset for value in [285, 46, 10, 170, 382]],
        )

        tokens = tokenizer.tokenize("I was born in 92000, and this is falsé.")
        self.assertListEqual(
            tokens,
            [
                SPIECE_UNDERLINE + "I",
                SPIECE_UNDERLINE + "was",
                SPIECE_UNDERLINE + "b",
                "or",
                "n",
                SPIECE_UNDERLINE + "in",
                SPIECE_UNDERLINE + "",
                "9",
                "2",
                "0",
                "0",
                "0",
                ",",
                SPIECE_UNDERLINE + "and",
                SPIECE_UNDERLINE + "this",
                SPIECE_UNDERLINE + "is",
                SPIECE_UNDERLINE + "f",
                "al",
                "s",
                "é",
                ".",
            ],
        )
        ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertListEqual(
            ids,
            [
                value + tokenizer.fairseq_offset
                for value in [8, 21, 84, 55, 24, 19, 7, 2, 602, 347, 347, 347, 3, 12, 66, 46, 72, 80, 6, 2, 4]
            ],
        )

        back_tokens = tokenizer.convert_ids_to_tokens(ids)
        self.assertListEqual(
            back_tokens,
            [
                SPIECE_UNDERLINE + "I",
                SPIECE_UNDERLINE + "was",
                SPIECE_UNDERLINE + "b",
                "or",
                "n",
                SPIECE_UNDERLINE + "in",
                SPIECE_UNDERLINE + "",
                "<unk>",
                "2",
                "0",
                "0",
                "0",
                ",",
                SPIECE_UNDERLINE + "and",
                SPIECE_UNDERLINE + "this",
                SPIECE_UNDERLINE + "is",
                SPIECE_UNDERLINE + "f",
                "al",
                "s",
                "<unk>",
                ".",
            ],
        )
        end = tokenizer.vocab_size
        language_tokens = [tokenizer.convert_ids_to_tokens(x) for x in range(end - 7, end)]

        self.assertListEqual(
            language_tokens, ["__java__", "__python__", "__en_XX__", "__javascript__", "__php__", "__ruby__", "__go__"]
        )
        code = "java.lang.Exception, python.lang.Exception, javascript, php, ruby, go"
        input_ids = tokenizer(code).input_ids
        self.assertEqual(
            tokenizer.decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False),
            code,
        )


@require_torch
@require_sentencepiece
@require_tokenizers
class PLBartPythonEnIntegrationTest(unittest.TestCase):
    checkpoint_name = "uclanlp/plbart-python-en_XX"
    src_text = [
        "def maximum(a,b,c):NEW_LINE_INDENTreturn max([a,b,c])",
        "def sum(a,b,c):NEW_LINE_INDENTreturn sum([a,b,c])",
    ]
    tgt_text = [
        "Returns the maximum value of a b c.",
        "Sums the values of a b c.",
    ]
    expected_src_tokens = [
        134,
        5452,
        33460,
        33441,
        33463,
        33465,
        33463,
        33449,
        988,
        20,
        33456,
        19,
        33456,
        771,
        39,
        4258,
        889,
        3318,
        33441,
        33463,
        33465,
        33463,
        33449,
        2471,
        2,
        PYTHON_CODE,
    ]

    @classmethod
    def setUpClass(cls):
        cls.tokenizer: PLBartTokenizer = PLBartTokenizer.from_pretrained(
            cls.checkpoint_name, language_codes="base", src_lang="python", tgt_lang="en_XX"
        )
        cls.pad_token_id = 1
        return cls

    def check_language_codes(self):
        self.assertEqual(self.tokenizer.fairseq_tokens_to_ids["__java__"], 50001)
        self.assertEqual(self.tokenizer.fairseq_tokens_to_ids["__python__"], 50002)
        self.assertEqual(self.tokenizer.fairseq_tokens_to_ids["__en_XX__"], 50003)

    def test_python_en_tokenizer_batch_encode_plus(self):
        ids = self.tokenizer.batch_encode_plus(self.src_text).input_ids[0]
        self.assertListEqual(self.expected_src_tokens, ids)

    def test_python_en_tokenizer_decode_ignores_language_codes(self):
        self.assertIn(PYTHON_CODE, self.tokenizer.all_special_ids)
        generated_ids = [EN_CODE, 9037, 33442, 57, 752, 153, 14, 56, 18, 9, 2]
        result = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        expected_english = self.tokenizer.decode(generated_ids[1:], skip_special_tokens=True)
        self.assertEqual(result, expected_english)
        self.assertNotIn(self.tokenizer.eos_token, result)

    def test_python_en_tokenizer_truncation(self):
        src_text = ["def sum(a,b,c):NEW_LINE_INDENTreturn sum([a,b,c])" * 20]
        self.assertIsInstance(src_text[0], str)
        desired_max_length = 10
        ids = self.tokenizer(src_text, max_length=desired_max_length, truncation=True).input_ids[0]
        self.assertEqual(ids[-2], 2)
        self.assertEqual(ids[-1], PYTHON_CODE)
        self.assertEqual(len(ids), desired_max_length)

    def test_mask_token(self):
        self.assertListEqual(self.tokenizer.convert_tokens_to_ids(["<mask>", "__java__"]), [50004, 50001])

    def test_special_tokens_unaffacted_by_save_load(self):
        tmpdirname = tempfile.mkdtemp()
        original_special_tokens = self.tokenizer.fairseq_tokens_to_ids
        self.tokenizer.save_pretrained(tmpdirname)
        new_tok = PLBartTokenizer.from_pretrained(tmpdirname)
        self.assertDictEqual(new_tok.fairseq_tokens_to_ids, original_special_tokens)

    @require_torch
    def test_batch_fairseq_parity(self):
        batch = self.tokenizer(self.src_text, text_target=self.tgt_text, padding=True, return_tensors="pt")
        batch["decoder_input_ids"] = shift_tokens_right(batch["labels"], self.tokenizer.pad_token_id)

        # fairseq batch: https://gist.github.com/sshleifer/cba08bc2109361a74ac3760a7e30e4f4
        self.assertEqual(batch.input_ids[1][-2:].tolist(), [2, PYTHON_CODE])
        self.assertEqual(batch.decoder_input_ids[1][0], EN_CODE)
        self.assertEqual(batch.decoder_input_ids[1][-1], 2)
        self.assertEqual(batch.labels[1][-2:].tolist(), [2, EN_CODE])

    @require_torch
    def test_python_en_tokenizer_prepare_batch(self):
        batch = self.tokenizer(
            self.src_text,
            text_target=self.tgt_text,
            padding=True,
            truncation=True,
            max_length=len(self.expected_src_tokens),
            return_tensors="pt",
        )
        batch["decoder_input_ids"] = shift_tokens_right(batch["labels"], self.tokenizer.pad_token_id)

        self.assertIsInstance(batch, BatchEncoding)

        self.assertEqual((2, 26), batch.input_ids.shape)
        self.assertEqual((2, 26), batch.attention_mask.shape)
        result = batch.input_ids.tolist()[0]
        self.assertListEqual(self.expected_src_tokens, result)
        self.assertEqual(2, batch.decoder_input_ids[0, -1])  # EOS
        # Test that special tokens are reset
        self.assertEqual(self.tokenizer.prefix_tokens, [])
        self.assertEqual(self.tokenizer.suffix_tokens, [self.tokenizer.eos_token_id, PYTHON_CODE])

    def test_seq2seq_max_length(self):
        batch = self.tokenizer(self.src_text, padding=True, truncation=True, max_length=3, return_tensors="pt")
        targets = self.tokenizer(
            text_target=self.tgt_text, padding=True, truncation=True, max_length=10, return_tensors="pt"
        )
        labels = targets["input_ids"]
        batch["decoder_input_ids"] = shift_tokens_right(labels, self.tokenizer.pad_token_id)

        self.assertEqual(batch.input_ids.shape[1], 3)
        self.assertEqual(batch.decoder_input_ids.shape[1], 10)

    @require_torch
    def test_tokenizer_translation(self):
        inputs = self.tokenizer._build_translation_inputs(
            "A test", return_tensors="pt", src_lang="en_XX", tgt_lang="java"
        )

        self.assertEqual(
            nested_simplify(inputs),
            {
                # A, test, EOS, en_XX
                "input_ids": [[150, 242, 2, 50003]],
                "attention_mask": [[1, 1, 1, 1]],
                # java
                "forced_bos_token_id": 50001,
            },
        )
