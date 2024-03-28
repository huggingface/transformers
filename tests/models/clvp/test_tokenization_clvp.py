# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from typing import List

from transformers import ClvpTokenizer

from ...test_tokenization_common import TokenizerTesterMixin, slow


class ClvpTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "susnato/clvp_dev"
    tokenizer_class = ClvpTokenizer
    test_rust_tokenizer = False
    from_pretrained_kwargs = {"add_prefix_space": True}
    test_seq2seq = False
    test_sentencepiece_ignore_case = True

    def setUp(self):
        super().setUp()

        # Adapted from Sennrich et al. 2015 and https://github.com/rsennrich/subword-nmt
        vocab = [
            "l",
            "o",
            "w",
            "e",
            "r",
            "s",
            "t",
            "i",
            "d",
            "n",
            "\u0120",
            "\u0120l",
            "\u0120n",
            "\u0120lo",
            "\u0120low",
            "er",
            "\u0120lowest",
            "\u0120newer",
            "\u0120wider",
            "<unk>",
            "<|endoftext|>",
            "[SPACE]",
        ]
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        merges = ["#version: 0.2", "\u0120 l", "\u0120l o", "\u0120lo w", "e r", ""]
        self.special_tokens_map = {"unk_token": "<unk>"}

        self.vocab_file = os.path.join(self.tmpdirname, "vocab.json")
        self.merges_file = os.path.join(self.tmpdirname, "merges.txt")
        with open(self.vocab_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(vocab_tokens) + "\n")
        with open(self.merges_file, "w", encoding="utf-8") as fp:
            fp.write("\n".join(merges))

    # Copied from transformers.tests.models.gpt2.test_tokenization_gpt2.GPT2TokenizationTest.get_tokenizer with GPT2->Clvp
    def get_tokenizer(self, **kwargs):
        kwargs.update(self.special_tokens_map)
        return ClvpTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    # Copied from transformers.tests.models.gpt2.test_tokenization_gpt2.GPT2TokenizationTest.get_input_output_texts
    def get_input_output_texts(self, tokenizer):
        input_text = "lower newer"
        output_text = "lower newer"
        return input_text, output_text

    # Copied from transformers.tests.models.layoutxlm.test_tokenization_layoutxlm.LayoutXLMTokenizationTest.test_add_special_tokens
    def test_add_special_tokens(self):
        tokenizers: List[ClvpTokenizer] = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                special_token = "[SPECIAL_TOKEN]"
                special_token_box = [1000, 1000, 1000, 1000]

                tokenizer.add_special_tokens({"cls_token": special_token})
                encoded_special_token = tokenizer.encode(
                    [special_token], boxes=[special_token_box], add_special_tokens=False
                )
                self.assertEqual(len(encoded_special_token), 1)

                decoded = tokenizer.decode(encoded_special_token, skip_special_tokens=True)
                self.assertTrue(special_token not in decoded)

    # Copied from transformers.tests.models.gpt2.test_tokenization_gpt2.GPT2TokenizationTest.test_rust_and_python_full_tokenizers
    def test_rust_and_python_full_tokenizers(self):
        if not self.test_rust_tokenizer:
            return

        tokenizer = self.get_tokenizer()
        rust_tokenizer = self.get_rust_tokenizer(add_prefix_space=True)

        sequence = "lower newer"

        # Testing tokenization
        tokens = tokenizer.tokenize(sequence, add_prefix_space=True)
        rust_tokens = rust_tokenizer.tokenize(sequence)
        self.assertListEqual(tokens, rust_tokens)

        # Testing conversion to ids without special tokens
        ids = tokenizer.encode(sequence, add_special_tokens=False, add_prefix_space=True)
        rust_ids = rust_tokenizer.encode(sequence, add_special_tokens=False)
        self.assertListEqual(ids, rust_ids)

        # Testing conversion to ids with special tokens
        rust_tokenizer = self.get_rust_tokenizer(add_prefix_space=True)
        ids = tokenizer.encode(sequence, add_prefix_space=True)
        rust_ids = rust_tokenizer.encode(sequence)
        self.assertListEqual(ids, rust_ids)

        # Testing the unknown token
        input_tokens = tokens + [rust_tokenizer.unk_token]
        input_bpe_tokens = [14, 15, 10, 9, 3, 2, 15, 19]
        self.assertListEqual(rust_tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)

    # Copied from transformers.tests.models.gpt2.test_tokenization_gpt2.GPT2TokenizationTest.test_padding
    def test_padding(self, max_length=15):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)

                # Simple input
                s = "This is a simple input"
                s2 = ["This is a simple input 1", "This is a simple input 2"]
                p = ("This is a simple input", "This is a pair")
                p2 = [
                    ("This is a simple input 1", "This is a simple input 2"),
                    ("This is a simple pair 1", "This is a simple pair 2"),
                ]

                # Simple input tests
                self.assertRaises(ValueError, tokenizer_r.encode, s, max_length=max_length, padding="max_length")

                # Simple input
                self.assertRaises(ValueError, tokenizer_r.encode_plus, s, max_length=max_length, padding="max_length")

                # Simple input
                self.assertRaises(
                    ValueError,
                    tokenizer_r.batch_encode_plus,
                    s2,
                    max_length=max_length,
                    padding="max_length",
                )

                # Pair input
                self.assertRaises(ValueError, tokenizer_r.encode, p, max_length=max_length, padding="max_length")

                # Pair input
                self.assertRaises(ValueError, tokenizer_r.encode_plus, p, max_length=max_length, padding="max_length")

                # Pair input
                self.assertRaises(
                    ValueError,
                    tokenizer_r.batch_encode_plus,
                    p2,
                    max_length=max_length,
                    padding="max_length",
                )

    # Copied from transformers.tests.models.gpt2.test_tokenization_gpt2.GPT2TokenizationTest.test_padding_if_pad_token_set_slow
    def test_padding_if_pad_token_set_slow(self):
        tokenizer = ClvpTokenizer.from_pretrained(self.tmpdirname, pad_token="<pad>")

        # Simple input
        s = "This is a simple input"
        s2 = ["This is a simple input looooooooong", "This is a simple input"]
        p = ("This is a simple input", "This is a pair")
        p2 = [
            ("This is a simple input loooooong", "This is a simple input"),
            ("This is a simple pair loooooong", "This is a simple pair"),
        ]

        pad_token_id = tokenizer.pad_token_id

        out_s = tokenizer(s, padding="max_length", max_length=30, return_tensors="np")
        out_s2 = tokenizer(s2, padding=True, truncate=True, return_tensors="np")
        out_p = tokenizer(*p, padding="max_length", max_length=60, return_tensors="np")
        out_p2 = tokenizer(p2, padding=True, truncate=True, return_tensors="np")

        # s
        # test single string max_length padding
        self.assertEqual(out_s["input_ids"].shape[-1], 30)
        self.assertTrue(pad_token_id in out_s["input_ids"])
        self.assertTrue(0 in out_s["attention_mask"])

        # s2
        # test automatic padding
        self.assertEqual(out_s2["input_ids"].shape[-1], 33)
        # long slice doesn't have padding
        self.assertFalse(pad_token_id in out_s2["input_ids"][0])
        self.assertFalse(0 in out_s2["attention_mask"][0])
        # short slice does have padding
        self.assertTrue(pad_token_id in out_s2["input_ids"][1])
        self.assertTrue(0 in out_s2["attention_mask"][1])

        # p
        # test single pair max_length padding
        self.assertEqual(out_p["input_ids"].shape[-1], 60)
        self.assertTrue(pad_token_id in out_p["input_ids"])
        self.assertTrue(0 in out_p["attention_mask"])

        # p2
        # test automatic padding pair
        self.assertEqual(out_p2["input_ids"].shape[-1], 52)
        # long slice pair doesn't have padding
        self.assertFalse(pad_token_id in out_p2["input_ids"][0])
        self.assertFalse(0 in out_p2["attention_mask"][0])
        # short slice pair does have padding
        self.assertTrue(pad_token_id in out_p2["input_ids"][1])
        self.assertTrue(0 in out_p2["attention_mask"][1])

    # Copied from transformers.tests.models.gpt2.test_tokenization_gpt2.GPT2TokenizationTest.test_special_tokens_mask_input_pairs_and_bos_token
    def test_special_tokens_mask_input_pairs_and_bos_token(self):
        # TODO: change to self.get_tokenizers() when the fast version is implemented
        tokenizers = [self.get_tokenizer(do_lower_case=False, add_bos_token=True)]
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                sequence_0 = "Encode this."
                sequence_1 = "This one too please."
                encoded_sequence = tokenizer.encode(sequence_0, add_special_tokens=False)
                encoded_sequence += tokenizer.encode(sequence_1, add_special_tokens=False)
                encoded_sequence_dict = tokenizer.encode_plus(
                    sequence_0,
                    sequence_1,
                    add_special_tokens=True,
                    return_special_tokens_mask=True,
                )
                encoded_sequence_w_special = encoded_sequence_dict["input_ids"]
                special_tokens_mask = encoded_sequence_dict["special_tokens_mask"]
                self.assertEqual(len(special_tokens_mask), len(encoded_sequence_w_special))

                filtered_sequence = [
                    (x if not special_tokens_mask[i] else None) for i, x in enumerate(encoded_sequence_w_special)
                ]
                filtered_sequence = [x for x in filtered_sequence if x is not None]
                self.assertEqual(encoded_sequence, filtered_sequence)

    def test_token_type_ids(self):
        tokenizer = self.get_tokenizer()
        seq_0 = "Test this method."

        # We want to have sequence 0 and sequence 1 are tagged
        # respectively with 0 and 1 token_ids
        # (regardless of whether the model use token type ids)
        # We use this assumption in the QA pipeline among other place
        output = tokenizer(seq_0, return_token_type_ids=True, add_special_tokens=True)
        self.assertIn(0, output["token_type_ids"])

    def test_full_tokenizer(self):
        tokenizer = ClvpTokenizer(self.vocab_file, self.merges_file, **self.special_tokens_map)
        text = "lower newer"
        bpe_tokens = ["l", "o", "w", "er", "[SPACE]", "n", "e", "w", "er"]
        tokens = tokenizer.tokenize(text, add_prefix_space=False)
        self.assertListEqual(tokens, bpe_tokens)

        input_tokens = tokens + [tokenizer.unk_token]
        input_bpe_tokens = [0, 1, 2, 15, 21, 9, 3, 2, 15, 19]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)

    @slow
    def test_outputs_with_numbers(self):
        text = "hello and this is an example text and I have $1000. my lucky number is 12345."
        tokenizer = ClvpTokenizer.from_pretrained("susnato/clvp_dev")

        # fmt: off
        EXPECTED_OUTPUT = [62, 84, 28, 2, 53, 2,147, 2, 54, 2, 43, 2, 169, 122, 29, 64, 2, 136, 37, 33, 2, 53, 2, 22,
                           2, 148, 2, 110, 2, 40, 206, 53, 2, 134, 84, 59, 32, 9, 2, 125, 2, 25, 34, 197, 38, 2, 27,
                           231, 15, 44, 2, 54, 2, 33, 100, 25, 76, 2, 40, 206, 53, 7, 2, 40, 46, 18, 2, 21, 97, 17,
                           219, 2, 87, 210, 8, 19, 22, 76, 9,
                           ]
        # fmt: on

        self.assertListEqual(tokenizer.encode(text, add_special_tokens=False), EXPECTED_OUTPUT)

    @slow
    def test_tokenizer_integration(self):
        sequences = [
            "Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides "
            "general-purpose architectures (BERT, RoBERTa, XLM, DistilBert, XLNet...) for Natural "
            "Language Understanding (NLU) and Natural Language Generation (NLG) with over multiple pretrained "
            "models and deep interoperability between Jax, PyTorch and TensorFlow.",
            "BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly "
            "conditioning on both left and right context in all layers.",
            "The quick brown fox jumps over the lazy dog.",
        ]

        # fmt: off
        expected_encoding = {'input_ids': [[144,  43,  32,  87,  26, 173,   2,   5,  87,  26,  44,  70,   2, 209, 27,   2,  55,   2,  29,  38,  51,  31,  71,   8, 144,  43,  32,  87, 26, 173,   2,  53,   2,  29,  38,  51,  31,  71,   8,  29,  46, 144, 137,  49,   8,  15,  44,  33,   6,   2, 187,  35,  83,  61,   2,  20, 50,  44,  56,   8,  29, 121, 139,  66,   2,  59,  71,  60,  18,  16, 33,  34, 175,   2,   5,  15,  44,  33,   7,   2,  89,  15,  44,  33, 14,   7,   2,  37,  25,  26,   7,   2,  17,  54,  78,  25,  15,  44, 33,   7,   2,  37,  25, 111,  33,   9,   9,   9,   6,   2,  87,   2, 27,  48, 121,  56,   2,  25,  43,  20,  34,  14, 112,   2,  97, 234, 63,  53,  52,   2,   5,  27,  25,  34,   6,   2,  53,   2,  27,  48, 121,  56,   2,  25,  43,  20,  34,  14, 112,   2,  20,  50,  44, 158, 2,   5,  27,  25,  20,   6,   2, 103,   2, 253,   2,  26, 167,  78, 29,  64,   2,  29,  46, 144, 137,  49,   2, 115, 126,  25,  32,   2, 53,   2, 126,  18,  29,   2,  41, 114, 161,  44, 109, 151, 240,   2, 67,  33, 100,  50,   2,  23,  14,  37,   7,   2,  29,  38,  51,  31, 71,   2,  53,   2,  33,  50,  32,  57,  19,  25,  69,   9], [ 15,  44,  33,   2,  54,   2,  17,  61,  22,  20,  27,  49,   2,  51, 2,  29,  46,   8, 144, 137,   2, 126,  18,  29,   2,  15,  83,  22, 46,  16, 181,  56,   2,  46,  29, 175,  86, 158,  32,   2, 154,   2, 97,  25,  14,  67,  25,  49,   2, 136,  37,  33,   2, 185,   2,  23, 28,  41,  33,  70,   2, 135,  17,  60, 107,  52,   2,  47,   2, 165, 40,   2,  64,  19,  33,   2,  53,   2, 101, 104,   2, 135, 136,  37, 33,   2,  41,   2, 108,   2,  25,  88, 173,   9,   0,   0,   0,   0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0], [ 42,   2, 194,  91,  24,   2, 243, 190,   2, 182,  37,   2,  23, 231, 29,  32,   2, 253,   2,  42,   2,  25,  14,  39,  38,   2, 134,  20, 9,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]], # noqa: E501
                             'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], # noqa: E501
                             }
        # fmt: on

        self.tokenizer_integration_test_util(
            sequences=sequences, expected_encoding=expected_encoding, model_name="susnato/clvp_dev", padding=True
        )
