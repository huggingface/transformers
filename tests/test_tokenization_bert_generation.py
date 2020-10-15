# coding=utf-8
# Copyright 2020 The Google AI Language Team Authors.
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


import os
import unittest

from transformers import BertGenerationTokenizer
from transformers.file_utils import cached_property
from transformers.testing_utils import require_sentencepiece, require_torch, slow

from .test_tokenization_common import TokenizerTesterMixin


SPIECE_UNDERLINE = "▁"

SAMPLE_VOCAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures/test_sentencepiece.model")


@require_sentencepiece
class BertGenerationTokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = BertGenerationTokenizer

    def setUp(self):
        super().setUp()

        tokenizer = BertGenerationTokenizer(SAMPLE_VOCAB, keep_accents=True)
        tokenizer.save_pretrained(self.tmpdirname)

    def test_full_tokenizer(self):
        tokenizer = BertGenerationTokenizer(SAMPLE_VOCAB, keep_accents=True)

        tokens = tokenizer.tokenize("This is a test")
        self.assertListEqual(tokens, ["▁This", "▁is", "▁a", "▁t", "est"])

        self.assertListEqual(
            tokenizer.convert_tokens_to_ids(tokens),
            [285, 46, 10, 170, 382],
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
            [8, 21, 84, 55, 24, 19, 7, 0, 602, 347, 347, 347, 3, 12, 66, 46, 72, 80, 6, 0, 4],
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

    @cached_property
    def big_tokenizer(self):
        return BertGenerationTokenizer.from_pretrained("google/bert_for_seq_generation_L-24_bbc_encoder")

    @slow
    def test_tokenization_base_easy_symbols(self):
        symbols = "Hello World!"
        original_tokenizer_encodings = [18536, 2260, 101]

        self.assertListEqual(original_tokenizer_encodings, self.big_tokenizer.encode(symbols))

    @slow
    def test_tokenization_base_hard_symbols(self):
        symbols = 'This is a very long text with a lot of weird characters, such as: . , ~ ? ( ) " [ ] ! : - . Also we will add words that should not exsist and be tokenized to <unk>, such as saoneuhaoesuth'
        original_tokenizer_encodings = [
            871,
            419,
            358,
            946,
            991,
            2521,
            452,
            358,
            1357,
            387,
            7751,
            3536,
            112,
            985,
            456,
            126,
            865,
            938,
            5400,
            5734,
            458,
            1368,
            467,
            786,
            2462,
            5246,
            1159,
            633,
            865,
            4519,
            457,
            582,
            852,
            2557,
            427,
            916,
            508,
            405,
            34324,
            497,
            391,
            408,
            11342,
            1244,
            385,
            100,
            938,
            985,
            456,
            574,
            362,
            12597,
            3200,
            3129,
            1172,
        ]

        self.assertListEqual(original_tokenizer_encodings, self.big_tokenizer.encode(symbols))

    @require_torch
    @slow
    def test_torch_encode_plus_sent_to_model(self):
        import torch

        from transformers import BertGenerationConfig, BertGenerationEncoder

        # Build sequence
        first_ten_tokens = list(self.big_tokenizer.get_vocab().keys())[:10]
        sequence = " ".join(first_ten_tokens)
        encoded_sequence = self.big_tokenizer.encode_plus(sequence, return_tensors="pt", return_token_type_ids=False)
        batch_encoded_sequence = self.big_tokenizer.batch_encode_plus(
            [sequence + " " + sequence], return_tensors="pt", return_token_type_ids=False
        )

        config = BertGenerationConfig()
        model = BertGenerationEncoder(config)

        assert model.get_input_embeddings().weight.shape[0] >= self.big_tokenizer.vocab_size

        with torch.no_grad():
            model(**encoded_sequence)
            model(**batch_encoded_sequence)
