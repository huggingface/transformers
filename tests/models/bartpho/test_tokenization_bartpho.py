# coding=utf-8
# Copyright 2021 HuggingFace Inc. team.
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

from transformers import BartphoTokenizer, BartphoTokenizerFast, convert_slow_tokenizer
from transformers.models.bartpho.tokenization_bartpho import VOCAB_FILES_NAMES
from transformers.models.bartpho.tokenization_bartpho_fast import VOCAB_FILES_NAMES as VOCAB_FILES_NAMES_F
from transformers.testing_utils import get_tests_dir, require_sentencepiece, require_tokenizers

from ...test_tokenization_common import TokenizerTesterMixin


SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece_bpe.model")


@require_sentencepiece
@require_tokenizers
class BartphoTokenizerTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = BartphoTokenizer
    rust_tokenizer_class = BartphoTokenizerFast
    test_rust_tokenizer = True
    test_sentencepiece = True

    def setUp(self):
        super().setUp()

        vocab = ["▁This", "▁is", "▁a", "▁t", "est"]
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        self.special_tokens_map = {"unk_token": "<unk>"}

        self.monolingual_vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["monolingual_vocab_file"])
        with open(self.monolingual_vocab_file, "w", encoding="utf-8") as fp:
            for token in vocab_tokens:
                fp.write(f"{token} {vocab_tokens[token]}\n")

        tokenizer = BartphoTokenizer(SAMPLE_VOCAB, self.monolingual_vocab_file)
        tokenizer.save_pretrained(self.tmpdirname)

        tokenizer_f = convert_slow_tokenizer.convert_slow_tokenizer(tokenizer)
        self.tokenizer_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES_F["tokenizer_file"])
        tokenizer_f.save(self.tokenizer_file)

    def get_tokenizer(self, **kwargs):
        kwargs.update(self.special_tokens_map)
        return BartphoTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_rust_tokenizer(self, **kwargs):
        kwargs.update(self.special_tokens_map)
        return BartphoTokenizerFast.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self, tokenizer):
        input_text = "This is a test"
        output_text = "This is a test"
        return input_text, output_text

    def test_full_tokenizer(self):
        tokenizer = BartphoTokenizer(SAMPLE_VOCAB, self.monolingual_vocab_file, **self.special_tokens_map)
        text = "This is a test"
        bpe_tokens = "▁This ▁is ▁a ▁t est".split()
        tokens = tokenizer.tokenize(text)
        self.assertListEqual(tokens, bpe_tokens)

        input_tokens = tokens + [tokenizer.unk_token]
        input_bpe_tokens = [4, 5, 6, 7, 8, 3]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)

    def test_rust_and_python_full_tokenizers(self):
        if not self.test_rust_tokenizer:
            return

        tokenizer = self.get_tokenizer()
        rust_tokenizer = self.get_rust_tokenizer()

        sequence = "I was born in 2000."

        tokens = tokenizer.tokenize(sequence)
        rust_tokens = rust_tokenizer.tokenize(sequence)
        self.assertListEqual(tokens, rust_tokens)

        ids = tokenizer.encode(sequence, add_special_tokens=False)
        rust_ids = rust_tokenizer.encode(sequence, add_special_tokens=False)
        self.assertListEqual(ids, rust_ids)

        rust_tokenizer = self.get_rust_tokenizer()
        ids = tokenizer.encode(sequence)
        rust_ids = rust_tokenizer.encode(sequence)
        self.assertListEqual(ids, rust_ids)

    def test_add_tokens_tokenizer(self):
        """
        The tokenizer, which involves an external monolingual vocabulary file and assigns unk_token_id to
        out-of-monolingual-vocabulary tokens, prevents it from adding new tokens that will later be mapped to
        unk_token_id. Thus it will not pass tests that involve `tokenizer.add_tokens(new_toks)` as in the original
        function `test_add_tokens_tokenizer`.
        """

        pass

    def test_encode_decode_with_spaces(self):
        """
        The tokenizer, which involves an external monolingual vocabulary file and assigns unk_token_id to
        out-of-monolingual-vocabulary tokens, prevents it from adding new tokens that will later be mapped to
        unk_token_id. Thus it will not pass tests that involve `tokenizer.add_tokens(new_toks)` as in the
        original function `test_encode_decode_with_spaces`.
        """

        pass

    def test_special_tokens_initialization(self):
        """
        The tokenizer, which involves an external monolingual vocabulary file and assigns unk_token_id to
        out-of-monolingual-vocabulary tokens, prevents it from adding new tokens that will later be mapped
        to unk_token_id. Thus it will not pass tests that involve
        `from_pretrained(pretrained_name, additional_special_tokens=added_tokens)` as in the original function
        `test_special_tokens_initialization`.
        """

        pass

    def test_save_pretrained(self):
        """
        BartphoTokenizer involves an external monolingual vocabulary file. Thus, it would not pass the original test.
        """

        pass

    def test_save_slow_from_fast_and_reload_fast(self):
        """
        BartphoTokenizer involves an external monolingual vocabulary file. Thus, it would not pass the original test.
        """

        pass

    def test_training_new_tokenizer(self):
        """
        The tokenizer, which involves an external monolingual vocabulary file and assigns unk_token_id to
        out-of-monolingual-vocabulary tokens, prevents it from training a new one from scratch.
        """

        pass

    def test_training_new_tokenizer_with_special_tokens_change(self):
        """
        The tokenizer, which involves an external monolingual vocabulary file and assigns unk_token_id to
        out-of-monolingual-vocabulary tokens, prevents it from training a new one from scratch.
        """

        pass
