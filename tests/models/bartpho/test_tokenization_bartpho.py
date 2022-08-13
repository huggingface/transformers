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

import inspect
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

        self._data = "Chúng tôi là những nghiên cứu viên."

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

    def test_tokenizer_fast_store_full_signature(self):
        """
        Override the original test as BartphoTokenizer requires a monolingual_vocab_file rather than a merges_file
        """

        if not self.test_rust_tokenizer:
            return

        signature = inspect.signature(self.rust_tokenizer_class.__init__)
        tokenizer = self.get_rust_tokenizer()

        for parameter_name, parameter in signature.parameters.items():
            if parameter.default != inspect.Parameter.empty and parameter_name not in [
                "vocab_file",
                "monolingual_vocab_file",  # "merges_file",
                "tokenizer_file",
            ]:
                self.assertIn(parameter_name, tokenizer.init_kwargs)

    def test_add_tokens_tokenizer(self):
        """
        Override the original test as in the fast tokenizer, the actual vocab_size is in fact mask_token_id + 1
        """

        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                vocab_size = tokenizer.vocab_size
                all_size = len(tokenizer)

                self.assertNotEqual(vocab_size, 0)

                # We usually have added tokens from the start in tests because our vocab fixtures are
                # smaller than the original vocabs - let's not assert this
                # self.assertEqual(vocab_size, all_size)

                new_toks = ["aaaaa bbbbbb", "cccccccccdddddddd"]
                added_toks = tokenizer.add_tokens(new_toks)
                vocab_size_2 = tokenizer.vocab_size
                all_size_2 = len(tokenizer)

                self.assertNotEqual(vocab_size_2, 0)
                self.assertEqual(vocab_size, vocab_size_2)
                self.assertEqual(added_toks, len(new_toks))
                self.assertEqual(all_size_2, all_size + len(new_toks))

                tokens = tokenizer.encode("aaaaa bbbbbb low cccccccccdddddddd l", add_special_tokens=False)

                self.assertGreaterEqual(len(tokens), 4)
                if tokenizer.__class__.__name__.endswith("Fast"):
                    self.assertGreater(tokens[0], tokenizer.mask_token_id)
                    self.assertGreater(tokens[-2], tokenizer.mask_token_id)
                else:
                    self.assertGreater(tokens[0], tokenizer.vocab_size - 1)
                    self.assertGreater(tokens[-2], tokenizer.vocab_size - 1)

                new_toks_2 = {"eos_token": ">>>>|||<||<<|<<", "pad_token": "<<<<<|||>|>>>>|>"}
                added_toks_2 = tokenizer.add_special_tokens(new_toks_2)
                vocab_size_3 = tokenizer.vocab_size
                all_size_3 = len(tokenizer)

                self.assertNotEqual(vocab_size_3, 0)
                self.assertEqual(vocab_size, vocab_size_3)
                self.assertEqual(added_toks_2, len(new_toks_2))
                self.assertEqual(all_size_3, all_size_2 + len(new_toks_2))

                tokens = tokenizer.encode(
                    ">>>>|||<||<<|<< aaaaabbbbbb low cccccccccdddddddd <<<<<|||>|>>>>|> l", add_special_tokens=False
                )

                self.assertGreaterEqual(len(tokens), 6)
                if tokenizer.__class__.__name__.endswith("Fast"):
                    self.assertGreater(tokens[0], tokenizer.mask_token_id)
                else:
                    self.assertGreater(tokens[0], tokenizer.vocab_size - 1)
                self.assertGreater(tokens[0], tokens[1])
                if tokenizer.__class__.__name__.endswith("Fast"):
                    self.assertGreater(tokens[-2], tokenizer.mask_token_id)
                else:
                    self.assertGreater(tokens[-2], tokenizer.vocab_size - 1)
                self.assertGreater(tokens[-2], tokens[-3])

                if tokenizer.__class__.__name__.endswith("Fast"):
                    _, id_mapping = tokenizer.get_added_vocab_hacking()
                    self.assertEqual(id_mapping[tokens[0]], tokenizer.eos_token_id)
                    self.assertEqual(id_mapping[tokens[-2]], tokenizer.pad_token_id)
                else:
                    self.assertEqual(tokens[0], tokenizer.eos_token_id)
                    self.assertEqual(tokens[-2], tokenizer.pad_token_id)

    def test_training_new_tokenizer(self):
        """
        The tokenizer involves an external monolingual vocabulary file and assigns unk_token_id to
        out-of-monolingual-vocabulary tokens, thus is not used for initialization to train a new one.
        """

        pass

    def test_training_new_tokenizer_with_special_tokens_change(self):
        """
        The tokenizer involves an external monolingual vocabulary file and assigns unk_token_id to
        out-of-monolingual-vocabulary tokens, thus is not used for initialization to train a new one.
        """

        pass
