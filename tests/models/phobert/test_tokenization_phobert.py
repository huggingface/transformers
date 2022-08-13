# coding=utf-8
# Copyright 2018 Salesforce and HuggingFace Inc. team.
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

from transformers import PhobertTokenizer, PhobertTokenizerFast, convert_slow_tokenizer
from transformers.models.phobert.tokenization_phobert import VOCAB_FILES_NAMES
from transformers.models.phobert.tokenization_phobert_fast import VOCAB_FILES_NAMES as VOCAB_FILES_NAMES_F
from transformers.testing_utils import require_tokenizers

from ...test_tokenization_common import TokenizerTesterMixin


@require_tokenizers
class PhobertTokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = PhobertTokenizer
    rust_tokenizer_class = PhobertTokenizerFast
    test_rust_tokenizer = True
    test_slow_tokenizer = True

    def setUp(self):
        super().setUp()

        # Adapted from Sennrich et al. 2015 and https://github.com/rsennrich/subword-nmt
        vocab = ["T@@", "i", "I", "R@@", "r", "e@@"]
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        merges = ["#version: 0.2", "l à</w>"]
        self.special_tokens_map = {"unk_token": "<unk>"}

        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        self.merges_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["merges_file"])

        with open(self.vocab_file, "w", encoding="utf-8") as fp:
            for token in vocab_tokens:
                fp.write(f"{token} {vocab_tokens[token]}\n")
        with open(self.merges_file, "w", encoding="utf-8") as fp:
            fp.write("\n".join(merges))

        tokenizer = PhobertTokenizer(self.vocab_file, self.merges_file)
        tokenizer.save_pretrained(self.tmpdirname)

        tokenizer_f = convert_slow_tokenizer.convert_slow_tokenizer(tokenizer)
        self.tokenizer_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES_F["tokenizer_file"])
        tokenizer_f.save(self.tokenizer_file)

        self._data = "Tôi là VinAI Research"

    def get_tokenizer(self, **kwargs):
        kwargs.update(self.special_tokens_map)
        return PhobertTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_rust_tokenizer(self, **kwargs):
        kwargs.update(self.special_tokens_map)
        return PhobertTokenizerFast.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self, tokenizer):
        input_text = "Tôi là VinAI Research"
        output_text = "T<unk> i <unk> <unk> <unk> <unk> <unk> <unk> I Re<unk> e<unk> <unk> <unk> <unk>"
        if tokenizer.__class__.__name__.endswith("Fast"):
            output_text = "T<unk>i <unk><unk><unk><unk><unk><unk>I Re<unk>e<unk><unk><unk><unk>"
        return input_text, output_text

    def test_full_tokenizer(self):
        tokenizer = PhobertTokenizer(self.vocab_file, self.merges_file, **self.special_tokens_map)
        text = "Tôi là VinAI Research"
        bpe_tokens = "T@@ ô@@ i l@@ à V@@ i@@ n@@ A@@ I R@@ e@@ s@@ e@@ a@@ r@@ c@@ h".split()
        tokens = tokenizer.tokenize(text)
        print(tokens)
        self.assertListEqual(tokens, bpe_tokens)

        input_tokens = tokens + [tokenizer.unk_token]

        input_bpe_tokens = [4, 3, 5, 3, 3, 3, 3, 3, 3, 6, 7, 9, 3, 9, 3, 3, 3, 3, 3]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)

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

    def test_embeded_special_tokens(self):
        """
        Override the original test as the slow & fast tokenizers use different suffix/prefix annotations for
        subword representation, i.e. '@@' used in the slow tokenizer and '</w>' used in the fast tokenizer.
        """

        if not self.test_slow_tokenizer:
            # as we don't have a slow version, we can't compare the outputs between slow and fast versions
            return

        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                tokenizer_p = self.tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                sentence = "This is not a Vietnamese sentence."
                tokens_r = tokenizer_r.encode_plus(
                    sentence,
                    add_special_tokens=True,
                )
                tokens_p = tokenizer_p.encode_plus(
                    sentence,
                    add_special_tokens=True,
                )

                for key in tokens_p.keys():
                    self.assertEqual(tokens_r[key], tokens_p[key])

                if "token_type_ids" in tokens_r:
                    self.assertEqual(sum(tokens_r["token_type_ids"]), sum(tokens_p["token_type_ids"]))

                tokens_r = tokenizer_r.convert_ids_to_tokens(tokens_r["input_ids"])
                tokens_p = tokenizer_p.convert_ids_to_tokens(tokens_p["input_ids"])
                # Map between '@@' used in the slow tokenizer and '</w>' used in the fast tokenizer
                tokens_p = (
                    [tokens_p[0]]
                    + [token[:-2] if token.endswith("@@") else token + "</w>" for token in tokens_p[1:-1]]
                    + [tokens_p[-1]]
                )
                self.assertSequenceEqual(tokens_r, tokens_p)

    def test_training_new_tokenizer(self):
        """
        There are subwords from the merges_file, which do not appear in the vocab_file. Thus, we assign those
        subwords with unk_token_id. As a result, the fast tokenizer is not used for initialization to train a new one.
        """

        pass

    def test_training_new_tokenizer_with_special_tokens_change(self):
        """
        There are subwords from the merges_file, which do not appear in the vocab_file. Thus, we assign those
        subwords with unk_token_id. As a result, the fast tokenizer is not used for initialization to train a new one.
        """

        pass
