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
import shutil
import tempfile
import unittest

from torch import nn

from transformers import BertweetTokenizer, BertweetTokenizerFast, Trainer, TrainingArguments, convert_slow_tokenizer
from transformers.models.bertweet.tokenization_bertweet import VOCAB_FILES_NAMES
from transformers.models.bertweet.tokenization_bertweet_fast import VOCAB_FILES_NAMES as VOCAB_FILES_NAMES_F
from transformers.testing_utils import require_tokenizers
from transformers.tokenization_utils import AddedToken

from ...test_tokenization_common import TokenizerTesterMixin


@require_tokenizers
class BertweetTokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = BertweetTokenizer
    rust_tokenizer_class = BertweetTokenizerFast
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

        tokenizer = BertweetTokenizer.from_pretrained(self.tmpdirname)
        tokenizer_f = convert_slow_tokenizer.convert_slow_tokenizer(tokenizer)
        self.tokenizer_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES_F["tokenizer_file"])
        tokenizer_f.save(self.tokenizer_file)

        self._data = "Tôi là VinAI Research"

    def get_tokenizer(self, **kwargs):
        kwargs.update(self.special_tokens_map)
        return BertweetTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_rust_tokenizer(self, **kwargs):
        kwargs.update(self.special_tokens_map)
        return BertweetTokenizerFast.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self, tokenizer):
        input_text = "Tôi là VinAI Research"
        output_text = "T<unk> i <unk> <unk> <unk> <unk> <unk> <unk> I Re<unk> e<unk> <unk> <unk> <unk>"
        if hasattr(tokenizer, "tokenizer_file"):
            output_text = "T<unk>i <unk><unk><unk><unk><unk><unk>I Re<unk>e<unk><unk><unk><unk>"
        return input_text, output_text

    def test_full_tokenizer(self):
        tokenizer = BertweetTokenizer(self.vocab_file, self.merges_file, **self.special_tokens_map)
        text = "Tôi là VinAI Research"
        bpe_tokens = "T@@ ô@@ i l@@ à V@@ i@@ n@@ A@@ I R@@ e@@ s@@ e@@ a@@ r@@ c@@ h".split()
        tokens = tokenizer.tokenize(text)
        print(tokens)
        self.assertListEqual(tokens, bpe_tokens)

        input_tokens = tokens + [tokenizer.unk_token]

        input_bpe_tokens = [4, 3, 5, 3, 3, 3, 3, 3, 3, 6, 7, 9, 3, 9, 3, 3, 3, 3, 3]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)

    def test_rust_and_python_full_tokenizers(self):
        if not self.test_rust_tokenizer:
            return

        tokenizer = self.get_tokenizer()
        rust_tokenizer = self.get_rust_tokenizer()

        sequence = "Tôi là VinAI"

        ids = tokenizer.encode(sequence, add_special_tokens=False)
        rust_ids = rust_tokenizer.encode(sequence, add_special_tokens=False)
        self.assertListEqual(ids, rust_ids)

        rust_tokenizer = self.get_rust_tokenizer()
        ids = tokenizer.encode(sequence)
        rust_ids = rust_tokenizer.encode(sequence)
        self.assertListEqual(ids, rust_ids)

    def test_add_tokens_tokenizer(self):
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
                    self.assertEqual(tokens[0], tokenizer.unk_token_id)
                    self.assertEqual(tokens[-2], tokenizer.unk_token_id)
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

    def test_embeded_special_tokens(self):
        if not self.test_slow_tokenizer:
            # as we don't have a slow version, we can't compare the outputs between slow and fast versions
            return

        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                tokenizer_p = self.tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                sentence = "A Vietnamese sentence."
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
                tokens_p = (
                    [tokens_p[0]]
                    + [token[:-2] if token.endswith("@@") else token + "</w>" for token in tokens_p[1:-1]]
                    + [tokens_p[-1]]
                )
                self.assertSequenceEqual(tokens_r, tokens_p)

    def test_encode_decode_with_spaces(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):

                input = "Tôi là VinAI Research"
                output = "T<unk> i <unk> <unk> <unk> <unk> <unk> <unk> I Re<unk> e<unk> <unk> <unk> <unk>"
                if hasattr(tokenizer, "tokenizer_file"):
                    output = "T<unk>i <unk><unk><unk><unk><unk><unk>I Re<unk>e<unk><unk><unk><unk>"
                encoded = tokenizer.encode(input, add_special_tokens=False)
                decoded = tokenizer.decode(encoded, spaces_between_special_tokens=self.space_between_special_tokens)
                self.assertIn(decoded, [output, output.lower()])

    def test_save_pretrained(self):
        if not self.test_slow_tokenizer:
            # as we don't have a slow version, we can't compare the outputs between slow and fast versions
            return

        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                tokenizer_p = self.tokenizer_class.from_pretrained(pretrained_name, **kwargs)

                tmpdirname2 = tempfile.mkdtemp()

                tokenizer_r_files = tokenizer_r.save_pretrained(tmpdirname2)
                tokenizer_p_files = tokenizer_p.save_pretrained(tmpdirname2)

                # Checks it save with the same files + the tokenizer.json file for the fast one
                self.assertTrue(any("tokenizer.json" in f for f in tokenizer_r_files))
                tokenizer_r_files = tuple(f for f in tokenizer_r_files if "tokenizer.json" not in f)
                self.assertSequenceEqual(tokenizer_r_files, tokenizer_p_files)

                # Checks everything loads correctly in the same way
                tokenizer_rp = tokenizer_r.from_pretrained(tmpdirname2)
                tokenizer_pp = tokenizer_p.from_pretrained(tmpdirname2)

                # Check special tokens are set accordingly on Rust and Python
                for key in tokenizer_pp.special_tokens_map:
                    self.assertTrue(hasattr(tokenizer_rp, key))
                    # self.assertEqual(getattr(tokenizer_rp, key), getattr(tokenizer_pp, key))
                    # self.assertEqual(getattr(tokenizer_rp, key + "_id"), getattr(tokenizer_pp, key + "_id"))

                shutil.rmtree(tmpdirname2)

                # Save tokenizer rust, legacy_format=False
                tmpdirname2 = tempfile.mkdtemp()

                tokenizer_r_files = tokenizer_r.save_pretrained(tmpdirname2, legacy_format=False)
                tokenizer_p_files = tokenizer_p.save_pretrained(tmpdirname2)

                # Checks it saved the tokenizer.json file
                self.assertTrue(any("tokenizer.json" in f for f in tokenizer_r_files))

                # Checks everything loads correctly in the same way
                tokenizer_rp = tokenizer_r.from_pretrained(tmpdirname2)
                tokenizer_pp = tokenizer_p.from_pretrained(tmpdirname2)

                # Check special tokens are set accordingly on Rust and Python
                for key in tokenizer_pp.special_tokens_map:
                    self.assertTrue(hasattr(tokenizer_rp, key))

                shutil.rmtree(tmpdirname2)

    def test_save_slow_from_fast_and_reload_fast(self):
        if not self.test_slow_tokenizer or not self.test_rust_tokenizer:
            # we need both slow and fast versions
            return

        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                with tempfile.TemporaryDirectory() as tmp_dir_1:
                    # Here we check that even if we have initialized a fast tokenizer with a tokenizer_file we can
                    # still save only the slow version and use these saved files to rebuild a tokenizer
                    tokenizer_fast_old_1 = self.rust_tokenizer_class.from_pretrained(
                        pretrained_name, **kwargs, use_fast=True
                    )
                    tokenizer_file = os.path.join(tmp_dir_1, "tokenizer.json")
                    tokenizer_fast_old_1.backend_tokenizer.save(tokenizer_file)

                    tokenizer_fast_old_2 = self.rust_tokenizer_class.from_pretrained(
                        pretrained_name, **kwargs, use_fast=True, tokenizer_file=tokenizer_file
                    )

                    tokenizer_fast_old_2.save_pretrained(tmp_dir_1, legacy_format=True)  # save only slow version

                    self.tokenizer_class.from_pretrained(tmp_dir_1)

    def test_saving_tokenizer_trainer(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    # Save the fast tokenizer files in a temporary directory
                    tokenizer_old = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs, use_fast=True)
                    tokenizer_old.save_pretrained(tmp_dir, legacy_format=True)  # save only fast version

                    # Initialize toy model for the trainer
                    model = nn.Module()

                    # Load tokenizer from a folder without legacy files
                    tokenizer = self.rust_tokenizer_class.from_pretrained(tmp_dir)
                    training_args = TrainingArguments(output_dir=tmp_dir, do_train=True, no_cuda=True)
                    trainer = Trainer(model=model, args=training_args, tokenizer=tokenizer)

                    # Should not raise an error
                    trainer.save_model(os.path.join(tmp_dir, "checkpoint"))
                    self.assertIn("tokenizer.json", os.listdir(os.path.join(tmp_dir, "checkpoint")))

    def test_special_tokens_initialization(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):

                added_tokens = [AddedToken("<special>", lstrip=True)]

                tokenizer_r = self.rust_tokenizer_class.from_pretrained(
                    pretrained_name, additional_special_tokens=added_tokens, **kwargs
                )
                r_output = tokenizer_r.encode("Hey this is a <special> token")

                special_token_id = tokenizer_r.encode("<special>", add_special_tokens=False)[0]

                self.assertTrue(special_token_id in r_output)

    def test_training_new_tokenizer(self):
        """
        The tokenizer's files, from fastBPE's outputs, prevent the tokenizer from training a new one in `tokenizers`.
        """

        pass

    def test_training_new_tokenizer_with_special_tokens_change(self):
        """
        The tokenizer's files, from fastBPE's outputs, prevent the tokenizer from training a new one in `tokenizers`.
        """

        pass
