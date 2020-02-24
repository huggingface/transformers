# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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


import logging
import unittest

from transformers import (
    BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP,
    AutoTokenizer,
    BertTokenizer,
    BertTokenizerFast,
    GPT2Tokenizer,
    GPT2TokenizerFast,
    RobertaTokenizer,
    RobertaTokenizerFast,
)
from transformers.tokenization_auto import TOKENIZER_MAPPING

from .utils import DUMMY_UNKWOWN_IDENTIFIER, SMALL_MODEL_IDENTIFIER, slow  # noqa: F401


class AutoTokenizerTest(unittest.TestCase):
    # @slow
    def test_tokenizer_from_pretrained(self):
        logging.basicConfig(level=logging.INFO)
        for model_name in (x for x in BERT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys() if "japanese" not in x):
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.assertIsNotNone(tokenizer)
            self.assertIsInstance(tokenizer, (BertTokenizer, BertTokenizerFast))
            self.assertGreater(len(tokenizer), 0)

        for model_name in GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP.keys():
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.assertIsNotNone(tokenizer)
            self.assertIsInstance(tokenizer, (GPT2Tokenizer, GPT2TokenizerFast))
            self.assertGreater(len(tokenizer), 0)

    def test_tokenizer_from_pretrained_identifier(self):
        logging.basicConfig(level=logging.INFO)
        tokenizer = AutoTokenizer.from_pretrained(SMALL_MODEL_IDENTIFIER)
        self.assertIsInstance(tokenizer, (BertTokenizer, BertTokenizerFast))
        self.assertEqual(tokenizer.vocab_size, 12)

    def test_tokenizer_from_model_type(self):
        logging.basicConfig(level=logging.INFO)
        tokenizer = AutoTokenizer.from_pretrained(DUMMY_UNKWOWN_IDENTIFIER)
        self.assertIsInstance(tokenizer, (RobertaTokenizer, RobertaTokenizerFast))
        self.assertEqual(tokenizer.vocab_size, 20)

    def test_tokenizer_identifier_with_correct_config(self):
        logging.basicConfig(level=logging.INFO)
        for tokenizer_class in [BertTokenizer, BertTokenizerFast, AutoTokenizer]:
            tokenizer = tokenizer_class.from_pretrained("wietsedv/bert-base-dutch-cased")
            self.assertIsInstance(tokenizer, (BertTokenizer, BertTokenizerFast))

            if isinstance(tokenizer, BertTokenizer):
                self.assertEqual(tokenizer.basic_tokenizer.do_lower_case, False)
            else:
                self.assertEqual(tokenizer.do_lower_case, False)

            self.assertEqual(tokenizer.max_len, 512)

    def test_tokenizer_identifier_non_existent(self):
        logging.basicConfig(level=logging.INFO)
        for tokenizer_class in [BertTokenizer, BertTokenizerFast, AutoTokenizer]:
            with self.assertRaises(EnvironmentError):
                _ = tokenizer_class.from_pretrained("julien-c/herlolip-not-exists")

    def test_parents_and_children_in_mappings(self):
        # Test that the children are placed before the parents in the mappings, as the `instanceof` will be triggered
        # by the parents and will return the wrong configuration type when using auto models

        mappings = (TOKENIZER_MAPPING,)

        for mapping in mappings:
            mapping = tuple(mapping.items())
            for index, (child_config, (child_model_py, child_model_fast)) in enumerate(mapping[1:]):
                for parent_config, (parent_model_py, parent_model_fast) in mapping[: index + 1]:
                    with self.subTest(
                        msg="Testing if {} is child of {}".format(child_config.__name__, parent_config.__name__)
                    ):
                        self.assertFalse(issubclass(child_config, parent_config))
                        self.assertFalse(issubclass(child_model_py, parent_model_py))

                        # Check for Fast tokenizer implementation if provided
                        if child_model_fast and parent_model_fast:
                            self.assertFalse(issubclass(child_model_fast, parent_model_fast))

    def test_from_pretrained_use_fast_toggle(self):
        self.assertIsInstance(AutoTokenizer.from_pretrained("bert-base-cased"), BertTokenizer)
        self.assertIsInstance(AutoTokenizer.from_pretrained("bert-base-cased", use_fast=True), BertTokenizerFast)
