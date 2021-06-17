# coding=utf-8
# Copyright 2020 The HuggingFace Team. All rights reserved.
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

from transformers import (
    BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP,
    AutoTokenizer,
    BertTokenizer,
    BertTokenizerFast,
    GPT2Tokenizer,
    GPT2TokenizerFast,
    PreTrainedTokenizerFast,
    RobertaTokenizer,
    RobertaTokenizerFast,
)
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING, get_tokenizer_config
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.testing_utils import (
    DUMMY_DIFF_TOKENIZER_IDENTIFIER,
    DUMMY_UNKWOWN_IDENTIFIER,
    SMALL_MODEL_IDENTIFIER,
    require_tokenizers,
    slow,
)


class AutoTokenizerTest(unittest.TestCase):
    @slow
    def test_tokenizer_from_pretrained(self):
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
        tokenizer = AutoTokenizer.from_pretrained(SMALL_MODEL_IDENTIFIER)
        self.assertIsInstance(tokenizer, (BertTokenizer, BertTokenizerFast))
        self.assertEqual(tokenizer.vocab_size, 12)

    def test_tokenizer_from_model_type(self):
        tokenizer = AutoTokenizer.from_pretrained(DUMMY_UNKWOWN_IDENTIFIER)
        self.assertIsInstance(tokenizer, (RobertaTokenizer, RobertaTokenizerFast))
        self.assertEqual(tokenizer.vocab_size, 20)

    def test_tokenizer_from_tokenizer_class(self):
        config = AutoConfig.from_pretrained(DUMMY_DIFF_TOKENIZER_IDENTIFIER)
        self.assertIsInstance(config, RobertaConfig)
        # Check that tokenizer_type ≠ model_type
        tokenizer = AutoTokenizer.from_pretrained(DUMMY_DIFF_TOKENIZER_IDENTIFIER, config=config)
        self.assertIsInstance(tokenizer, (BertTokenizer, BertTokenizerFast))
        self.assertEqual(tokenizer.vocab_size, 12)

    @require_tokenizers
    def test_tokenizer_identifier_with_correct_config(self):
        for tokenizer_class in [BertTokenizer, BertTokenizerFast, AutoTokenizer]:
            tokenizer = tokenizer_class.from_pretrained("wietsedv/bert-base-dutch-cased")
            self.assertIsInstance(tokenizer, (BertTokenizer, BertTokenizerFast))

            if isinstance(tokenizer, BertTokenizer):
                self.assertEqual(tokenizer.basic_tokenizer.do_lower_case, False)
            else:
                self.assertEqual(tokenizer.do_lower_case, False)

            self.assertEqual(tokenizer.model_max_length, 512)

    @require_tokenizers
    def test_tokenizer_identifier_non_existent(self):
        for tokenizer_class in [BertTokenizer, BertTokenizerFast, AutoTokenizer]:
            with self.assertRaises(EnvironmentError):
                _ = tokenizer_class.from_pretrained("julien-c/herlolip-not-exists")

    def test_parents_and_children_in_mappings(self):
        # Test that the children are placed before the parents in the mappings, as the `instanceof` will be triggered
        # by the parents and will return the wrong configuration type when using auto models

        mappings = (TOKENIZER_MAPPING,)

        for mapping in mappings:
            mapping = tuple(mapping.items())
            for index, (child_config, _) in enumerate(mapping[1:]):
                for parent_config, _ in mapping[: index + 1]:
                    with self.subTest(msg=f"Testing if {child_config.__name__} is child of {parent_config.__name__}"):
                        self.assertFalse(issubclass(child_config, parent_config))

    @require_tokenizers
    def test_from_pretrained_use_fast_toggle(self):
        self.assertIsInstance(AutoTokenizer.from_pretrained("bert-base-cased", use_fast=False), BertTokenizer)
        self.assertIsInstance(AutoTokenizer.from_pretrained("bert-base-cased"), BertTokenizerFast)

    @require_tokenizers
    def test_do_lower_case(self):
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", do_lower_case=False)
        sample = "Hello, world. How are you?"
        tokens = tokenizer.tokenize(sample)
        self.assertEqual("[UNK]", tokens[0])

        tokenizer = AutoTokenizer.from_pretrained("microsoft/mpnet-base", do_lower_case=False)
        tokens = tokenizer.tokenize(sample)
        self.assertEqual("[UNK]", tokens[0])

    @require_tokenizers
    def test_PreTrainedTokenizerFast_from_pretrained(self):
        tokenizer = AutoTokenizer.from_pretrained("robot-test/dummy-tokenizer-fast-with-model-config")
        self.assertEqual(type(tokenizer), PreTrainedTokenizerFast)
        self.assertEqual(tokenizer.model_max_length, 512)
        self.assertEqual(tokenizer.vocab_size, 30000)
        self.assertEqual(tokenizer.unk_token, "[UNK]")
        self.assertEqual(tokenizer.padding_side, "right")

    def test_auto_tokenizer_from_local_folder(self):
        tokenizer = AutoTokenizer.from_pretrained(SMALL_MODEL_IDENTIFIER)
        self.assertIsInstance(tokenizer, (BertTokenizer, BertTokenizerFast))
        with tempfile.TemporaryDirectory() as tmp_dir:
            tokenizer.save_pretrained(tmp_dir)
            tokenizer2 = AutoTokenizer.from_pretrained(tmp_dir)

        self.assertIsInstance(tokenizer2, tokenizer.__class__)
        self.assertEqual(tokenizer2.vocab_size, 12)

    def test_get_tokenizer_config(self):
        # Check we can load the tokenizer config of an online model.
        config = get_tokenizer_config("bert-base-cased")
        # If we ever update bert-base-cased tokenizer config, this dict here will need to be updated.
        self.assertEqual(config, {"do_lower_case": False})

        # This model does not have a tokenizer_config so we get back an empty dict.
        config = get_tokenizer_config(SMALL_MODEL_IDENTIFIER)
        self.assertDictEqual(config, {})

        # A tokenizer saved with `save_pretrained` always creates a tokenizer config.
        tokenizer = AutoTokenizer.from_pretrained(SMALL_MODEL_IDENTIFIER)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tokenizer.save_pretrained(tmp_dir)
            config = get_tokenizer_config(tmp_dir)

        # Check the class of the tokenizer was properly saved (note that it always saves the slow class).
        self.assertEqual(config["tokenizer_class"], "BertTokenizer")
        # Check other keys just to make sure the config was properly saved /reloaded.
        self.assertEqual(config["name_or_path"], SMALL_MODEL_IDENTIFIER)
