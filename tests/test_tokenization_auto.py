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

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import pytest

from transformers import (
    BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP,
    AutoTokenizer,
    BertConfig,
    BertTokenizer,
    BertTokenizerFast,
    CTRLTokenizer,
    GPT2Tokenizer,
    GPT2TokenizerFast,
    PreTrainedTokenizerFast,
    RobertaTokenizer,
    RobertaTokenizerFast,
    is_tokenizers_available,
)
from transformers.models.auto.configuration_auto import CONFIG_MAPPING, AutoConfig
from transformers.models.auto.tokenization_auto import (
    TOKENIZER_MAPPING,
    get_tokenizer_config,
    tokenizer_class_from_name,
)
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.testing_utils import (
    DUMMY_DIFF_TOKENIZER_IDENTIFIER,
    DUMMY_UNKNOWN_IDENTIFIER,
    SMALL_MODEL_IDENTIFIER,
    require_tokenizers,
    slow,
)


sys.path.append(str(Path(__file__).parent.parent / "utils"))

from test_module.custom_configuration import CustomConfig  # noqa E402
from test_module.custom_tokenization import CustomTokenizer  # noqa E402


if is_tokenizers_available():
    from test_module.custom_tokenization_fast import CustomTokenizerFast


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
        tokenizer = AutoTokenizer.from_pretrained(DUMMY_UNKNOWN_IDENTIFIER)
        self.assertIsInstance(tokenizer, (RobertaTokenizer, RobertaTokenizerFast))
        self.assertEqual(tokenizer.vocab_size, 20)

    def test_tokenizer_from_tokenizer_class(self):
        config = AutoConfig.from_pretrained(DUMMY_DIFF_TOKENIZER_IDENTIFIER)
        self.assertIsInstance(config, RobertaConfig)
        # Check that tokenizer_type ≠ model_type
        tokenizer = AutoTokenizer.from_pretrained(DUMMY_DIFF_TOKENIZER_IDENTIFIER, config=config)
        self.assertIsInstance(tokenizer, (BertTokenizer, BertTokenizerFast))
        self.assertEqual(tokenizer.vocab_size, 12)

    def test_tokenizer_from_type(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            shutil.copy("./tests/fixtures/vocab.txt", os.path.join(tmp_dir, "vocab.txt"))

            tokenizer = AutoTokenizer.from_pretrained(tmp_dir, tokenizer_type="bert", use_fast=False)
            self.assertIsInstance(tokenizer, BertTokenizer)

        with tempfile.TemporaryDirectory() as tmp_dir:
            shutil.copy("./tests/fixtures/vocab.json", os.path.join(tmp_dir, "vocab.json"))
            shutil.copy("./tests/fixtures/merges.txt", os.path.join(tmp_dir, "merges.txt"))

            tokenizer = AutoTokenizer.from_pretrained(tmp_dir, tokenizer_type="gpt2", use_fast=False)
            self.assertIsInstance(tokenizer, GPT2Tokenizer)

    @require_tokenizers
    def test_tokenizer_from_type_fast(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            shutil.copy("./tests/fixtures/vocab.txt", os.path.join(tmp_dir, "vocab.txt"))

            tokenizer = AutoTokenizer.from_pretrained(tmp_dir, tokenizer_type="bert")
            self.assertIsInstance(tokenizer, BertTokenizerFast)

        with tempfile.TemporaryDirectory() as tmp_dir:
            shutil.copy("./tests/fixtures/vocab.json", os.path.join(tmp_dir, "vocab.json"))
            shutil.copy("./tests/fixtures/merges.txt", os.path.join(tmp_dir, "merges.txt"))

            tokenizer = AutoTokenizer.from_pretrained(tmp_dir, tokenizer_type="gpt2")
            self.assertIsInstance(tokenizer, GPT2TokenizerFast)

    def test_tokenizer_from_type_incorrect_name(self):
        with pytest.raises(ValueError):
            AutoTokenizer.from_pretrained("./", tokenizer_type="xxx")

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
            with self.assertRaisesRegex(
                EnvironmentError,
                "julien-c/herlolip-not-exists is not a local folder and is not a valid model identifier",
            ):
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

    def test_model_name_edge_cases_in_mappings(self):
        # tests: https://github.com/huggingface/transformers/pull/13251
        # 1. models with `-`, e.g. xlm-roberta -> xlm_roberta
        # 2. models that don't remap 1-1 from model-name to model file, e.g., openai-gpt -> openai
        tokenizers = TOKENIZER_MAPPING.values()
        tokenizer_names = []

        for slow_tok, fast_tok in tokenizers:
            if slow_tok is not None:
                tokenizer_names.append(slow_tok.__name__)

            if fast_tok is not None:
                tokenizer_names.append(fast_tok.__name__)

        for tokenizer_name in tokenizer_names:
            # must find the right class
            tokenizer_class_from_name(tokenizer_name)

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
        self.assertEqual(tokenizer.truncation_side, "right")

    def test_auto_tokenizer_from_local_folder(self):
        tokenizer = AutoTokenizer.from_pretrained(SMALL_MODEL_IDENTIFIER)
        self.assertIsInstance(tokenizer, (BertTokenizer, BertTokenizerFast))
        with tempfile.TemporaryDirectory() as tmp_dir:
            tokenizer.save_pretrained(tmp_dir)
            tokenizer2 = AutoTokenizer.from_pretrained(tmp_dir)

        self.assertIsInstance(tokenizer2, tokenizer.__class__)
        self.assertEqual(tokenizer2.vocab_size, 12)

    def test_auto_tokenizer_fast_no_slow(self):
        tokenizer = AutoTokenizer.from_pretrained("ctrl")
        # There is no fast CTRL so this always gives us a slow tokenizer.
        self.assertIsInstance(tokenizer, CTRLTokenizer)

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

    def test_new_tokenizer_registration(self):
        try:
            AutoConfig.register("custom", CustomConfig)

            AutoTokenizer.register(CustomConfig, slow_tokenizer_class=CustomTokenizer)
            # Trying to register something existing in the Transformers library will raise an error
            with self.assertRaises(ValueError):
                AutoTokenizer.register(BertConfig, slow_tokenizer_class=BertTokenizer)

            tokenizer = CustomTokenizer.from_pretrained(SMALL_MODEL_IDENTIFIER)
            with tempfile.TemporaryDirectory() as tmp_dir:
                tokenizer.save_pretrained(tmp_dir)

                new_tokenizer = AutoTokenizer.from_pretrained(tmp_dir)
                self.assertIsInstance(new_tokenizer, CustomTokenizer)

        finally:
            if "custom" in CONFIG_MAPPING._extra_content:
                del CONFIG_MAPPING._extra_content["custom"]
            if CustomConfig in TOKENIZER_MAPPING._extra_content:
                del TOKENIZER_MAPPING._extra_content[CustomConfig]

    @require_tokenizers
    def test_new_tokenizer_fast_registration(self):
        try:
            AutoConfig.register("custom", CustomConfig)

            # Can register in two steps
            AutoTokenizer.register(CustomConfig, slow_tokenizer_class=CustomTokenizer)
            self.assertEqual(TOKENIZER_MAPPING[CustomConfig], (CustomTokenizer, None))
            AutoTokenizer.register(CustomConfig, fast_tokenizer_class=CustomTokenizerFast)
            self.assertEqual(TOKENIZER_MAPPING[CustomConfig], (CustomTokenizer, CustomTokenizerFast))

            del TOKENIZER_MAPPING._extra_content[CustomConfig]
            # Can register in one step
            AutoTokenizer.register(
                CustomConfig, slow_tokenizer_class=CustomTokenizer, fast_tokenizer_class=CustomTokenizerFast
            )
            self.assertEqual(TOKENIZER_MAPPING[CustomConfig], (CustomTokenizer, CustomTokenizerFast))

            # Trying to register something existing in the Transformers library will raise an error
            with self.assertRaises(ValueError):
                AutoTokenizer.register(BertConfig, fast_tokenizer_class=BertTokenizerFast)

            # We pass through a bert tokenizer fast cause there is no converter slow to fast for our new toknizer
            # and that model does not have a tokenizer.json
            with tempfile.TemporaryDirectory() as tmp_dir:
                bert_tokenizer = BertTokenizerFast.from_pretrained(SMALL_MODEL_IDENTIFIER)
                bert_tokenizer.save_pretrained(tmp_dir)
                tokenizer = CustomTokenizerFast.from_pretrained(tmp_dir)

            with tempfile.TemporaryDirectory() as tmp_dir:
                tokenizer.save_pretrained(tmp_dir)

                new_tokenizer = AutoTokenizer.from_pretrained(tmp_dir)
                self.assertIsInstance(new_tokenizer, CustomTokenizerFast)

                new_tokenizer = AutoTokenizer.from_pretrained(tmp_dir, use_fast=False)
                self.assertIsInstance(new_tokenizer, CustomTokenizer)

        finally:
            if "custom" in CONFIG_MAPPING._extra_content:
                del CONFIG_MAPPING._extra_content["custom"]
            if CustomConfig in TOKENIZER_MAPPING._extra_content:
                del TOKENIZER_MAPPING._extra_content[CustomConfig]

    def test_repo_not_found(self):
        with self.assertRaisesRegex(
            EnvironmentError, "bert-base is not a local folder and is not a valid model identifier"
        ):
            _ = AutoTokenizer.from_pretrained("bert-base")

    def test_revision_not_found(self):
        with self.assertRaisesRegex(
            EnvironmentError, r"aaaaaa is not a valid git identifier \(branch name, tag name or commit id\)"
        ):
            _ = AutoTokenizer.from_pretrained(DUMMY_UNKNOWN_IDENTIFIER, revision="aaaaaa")
