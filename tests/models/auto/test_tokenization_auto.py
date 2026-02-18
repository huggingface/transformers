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

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pytest

import transformers
from transformers import (
    AutoTokenizer,
    BertConfig,
    BertTokenizer,
    BertTokenizerFast,
    CTRLTokenizer,
    GPT2Tokenizer,
    HerbertTokenizer,
    PreTrainedTokenizerFast,
    PythonBackend,
    Qwen2Tokenizer,
    Qwen2TokenizerFast,
    Qwen3MoeConfig,
    RobertaTokenizer,
    TokenizersBackend,
    is_tokenizers_available,
    logging,
)
from transformers.models.auto.configuration_auto import CONFIG_MAPPING, AutoConfig
from transformers.models.auto.tokenization_auto import (
    REGISTERED_FAST_ALIASES,
    REGISTERED_TOKENIZER_CLASSES,
    TOKENIZER_MAPPING,
    TOKENIZER_MAPPING_NAMES,
    get_tokenizer_config,
    tokenizer_class_from_name,
)
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.testing_utils import (
    DUMMY_DIFF_TOKENIZER_IDENTIFIER,
    DUMMY_UNKNOWN_IDENTIFIER,
    SMALL_MODEL_IDENTIFIER,
    CaptureLogger,
    RequestCounter,
    require_tokenizers,
    slow,
)


sys.path.append(str(Path(__file__).parent.parent.parent.parent / "utils"))

from test_module.custom_configuration import CustomConfig  # noqa E402
from test_module.custom_tokenization import CustomTokenizer  # noqa E402


if is_tokenizers_available():
    from test_module.custom_tokenization_fast import CustomTokenizerFast


class AutoTokenizerTest(unittest.TestCase):
    def setUp(self):
        transformers.dynamic_module_utils.TIME_OUT_REMOTE_CODE = 0

    @slow
    def test_tokenizer_from_pretrained(self):
        for model_name in ("google-bert/bert-base-uncased", "google-bert/bert-base-cased"):
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.assertIsNotNone(tokenizer)
            self.assertIsInstance(tokenizer, (BertTokenizer))
            self.assertGreater(len(tokenizer), 0)

        for model_name in ["openai-community/gpt2", "openai-community/gpt2-medium"]:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.assertIsNotNone(tokenizer)
            self.assertIsInstance(tokenizer, (GPT2Tokenizer))
            self.assertGreater(len(tokenizer), 0)

    def test_tokenizer_from_pretrained_identifier(self):
        tokenizer = AutoTokenizer.from_pretrained(SMALL_MODEL_IDENTIFIER)
        self.assertIsInstance(tokenizer, (BertTokenizer))
        self.assertEqual(tokenizer.vocab_size, 12)

    def test_tokenizer_from_model_type(self):
        tokenizer = AutoTokenizer.from_pretrained(DUMMY_UNKNOWN_IDENTIFIER)
        self.assertIsInstance(tokenizer, (RobertaTokenizer))
        self.assertEqual(tokenizer.vocab_size, 20)

    def test_tokenizer_from_tokenizer_class(self):
        config = AutoConfig.from_pretrained(DUMMY_DIFF_TOKENIZER_IDENTIFIER)
        self.assertIsInstance(config, RobertaConfig)
        # Check that tokenizer_type ≠ model_type
        tokenizer = AutoTokenizer.from_pretrained(DUMMY_DIFF_TOKENIZER_IDENTIFIER, config=config)
        self.assertIsInstance(tokenizer, (BertTokenizer))
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
            self.assertIsInstance(tokenizer, PreTrainedTokenizerFast)

        with tempfile.TemporaryDirectory() as tmp_dir:
            shutil.copy("./tests/fixtures/vocab.json", os.path.join(tmp_dir, "vocab.json"))
            shutil.copy("./tests/fixtures/merges.txt", os.path.join(tmp_dir, "merges.txt"))

            tokenizer = AutoTokenizer.from_pretrained(tmp_dir, tokenizer_type="gpt2")
            self.assertIsInstance(tokenizer, PreTrainedTokenizerFast)

    def test_tokenizer_from_type_incorrect_name(self):
        with pytest.raises(ValueError):
            AutoTokenizer.from_pretrained("./", tokenizer_type="xxx")

    @require_tokenizers
    def test_tokenizer_identifier_with_correct_config(self):
        for tokenizer_class in [BertTokenizer, AutoTokenizer]:
            tokenizer = tokenizer_class.from_pretrained("wietsedv/bert-base-dutch-cased")
            self.assertIsInstance(tokenizer, (BertTokenizer))

            self.assertEqual(tokenizer.do_lower_case, False)

            self.assertEqual(tokenizer.model_max_length, 512)

    @require_tokenizers
    def test_tokenizer_identifier_non_existent(self):
        for tokenizer_class in [BertTokenizer, AutoTokenizer]:
            with self.assertRaisesRegex(
                EnvironmentError,
                "julien-c/herlolip-not-exists is not a local folder and is not a valid model identifier",
            ):
                _ = tokenizer_class.from_pretrained("julien-c/herlolip-not-exists")

    def test_model_name_edge_cases_in_mappings(self):
        # tests: https://github.com/huggingface/transformers/pull/13251
        # 1. models with `-`, e.g. xlm-roberta -> xlm_roberta
        # 2. models that don't remap 1-1 from model-name to model file, e.g., openai-gpt -> openai
        tokenizers = TOKENIZER_MAPPING.values()
        tokenizer_names = []

        for tokenizer_entry in tokenizers:
            candidates = tokenizer_entry if isinstance(tokenizer_entry, tuple) else (tokenizer_entry,)
            for tokenizer_cls in candidates:
                if tokenizer_cls is not None:
                    tokenizer_names.append(tokenizer_cls.__name__)

        for tokenizer_name in tokenizer_names:
            # must find the right class
            tokenizer_class_from_name(tokenizer_name)

    def test_tokenizer_mapping_names_use_single_entries(self):
        # this is just to ensure tokenizer mapping names are correct and map to strings!
        invalid_entries = [
            model_name
            for model_name, tokenizer_entry in TOKENIZER_MAPPING_NAMES.items()
            if isinstance(tokenizer_entry, (tuple, list))
        ]
        self.assertListEqual(
            invalid_entries,
            [],
            msg=(
                "TOKENIZER_MAPPING_NAMES should map model types to single tokenizer class names. "
                f"Found invalid mappings for: {invalid_entries}"
            ),
        )

    @require_tokenizers
    def test_from_pretrained_use_fast_toggle(self):
        self.assertIsInstance(
            AutoTokenizer.from_pretrained("google-bert/bert-base-cased", use_fast=False), BertTokenizer
        )
        self.assertIsInstance(AutoTokenizer.from_pretrained("google-bert/bert-base-cased"), BertTokenizerFast)

    @require_tokenizers
    @slow
    def test_custom_tokenizer_from_hub(self):
        tokenizer = AutoTokenizer.from_pretrained(
            "openbmb/MiniCPM-Llama3-V-2_5", trust_remote_code=True, revision="fd7f352fac0e06d0d818b23f98e3ec8c64267a57"
        )
        self.assertTrue(tokenizer.__class__.__module__.startswith("transformers_modules."))

    @require_tokenizers
    def test_voxtral_tokenizer_converts_from_tekken(self):
        # Test that voxtral tokenizer loads correctly when falling back to TokenizersBackend
        # (i.e., when MistralCommonBackend is not available)
        repo_id = "mistralai/Voxtral-Mini-3B-2507"

        # Simulate the fallback path by temporarily changing the mapping for voxtral
        # from MistralCommonBackend to TokenizersBackend
        with mock.patch.dict(TOKENIZER_MAPPING_NAMES, {"voxtral": "TokenizersBackend"}):
            tokenizer = AutoTokenizer.from_pretrained(repo_id)

        self.assertIsInstance(tokenizer, PreTrainedTokenizerFast)
        self.assertTrue(tokenizer.is_fast)
        self.assertGreater(len(tokenizer("Voxtral")["input_ids"]), 0)

    @require_tokenizers
    def test_do_lower_case(self):
        tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased", do_lower_case=False)
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
        self.assertIsInstance(tokenizer, (BertTokenizer))
        with tempfile.TemporaryDirectory() as tmp_dir:
            tokenizer.save_pretrained(tmp_dir)
            tokenizer2 = AutoTokenizer.from_pretrained(tmp_dir)

        self.assertIsInstance(tokenizer2, tokenizer.__class__)
        self.assertEqual(tokenizer2.vocab_size, 12)

    def test_auto_tokenizer_from_local_folder_mistral_detection(self):
        """See #42374 for reference, ensuring proper mistral detection on local tokenizers"""
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-235B-A22B-Thinking-2507")
        config = Qwen3MoeConfig.from_pretrained("Qwen/Qwen3-235B-A22B-Thinking-2507")
        self.assertIsInstance(tokenizer, (Qwen2Tokenizer, Qwen2TokenizerFast))

        with tempfile.TemporaryDirectory() as tmp_dir:
            tokenizer.save_pretrained(tmp_dir)

            # Case 1: Tokenizer with no config associated
            logger = logging.get_logger("transformers.tokenization_utils_base")
            with CaptureLogger(logger) as cl:
                AutoTokenizer.from_pretrained(tmp_dir)
            self.assertNotIn(
                "with an incorrect regex pattern: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84#69121093e8b480e709447d5e",
                cl.out,
            )

            # Case 2: Tokenizer with config associated
            # Needed to be saved along the tokenizer to detect (non)mistral
            # for a version where the regex bug occurs
            config_dict = config.to_diff_dict()
            config_dict["transformers_version"] = "4.57.2"

            # Manually saving to avoid versioning clashes
            config_path = os.path.join(tmp_dir, "config.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2, sort_keys=True)

            tokenizer2 = AutoTokenizer.from_pretrained(tmp_dir)

        self.assertIsInstance(tokenizer2, tokenizer.__class__)
        self.assertTrue(tokenizer2.vocab_size > 100_000)

    @require_tokenizers
    def test_auto_tokenizer_loads_bloom_repo_without_tokenizer_class(self):
        tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-BloomForCausalLM")
        self.assertIsInstance(tokenizer, TokenizersBackend)
        self.assertTrue(tokenizer.is_fast)

    @require_tokenizers
    def test_auto_tokenizer_loads_sentencepiece_only_repo(self):
        tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-mbart")
        self.assertIsInstance(tokenizer, TokenizersBackend)
        self.assertTrue(tokenizer.is_fast)

    def test_auto_tokenizer_fast_no_slow(self):
        tokenizer = AutoTokenizer.from_pretrained("Salesforce/ctrl")
        # There is no fast CTRL so this always gives us a slow tokenizer.
        self.assertIsInstance(tokenizer, CTRLTokenizer)

    def test_get_tokenizer_config(self):
        # Check we can load the tokenizer config of an online model.
        config = get_tokenizer_config("google-bert/bert-base-cased")
        _ = config.pop("_commit_hash", None)
        # If we ever update google-bert/bert-base-cased tokenizer config, this dict here will need to be updated.
        self.assertEqual(config, {"do_lower_case": False, "model_max_length": 512})

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
                self.assertIsInstance(new_tokenizer, TokenizersBackend)

        finally:
            if "custom" in CONFIG_MAPPING._extra_content:
                del CONFIG_MAPPING._extra_content["custom"]
            if CustomConfig in TOKENIZER_MAPPING._extra_content:
                del TOKENIZER_MAPPING._extra_content[CustomConfig]
            REGISTERED_TOKENIZER_CLASSES.pop("CustomTokenizer", None)

    @require_tokenizers
    def test_new_tokenizer_fast_registration(self):
        try:
            AutoConfig.register("custom", CustomConfig)

            # Can register in two steps (fast takes precedence)
            AutoTokenizer.register(CustomConfig, slow_tokenizer_class=CustomTokenizer)
            self.assertEqual(TOKENIZER_MAPPING[CustomConfig], CustomTokenizer)
            AutoTokenizer.register(CustomConfig, fast_tokenizer_class=CustomTokenizerFast)
            self.assertEqual(TOKENIZER_MAPPING[CustomConfig], CustomTokenizerFast)

            del TOKENIZER_MAPPING._extra_content[CustomConfig]
            # Can register in one step
            AutoTokenizer.register(
                CustomConfig, slow_tokenizer_class=CustomTokenizer, fast_tokenizer_class=CustomTokenizerFast
            )
            self.assertEqual(TOKENIZER_MAPPING[CustomConfig], CustomTokenizerFast)

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
                self.assertIsInstance(new_tokenizer, CustomTokenizerFast)

        finally:
            if "custom" in CONFIG_MAPPING._extra_content:
                del CONFIG_MAPPING._extra_content["custom"]
            if CustomConfig in TOKENIZER_MAPPING._extra_content:
                del TOKENIZER_MAPPING._extra_content[CustomConfig]
            REGISTERED_TOKENIZER_CLASSES.pop("CustomTokenizer", None)
            REGISTERED_TOKENIZER_CLASSES.pop("CustomTokenizerFast", None)
            REGISTERED_FAST_ALIASES.pop("CustomTokenizer", None)

    def test_from_pretrained_dynamic_tokenizer(self):
        # If remote code is not set, we will time out when asking whether to load the model.
        with self.assertRaises(ValueError):
            tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/test_dynamic_tokenizer")
        # If remote code is disabled, we can't load this config.
        with self.assertRaises(ValueError):
            tokenizer = AutoTokenizer.from_pretrained(
                "hf-internal-testing/test_dynamic_tokenizer", trust_remote_code=False
            )

        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/test_dynamic_tokenizer", trust_remote_code=True)
        self.assertTrue(tokenizer.special_attribute_present)

        # Test the dynamic module is loaded only once.
        reloaded_tokenizer = AutoTokenizer.from_pretrained(
            "hf-internal-testing/test_dynamic_tokenizer", trust_remote_code=True
        )
        self.assertIs(tokenizer.__class__, reloaded_tokenizer.__class__)

        # Test tokenizer can be reloaded.
        with tempfile.TemporaryDirectory() as tmp_dir:
            tokenizer.save_pretrained(tmp_dir)
            reloaded_tokenizer = AutoTokenizer.from_pretrained(tmp_dir, trust_remote_code=True)
        self.assertTrue(reloaded_tokenizer.special_attribute_present)

        if is_tokenizers_available():
            self.assertEqual(tokenizer.__class__.__name__, "NewTokenizerFast")
            self.assertEqual(reloaded_tokenizer.__class__.__name__, "NewTokenizerFast")

            # Test we can also load the slow version
            tokenizer = AutoTokenizer.from_pretrained(
                "hf-internal-testing/test_dynamic_tokenizer", trust_remote_code=True, use_fast=False
            )
            self.assertTrue(tokenizer.special_attribute_present)
            self.assertEqual(tokenizer.__class__.__name__, "NewTokenizerFast")
            # Test tokenizer can be reloaded.
            with tempfile.TemporaryDirectory() as tmp_dir:
                tokenizer.save_pretrained(tmp_dir)
                reloaded_tokenizer = AutoTokenizer.from_pretrained(tmp_dir, trust_remote_code=True, use_fast=False)
                self.assertTrue(
                    os.path.exists(os.path.join(tmp_dir, "tokenization.py"))
                )  # Assert we saved tokenizer code
                self.assertEqual(reloaded_tokenizer._auto_class, "AutoTokenizer")
                with open(os.path.join(tmp_dir, "tokenizer_config.json"), "r") as f:
                    tokenizer_config = json.load(f)
                # Assert we're pointing at local code and not another remote repo
                self.assertEqual(
                    tokenizer_config["auto_map"]["AutoTokenizer"],
                    ["tokenization.NewTokenizer", "tokenization_fast.NewTokenizerFast"],
                )
            self.assertEqual(reloaded_tokenizer.__class__.__name__, "NewTokenizerFast")
            self.assertTrue(reloaded_tokenizer.special_attribute_present)
        else:
            self.assertEqual(tokenizer.__class__.__name__, "NewTokenizer")
            self.assertEqual(reloaded_tokenizer.__class__.__name__, "NewTokenizer")

        # Test the dynamic module is reloaded if we force it.
        reloaded_tokenizer = AutoTokenizer.from_pretrained(
            "hf-internal-testing/test_dynamic_tokenizer", trust_remote_code=True, force_download=True
        )
        self.assertIsNot(tokenizer.__class__, reloaded_tokenizer.__class__)
        self.assertTrue(reloaded_tokenizer.special_attribute_present)

    @slow
    def test_custom_tokenizer_init(self):
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen-VL", trust_remote_code=True, revision="0547ed36a86561e2e42fecec8fd0c4f6953e33c4"
        )
        self.assertIsInstance(tokenizer, PythonBackend)
        self.assertGreater(len(tokenizer.get_vocab()), 0)

    @require_tokenizers
    def test_from_pretrained_dynamic_tokenizer_conflict(self):
        class NewTokenizer(BertTokenizer):
            special_attribute_present = False

        try:
            AutoConfig.register("custom", CustomConfig)
            AutoTokenizer.register(CustomConfig, slow_tokenizer_class=NewTokenizer)
            # If remote code is not set, the default is to use local
            tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/test_dynamic_tokenizer", use_fast=False)
            self.assertEqual(tokenizer.__class__.__name__, "NewTokenizer")
            self.assertFalse(tokenizer.special_attribute_present)

            tokenizer = AutoTokenizer.from_pretrained(
                "hf-internal-testing/test_dynamic_tokenizer", trust_remote_code=False, use_fast=False
            )
            self.assertEqual(tokenizer.__class__.__name__, "NewTokenizer")
            self.assertFalse(tokenizer.special_attribute_present)

            tokenizer = AutoTokenizer.from_pretrained(
                "hf-internal-testing/test_dynamic_tokenizer", trust_remote_code=True, use_fast=False
            )
            self.assertEqual(tokenizer.__class__.__name__, "NewTokenizerFast")
            self.assertTrue(tokenizer.special_attribute_present)

        finally:
            if "custom" in CONFIG_MAPPING._extra_content:
                del CONFIG_MAPPING._extra_content["custom"]
            if CustomConfig in TOKENIZER_MAPPING._extra_content:
                del TOKENIZER_MAPPING._extra_content[CustomConfig]
            REGISTERED_TOKENIZER_CLASSES.pop("NewTokenizer", None)

    def test_from_pretrained_dynamic_tokenizer_legacy_format(self):
        tokenizer = AutoTokenizer.from_pretrained(
            "hf-internal-testing/test_dynamic_tokenizer_legacy", trust_remote_code=True
        )
        self.assertTrue(tokenizer.special_attribute_present)
        if is_tokenizers_available():
            self.assertEqual(tokenizer.__class__.__name__, "NewTokenizerFast")

            # Test we can also load the slow version
            tokenizer = AutoTokenizer.from_pretrained(
                "hf-internal-testing/test_dynamic_tokenizer_legacy", trust_remote_code=True, use_fast=False
            )
            self.assertTrue(tokenizer.special_attribute_present)
            self.assertEqual(tokenizer.__class__.__name__, "NewTokenizerFast")
        else:
            self.assertEqual(tokenizer.__class__.__name__, "NewTokenizer")

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

    @unittest.skip("This test is failing on main")  # TODO Matt/ydshieh, fix this test!
    def test_cached_tokenizer_has_minimum_calls_to_head(self):
        # Make sure we have cached the tokenizer.
        _ = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bert")
        with RequestCounter() as counter:
            _ = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bert")
        self.assertEqual(counter["GET"], 0)
        self.assertEqual(counter["HEAD"], 1)
        self.assertEqual(counter.total_calls, 1)

    def test_init_tokenizer_with_trust(self):
        nop_tokenizer_code = """
import transformers

class NopTokenizer(transformers.PreTrainedTokenizer):
    def get_vocab(self):
        return {}
"""

        nop_config_code = """
from transformers import PreTrainedConfig

class NopConfig(PreTrainedConfig):
    model_type = "test_unregistered_dynamic"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
"""

        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_model_id = "hf-internal-testing/test_unregistered_dynamic"
            fake_repo = os.path.join(tmp_dir, fake_model_id)
            os.makedirs(fake_repo)

            tokenizer_src_file = os.path.join(fake_repo, "tokenizer.py")
            with open(tokenizer_src_file, "w") as wfp:
                wfp.write(nop_tokenizer_code)

            model_config_src_file = os.path.join(fake_repo, "config.py")
            with open(model_config_src_file, "w") as wfp:
                wfp.write(nop_config_code)

            config = {
                "model_type": "test_unregistered_dynamic",
                "auto_map": {"AutoConfig": f"{fake_model_id}--config.NopConfig"},
            }

            config_file = os.path.join(fake_repo, "config.json")
            with open(config_file, "w") as wfp:
                json.dump(config, wfp, indent=2)

            tokenizer_config = {
                "auto_map": {
                    "AutoTokenizer": [
                        f"{fake_model_id}--tokenizer.NopTokenizer",
                        None,
                    ]
                }
            }

            tokenizer_config_file = os.path.join(fake_repo, "tokenizer_config.json")
            with open(tokenizer_config_file, "w") as wfp:
                json.dump(tokenizer_config, wfp, indent=2)

            prev_dir = os.getcwd()
            try:
                # it looks like subdir= is broken in the from_pretrained also, so this is necessary
                os.chdir(tmp_dir)

                # this should work because we trust the code
                _ = AutoTokenizer.from_pretrained(fake_model_id, local_files_only=True, trust_remote_code=True)
                try:
                    # this should fail because we don't trust and we're not at a terminal for interactive response
                    _ = AutoTokenizer.from_pretrained(fake_model_id, local_files_only=True, trust_remote_code=False)
                    self.fail("AutoTokenizer.from_pretrained with trust_remote_code=False should raise ValueException")
                except ValueError:
                    pass
            finally:
                os.chdir(prev_dir)

    def test_tokenization_class_priority(self):
        from transformers import AutoProcessor

        tok = AutoTokenizer.from_pretrained("mlx-community/MiniMax-M2.1-4bit")
        self.assertTrue(tok.__class__ == TokenizersBackend)

        tok = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
        self.assertTrue(tok.__class__ == HerbertTokenizer)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tok.save_pretrained(tmp_dir)
            tok2 = AutoTokenizer.from_pretrained(tmp_dir)
            self.assertTrue(tok2.__class__ == HerbertTokenizer)

        tok = AutoProcessor.from_pretrained("mistralai/Ministral-3-8B-Instruct-2512-BF16").tokenizer
        self.assertTrue(tok.__class__ == TokenizersBackend)

    def test_custom_tokenizer_with_mismatched_tokenizer_class(self):
        nop_tokenizer_code = """
import transformers

class NopTokenizer(transformers.PreTrainedTokenizer):
    special_attribute_present = True

    def get_vocab(self):
        return {}
"""

        nop_config_code = """
from transformers import PreTrainedConfig

class NopConfig(PreTrainedConfig):
    model_type = "test_unregistered_dynamic"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
"""

        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_model_id = "hf-internal-testing/test_unregistered_dynamic"
            fake_repo = os.path.join(tmp_dir, fake_model_id)
            os.makedirs(fake_repo)

            tokenizer_src_file = os.path.join(fake_repo, "tokenizer.py")
            with open(tokenizer_src_file, "w") as wfp:
                wfp.write(nop_tokenizer_code)

            model_config_src_file = os.path.join(fake_repo, "config.py")
            with open(model_config_src_file, "w") as wfp:
                wfp.write(nop_config_code)

            config = {
                "model_type": "test_unregistered_dynamic",
                "auto_map": {"AutoConfig": f"{fake_model_id}--config.NopConfig"},
            }

            config_file = os.path.join(fake_repo, "config.json")
            with open(config_file, "w") as wfp:
                json.dump(config, wfp, indent=2)

            tokenizer_config = {
                "tokenizer_class": "NopTokenizer",
                "auto_map": {
                    "AutoTokenizer": [
                        f"{fake_model_id}--tokenizer.NopTokenizer",
                        None,
                    ]
                },
            }

            tokenizer_config_file = os.path.join(fake_repo, "tokenizer_config.json")
            with open(tokenizer_config_file, "w") as wfp:
                json.dump(tokenizer_config, wfp, indent=2)

            prev_dir = os.getcwd()
            try:
                os.chdir(tmp_dir)

                tokenizer = AutoTokenizer.from_pretrained(fake_model_id, local_files_only=True, trust_remote_code=True)
                self.assertEqual(tokenizer.__class__.__name__, "NopTokenizer")
                self.assertTrue(tokenizer.special_attribute_present)
            finally:
                os.chdir(prev_dir)
