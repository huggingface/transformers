# Copyright 2019-present, the HuggingFace Inc. team.
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

import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import transformers
import transformers.models.auto
from transformers.models.auto.configuration_auto import CONFIG_MAPPING, AutoConfig
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.testing_utils import DUMMY_UNKNOWN_IDENTIFIER, get_tests_dir


sys.path.append(str(Path(__file__).parent.parent.parent.parent / "utils"))

from test_module.custom_configuration import CustomConfig  # noqa E402


SAMPLE_ROBERTA_CONFIG = get_tests_dir("fixtures/dummy-config.json")


class AutoConfigTest(unittest.TestCase):
    def setUp(self):
        transformers.dynamic_module_utils.TIME_OUT_REMOTE_CODE = 0

    def test_module_spec(self):
        self.assertIsNotNone(transformers.models.auto.__spec__)
        self.assertIsNotNone(importlib.util.find_spec("transformers.models.auto"))

    def test_config_from_model_shortcut(self):
        config = AutoConfig.from_pretrained("google-bert/bert-base-uncased")
        self.assertIsInstance(config, BertConfig)

    def test_config_model_type_from_local_file(self):
        config = AutoConfig.from_pretrained(SAMPLE_ROBERTA_CONFIG)
        self.assertIsInstance(config, RobertaConfig)

    def test_config_model_type_from_model_identifier(self):
        config = AutoConfig.from_pretrained(DUMMY_UNKNOWN_IDENTIFIER)
        self.assertIsInstance(config, RobertaConfig)

    def test_config_for_model_str(self):
        config = AutoConfig.for_model("roberta")
        self.assertIsInstance(config, RobertaConfig)

    def test_pattern_matching_fallback(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # This model name contains bert and roberta, but roberta ends up being picked.
            folder = os.path.join(tmp_dir, "fake-roberta")
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, "config.json"), "w") as f:
                f.write(json.dumps({}))
            config = AutoConfig.from_pretrained(folder)
            self.assertEqual(type(config), RobertaConfig)

    def test_new_config_registration(self):
        try:
            AutoConfig.register("custom", CustomConfig)
            # Wrong model type will raise an error
            with self.assertRaises(ValueError):
                AutoConfig.register("model", CustomConfig)
            # Trying to register something existing in the Transformers library will raise an error
            with self.assertRaises(ValueError):
                AutoConfig.register("bert", BertConfig)

            # Now that the config is registered, it can be used as any other config with the auto-API
            config = CustomConfig()
            with tempfile.TemporaryDirectory() as tmp_dir:
                config.save_pretrained(tmp_dir)
                new_config = AutoConfig.from_pretrained(tmp_dir)
                self.assertIsInstance(new_config, CustomConfig)

        finally:
            if "custom" in CONFIG_MAPPING._extra_content:
                del CONFIG_MAPPING._extra_content["custom"]

    def test_repo_not_found(self):
        with self.assertRaisesRegex(
            EnvironmentError, "bert-base is not a local folder and is not a valid model identifier"
        ):
            _ = AutoConfig.from_pretrained("bert-base")

    def test_revision_not_found(self):
        with self.assertRaisesRegex(
            EnvironmentError, r"aaaaaa is not a valid git identifier \(branch name, tag name or commit id\)"
        ):
            _ = AutoConfig.from_pretrained(DUMMY_UNKNOWN_IDENTIFIER, revision="aaaaaa")

    def test_from_pretrained_dynamic_config(self):
        # If remote code is not set, we will time out when asking whether to load the model.
        with self.assertRaises(ValueError):
            config = AutoConfig.from_pretrained("hf-internal-testing/test_dynamic_model")
        # If remote code is disabled, we can't load this config.
        with self.assertRaises(ValueError):
            config = AutoConfig.from_pretrained("hf-internal-testing/test_dynamic_model", trust_remote_code=False)

        config = AutoConfig.from_pretrained("hf-internal-testing/test_dynamic_model", trust_remote_code=True)
        self.assertEqual(config.__class__.__name__, "NewModelConfig")

        # Test the dynamic module is loaded only once.
        reloaded_config = AutoConfig.from_pretrained("hf-internal-testing/test_dynamic_model", trust_remote_code=True)
        self.assertIs(config.__class__, reloaded_config.__class__)

        # Test config can be reloaded.
        with tempfile.TemporaryDirectory() as tmp_dir:
            config.save_pretrained(tmp_dir)
            reloaded_config = AutoConfig.from_pretrained(tmp_dir, trust_remote_code=True)
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "configuration.py")))  # Assert we saved config code
            # Assert we're pointing at local code and not another remote repo
            self.assertEqual(reloaded_config.auto_map["AutoConfig"], "configuration.NewModelConfig")
        self.assertEqual(reloaded_config.__class__.__name__, "NewModelConfig")

    def test_from_pretrained_dynamic_config_conflict(self):
        class NewModelConfigLocal(BertConfig):
            model_type = "new-model"

        try:
            AutoConfig.register("new-model", NewModelConfigLocal)
            # If remote code is not set, the default is to use local
            config = AutoConfig.from_pretrained("hf-internal-testing/test_dynamic_model")
            self.assertEqual(config.__class__.__name__, "NewModelConfigLocal")

            # If remote code is disabled, we load the local one.
            config = AutoConfig.from_pretrained("hf-internal-testing/test_dynamic_model", trust_remote_code=False)
            self.assertEqual(config.__class__.__name__, "NewModelConfigLocal")

            # If remote is enabled, we load from the Hub
            config = AutoConfig.from_pretrained("hf-internal-testing/test_dynamic_model", trust_remote_code=True)
            self.assertEqual(config.__class__.__name__, "NewModelConfig")

        finally:
            if "new-model" in CONFIG_MAPPING._extra_content:
                del CONFIG_MAPPING._extra_content["new-model"]
