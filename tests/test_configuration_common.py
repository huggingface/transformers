# coding=utf-8
# Copyright 2019 HuggingFace Inc.
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

import copy
import json
import os
import shutil
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from huggingface_hub import Repository, delete_repo, login
from requests.exceptions import HTTPError
from transformers import AutoConfig, BertConfig, GPT2Config, is_torch_available
from transformers.configuration_utils import PretrainedConfig
from transformers.testing_utils import PASS, USER, is_staging_test


sys.path.append(str(Path(__file__).parent.parent / "utils"))

from test_module.custom_configuration import CustomConfig  # noqa E402


config_common_kwargs = {
    "return_dict": False,
    "output_hidden_states": True,
    "output_attentions": True,
    "torchscript": True,
    "torch_dtype": "float16",
    "use_bfloat16": True,
    "pruned_heads": {"a": 1},
    "tie_word_embeddings": False,
    "is_decoder": True,
    "cross_attention_hidden_size": 128,
    "add_cross_attention": True,
    "tie_encoder_decoder": True,
    "max_length": 50,
    "min_length": 3,
    "do_sample": True,
    "early_stopping": True,
    "num_beams": 3,
    "num_beam_groups": 3,
    "diversity_penalty": 0.5,
    "temperature": 2.0,
    "top_k": 10,
    "top_p": 0.7,
    "repetition_penalty": 0.8,
    "length_penalty": 0.8,
    "no_repeat_ngram_size": 5,
    "encoder_no_repeat_ngram_size": 5,
    "bad_words_ids": [1, 2, 3],
    "num_return_sequences": 3,
    "chunk_size_feed_forward": 5,
    "output_scores": True,
    "return_dict_in_generate": True,
    "forced_bos_token_id": 2,
    "forced_eos_token_id": 3,
    "remove_invalid_values": True,
    "architectures": ["BertModel"],
    "finetuning_task": "translation",
    "id2label": {0: "label"},
    "label2id": {"label": "0"},
    "tokenizer_class": "BertTokenizerFast",
    "prefix": "prefix",
    "bos_token_id": 6,
    "pad_token_id": 7,
    "eos_token_id": 8,
    "sep_token_id": 9,
    "decoder_start_token_id": 10,
    "task_specific_params": {"translation": "some_params"},
    "problem_type": "regression",
}


class ConfigTester(object):
    def __init__(self, parent, config_class=None, has_text_modality=True, **kwargs):
        self.parent = parent
        self.config_class = config_class
        self.has_text_modality = has_text_modality
        self.inputs_dict = kwargs

    def create_and_test_config_common_properties(self):
        config = self.config_class(**self.inputs_dict)
        common_properties = ["hidden_size", "num_attention_heads", "num_hidden_layers"]

        # Add common fields for text models
        if self.has_text_modality:
            common_properties.extend(["vocab_size"])

        # Test that config has the common properties as getters
        for prop in common_properties:
            self.parent.assertTrue(hasattr(config, prop), msg=f"`{prop}` does not exist")

        # Test that config has the common properties as setter
        for idx, name in enumerate(common_properties):
            try:
                setattr(config, name, idx)
                self.parent.assertEqual(
                    getattr(config, name), idx, msg=f"`{name} value {idx} expected, but was {getattr(config, name)}"
                )
            except NotImplementedError:
                # Some models might not be able to implement setters for common_properties
                # In that case, a NotImplementedError is raised
                pass

        # Test if config class can be called with Config(prop_name=..)
        for idx, name in enumerate(common_properties):
            try:
                config = self.config_class(**{name: idx})
                self.parent.assertEqual(
                    getattr(config, name), idx, msg=f"`{name} value {idx} expected, but was {getattr(config, name)}"
                )
            except NotImplementedError:
                # Some models might not be able to implement setters for common_properties
                # In that case, a NotImplementedError is raised
                pass

    def create_and_test_config_to_json_string(self):
        config = self.config_class(**self.inputs_dict)
        obj = json.loads(config.to_json_string())
        for key, value in self.inputs_dict.items():
            self.parent.assertEqual(obj[key], value)

    def create_and_test_config_to_json_file(self):
        config_first = self.config_class(**self.inputs_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            json_file_path = os.path.join(tmpdirname, "config.json")
            config_first.to_json_file(json_file_path)
            config_second = self.config_class.from_json_file(json_file_path)

        self.parent.assertEqual(config_second.to_dict(), config_first.to_dict())

    def create_and_test_config_from_and_save_pretrained(self):
        config_first = self.config_class(**self.inputs_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            config_first.save_pretrained(tmpdirname)
            config_second = self.config_class.from_pretrained(tmpdirname)

        self.parent.assertEqual(config_second.to_dict(), config_first.to_dict())

    def create_and_test_config_with_num_labels(self):
        config = self.config_class(**self.inputs_dict, num_labels=5)
        self.parent.assertEqual(len(config.id2label), 5)
        self.parent.assertEqual(len(config.label2id), 5)

        config.num_labels = 3
        self.parent.assertEqual(len(config.id2label), 3)
        self.parent.assertEqual(len(config.label2id), 3)

    def check_config_can_be_init_without_params(self):
        if self.config_class.is_composition:
            return
        config = self.config_class()
        self.parent.assertIsNotNone(config)

    def check_config_arguments_init(self):
        kwargs = copy.deepcopy(config_common_kwargs)
        config = self.config_class(**kwargs)
        wrong_values = []
        for key, value in config_common_kwargs.items():
            if key == "torch_dtype":
                if not is_torch_available():
                    continue
                else:
                    import torch

                    if config.torch_dtype != torch.float16:
                        wrong_values.append(("torch_dtype", config.torch_dtype, torch.float16))
            elif getattr(config, key) != value:
                wrong_values.append((key, getattr(config, key), value))

        if len(wrong_values) > 0:
            errors = "\n".join([f"- {v[0]}: got {v[1]} instead of {v[2]}" for v in wrong_values])
            raise ValueError(f"The following keys were not properly set in the config:\n{errors}")

    def run_common_tests(self):
        self.create_and_test_config_common_properties()
        self.create_and_test_config_to_json_string()
        self.create_and_test_config_to_json_file()
        self.create_and_test_config_from_and_save_pretrained()
        self.create_and_test_config_with_num_labels()
        self.check_config_can_be_init_without_params()
        self.check_config_arguments_init()


@is_staging_test
class ConfigPushToHubTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._token = login(username=USER, password=PASS)

    @classmethod
    def tearDownClass(cls):
        try:
            delete_repo(token=cls._token, name="test-config")
        except HTTPError:
            pass

        try:
            delete_repo(token=cls._token, name="test-config-org", organization="valid_org")
        except HTTPError:
            pass

        try:
            delete_repo(token=cls._token, name="test-dynamic-config")
        except HTTPError:
            pass

    def test_push_to_hub(self):
        config = BertConfig(
            vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            config.save_pretrained(os.path.join(tmp_dir, "test-config"), push_to_hub=True, use_auth_token=self._token)

            new_config = BertConfig.from_pretrained(f"{USER}/test-config")
            for k, v in config.__dict__.items():
                if k != "transformers_version":
                    self.assertEqual(v, getattr(new_config, k))

    def test_push_to_hub_in_organization(self):
        config = BertConfig(
            vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            config.save_pretrained(
                os.path.join(tmp_dir, "test-config-org"),
                push_to_hub=True,
                use_auth_token=self._token,
                organization="valid_org",
            )

            new_config = BertConfig.from_pretrained("valid_org/test-config-org")
            for k, v in config.__dict__.items():
                if k != "transformers_version":
                    self.assertEqual(v, getattr(new_config, k))

    def test_push_to_hub_dynamic_config(self):
        CustomConfig.register_for_auto_class()
        config = CustomConfig(attribute=42)

        with tempfile.TemporaryDirectory() as tmp_dir:
            repo = Repository(tmp_dir, clone_from=f"{USER}/test-dynamic-config", use_auth_token=self._token)
            config.save_pretrained(tmp_dir)

            # This has added the proper auto_map field to the config
            self.assertDictEqual(config.auto_map, {"AutoConfig": "custom_configuration.CustomConfig"})
            # The code has been copied from fixtures
            self.assertTrue(os.path.isfile(os.path.join(tmp_dir, "custom_configuration.py")))

            repo.push_to_hub()

        new_config = AutoConfig.from_pretrained(f"{USER}/test-dynamic-config", trust_remote_code=True)
        # Can't make an isinstance check because the new_config is from the FakeConfig class of a dynamic module
        self.assertEqual(new_config.__class__.__name__, "CustomConfig")
        self.assertEqual(new_config.attribute, 42)


class ConfigTestUtils(unittest.TestCase):
    def test_config_from_string(self):
        c = GPT2Config()

        # attempt to modify each of int/float/bool/str config records and verify they were updated
        n_embd = c.n_embd + 1  # int
        resid_pdrop = c.resid_pdrop + 1.0  # float
        scale_attn_weights = not c.scale_attn_weights  # bool
        summary_type = c.summary_type + "foo"  # str
        c.update_from_string(
            f"n_embd={n_embd},resid_pdrop={resid_pdrop},scale_attn_weights={scale_attn_weights},summary_type={summary_type}"
        )
        self.assertEqual(n_embd, c.n_embd, "mismatch for key: n_embd")
        self.assertEqual(resid_pdrop, c.resid_pdrop, "mismatch for key: resid_pdrop")
        self.assertEqual(scale_attn_weights, c.scale_attn_weights, "mismatch for key: scale_attn_weights")
        self.assertEqual(summary_type, c.summary_type, "mismatch for key: summary_type")

    def test_config_common_kwargs_is_complete(self):
        base_config = PretrainedConfig()
        missing_keys = [key for key in base_config.__dict__ if key not in config_common_kwargs]
        # If this part of the test fails, you have arguments to addin config_common_kwargs above.
        self.assertListEqual(missing_keys, ["is_encoder_decoder", "_name_or_path", "transformers_version"])
        keys_with_defaults = [key for key, value in config_common_kwargs.items() if value == getattr(base_config, key)]
        if len(keys_with_defaults) > 0:
            raise ValueError(
                "The following keys are set with the default values in `test_configuration_common.config_common_kwargs` "
                f"pick another value for them: {', '.join(keys_with_defaults)}."
            )


class ConfigurationVersioningTest(unittest.TestCase):
    def test_local_versioning(self):
        configuration = AutoConfig.from_pretrained("bert-base-cased")
        configuration.configuration_files = ["config.4.0.0.json"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            configuration.save_pretrained(tmp_dir)
            configuration.hidden_size = 2
            json.dump(configuration.to_dict(), open(os.path.join(tmp_dir, "config.4.0.0.json"), "w"))

            # This should pick the new configuration file as the version of Transformers is > 4.0.0
            new_configuration = AutoConfig.from_pretrained(tmp_dir)
            self.assertEqual(new_configuration.hidden_size, 2)

            # Will need to be adjusted if we reach v42 and this test is still here.
            # Should pick the old configuration file as the version of Transformers is < 4.42.0
            configuration.configuration_files = ["config.42.0.0.json"]
            configuration.hidden_size = 768
            configuration.save_pretrained(tmp_dir)
            shutil.move(os.path.join(tmp_dir, "config.4.0.0.json"), os.path.join(tmp_dir, "config.42.0.0.json"))
            new_configuration = AutoConfig.from_pretrained(tmp_dir)
            self.assertEqual(new_configuration.hidden_size, 768)

    def test_repo_versioning_before(self):
        # This repo has two configuration files, one for v4.0.0 and above with a different hidden size.
        repo = "hf-internal-testing/test-two-configs"

        import transformers as new_transformers

        new_transformers.configuration_utils.__version__ = "v4.0.0"
        new_configuration = new_transformers.models.auto.AutoConfig.from_pretrained(repo)
        self.assertEqual(new_configuration.hidden_size, 2)

        # Testing an older version by monkey-patching the version in the module it's used.
        import transformers as old_transformers

        old_transformers.configuration_utils.__version__ = "v3.0.0"
        old_configuration = old_transformers.models.auto.AutoConfig.from_pretrained(repo)
        self.assertEqual(old_configuration.hidden_size, 768)
