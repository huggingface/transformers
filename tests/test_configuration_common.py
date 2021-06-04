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


import json
import os
import tempfile
import unittest

from huggingface_hub import HfApi
from requests.exceptions import HTTPError
from transformers import BertConfig, GPT2Config
from transformers.testing_utils import ENDPOINT_STAGING, PASS, USER, is_staging_test


class ConfigTester(object):
    def __init__(self, parent, config_class=None, has_text_modality=True, **kwargs):
        self.parent = parent
        self.config_class = config_class
        self.has_text_modality = has_text_modality
        self.inputs_dict = kwargs

    def create_and_test_config_common_properties(self):
        config = self.config_class(**self.inputs_dict)
        if self.has_text_modality:
            self.parent.assertTrue(hasattr(config, "vocab_size"))
        self.parent.assertTrue(hasattr(config, "hidden_size"))
        self.parent.assertTrue(hasattr(config, "num_attention_heads"))
        self.parent.assertTrue(hasattr(config, "num_hidden_layers"))

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

    def run_common_tests(self):
        self.create_and_test_config_common_properties()
        self.create_and_test_config_to_json_string()
        self.create_and_test_config_to_json_file()
        self.create_and_test_config_from_and_save_pretrained()
        self.create_and_test_config_with_num_labels()
        self.check_config_can_be_init_without_params()


@is_staging_test
class ConfigPushToHubTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._api = HfApi(endpoint=ENDPOINT_STAGING)
        cls._token = cls._api.login(username=USER, password=PASS)

    @classmethod
    def tearDownClass(cls):
        try:
            cls._api.delete_repo(token=cls._token, name="test-config")
        except HTTPError:
            pass

        try:
            cls._api.delete_repo(token=cls._token, name="test-config-org", organization="valid_org")
        except HTTPError:
            pass

    def test_push_to_hub(self):
        config = BertConfig(
            vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            config.save_pretrained(tmp_dir, push_to_hub=True, repo_name="test-config", use_auth_token=self._token)

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
                tmp_dir,
                push_to_hub=True,
                repo_name="test-config-org",
                use_auth_token=self._token,
                organization="valid_org",
            )

            new_config = BertConfig.from_pretrained("valid_org/test-config-org")
            for k, v in config.__dict__.items():
                if k != "transformers_version":
                    self.assertEqual(v, getattr(new_config, k))


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
