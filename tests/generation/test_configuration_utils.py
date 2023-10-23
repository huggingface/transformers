# coding=utf-8
# Copyright 2022 The HuggingFace Team Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a clone of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import os
import tempfile
import unittest
import warnings

from huggingface_hub import HfFolder, delete_repo
from parameterized import parameterized
from requests.exceptions import HTTPError

from transformers import AutoConfig, GenerationConfig
from transformers.testing_utils import TOKEN, USER, is_staging_test


class GenerationConfigTest(unittest.TestCase):
    @parameterized.expand([(None,), ("foo.json",)])
    def test_save_load_config(self, config_name):
        config = GenerationConfig(
            do_sample=True,
            temperature=0.7,
            length_penalty=1.0,
            bad_words_ids=[[1, 2, 3], [4, 5]],
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            config.save_pretrained(tmp_dir, config_name=config_name)
            loaded_config = GenerationConfig.from_pretrained(tmp_dir, config_name=config_name)

        # Checks parameters that were specified
        self.assertEqual(loaded_config.do_sample, True)
        self.assertEqual(loaded_config.temperature, 0.7)
        self.assertEqual(loaded_config.length_penalty, 1.0)
        self.assertEqual(loaded_config.bad_words_ids, [[1, 2, 3], [4, 5]])

        # Checks parameters that were not specified (defaults)
        self.assertEqual(loaded_config.top_k, 50)
        self.assertEqual(loaded_config.max_length, 20)
        self.assertEqual(loaded_config.max_time, None)

    def test_from_model_config(self):
        model_config = AutoConfig.from_pretrained("gpt2")
        generation_config_from_model = GenerationConfig.from_model_config(model_config)
        default_generation_config = GenerationConfig()

        # The generation config has loaded a few non-default parameters from the model config
        self.assertNotEqual(generation_config_from_model, default_generation_config)

        # One of those parameters is eos_token_id -- check if it matches
        self.assertNotEqual(generation_config_from_model.eos_token_id, default_generation_config.eos_token_id)
        self.assertEqual(generation_config_from_model.eos_token_id, model_config.eos_token_id)

    def test_update(self):
        generation_config = GenerationConfig()
        update_kwargs = {
            "max_new_tokens": 1024,
            "foo": "bar",
        }
        update_kwargs_copy = copy.deepcopy(update_kwargs)
        unused_kwargs = generation_config.update(**update_kwargs)

        # update_kwargs was not modified (no side effects)
        self.assertEqual(update_kwargs, update_kwargs_copy)

        # update_kwargs was used to update the config on valid attributes
        self.assertEqual(generation_config.max_new_tokens, 1024)

        # `.update()` returns a dictionary of unused kwargs
        self.assertEqual(unused_kwargs, {"foo": "bar"})

    def test_initialize_new_kwargs(self):
        generation_config = GenerationConfig()
        generation_config.foo = "bar"

        with tempfile.TemporaryDirectory("test-generation-config") as tmp_dir:
            generation_config.save_pretrained(tmp_dir)

            new_config = GenerationConfig.from_pretrained(tmp_dir)
        # update_kwargs was used to update the config on valid attributes
        self.assertEqual(new_config.foo, "bar")

        generation_config = GenerationConfig.from_model_config(new_config)
        assert not hasattr(generation_config, "foo")  # no new kwargs should be initialized if from config

    def test_kwarg_init(self):
        """Tests that we can overwrite attributes at `from_pretrained` time."""
        default_config = GenerationConfig()
        self.assertEqual(default_config.temperature, 1.0)
        self.assertEqual(default_config.do_sample, False)
        self.assertEqual(default_config.num_beams, 1)

        config = GenerationConfig(
            do_sample=True,
            temperature=0.7,
            length_penalty=1.0,
            bad_words_ids=[[1, 2, 3], [4, 5]],
        )
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.do_sample, True)
        self.assertEqual(config.num_beams, 1)

        with tempfile.TemporaryDirectory() as tmp_dir:
            config.save_pretrained(tmp_dir)
            loaded_config = GenerationConfig.from_pretrained(tmp_dir, temperature=1.0)

        self.assertEqual(loaded_config.temperature, 1.0)
        self.assertEqual(loaded_config.do_sample, True)
        self.assertEqual(loaded_config.num_beams, 1)  # default value

    def test_refuse_to_save(self):
        """Tests that we refuse to save a generation config that fails validation."""

        # setting the temperature alone is invalid, as we also need to set do_sample to True -> throws a warning that
        # is caught, doesn't save, and raises a warning
        config = GenerationConfig()
        config.temperature = 0.5
        with tempfile.TemporaryDirectory() as tmp_dir:
            with warnings.catch_warnings(record=True) as captured_warnings:
                config.save_pretrained(tmp_dir)
            self.assertEqual(len(captured_warnings), 1)
            self.assertTrue("Fix these issues to save the configuration." in str(captured_warnings[0].message))
            self.assertTrue(len(os.listdir(tmp_dir)) == 0)

        # greedy decoding throws an exception if we try to return multiple sequences -> throws an exception that is
        # caught, doesn't save, and raises a warning
        config = GenerationConfig()
        config.num_return_sequences = 2
        with tempfile.TemporaryDirectory() as tmp_dir:
            with warnings.catch_warnings(record=True) as captured_warnings:
                config.save_pretrained(tmp_dir)
            self.assertEqual(len(captured_warnings), 1)
            self.assertTrue("Fix these issues to save the configuration." in str(captured_warnings[0].message))
            self.assertTrue(len(os.listdir(tmp_dir)) == 0)

        # final check: no warnings thrown if it is correct, and file is saved
        config = GenerationConfig()
        with tempfile.TemporaryDirectory() as tmp_dir:
            with warnings.catch_warnings(record=True) as captured_warnings:
                config.save_pretrained(tmp_dir)
            self.assertEqual(len(captured_warnings), 0)
            self.assertTrue(len(os.listdir(tmp_dir)) == 1)


@is_staging_test
class ConfigPushToHubTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._token = TOKEN
        HfFolder.save_token(TOKEN)

    @classmethod
    def tearDownClass(cls):
        try:
            delete_repo(token=cls._token, repo_id="test-generation-config")
        except HTTPError:
            pass

        try:
            delete_repo(token=cls._token, repo_id="valid_org/test-generation-config-org")
        except HTTPError:
            pass

    def test_push_to_hub(self):
        config = GenerationConfig(
            do_sample=True,
            temperature=0.7,
            length_penalty=1.0,
        )
        config.push_to_hub("test-generation-config", token=self._token)

        new_config = GenerationConfig.from_pretrained(f"{USER}/test-generation-config")
        for k, v in config.to_dict().items():
            if k != "transformers_version":
                self.assertEqual(v, getattr(new_config, k))

        # Reset repo
        delete_repo(token=self._token, repo_id="test-generation-config")

        # Push to hub via save_pretrained
        with tempfile.TemporaryDirectory() as tmp_dir:
            config.save_pretrained(tmp_dir, repo_id="test-generation-config", push_to_hub=True, token=self._token)

        new_config = GenerationConfig.from_pretrained(f"{USER}/test-generation-config")
        for k, v in config.to_dict().items():
            if k != "transformers_version":
                self.assertEqual(v, getattr(new_config, k))

    def test_push_to_hub_in_organization(self):
        config = GenerationConfig(
            do_sample=True,
            temperature=0.7,
            length_penalty=1.0,
        )
        config.push_to_hub("valid_org/test-generation-config-org", token=self._token)

        new_config = GenerationConfig.from_pretrained("valid_org/test-generation-config-org")
        for k, v in config.to_dict().items():
            if k != "transformers_version":
                self.assertEqual(v, getattr(new_config, k))

        # Reset repo
        delete_repo(token=self._token, repo_id="valid_org/test-generation-config-org")

        # Push to hub via save_pretrained
        with tempfile.TemporaryDirectory() as tmp_dir:
            config.save_pretrained(
                tmp_dir, repo_id="valid_org/test-generation-config-org", push_to_hub=True, token=self._token
            )

        new_config = GenerationConfig.from_pretrained("valid_org/test-generation-config-org")
        for k, v in config.to_dict().items():
            if k != "transformers_version":
                self.assertEqual(v, getattr(new_config, k))
