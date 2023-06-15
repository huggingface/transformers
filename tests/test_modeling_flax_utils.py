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

import numpy as np
from huggingface_hub import HfFolder, delete_repo
from requests.exceptions import HTTPError

from transformers import BertConfig, is_flax_available
from transformers.testing_utils import TOKEN, USER, is_staging_test, require_flax


if is_flax_available():
    import os

    from flax.core.frozen_dict import unfreeze
    from flax.traverse_util import flatten_dict

    from transformers import FlaxBertModel

    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.12"  # assumed parallelism: 8


@require_flax
@is_staging_test
class FlaxModelPushToHubTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._token = TOKEN
        HfFolder.save_token(TOKEN)

    @classmethod
    def tearDownClass(cls):
        try:
            delete_repo(token=cls._token, repo_id="test-model-flax")
        except HTTPError:
            pass

        try:
            delete_repo(token=cls._token, repo_id="valid_org/test-model-flax-org")
        except HTTPError:
            pass

    def test_push_to_hub(self):
        config = BertConfig(
            vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
        )
        model = FlaxBertModel(config)
        model.push_to_hub("test-model-flax", use_auth_token=self._token)

        new_model = FlaxBertModel.from_pretrained(f"{USER}/test-model-flax")

        base_params = flatten_dict(unfreeze(model.params))
        new_params = flatten_dict(unfreeze(new_model.params))

        for key in base_params.keys():
            max_diff = (base_params[key] - new_params[key]).sum().item()
            self.assertLessEqual(max_diff, 1e-3, msg=f"{key} not identical")

        # Reset repo
        delete_repo(token=self._token, repo_id="test-model-flax")

        # Push to hub via save_pretrained
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, repo_id="test-model-flax", push_to_hub=True, use_auth_token=self._token)

        new_model = FlaxBertModel.from_pretrained(f"{USER}/test-model-flax")

        base_params = flatten_dict(unfreeze(model.params))
        new_params = flatten_dict(unfreeze(new_model.params))

        for key in base_params.keys():
            max_diff = (base_params[key] - new_params[key]).sum().item()
            self.assertLessEqual(max_diff, 1e-3, msg=f"{key} not identical")

    def test_push_to_hub_in_organization(self):
        config = BertConfig(
            vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
        )
        model = FlaxBertModel(config)
        model.push_to_hub("valid_org/test-model-flax-org", use_auth_token=self._token)

        new_model = FlaxBertModel.from_pretrained("valid_org/test-model-flax-org")

        base_params = flatten_dict(unfreeze(model.params))
        new_params = flatten_dict(unfreeze(new_model.params))

        for key in base_params.keys():
            max_diff = (base_params[key] - new_params[key]).sum().item()
            self.assertLessEqual(max_diff, 1e-3, msg=f"{key} not identical")

        # Reset repo
        delete_repo(token=self._token, repo_id="valid_org/test-model-flax-org")

        # Push to hub via save_pretrained
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(
                tmp_dir, repo_id="valid_org/test-model-flax-org", push_to_hub=True, use_auth_token=self._token
            )

        new_model = FlaxBertModel.from_pretrained("valid_org/test-model-flax-org")

        base_params = flatten_dict(unfreeze(model.params))
        new_params = flatten_dict(unfreeze(new_model.params))

        for key in base_params.keys():
            max_diff = (base_params[key] - new_params[key]).sum().item()
            self.assertLessEqual(max_diff, 1e-3, msg=f"{key} not identical")


def check_models_equal(model1, model2):
    models_are_equal = True
    flat_params_1 = flatten_dict(model1.params)
    flat_params_2 = flatten_dict(model2.params)
    for key in flat_params_1.keys():
        if np.sum(np.abs(flat_params_1[key] - flat_params_2[key])) > 1e-4:
            models_are_equal = False

    return models_are_equal


@require_flax
class FlaxModelUtilsTest(unittest.TestCase):
    def test_model_from_pretrained_subfolder(self):
        config = BertConfig.from_pretrained("hf-internal-testing/tiny-bert-flax-only")
        model = FlaxBertModel(config)

        subfolder = "bert"
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(os.path.join(tmp_dir, subfolder))

            with self.assertRaises(OSError):
                _ = FlaxBertModel.from_pretrained(tmp_dir)

            model_loaded = FlaxBertModel.from_pretrained(tmp_dir, subfolder=subfolder)

        self.assertTrue(check_models_equal(model, model_loaded))

    def test_model_from_pretrained_subfolder_sharded(self):
        config = BertConfig.from_pretrained("hf-internal-testing/tiny-bert-flax-only")
        model = FlaxBertModel(config)

        subfolder = "bert"
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(os.path.join(tmp_dir, subfolder), max_shard_size="10KB")

            with self.assertRaises(OSError):
                _ = FlaxBertModel.from_pretrained(tmp_dir)

            model_loaded = FlaxBertModel.from_pretrained(tmp_dir, subfolder=subfolder)

        self.assertTrue(check_models_equal(model, model_loaded))

    def test_model_from_pretrained_hub_subfolder(self):
        subfolder = "bert"
        model_id = "hf-internal-testing/tiny-random-bert-subfolder"

        with self.assertRaises(OSError):
            _ = FlaxBertModel.from_pretrained(model_id)

        model = FlaxBertModel.from_pretrained(model_id, subfolder=subfolder)

        self.assertIsNotNone(model)

    def test_model_from_pretrained_hub_subfolder_sharded(self):
        subfolder = "bert"
        model_id = "hf-internal-testing/tiny-random-bert-sharded-subfolder"
        with self.assertRaises(OSError):
            _ = FlaxBertModel.from_pretrained(model_id)

        model = FlaxBertModel.from_pretrained(model_id, subfolder=subfolder)

        self.assertIsNotNone(model)
