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
import tempfile
import unittest

from parameterized import parameterized
from transformers import AutoConfig, GenerationConfig


class LogitsProcessorTest(unittest.TestCase):
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
