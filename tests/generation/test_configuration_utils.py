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

import tempfile
import unittest

from parameterized import parameterized
from transformers.generation import GenerationConfig


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
