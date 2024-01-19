# coding=utf-8
# Copyright 2020 The HuggingFace Team Inc.
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

import unittest
from parameterized import parameterized

from transformers import AutoModelForCausalLM, AutoTokenizer

class TokenHealingTestCase(unittest.TestCase):
    model_name_or_path = 'TheBloke/deepseek-llm-7B-base-GPTQ'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    completion_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map='auto',
        trust_remote_code=False,
        revision='main',
        use_cache=True,
    )

    @parameterized.expand(
        [
            ('square_bracket', 'An example ["like this"] and another example [', 'An example ["like this"] and another example ["'),
            ('url', 'The link is <a href="http:', 'The link is <a href="http://'),
            ('aggressive_healing', 'The link is <a href="http', 'The link is <a href="http'),
            ('trailing_whitespace', 'I read a book about ', 'I read a book about'),
            ('nothing_to_heal', 'I read a book about', 'I read a book about'),
            ('single_token', 'I', 'I'),
            ('empty_prompt', '', ''),
        ]
    )
    def test_prompts(self, name, input, expected):
        input_ids = self.tokenizer(input, return_tensors='pt').input_ids.to(self.completion_model.device)

        healed_ids = self.completion_model.heal_tokens(input_ids)
        predicted = self.tokenizer.decode(healed_ids[0], skip_special_tokens=True)

        self.assertEqual(predicted, expected)
