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
from transformers.generation import GenerationConfig

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
    generation_config = GenerationConfig(
        token_healing=True,
        temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=16,
        pad_token_id=completion_model.config.pad_token_id,
    )

    @parameterized.expand(
        [
            ('square_bracket', 'An example ["like this"] and another example [', 'An example ["like this"] and another example ["')
            ('url', 'The link is <a href="http:', 'The link is <a href="http://')
            ('aggressive_healing', 'The link is <a href="http', 'The link is <a href="http')
            ('trailing_whitespace', 'I read a book about ', 'I read a book about a')
            ('no_op', 'I read a book about', 'I read a book about')
            ('single_token', 'I', 'I')
        ]
    )
    def test_prompts(self, name, input, expected):
        input_ids = self.tokenizer(input, return_tensors='pt').input_ids.cuda()
        predicted = self.completion_model.generate(inputs=input_ids, generation_config=self.generation_config)
        self.assertEqual(predicted, expected)
