# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PPChart2Table model."""

import unittest

from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.testing_utils import cleanup, require_torch, require_vision, slow, torch_device


@slow
@require_vision
@require_torch
class PPChart2TableIntegrationTest(unittest.TestCase):
    def setUp(self):
        model_path = "PaddlePaddle/PP-Chart2Table_safetensors"
        self.model = AutoModelForImageTextToText.from_pretrained(model_path).to(torch_device)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.conversation = [
            {
                "role": "system",
                "content": [],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/chart_parsing_02.png",
                    },
                ],
            },
        ]

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_small_model_integration_test_pp_chart2table(self):
        inputs = self.processor.apply_chat_template(
            self.conversation,
            tokenize=True,
            add_generation_prompt=True,
            truncation=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        generated_ids = self.model.generate(**inputs, do_sample=False, max_new_tokens=32)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        decoded_output = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        expected_output = ["年份 | 单家五星级旅游饭店年平均营收 (百万元) | 单家五星级旅游饭店年平均利润 (百万元)\n"]
        self.assertEqual(decoded_output, expected_output)

    def test_small_model_integration_test_pp_chart2table_batched(self):
        inputs = self.processor.apply_chat_template(
            [self.conversation, self.conversation],
            tokenize=True,
            add_generation_prompt=True,
            truncation=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        generated_ids = self.model.generate(**inputs, do_sample=False, max_new_tokens=6)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        decoded_output = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        expected_output = ["年份 | 单家", "年份 | 单家"]
        self.assertEqual(decoded_output, expected_output)
