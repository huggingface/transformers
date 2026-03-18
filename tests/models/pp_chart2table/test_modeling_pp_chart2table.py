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

from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    is_vision_available,
)
from transformers.testing_utils import cleanup, require_torch, require_vision, slow, torch_device


if is_vision_available():
    from transformers.image_utils import load_image


@slow
@require_vision
@require_torch
class PPChart2TableIntegrationTest(unittest.TestCase):
    def setUp(self):
        model_path = "PaddlePaddle/PP-Chart2Table_safetensors"
        self.model = AutoModelForImageTextToText.from_pretrained(model_path).to(torch_device)
        self.processor = AutoProcessor.from_pretrained(model_path)
        url = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/chart_parsing_02.png"
        self.image = load_image(url)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_small_model_integration_test_pp_chart2table(self):
        inputs = self.processor(self.image, return_tensors="pt").to(torch_device)
        generate_ids = self.model.generate(
            **inputs,
            use_cache=True,
            do_sample=False,
            max_new_tokens=32,
        )
        decoded_output = self.processor.decode(
            generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        expected_output = "年份 | 单家五星级旅游饭店年平均营收 (百万元) | 单家五星级旅游饭店年平均利润 (百万元)\n"
        self.assertEqual(decoded_output, expected_output)

    def test_small_model_integration_test_pp_chart2table_batched(self):
        inputs = self.processor([self.image, self.image], return_tensors="pt").to(torch_device)
        generate_ids = self.model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=6)
        decoded_output = self.processor.batch_decode(
            generate_ids[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        expected_output = ["年份 | 单家", "年份 | 单家"]
        self.assertEqual(decoded_output, expected_output)
