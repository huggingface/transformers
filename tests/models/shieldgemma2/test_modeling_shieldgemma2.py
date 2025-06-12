# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Gemma3 model."""

import unittest
from io import BytesIO

import requests
from PIL import Image

from transformers import is_torch_available
from transformers.testing_utils import (
    cleanup,
    require_torch_accelerator,
    slow,
    torch_device,
)


if is_torch_available():
    import torch

    from transformers import ShieldGemma2ForImageClassification, ShieldGemma2Processor


@slow
@require_torch_accelerator
# @require_read_token
class ShieldGemma2IntegrationTest(unittest.TestCase):
    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_model(self):
        model_id = "google/shieldgemma-2-4b-it"

        processor = ShieldGemma2Processor.from_pretrained(model_id, padding_side="left")
        url = "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png"
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))

        model = ShieldGemma2ForImageClassification.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(
            torch_device
        )

        inputs = processor(images=[image]).to(torch_device)
        output = model(**inputs)
        self.assertEqual(len(output.probabilities), 3)
        for element in output.probabilities:
            self.assertEqual(len(element), 2)
