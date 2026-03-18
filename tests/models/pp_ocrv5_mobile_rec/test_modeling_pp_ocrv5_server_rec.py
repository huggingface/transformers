# coding = utf-8
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
"""Testing suite for the PPOCRV5ServerRec model."""

import unittest

import requests

from transformers import (
    AutoImageProcessor,
    PPOCRV5ServerRecForTextRecognition,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import (
    require_torch,
    require_vision,
    slow,
    torch_device,
)


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image


@require_torch
@require_vision
@slow
class PPOCRV5ServerRecModelIntegrationTest(unittest.TestCase):
    def setUp(self):
        model_path = "PaddlePaddle/PP-OCRv5_mobile_rec_safetensors"
        self.model = PPOCRV5ServerRecForTextRecognition.from_pretrained(model_path).to(torch_device)
        self.image_processor = (
            AutoImageProcessor.from_pretrained(model_path, return_tensors="pt") if is_vision_available() else None
        )
        url = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_rec_001.png"
        self.image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    def test_inference_text_recognition_head(self):
        inputs = self.image_processor(images=self.image, return_tensors="pt").to(torch_device)
        outputs = self.model(**inputs)

        results = self.image_processor.post_process_text_recognition(outputs)
        expected_text = "绿洲仕格维花园公寓"
        expected_score = 0.9909055233001709

        self.assertEqual(results[0]["text"], expected_text)
        torch.testing.assert_close(results[0]["score"], expected_score, rtol=2e-2, atol=2e-2)
