# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
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

import unittest

from huggingface_hub import hf_hub_download
from transformers import is_torch_available, is_vision_available
from transformers.testing_utils import require_torch, require_vision, slow, torch_device


if is_torch_available():
    import torch

    from transformers import AutoModelForObjectDetection

if is_vision_available():
    from PIL import Image

    from transformers import AutoFeatureExtractor


@require_torch
@require_vision
class TableTransformerIntegrationTest(unittest.TestCase):
    @slow
    def test_table_detection(self):
        # TODO update to microsoft
        feature_extractor = AutoFeatureExtractor.from_pretrained("nielsr/detr-table-detection")
        model = AutoModelForObjectDetection.from_pretrained("nielsr/detr-table-detection")
        model.to(torch_device)

        file_path = hf_hub_download(repo_id="nielsr/example-pdf", repo_type="dataset", filename="example_pdf.png")
        image = Image.open(file_path).convert("RGB")
        inputs = feature_extractor(image, return_tensors="pt")

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        expected_shape = (1, 15, 3)
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_logits = torch.tensor(
            [[-6.7329, -16.9590, 6.7447], [-8.0038, -22.3071, 6.9288], [-7.2445, -20.9855, 7.3465]]
        )
        self.assertTrue(torch.allclose(outputs.logits[0, :3, :3], expected_logits, atol=1e-4))

        expected_boxes = torch.tensor([[0.4868, 0.1764, 0.6729], [0.6674, 0.4621, 0.3864], [0.4720, 0.1757, 0.6362]])
        self.assertTrue(torch.allclose(outputs.pred_boxes[0, :3, :3], expected_boxes, atol=1e-3))
