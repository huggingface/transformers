# coding=utf-8
# Copyright 2021 HuggingFace Inc.
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

from transformers.file_utils import is_torch_available, is_vision_available
from transformers.testing_utils import require_torch, require_vision, slow

from .test_feature_extraction_common import FeatureExtractionSavingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    from transformers import ImageGPTFeatureExtractor


class ImageGPTFeatureExtractorIntegrationTest(unittest.TestCase):
    @slow
    def test_image(self):
        feature_extractor = ImageGPTFeatureExtractor.from_pretrained("openai/imagegpt-small")

        image = Image.open("data/images/image_test.jpg")

        encoding = feature_extractor(image, return_tensors="pt")

        self.assertIsInstance(encoding, torch.Tensor)
        self.assertEqual(encoding.pixel_values.shape, (1, 1024))
