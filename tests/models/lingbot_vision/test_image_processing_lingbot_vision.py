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

import unittest

from transformers import LingbotVisionImageProcessor, LingbotVisionImageProcessorPil
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch


if is_vision_available():
    from PIL import Image


class LingbotVisionImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        num_channels=3,
        min_resolution=30,
        max_resolution=60,
        size=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.size = size if size is not None else {"height": 32, "width": 32}

    def prepare_image_processor_dict(self):
        return {
            "do_resize": True,
            "size": self.size,
            "do_rescale": True,
            "do_normalize": True,
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
        }

    def expected_output_image_shape(self, images):
        return self.num_channels, self.size["height"], self.size["width"]

    def prepare_image_inputs(self, equal_resolution=False, numpify=False, torchify=False):
        return prepare_image_inputs(
            batch_size=self.batch_size,
            num_channels=self.num_channels,
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
            equal_resolution=equal_resolution,
            numpify=numpify,
            torchify=torchify,
        )


@require_torch
@require_vision
class LingbotVisionImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.image_processor_tester = LingbotVisionImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_defaults(self):
        image_processor = LingbotVisionImageProcessor()
        self.assertEqual(image_processor.size.height, 512)
        self.assertEqual(image_processor.size.width, 512)
        self.assertEqual(image_processor.image_mean, (0.485, 0.456, 0.406))
        self.assertEqual(image_processor.image_std, (0.229, 0.224, 0.225))

    def test_image_processor_matches_pil_backend(self):
        image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
        default_pixel_values = LingbotVisionImageProcessor()(image, return_tensors="pt").pixel_values
        pil_pixel_values = LingbotVisionImageProcessorPil()(image, return_tensors="pt").pixel_values
        torch.testing.assert_close(default_pixel_values, pil_pixel_values, rtol=0, atol=1e-6)
