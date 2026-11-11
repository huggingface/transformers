# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available

from ...test_image_processing_common import ImageProcessingTester, ImageProcessingTestMixin


if is_torch_available():
    import torch


class PI0ImageProcessingTester(ImageProcessingTester):
    def __init__(self, **kwargs):
        kwargs.setdefault("size", {"max_height": 64, "max_width": 64})
        kwargs.setdefault("pad_size", {"height": 64, "width": 64})
        super().__init__(**kwargs)

    def expected_output_image_shape(self, images):
        return self.num_channels, self.pad_size["height"], self.pad_size["width"]


@require_torch
@require_vision
class PI0ImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processor_tester_class = PI0ImageProcessingTester

    def test_resize_keeps_aspect_ratio_and_pads(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(**self.image_processor_dict)
            image = torch.full((3, 40, 80), 255, dtype=torch.uint8)
            pixel_values = image_processor(image, do_normalize=False, do_rescale=False, return_tensors="pt")[
                "pixel_values"
            ]
            self.assertEqual(tuple(pixel_values.shape), (1, 3, 64, 64))
            self.assertTrue(pixel_values[0, :, 32:].eq(0).all())
