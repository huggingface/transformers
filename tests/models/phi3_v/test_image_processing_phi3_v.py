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

from transformers.testing_utils import require_torch, require_vision

from ...test_image_processing_common import ImageProcessingTester, ImageProcessingTestMixin


class Phi3VImageProcessingTester(ImageProcessingTester):
    def __init__(self, **kwargs):
        kwargs.setdefault("size", {"height": 28, "width": 28})
        kwargs.setdefault("num_crops", 1)
        super().__init__(**kwargs)

    def expected_output_image_shape(self, images):
        # With `num_crops=1` every image becomes the global image and one crop
        return 2, self.num_channels, self.size["height"], self.size["width"]


@require_torch
@require_vision
class Phi3VImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processor_tester_class = Phi3VImageProcessingTester

    def test_num_crops(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(**self.image_processor_dict)
            image = self.image_processor_tester.prepare_image_inputs(batch_size=1, equal_resolution=True)[0]
            image = image.resize((50, 100))
            # Portrait 1:2 with 4 crops: the long side is resized to 2 * 28, then 2x1 crops plus the global image
            encoding = image_processor(image, num_crops=4, return_tensors="pt")
            self.assertEqual(tuple(encoding.pixel_values.shape), (1, 3, 3, 28, 28))
            self.assertEqual(encoding.image_sizes.tolist(), [[56, 28]])
