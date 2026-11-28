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


class NemotronParseImageProcessingTester(ImageProcessingTester):
    def __init__(self, **kwargs):
        kwargs.setdefault("size", {"height": 18, "width": 20})
        super().__init__(**kwargs)


@require_torch
@require_vision
class NemotronParseImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processor_tester_class = NemotronParseImageProcessingTester

    def test_thumbnail_and_white_padding(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(size={"height": 18, "width": 20}, do_normalize=False)
            # 60x50 only exceeds the height: 18x15, padded by 2 + 3 columns
            tall = image_processor(torch.zeros(3, 60, 50, dtype=torch.uint8), return_tensors="pt").pixel_values[0]
            self.assertTrue(
                (tall[:, :, 2:17] == 0).all() and (tall[:, :, :2] == 1).all() and (tall[:, :, 17:] == 1).all()
            )
            # 20x100 exceeds both: 18x90 after the height clamp, then 4x20, padded by 7 + 7 rows
            wide = image_processor(torch.zeros(3, 20, 100, dtype=torch.uint8), return_tensors="pt").pixel_values[0]
            self.assertTrue((wide[:, 7:11] == 0).all() and (wide[:, :7] == 1).all() and (wide[:, 11:] == 1).all())

    def test_size_with_longest_edge(self):
        # the released `preprocessor_config.json` files store `longest_edge` next to `height` and `width`
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(
                size={"height": 2048, "width": 1664, "longest_edge": [2048, 1664]}
            )
            self.assertEqual(image_processor.size, {"height": 2048, "width": 1664})
