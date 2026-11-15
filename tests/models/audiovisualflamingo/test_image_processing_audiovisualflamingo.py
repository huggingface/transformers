# Copyright 2026 NVIDIA CORPORATION and the HuggingFace Inc. team. All rights reserved.
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


class AudioVisualFlamingoImageProcessingTester(ImageProcessingTester):
    def __init__(self, **kwargs):
        # Images are not tiled by default, tiling is tested in `test_s2_tiling`
        kwargs.setdefault("size", {"height": 20, "width": 20})
        kwargs.setdefault("s2_scales", [20])
        kwargs.setdefault("max_tiles", 1)
        super().__init__(**kwargs)


@require_torch
@require_vision
class AudioVisualFlamingoImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processor_tester_class = AudioVisualFlamingoImageProcessingTester

    def test_s2_tiling(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(
                **{**self.image_processor_dict, "s2_scales": [20, 40, 60], "max_tiles": 12}
            )
            image = torch.randint(0, 256, (3, 30, 80), dtype=torch.uint8)

            # A single image is tiled at every scale: 1 + 4 tiles, then the 2x5 grid closest to its aspect ratio
            inputs = image_processor(image, return_tensors="pt")
            self.assertEqual(inputs.pixel_values.shape, (15, 3, 20, 20))
            self.assertEqual(inputs.image_grid_hw.tolist(), [[2, 5]])

            # Images of multi-image samples are resized to a single tile
            inputs = image_processor([[image, image], [image]], return_tensors="pt")
            self.assertEqual(inputs.pixel_values.shape, (17, 3, 20, 20))
            self.assertEqual(inputs.image_grid_hw.tolist(), [[1, 1], [1, 1], [2, 5]])
