# Copyright 2026 the HuggingFace Team. All rights reserved.
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

import numpy as np

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTester, ImageProcessingTestMixin


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image


class Inkling2ImageProcessingTester(ImageProcessingTester):
    def __init__(self, parent, **kwargs):
        # images up to 40x40 fit in one patch, so `pixel_values` has one row per image
        kwargs.setdefault("max_resolution", 40)
        kwargs.setdefault("size", {"height": 40, "width": 40})
        super().__init__(parent, **kwargs)

    def expected_output_image_shape(self, images):
        return self.size["height"], self.size["width"], self.num_channels


@require_torch
@require_vision
class Inkling2ImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processor_tester_class = Inkling2ImageProcessingTester

    def test_patches_and_grid(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            pixels = np.random.randint(0, 256, (50, 100, 3), dtype=np.uint8)
            encoding = image_processing(Image.fromarray(pixels), return_tensors="pt")

            self.assertEqual(encoding.image_grid_thw.tolist(), [[1, 2, 3]])
            self.assertEqual(tuple(encoding.pixel_values.shape), (6, 40, 40, 3))
            expected = torch.from_numpy(pixels).float() / 127.5 - 1.0
            torch.testing.assert_close(encoding.pixel_values[0], expected[:40, :40])
            # the bottom row of patches is padded with -1 on the 0-255 scale
            torch.testing.assert_close(encoding.pixel_values[3, 10:], torch.full((30, 40, 3), -1.0 - 2.0 / 255))

    def test_transparent_image_shows_checkerboard(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            pixels = np.random.randint(0, 256, (16, 16, 4), dtype=np.uint8)
            pixels[..., 3] = 0
            encoding = image_processing(Image.fromarray(pixels), return_tensors="pt")

            self.assertEqual(encoding.image_grid_thw.tolist(), [[1, 1, 1]])
            light_square = torch.full((8, 8, 3), 204 / 127.5 - 1.0)
            dark_square = torch.full((8, 8, 3), 153 / 127.5 - 1.0)
            torch.testing.assert_close(encoding.pixel_values[0, :8, :8], light_square)
            torch.testing.assert_close(encoding.pixel_values[0, :8, 8:16], dark_square)
            torch.testing.assert_close(encoding.pixel_values[0, 8:16, :8], dark_square)
            torch.testing.assert_close(encoding.pixel_values[0, 8:16, 8:16], light_square)
