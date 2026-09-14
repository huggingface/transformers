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
"""Focused tests for Apertus 1.5 image preprocessing."""

import unittest

import numpy as np

from transformers import Apertus1p5ImageProcessor, is_torch_available
from transformers.testing_utils import require_torch, require_torchvision, require_vision


if is_torch_available():
    import torch


@require_torch
@require_vision
@require_torchvision
class Apertus1p5ImageProcessingTest(unittest.TestCase):
    def setUp(self):
        self.image_processor = Apertus1p5ImageProcessor(min_pixels=32**2, max_pixels=64**2, spatial_factor=16)

    def test_resize_and_grid_metadata(self):
        for input_size, expected_size in [((16, 16), (32, 32)), ((128, 128), (64, 64)), ((32, 64), (32, 64))]:
            with self.subTest(input_size=input_size):
                image = np.zeros((*input_size, 3), dtype=np.uint8)
                output = self.image_processor(image, return_tensors="pt")
                self.assertEqual(output.pixel_values.shape, (1, 3, *expected_size))
                self.assertEqual(output.image_sizes.tolist(), [list(expected_size)])
                self.assertEqual(output.image_grids.tolist(), [[side // 16 for side in expected_size]])

    def test_normalization(self):
        for pixel_value, expected_value in [(0, -1.0), (255, 1.0)]:
            with self.subTest(pixel_value=pixel_value):
                image = np.full((32, 32, 3), pixel_value, dtype=np.uint8)
                pixels = self.image_processor(image, return_tensors="pt").pixel_values
                self.assertEqual(pixels.dtype, torch.float32)
                torch.testing.assert_close(pixels, torch.full_like(pixels, expected_value))

    def test_mixed_size_batch_padding_and_order(self):
        sizes = [(32, 48), (48, 32), (32, 48)]
        images = [np.full((*size, 3), value, dtype=np.uint8) for size, value in zip(sizes, [0, 127, 255])]
        for disable_grouping in [False, True]:
            with self.subTest(disable_grouping=disable_grouping):
                output = self.image_processor(images, disable_grouping=disable_grouping, return_tensors="pt")
                self.assertEqual(output.pixel_values.shape, (3, 3, 48, 48))
                self.assertEqual(output.image_sizes.tolist(), [list(size) for size in sizes])
                self.assertEqual(output.image_grids.tolist(), [[2, 3], [3, 2], [2, 3]])
                for index, (image, (height, width)) in enumerate(zip(images, sizes)):
                    individual = self.image_processor(image, return_tensors="pt").pixel_values[0]
                    expected = torch.zeros_like(output.pixel_values[index])
                    expected[:, :height, :width] = individual
                    torch.testing.assert_close(output.pixel_values[index], expected)
