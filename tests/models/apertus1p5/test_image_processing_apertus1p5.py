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

import tempfile
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
        self.image_processor = Apertus1p5ImageProcessor(
            size={"min_pixels": 32**2, "max_pixels": 64**2}, spatial_factor=16
        )

    def test_resize_and_image_sizes(self):
        for input_size, expected_size in [
            ((16, 16), (32, 32)),
            ((128, 128), (64, 64)),
            ((32, 64), (32, 64)),
            ((40, 40), (32, 32)),
            ((16, 32), (32, 48)),
            ((96, 64), (64, 48)),
        ]:
            with self.subTest(input_size=input_size):
                image = np.zeros((*input_size, 3), dtype=np.uint8)
                output = self.image_processor(image, return_tensors="pt")
                self.assertEqual(output.pixel_values.shape, (1, 3, *expected_size))
                self.assertEqual(output.image_sizes.tolist(), [list(expected_size)])
                self.assertEqual(
                    self.image_processor.get_number_of_image_patches(*input_size),
                    (expected_size[0] // 16) * (expected_size[1] // 16),
                )

    def test_patch_count_respects_do_resize(self):
        image = np.zeros((32, 48, 3), dtype=np.uint8)
        for default in (True, False):
            processor = Apertus1p5ImageProcessor(do_resize=default)
            for kwargs in ({}, {"do_resize": True}, {"do_resize": False, "size": None}):
                with self.subTest(default=default, kwargs=kwargs):
                    output = processor(image, return_tensors="pt", **kwargs)
                    expected_count = 70 if kwargs.get("do_resize", default) else 6
                    height, width = output.pixel_values.shape[-2:]
                    self.assertEqual(
                        (height // processor.spatial_factor) * (width // processor.spatial_factor), expected_count
                    )
                    self.assertEqual(processor.get_number_of_image_patches(32, 48, kwargs), expected_count)

    def test_size_override_and_save_load(self):
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        size = {"min_pixels": 48**2, "max_pixels": 64**2}
        output = self.image_processor(image, size=size, return_tensors="pt")
        self.assertEqual(output.image_sizes.tolist(), [[48, 48]])
        self.assertEqual(self.image_processor.get_number_of_image_patches(16, 16, {"size": size}), 9)
        self.assertEqual(self.image_processor(image, return_tensors="pt").image_sizes.tolist(), [[32, 32]])

        with tempfile.TemporaryDirectory() as directory:
            processor = Apertus1p5ImageProcessor(size=size)
            processor.save_pretrained(directory)
            restored = Apertus1p5ImageProcessor.from_pretrained(directory)
        self.assertEqual(restored.to_dict()["size"], size)
        self.assertNotIn("min_pixels", restored.to_dict())
        self.assertNotIn("max_pixels", restored.to_dict())
        torch.testing.assert_close(restored(image, return_tensors="pt").pixel_values, output.pixel_values)

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
                for index, (image, (height, width)) in enumerate(zip(images, sizes)):
                    individual = self.image_processor(image, return_tensors="pt").pixel_values[0]
                    expected = torch.zeros_like(output.pixel_values[index])
                    expected[:, :height, :width] = individual
                    torch.testing.assert_close(output.pixel_values[index], expected)
