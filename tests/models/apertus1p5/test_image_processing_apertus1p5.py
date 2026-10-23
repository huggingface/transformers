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
"""Testing suite for the Apertus 1.5 image processor."""

import tempfile
import unittest

import numpy as np

from transformers import is_torch_available
from transformers.testing_utils import require_torch, require_torchvision, require_vision
from transformers.utils import is_torchvision_available

from ...test_image_processing_common import ImageProcessingTester, ImageProcessingTestMixin


if is_torch_available():
    import torch

if is_torch_available() and is_torchvision_available():
    from transformers.models.apertus1p5.image_processing_apertus1p5 import smart_resize


class Apertus1p5ImageProcessingTester(ImageProcessingTester):
    def __init__(self, **kwargs):
        kwargs.setdefault("min_resolution", 16)
        kwargs.setdefault("max_resolution", 96)
        kwargs.setdefault("size", {"min_pixels": 32**2, "max_pixels": 64**2})
        kwargs.setdefault("spatial_factor", 16)
        super().__init__(**kwargs)

    def expected_output_image_shape(self, images):
        sizes = []
        for image in images:
            height, width = image.shape[-2:] if isinstance(image, torch.Tensor) else np.asarray(image).shape[:2]
            sizes.append(smart_resize(height, width, factor=self.spatial_factor, **self.size))
        return self.num_channels, max(height for height, _ in sizes), max(width for _, width in sizes)


@require_torch
@require_vision
@require_torchvision
class Apertus1p5ImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processor_tester_class = Apertus1p5ImageProcessingTester
    test_cast_dtype = True

    def test_resize_and_image_sizes(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
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
                    output = image_processing(image, return_tensors="pt")
                    self.assertEqual(output.pixel_values.shape, (1, 3, *expected_size))
                    self.assertEqual(output.image_sizes.tolist(), [list(expected_size)])
                    self.assertEqual(
                        image_processing.get_number_of_image_patches(*input_size),
                        (expected_size[0] // 16) * (expected_size[1] // 16),
                    )

    def test_patch_count_respects_do_resize(self):
        image = np.zeros((32, 48, 3), dtype=np.uint8)
        # with the default budget, 32x48 is upscaled to 112x160 (a 7x10 grid); unresized it is a 2x3 grid
        expected_counts = {True: 70, False: 6}
        for image_processing_class in self.image_processing_classes.values():
            for default in (True, False):
                image_processing = image_processing_class(do_resize=default)
                for kwargs in ({}, {"do_resize": True}, {"do_resize": False}):
                    with self.subTest(default=default, kwargs=kwargs):
                        expected_count = expected_counts[kwargs.get("do_resize", default)]
                        height, width = image_processing(image, return_tensors="pt", **kwargs).pixel_values.shape[-2:]
                        factor = image_processing.spatial_factor
                        self.assertEqual((height // factor) * (width // factor), expected_count)
                        self.assertEqual(image_processing.get_number_of_image_patches(32, 48, kwargs), expected_count)

    def test_size_dict_serialization(self):
        """The pixel budget is saved inside `size`, not as top-level `min_pixels` / `max_pixels` keys."""
        size = {"min_pixels": 48**2, "max_pixels": 64**2}
        for image_processing_class in self.image_processing_classes.values():
            with tempfile.TemporaryDirectory() as directory:
                image_processing_class(size=size).save_pretrained(directory)
                restored = image_processing_class.from_pretrained(directory).to_dict()
            self.assertEqual(restored["size"], size)
            self.assertNotIn("min_pixels", restored)
            self.assertNotIn("max_pixels", restored)

    def test_normalization(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            for pixel_value, expected_value in [(0, -1.0), (255, 1.0)]:
                with self.subTest(pixel_value=pixel_value):
                    image = np.full((32, 32, 3), pixel_value, dtype=np.uint8)
                    pixels = image_processing(image, return_tensors="pt").pixel_values
                    self.assertEqual(pixels.dtype, torch.float32)
                    torch.testing.assert_close(pixels, torch.full_like(pixels, expected_value))

    def test_mixed_size_batch_padding_and_order(self):
        sizes = [(32, 48), (48, 32), (32, 48)]
        images = [np.full((*size, 3), value, dtype=np.uint8) for size, value in zip(sizes, [0, 127, 255])]
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            for disable_grouping in [False, True]:
                with self.subTest(disable_grouping=disable_grouping):
                    output = image_processing(images, disable_grouping=disable_grouping, return_tensors="pt")
                    self.assertEqual(output.pixel_values.shape, (3, 3, 48, 48))
                    self.assertEqual(output.image_sizes.tolist(), [list(size) for size in sizes])
                    for index, (image, (height, width)) in enumerate(zip(images, sizes)):
                        individual = image_processing(image, return_tensors="pt").pixel_values[0]
                        expected = torch.zeros_like(output.pixel_values[index])
                        expected[:, :height, :width] = individual
                        torch.testing.assert_close(output.pixel_values[index], expected)
