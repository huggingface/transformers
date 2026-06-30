# Copyright 2026 NVIDIA and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the LocateAnything image processor."""

import unittest

import numpy as np

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    from transformers import LocateAnythingImageProcessor


@require_torch
@require_vision
class LocateAnythingImageProcessorTest(unittest.TestCase):
    def setUp(self):
        self.patch_size = 14
        self.merge = 2
        self.image_processor = LocateAnythingImageProcessor(
            patch_size=self.patch_size, merge_kernel_size=[self.merge, self.merge]
        )

    def _make_image(self, width, height):
        return Image.fromarray(np.uint8(np.random.rand(height, width, 3) * 255))

    def test_output_keys_and_patch_shape(self):
        image = self._make_image(112, 112)
        out = self.image_processor(images=image, return_tensors="pt")
        self.assertIn("pixel_values", out)
        self.assertIn("image_grid_hws", out)
        # Patches are (num_patches, 3, patch_size, patch_size).
        self.assertEqual(out["pixel_values"].shape[1:], (3, self.patch_size, self.patch_size))
        grid_h, grid_w = out["image_grid_hws"][0].tolist()
        self.assertEqual(out["pixel_values"].shape[0], grid_h * grid_w)

    def test_grid_divisible_by_merge_kernel(self):
        image = self._make_image(100, 60)
        out = self.image_processor(images=image, return_tensors="pt")
        grid_h, grid_w = out["image_grid_hws"][0].tolist()
        self.assertEqual(grid_h % self.merge, 0)
        self.assertEqual(grid_w % self.merge, 0)

    def test_multiple_images_are_concatenated(self):
        images = [self._make_image(112, 112), self._make_image(56, 56)]
        out = self.image_processor(images=images, return_tensors="pt")
        self.assertEqual(out["image_grid_hws"].shape[0], 2)
        total_patches = sum(int(h) * int(w) for h, w in out["image_grid_hws"])
        self.assertEqual(out["pixel_values"].shape[0], total_patches)

    def test_normalization_applied(self):
        image = self._make_image(56, 56)
        out = self.image_processor(images=image, return_tensors="pt")
        self.assertTrue(torch.isfinite(out["pixel_values"]).all())
