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

import math
import unittest

import numpy as np

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTester, ImageProcessingTestMixin


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image


class MiniCPMV2_5ImageProcessingTester(ImageProcessingTester):
    def __init__(self, **kwargs):
        kwargs.setdefault("batch_size", 2)
        kwargs.setdefault("min_resolution", 64)
        kwargs.setdefault("max_resolution", 128)
        kwargs.setdefault("scale_resolution", 448)
        kwargs.setdefault("patch_size", 14)
        super().__init__(**kwargs)

    def num_patches(self, image):
        """Patches of an image too small to be sliced: upscaled to ~scale_resolution**2 px, sides rounded to 14 px."""
        if isinstance(image, Image.Image):
            height, width = image.height, image.width
        elif isinstance(image, np.ndarray):  # the tester's NumPy images are channels-last
            height, width = image.shape[:2]
        else:  # and its PyTorch images channels-first
            height, width = image.shape[-2:]
        aspect_ratio = width / height
        new_height = int(self.scale_resolution / math.sqrt(aspect_ratio))
        new_width = int(new_height * aspect_ratio)
        best_height = max(round(new_height / self.patch_size) * self.patch_size, self.patch_size)
        best_width = max(round(new_width / self.patch_size) * self.patch_size, self.patch_size)
        return best_height * best_width // self.patch_size**2

    def expected_output_image_shape(self, image_inputs):
        """Shape of one slice in `pixel_values`: a single row of patches, padded to the longest slice."""
        max_patches = max(self.num_patches(image) for image in image_inputs)
        return [self.num_channels, self.patch_size, max_patches * self.patch_size]


@require_torch
@require_vision
class MiniCPMV2_5ImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processor_tester_class = MiniCPMV2_5ImageProcessingTester

    def test_call_returns_expected_keys(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(**self.image_processor_dict)
            images = self.image_processor_tester.prepare_image_inputs(torchify=True)
            result = image_processor(images, return_tensors="pt")
            for key in ("pixel_values", "pixel_attention_mask", "patch_position_ids", "grids"):
                self.assertIn(key, result)

    def test_padding_mask_and_position_ids(self):
        """Each slice's real patches are masked in, padding is zero, and positions follow the slice's patch grid."""
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(**self.image_processor_dict)
            images = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)
            result = image_processor(images, return_tensors="pt")
            mask, positions, pixel_values = (
                result["pixel_attention_mask"],
                result["patch_position_ids"],
                result["pixel_values"],
            )
            patch_size = self.image_processor_tester.patch_size
            patches = pixel_values.unflatten(-1, (-1, patch_size)).permute(0, 3, 1, 2, 4)
            for slice_idx in range(pixel_values.shape[0]):
                num_patches = mask[slice_idx].sum().item()
                rows, cols = (positions[slice_idx, :num_patches].max(dim=0).values + 1).tolist()
                self.assertEqual(rows * cols, num_patches)
                self.assertTrue(mask[slice_idx, :num_patches].all())
                self.assertEqual(patches[slice_idx, num_patches:].abs().sum().item(), 0)
                expected = torch.stack(torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing="ij"), -1)
                torch.testing.assert_close(positions[slice_idx, :num_patches], expected.reshape(-1, 2))
                self.assertEqual(positions[slice_idx, num_patches:].abs().sum().item(), 0)

    def test_crop_to_patches_aspect_ratio(self):
        """Row/column order is correct when slicing non-square images."""
        for image_processing_class in self.image_processing_classes.values():
            patch_size = 14
            image_processor = image_processing_class(slice_mode=True, max_slice_nums=7, scale_resolution=10)
            # 6 slices of one 14 x 14 patch, plus the overview: 1 patch, or 2 for the very elongated 1 x 6 / 6 x 1
            # images (e.g. 14 x 84 px shrinks to ~4 x 24 px, rounded to 14 x 28 px).
            expected_overview_patches = {(2, 3): 1, (3, 2): 1, (1, 6): 2, (6, 1): 2}
            for (num_rows, num_cols), overview_patches in expected_overview_patches.items():
                test_image = Image.new("RGB", (patch_size * num_cols, patch_size * num_rows))
                result = image_processor(test_image, return_tensors="pt")
                self.assertEqual(result.grids, [[num_rows, num_cols]])
                self.assertEqual(tuple(result.pixel_values.shape), (7, 3, patch_size, overview_patches * patch_size))
                self.assertEqual(result.pixel_attention_mask.sum(-1).tolist(), [overview_patches] + [1] * 6)

    def test_sizes_rounded_to_patch_size(self):
        # 300 x 500 px is under 448**2, so it is upscaled to ~448**2 px and each side rounded to a multiple of 14
        # (MiniCPM-V 4.6 would round to multiples of 56 instead).
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            image = np.random.randint(255, size=(3, 300, 500), dtype=np.uint8)
            positions = image_processing(image, return_tensors="pt")["patch_position_ids"]
            self.assertListEqual((positions[0].max(dim=0).values + 1).tolist(), [25, 41])  # 350 x 574 px
