# Copyright 2026 The StepFun and HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the Step3p7 image processor."""

import unittest

from transformers.testing_utils import require_torch, require_torchvision, require_vision
from transformers.utils import is_torch_available

from ...test_image_processing_common import ImageProcessingTester, ImageProcessingTestMixin


if is_torch_available():
    import torch


class Step3p7ImageProcessingTester(ImageProcessingTester):
    def __init__(self, **kwargs):
        kwargs.setdefault("batch_size", 2)
        kwargs.setdefault("num_channels", 3)
        kwargs.setdefault("min_resolution", 30)
        kwargs.setdefault("max_resolution", 50)
        kwargs.setdefault("do_resize", True)
        kwargs.setdefault("size", {"height": 64, "width": 64})
        kwargs.setdefault("patch_size", 32)
        kwargs.setdefault("do_rescale", True)
        kwargs.setdefault("rescale_factor", 1 / 255)
        kwargs.setdefault("do_normalize", True)
        kwargs.setdefault("image_mean", [0.5, 0.5, 0.5])
        kwargs.setdefault("image_std", [0.5, 0.5, 0.5])
        kwargs.setdefault("do_convert_rgb", True)
        super().__init__(**kwargs)

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "patch_size": self.patch_size,
            "do_rescale": self.do_rescale,
            "rescale_factor": self.rescale_factor,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_convert_rgb": self.do_convert_rgb,
        }


@require_torch
@require_vision
@require_torchvision
class Step3p7ImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_tester_class = Step3p7ImageProcessingTester

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def _processor(self):
        image_processing_class = next(iter(self.image_processing_classes.values()))
        return image_processing_class(**self.image_processor_dict)

    def test_no_local_patches_for_image_fitting_global_view(self):
        # 48x48 fits within `size` (64) with an aspect ratio too square to tile (< 1.5).
        image_processor = self._processor()
        image = torch.randint(0, 256, (3, 48, 48), dtype=torch.uint8)

        num_patches = image_processor.get_number_of_image_patches(height=48, width=48)
        self.assertEqual(num_patches, 0)

        result = image_processor([image], return_tensors="pt")
        self.assertEqual(list(result["pixel_values"].shape), [1, 3, 64, 64])
        self.assertEqual(result["num_local_patches"].tolist(), [0])
        self.assertNotIn("pixel_values_local", result)
        self.assertNotIn("patch_newline_masks", result)

    def test_local_patches_for_wide_image(self):
        # 200x64 (W x H): long_side=200 > image_size=64, ratio 3.125 <= 4 -> window_size = patch_size (32).
        # Snapped crop is 224x64 -> 7x2 = 14 patches, 1 newline row.
        image_processor = self._processor()
        image = torch.randint(0, 256, (3, 64, 200), dtype=torch.uint8)  # (C, H, W)

        num_patches = image_processor.get_number_of_image_patches(height=64, width=200)
        self.assertEqual(num_patches, 14)

        result = image_processor([image], return_tensors="pt")
        self.assertEqual(list(result["pixel_values"].shape), [1, 3, 64, 64])
        self.assertEqual(result["num_local_patches"].tolist(), [14])
        self.assertIn("pixel_values_local", result)
        self.assertEqual(list(result["pixel_values_local"].shape), [14, 3, 32, 32])
        self.assertIn("patch_newline_masks", result)
        self.assertEqual(len(result["patch_newline_masks"][0]), 14)

    def test_patch_newline_masks_padded_across_batch(self):
        # Same layout as above (14 patches) plus a smaller 96x32 image (3x1 = 3 patches, no newline row).
        image_processor = self._processor()
        wide_image = torch.randint(0, 256, (3, 64, 200), dtype=torch.uint8)
        small_wide_image = torch.randint(0, 256, (3, 32, 96), dtype=torch.uint8)

        result = image_processor([wide_image, small_wide_image], return_tensors="pt")
        self.assertEqual(result["num_local_patches"].tolist(), [14, 3])
        self.assertEqual(list(result["pixel_values_local"].shape), [17, 3, 32, 32])
        # Every image's mask is padded to the batch max (14).
        self.assertEqual(len(result["patch_newline_masks"][0]), 14)
        self.assertEqual(len(result["patch_newline_masks"][1]), 14)
        self.assertTrue(all(v is False for v in result["patch_newline_masks"][1][3:]))

    def test_extreme_aspect_ratio_is_square_padded(self):
        # min_side=20 < 32 and ratio=10 > 4 -> squared to 200x200 before tiling.
        image_processor = self._processor()
        image = torch.randint(0, 256, (3, 20, 200), dtype=torch.uint8)  # (C, H, W)

        num_patches = image_processor.get_number_of_image_patches(height=20, width=200)
        self.assertEqual(num_patches, 49)

        result = image_processor([image], return_tensors="pt")
        # The global view is still squared to `size` regardless of the padding path.
        self.assertEqual(list(result["pixel_values"].shape), [1, 3, 64, 64])
        self.assertEqual(result["num_local_patches"].tolist(), [49])
        self.assertEqual(list(result["pixel_values_local"].shape), [49, 3, 32, 32])
