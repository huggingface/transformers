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

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch


class Step3p7ImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        num_channels=3,
        min_resolution=30,
        max_resolution=50,
        do_resize=True,
        size=None,
        patch_size=32,
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        do_convert_rgb=True,
    ):
        size = size if size is not None else {"height": 64, "width": 64}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.patch_size = patch_size
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_convert_rgb = do_convert_rgb

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

    def expected_output_image_shape(self, images):
        # The global view is always resized/squared to `size`, regardless of input resolution.
        return [self.num_channels, self.size["height"], self.size["width"]]

    def prepare_image_inputs(self, equal_resolution=False, numpify=False, torchify=False):
        return prepare_image_inputs(
            batch_size=self.batch_size,
            num_channels=self.num_channels,
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
            equal_resolution=equal_resolution,
            numpify=numpify,
            torchify=torchify,
        )


@require_torch
@require_vision
@require_torchvision
class Step3p7ImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.image_processor_tester = Step3p7ImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            self.assertTrue(hasattr(image_processing, "do_resize"))
            self.assertTrue(hasattr(image_processing, "size"))
            self.assertTrue(hasattr(image_processing, "patch_size"))
            self.assertTrue(hasattr(image_processing, "do_rescale"))
            self.assertTrue(hasattr(image_processing, "rescale_factor"))
            self.assertTrue(hasattr(image_processing, "do_normalize"))
            self.assertTrue(hasattr(image_processing, "image_mean"))
            self.assertTrue(hasattr(image_processing, "image_std"))

    def _processor(self):
        image_processing_class = next(iter(self.image_processing_classes.values()))
        return image_processing_class(**self.image_processor_dict)

    def test_no_local_patches_for_image_fitting_global_view(self):
        # 48x48 fits within `size` (64) with an aspect ratio too square to tile (< 1.5).
        image_processor = self._processor()
        image = torch.randint(0, 256, (3, 48, 48), dtype=torch.uint8)

        num_patches, num_newlines = image_processor.get_number_of_image_patches(height=48, width=48)
        self.assertEqual((num_patches, num_newlines), (0, 0))

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

        num_patches, num_newlines = image_processor.get_number_of_image_patches(height=64, width=200)
        self.assertEqual((num_patches, num_newlines), (14, 1))

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

        num_patches, num_newlines = image_processor.get_number_of_image_patches(height=20, width=200)
        self.assertEqual((num_patches, num_newlines), (49, 6))

        result = image_processor([image], return_tensors="pt")
        # The global view is still squared to `size` regardless of the padding path.
        self.assertEqual(list(result["pixel_values"].shape), [1, 3, 64, 64])
        self.assertEqual(result["num_local_patches"].tolist(), [49])
        self.assertEqual(list(result["pixel_values_local"].shape), [49, 3, 32, 32])
