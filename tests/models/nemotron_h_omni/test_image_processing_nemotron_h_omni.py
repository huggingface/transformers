# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Testing suite for the NemotronH_Omni image processor."""

import unittest

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    from transformers import NemotronH_Omni_Reasoning_V3ImageProcessor


@require_torch
@require_vision
class NemotronH_Omni_Reasoning_V3ImageProcessingTest(unittest.TestCase):
    patch_size = 16
    downsample_ratio = 0.5

    def image_processor(self, **kwargs):
        defaults = {
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "patch_size": self.patch_size,
            "downsample_ratio": self.downsample_ratio,
            "min_num_patches": 4,
            "max_num_patches": 64,
            "max_model_len": 1024,
        }
        defaults.update(kwargs)
        return NemotronH_Omni_Reasoning_V3ImageProcessor(**defaults)

    def test_properties_round_trip(self):
        processor = self.image_processor()
        restored = NemotronH_Omni_Reasoning_V3ImageProcessor.from_dict(processor.to_dict())
        for attr in ("image_mean", "image_std", "patch_size", "min_num_patches", "max_num_patches"):
            self.assertEqual(getattr(restored, attr), getattr(processor, attr))

    def test_single_image_output_shape(self):
        processor = self.image_processor()
        image = Image.new("RGB", (128, 96))
        out = processor(images=image, return_tensors="pt")

        pixel_values, image_grid_hw = out["pixel_values"], out["image_grid_hw"]
        self.assertEqual(image_grid_hw.shape, (1, 2))
        self.assertEqual(pixel_values.shape, (int(image_grid_hw.prod()), 3 * self.patch_size**2))
        # both sides of the grid are multiples of the pixel-shuffle factor
        self.assertTrue((image_grid_hw % 2 == 0).all())

    def test_patch_budget_is_respected(self):
        processor = self.image_processor(min_num_patches=4, max_num_patches=16)
        out = processor(images=Image.new("RGB", (512, 512)), return_tensors="pt")

        num_patches = int(out["image_grid_hw"].prod())
        self.assertEqual(out["pixel_values"].shape[0], num_patches)
        self.assertGreaterEqual(num_patches, 4)
        self.assertLessEqual(num_patches, 16)

    def test_batched_images(self):
        processor = self.image_processor()
        images = [Image.new("RGB", (128, 96)), Image.new("RGB", (128, 96))]
        out = processor(images=images, return_tensors="pt")
        self.assertEqual(out["image_grid_hw"].shape, (2, 2))
        self.assertEqual(out["pixel_values"].shape[0], int(out["image_grid_hw"].prod(-1).sum()))

    def test_normalization_uses_configured_statistics(self):
        processor = self.image_processor(image_mean=[0.0, 0.0, 0.0], image_std=[1.0, 1.0, 1.0])
        out = processor(images=Image.new("RGB", (128, 96), color=(255, 255, 255)), return_tensors="pt")
        # with mean 0 / std 1 a fully white image stays at 1.0
        self.assertTrue(torch.allclose(out["pixel_values"], torch.ones_like(out["pixel_values"])))

    def test_legacy_norm_statistics_are_accepted(self):
        processor = NemotronH_Omni_Reasoning_V3ImageProcessor(norm_mean=[0.1, 0.2, 0.3], norm_std=[0.4, 0.5, 0.6])
        self.assertEqual(tuple(processor.image_mean), (0.1, 0.2, 0.3))
        self.assertEqual(tuple(processor.image_std), (0.4, 0.5, 0.6))
        self.assertNotIn("norm_mean", processor.to_dict())

    def test_call_kwargs_override_defaults(self):
        processor = self.image_processor(min_num_patches=4, max_num_patches=64)
        image = Image.new("RGB", (512, 512))
        default = processor(images=image, return_tensors="pt")["image_grid_hw"]
        capped = processor(images=image, max_num_patches=16, return_tensors="pt")["image_grid_hw"]
        self.assertEqual(int(default.prod()), 64)
        self.assertEqual(int(capped.prod()), 16)

    def test_images_of_different_sizes_are_packed(self):
        processor = self.image_processor()
        out = processor(images=[Image.new("RGB", (128, 96)), Image.new("RGB", (96, 128))], return_tensors="pt")
        self.assertIsInstance(out["pixel_values"], torch.Tensor)
        self.assertNotEqual(out["image_grid_hw"][0].tolist(), out["image_grid_hw"][1].tolist())
        self.assertEqual(out["pixel_values"].shape[0], int(out["image_grid_hw"].prod(-1).sum()))

    def test_patches_are_channel_major(self):
        processor = self.image_processor(image_mean=[0.0, 0.0, 0.0], image_std=[1.0, 1.0, 1.0])
        out = processor(images=Image.new("RGB", (64, 64), color=(255, 0, 0)), return_tensors="pt")
        first_patch = out["pixel_values"][0].view(3, self.patch_size, self.patch_size)
        self.assertTrue(torch.allclose(first_patch[0], torch.ones_like(first_patch[0])))
        self.assertTrue(torch.allclose(first_patch[1:], torch.zeros_like(first_patch[1:])))
