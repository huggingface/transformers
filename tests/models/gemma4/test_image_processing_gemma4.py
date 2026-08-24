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
from parameterized import parameterized

from transformers.models.gemma4.image_processing_pil_gemma4 import get_aspect_ratio_preserving_size
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_torchvision_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    if is_torchvision_available():
        pass


class Gemma4ImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        min_resolution=30,
        max_resolution=400,
        do_resize=True,
        do_normalize=False,
        image_mean=None,
        image_std=None,
        do_convert_rgb=True,
        patch_size=6,
        max_soft_tokens=70,
        pooling_kernel_size=1,
    ):
        super().__init__()
        image_mean = image_mean if image_mean is not None else [0.0, 0.0, 0.0]
        image_std = image_std if image_std is not None else [1.0, 1.0, 1.0]
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_convert_rgb = do_convert_rgb
        self.patch_size = patch_size
        self.max_soft_tokens = max_soft_tokens
        self.pooling_kernel_size = pooling_kernel_size

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_convert_rgb": self.do_convert_rgb,
            "patch_size": self.patch_size,
            "max_soft_tokens": self.max_soft_tokens,
            "pooling_kernel_size": self.pooling_kernel_size,
        }

    # Copied from tests.models.clip.test_image_processing_clip.CLIPImageProcessingTester.prepare_image_inputs
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

    def expected_output_image_shape(self, images=None):
        """Return the expected per-image output shape: (max_patches, patch_pixels)."""
        max_patches = self.max_soft_tokens * self.pooling_kernel_size**2
        # Images are always converted to RGB (3 channels) before patchification
        patch_pixels = self.patch_size**2 * 3
        return max_patches, patch_pixels


@require_torch
@require_vision
class Gemma4ImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.image_processor_tester = Gemma4ImageProcessingTester(self)

    @unittest.skip("Gemma4 patchification requires RGB (3-channel) images; 4-channel inputs are unsupported.")
    def test_call_numpy_4_channels(self):
        pass

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        """Test that all expected attributes are present."""
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            self.assertTrue(hasattr(image_processing, "do_resize"))
            self.assertTrue(hasattr(image_processing, "do_normalize"))
            self.assertTrue(hasattr(image_processing, "image_mean"))
            self.assertTrue(hasattr(image_processing, "image_std"))
            self.assertTrue(hasattr(image_processing, "do_convert_rgb"))
            self.assertTrue(hasattr(image_processing, "patch_size"))
            self.assertTrue(hasattr(image_processing, "max_soft_tokens"))
            self.assertTrue(hasattr(image_processing, "pooling_kernel_size"))

    def test_image_processor_defaults(self):
        """Test default parameter values for Gemma4 matching VARASP_SL280_K3."""
        for image_processing_class in self.image_processing_classes.values():
            proc = image_processing_class()
            self.assertEqual(proc.patch_size, 16)
            self.assertEqual(proc.max_soft_tokens, 280)
            self.assertEqual(proc.pooling_kernel_size, 3)
            self.assertFalse(proc.do_normalize)
            self.assertEqual(list(proc.image_mean), [0.0, 0.0, 0.0])
            self.assertEqual(list(proc.image_std), [1.0, 1.0, 1.0])
            self.assertEqual(proc.resample, 3)

    def test_image_processor_from_dict_with_kwargs(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class.from_dict(self.image_processor_dict)
            self.assertEqual(image_processor.patch_size, 6)
            self.assertEqual(image_processor.max_soft_tokens, 70)

            image_processor = image_processing_class.from_dict(self.image_processor_dict, patch_size=18)
            self.assertEqual(image_processor.patch_size, 18)

    def test_output_keys(self):
        """Test that the output contains pixel_values, image_position_ids, and num_soft_tokens_per_image."""
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            image = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
            result = image_processing(image, return_tensors="pt")
            self.assertIn("pixel_values", result)
            self.assertIn("image_position_ids", result)
            self.assertIn("num_soft_tokens_per_image", result)

    def test_aspect_ratio_preserving_resize_dimensions(self):
        """Test resize dimension calculations match C++ source of truth VisionAspectRatioTests."""
        for patch_size, max_patches, pooling_kernel_size, height, width, expectation in [
            (16, 256, 1, 256, 256, (256, 256)),
            (16, 256, 1, 512, 512, (256, 256)),
            (10, 200, 1, 50, 10000, (10, 2000)),
            (10, 200, 1, 25, 10000, (10, 2000)),
            (16, 2304, 6, 2785, 34, (6144, 96)),
            (10, 200, 1, 25, 20000, (10, 2000)),
            (4, 64, 2, 50, 1000, (8, 128)),
            (5, 100, 3, 100, 100, (45, 45)),
            (5, 20, 3, 5, 100, (15, 30)),
        ]:
            target_h, target_w = get_aspect_ratio_preserving_size(
                height=height,
                width=width,
                patch_size=patch_size,
                max_patches=max_patches,
                pooling_kernel_size=pooling_kernel_size,
            )
            side_mult = patch_size * pooling_kernel_size

            self.assertEqual((target_h, target_w), expectation)
            self.assertEqual(target_h % side_mult, 0, f"Resized height {target_h} not divisible by {side_mult}")
            self.assertEqual(target_w % side_mult, 0, f"Resized width {target_w} not divisible by {side_mult}")

    @parameterized.expand([(70), (140), (280), (560), (1120)])
    def test_max_soft_tokens_values(self, max_soft_tokens):
        """Test that the processor produces valid patchified output for each supported max_soft_tokens value."""
        for image_processing_class in self.image_processing_classes.values():
            processor = image_processing_class(patch_size=16, max_soft_tokens=max_soft_tokens, pooling_kernel_size=3)
            image = Image.fromarray(np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8))
            result = processor(image, return_tensors="pt")

            max_patches = max_soft_tokens * 3**2
            patch_pixels = 16 * 16 * 3
            self.assertEqual(result.pixel_values.shape, (1, max_patches, patch_pixels))
            self.assertEqual(result.image_position_ids.shape, (1, max_patches, 2))

            # Verify real patches don't exceed the budget
            real_mask = result.image_position_ids[0, :, 0] >= 0
            num_real = real_mask.sum().item()
            self.assertLessEqual(num_real, max_patches)

    def test_position_ids_structure(self):
        """Test that image_position_ids has correct real and padding structure."""
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            image = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
            result = image_processing(image, return_tensors="pt")

            position_ids = result.image_position_ids[0]  # (max_patches, 2)
            max_patches = (
                self.image_processor_tester.max_soft_tokens * self.image_processor_tester.pooling_kernel_size**2
            )

            # Real positions should be non-negative
            real_mask = position_ids[:, 0] >= 0
            num_real = real_mask.sum().item()
            self.assertGreater(num_real, 0)
            self.assertLessEqual(num_real, max_patches)

            # Padding positions should be (-1, -1)
            pad_mask = ~real_mask
            if pad_mask.any():
                pad_positions = position_ids[pad_mask]
                self.assertTrue((pad_positions == -1).all())

            # Real positions should come before padding positions
            if pad_mask.any():
                last_real_idx = torch.where(real_mask)[0][-1].item()
                first_pad_idx = torch.where(pad_mask)[0][0].item()
                self.assertEqual(last_real_idx + 1, first_pad_idx)

    def test_padding_patches_are_zero(self):
        """Test that padding patches in pixel_values are filled with zeros."""
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            image = Image.fromarray(np.random.randint(1, 255, (100, 100, 3), dtype=np.uint8))
            result = image_processing(image, return_tensors="pt")

            position_ids = result.image_position_ids[0]
            pad_mask = position_ids[:, 0] < 0
            if pad_mask.any():
                pad_patches = result.pixel_values[0, pad_mask]
                self.assertTrue((pad_patches == 0).all())
