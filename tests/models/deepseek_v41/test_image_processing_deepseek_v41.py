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
import unittest

import numpy as np

from transformers.testing_utils import require_torch, require_torchvision, require_vision
from transformers.utils import is_torch_available, is_torchvision_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTester, ImageProcessingTestMixin


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image, ImageOps

if is_torchvision_available():
    from transformers.models.deepseek_v41.image_processing_deepseek_v41 import get_resized_image_size


# Small enough that a test image is a handful of patches: 2 x 2 patches, merged 3 x 3 into a token
TINY_LIMITS = {"patch_size": 2, "downsample_ratio": 3, "max_image_tokens": 40, "min_pixels": 64}


class DeepseekV41ImageProcessingTester(ImageProcessingTester):
    def __init__(self, parent, **kwargs):
        # An image no larger than a patch is rounded up to exactly one patch, so `pixel_values` has one row per
        # image and the common tests know what shape to expect
        kwargs.setdefault("min_resolution", 8)
        kwargs.setdefault("max_resolution", 32)
        kwargs.setdefault("patch_size", 32)
        kwargs.setdefault("downsample_ratio", 1)
        kwargs.setdefault("max_image_tokens", 64)
        kwargs.setdefault("min_pixels", 0)
        super().__init__(parent, **kwargs)

    def expected_output_image_shape(self, images):
        return (self.num_channels * self.patch_size**2,)


@require_torch
@require_vision
@require_torchvision
class DeepseekV41ImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processor_tester_class = DeepseekV41ImageProcessingTester

    def _assert_has_attributes(self, processor, expected_attributes):
        # The common check looks for `min_pixels` inside `size`, which is where the Qwen2-VL family keeps it. Here
        # it is an attribute of its own, named like the field of the vision config.
        expected_attributes = dict(expected_attributes)
        if "min_pixels" in expected_attributes:
            self.assertEqual(processor.min_pixels, expected_attributes.pop("min_pixels"))
        super()._assert_has_attributes(processor, expected_attributes)

    def _get_image_processor(self, **kwargs):
        image_processing_class = self.image_processing_classes["torchvision"]
        return image_processing_class(**{**TINY_LIMITS, **kwargs})

    def test_resized_image_size(self):
        """The sizes that the original preprocessing plans, with the limits of the released checkpoint."""
        limits = {"patch_size": 14, "downsample_ratio": 3, "max_image_tokens": 1024, "min_pixels": 295936}
        expected_sizes = {
            # Only rounded up to whole patches
            (701, 1024): (714, 1036),
            # Scaled up to `min_pixels` first
            (200, 320): (434, 700),
            (1, 1): (546, 546),
            (40, 4000): (56, 5446),
            # Would take more than 1024 positions, so scaled down
            (2000, 3000): (1050, 1568),
            (2600, 500): (2604, 504),
            # Too tall or too wide for a full column or row of tokens at their own aspect ratio
            (100000, 100): (21462, 42),
            (100, 200000): (42, 42882),
        }
        for (height, width), expected_size in expected_sizes.items():
            with self.subTest(height=height, width=width):
                self.assertEqual(get_resized_image_size(height, width, **limits), expected_size)
        # With a limit on the aspect ratio, a wider image is planned as if it had exactly that ratio
        self.assertEqual(get_resized_image_size(300, 1500, **limits, max_wh_ratio=2), (392, 770))
        self.assertEqual(get_resized_image_size(300, 600, **limits, max_wh_ratio=2), (392, 770))

    def test_patches_and_grid(self):
        """Patches come out in reading order, each one flattened as (channels, patch height, patch width)."""
        image_processor = self._get_image_processor()
        # 12 x 18 is already whole patches and within the limits, so the pixels are not resampled
        pixels = np.random.randint(0, 256, (12, 18, 3), dtype=np.uint8)
        encoding = image_processor(Image.fromarray(pixels), return_tensors="pt")

        self.assertEqual(encoding.image_grid_hw.tolist(), [[6, 9]])
        self.assertEqual(tuple(encoding.pixel_values.shape), (54, 3 * 2 * 2))
        normalized = torch.from_numpy(pixels).permute(2, 0, 1).float() / 127.5 - 1.0
        torch.testing.assert_close(encoding.pixel_values[0], normalized[:, :2, :2].flatten())
        # Patch 13 is in the second row of patches (9 per row), and the fifth column
        torch.testing.assert_close(encoding.pixel_values[13], normalized[:, 2:4, 8:10].flatten())

    def test_several_images(self):
        """The patches of all the images are concatenated, and each image has its own row in the grid."""
        image_processor = self._get_image_processor()
        images = [Image.new("RGB", (18, 12)), Image.new("RGB", (7, 5)), Image.new("RGB", (100, 100))]
        encoding = image_processor(images, return_tensors="pt")

        self.assertEqual(encoding.image_grid_hw.tolist(), [[6, 9], [3, 5], [15, 15]])
        self.assertEqual(encoding.pixel_values.shape[0], 6 * 9 + 3 * 5 + 15 * 15)

    def test_padding_matches_original_preprocessing(self):
        """
        The original preprocessing resizes with `PIL.ImageOps.pad` onto a gray canvas. A single colour does not
        change when it is resampled, so the outputs must be equal exactly: same size, same placement, same gray.
        """
        image_processor = self._get_image_processor()
        color, gray = (200, 50, 10), (127, 127, 127)
        for height, width in [(5, 7), (7, 5), (4, 4), (9, 31), (31, 9), (30, 40), (60, 10), (100, 100)]:
            with self.subTest(height=height, width=width):
                image = Image.new("RGB", (width, height), color)
                resized_height, resized_width = get_resized_image_size(height, width, **TINY_LIMITS)
                expected = np.asarray(ImageOps.pad(image, (resized_width, resized_height), color=gray))
                expected = torch.from_numpy(expected).permute(2, 0, 1).float() / 127.5 - 1.0
                rows, columns = resized_height // 2, resized_width // 2
                expected = expected.reshape(3, rows, 2, columns, 2).permute(1, 3, 0, 2, 4).reshape(rows * columns, -1)

                encoding = image_processor(image, return_tensors="pt")
                self.assertEqual(encoding.image_grid_hw.tolist(), [[rows, columns]])
                torch.testing.assert_close(encoding.pixel_values, expected)

    def test_transparency_is_dropped(self):
        """Like the original preprocessing, the alpha channel is dropped rather than shown on a white background."""
        image_processor = self._get_image_processor()
        # Fully transparent, so compositing would turn every pixel white
        image = Image.new("RGBA", (18, 12), (200, 50, 10, 0))
        encoding = image_processor(image, return_tensors="pt")

        expected_color = torch.tensor([200, 50, 10]).float() / 127.5 - 1.0
        torch.testing.assert_close(encoding.pixel_values[0].view(3, 4), expected_color[:, None].expand(3, 4))

    def test_max_wh_ratio(self):
        """An image wider than `max_wh_ratio` is squashed to that ratio, with no gray bars."""
        image_processor = self._get_image_processor(max_wh_ratio=2)
        image = Image.new("RGB", (60, 10), (200, 50, 10))
        encoding = image_processor(image, return_tensors="pt")

        self.assertEqual(encoding.image_grid_hw.tolist(), [[5, 10]])
        expected_color = torch.tensor([200, 50, 10]).float() / 127.5 - 1.0
        expected = expected_color[:, None].expand(3, 4).flatten().expand(50, 12)
        torch.testing.assert_close(encoding.pixel_values, expected)

    def test_do_resize_false_needs_whole_patches(self):
        image_processor = self._get_image_processor()
        encoding = image_processor(Image.new("RGB", (18, 12)), do_resize=False, return_tensors="pt")
        self.assertEqual(encoding.image_grid_hw.tolist(), [[6, 9]])
        with self.assertRaisesRegex(ValueError, "multiples of the patch size"):
            image_processor(Image.new("RGB", (7, 5)), do_resize=False, return_tensors="pt")
