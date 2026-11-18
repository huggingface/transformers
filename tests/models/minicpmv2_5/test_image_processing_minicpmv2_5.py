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

from transformers.image_utils import get_image_size
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_vision_available

# Import the module, not the classes, so pytest doesn't collect MiniCPM-V 4.6's tests a second time from this file.
from ..minicpmv4_6 import test_image_processing_minicpmv4_6 as minicpmv4_6_tests


if is_vision_available():
    from PIL import Image


class MiniCPMV2_5ImageProcessingTester(minicpmv4_6_tests.MiniCPMV4_6ImageProcessingTester):
    def expected_output_image_shape(self, image_inputs):
        """Packed shape [C, P, total_L] of pixel_values[0]. Same as MiniCPM-V 4.6, but sides are rounded to a
        multiple of `patch_size` (2.5 has no token merger) instead of `patch_size * 4`."""
        total_length = 0
        for image in image_inputs:
            height, width = (image.height, image.width) if isinstance(image, Image.Image) else get_image_size(image)
            aspect_ratio = width / height
            new_height = int(self.scale_resolution / math.sqrt(aspect_ratio))
            new_width = int(new_height * aspect_ratio)
            best_height = max(round(new_height / self.patch_size) * self.patch_size, self.patch_size)
            best_width = max(round(new_width / self.patch_size) * self.patch_size, self.patch_size)
            total_length += best_height * best_width // self.patch_size
        return [self.num_channels, self.patch_size, total_length]


@require_torch
@require_vision
class MiniCPMV2_5ImageProcessingTest(minicpmv4_6_tests.MiniCPMV4_6ImageProcessingTest):
    image_processor_tester_class = MiniCPMV2_5ImageProcessingTester

    @unittest.skip(reason="MiniCPM-Llama3-V 2.5 has no visual-token downsampling, so `downsample_mode` is unused")
    def test_downsample_modes(self):
        pass

    def test_crop_to_patches_aspect_ratio(self):
        """Row/column order is correct when slicing non-square images (same as 4.6, with 14 px rounding)."""
        for image_processing_class in self.image_processing_classes.values():
            patch_size = 14
            image_processor = image_processing_class(slice_mode=True, max_slice_nums=7, scale_resolution=10)
            # 6 slices of one 14 x 14 patch each, plus the overview: 1 patch, or 2 for the very elongated 1 x 6 / 6 x 1
            # images (e.g. 14 x 84 px shrinks to ~4 x 24 px, rounded to 14 x 28 px).
            expected_lengths = {(2, 3): 7 * 14, (3, 2): 7 * 14, (1, 6): 8 * 14, (6, 1): 8 * 14}
            for (num_rows, num_cols), expected_length in expected_lengths.items():
                test_image = Image.new("RGB", (patch_size * num_cols, patch_size * num_rows))
                result = image_processor(test_image, return_tensors="pt")
                self.assertEqual(result.grids, [[num_rows, num_cols]])
                self.assertEqual(tuple(result.pixel_values.shape), (1, 3, patch_size, expected_length))

    def test_sizes_rounded_to_patch_size(self):
        # 300 x 500 px is under 448**2, so it is upscaled to ~448**2 px and each side rounded to a multiple of 14
        # (MiniCPM-V 4.6 would round to multiples of 56 instead).
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            image = np.random.randint(255, size=(3, 300, 500), dtype=np.uint8)
            target_sizes = image_processing(image, return_tensors="pt")["target_sizes"]
            self.assertListEqual(target_sizes.tolist(), [[25, 41]])  # 350 x 574 px
