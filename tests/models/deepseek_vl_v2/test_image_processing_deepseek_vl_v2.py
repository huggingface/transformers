# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    from transformers import DeepseekVLV2ImageProcessor


@require_torch
@require_vision
class DeepseekVLV2ImageProcessingTest(unittest.TestCase):
    def setUp(self):
        self.image_processor = DeepseekVLV2ImageProcessor()

    def test_tiling_follows_best_canvas(self):
        # (width, height) -> expected [width_tiles, height_tiles], from the original select_best_resolution
        cases = {(640, 480): [2, 2], (300, 1200): [1, 4], (384, 384): [1, 1], (2000, 500): [4, 1], (1280, 720): [4, 2]}
        for (width, height), expected in cases.items():
            outputs = self.image_processor(Image.new("RGB", (width, height), (200, 10, 10)), return_tensors="pt")
            self.assertEqual(outputs.image_spatial_crop.tolist(), [expected])
            self.assertEqual(tuple(outputs.pixel_values.shape), (1, 3, 384, 384))
            self.assertEqual(tuple(outputs.pixel_values_local.shape), (expected[0] * expected[1], 3, 384, 384))

    def test_without_cropping_each_image_gets_one_tile(self):
        images = [Image.new("RGB", (640, 480)), Image.new("RGB", (2000, 500)), Image.new("RGB", (300, 1200))]
        outputs = self.image_processor(images, crop_to_patches=False, return_tensors="pt")
        self.assertEqual(outputs.image_spatial_crop.tolist(), [[1, 1]] * 3)
        self.assertEqual(tuple(outputs.pixel_values_local.shape), (3, 3, 384, 384))

    def test_padding_uses_the_mean_colour(self):
        # A wide black image is padded top and bottom; the padding normalises to 127/255 * 2 - 1.
        outputs = self.image_processor(Image.new("RGB", (768, 192), (0, 0, 0)), return_tensors="pt")
        global_view = outputs.pixel_values[0]
        expected_pad = 127 / 255 * 2 - 1
        torch.testing.assert_close(global_view[:, 0, 0], torch.full((3,), expected_pad), atol=1e-4, rtol=0)
        torch.testing.assert_close(global_view[:, 192, 192], torch.full((3,), -1.0), atol=1e-4, rtol=0)
