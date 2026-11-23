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

from ...test_image_processing_common import ImageProcessingTester, ImageProcessingTestMixin


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image


class DeepseekVLV2ImageProcessingTester(ImageProcessingTester):
    def __init__(self, parent, **kwargs):
        kwargs.setdefault("size", {"height": 384, "width": 384})
        super().__init__(parent, **kwargs)


@require_torch
@require_vision
class DeepseekVLV2ImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processor_tester_class = DeepseekVLV2ImageProcessingTester

    @unittest.skip(reason="The mean-colour padding is defined for RGB images only")
    def test_call_numpy_4_channels(self):
        pass

    def test_tiling_follows_best_canvas(self):
        image_processor = self.image_processing_classes["torchvision"]()
        # (width, height) -> expected [width_tiles, height_tiles], from the original select_best_resolution
        cases = {(640, 480): [2, 2], (300, 1200): [1, 4], (384, 384): [1, 1], (2000, 500): [4, 1], (1280, 720): [4, 2]}
        for (width, height), expected in cases.items():
            outputs = image_processor(Image.new("RGB", (width, height), (200, 10, 10)), return_tensors="pt")
            self.assertEqual(outputs.image_spatial_crop.tolist(), [expected])
            self.assertEqual(tuple(outputs.pixel_values.shape), (1, 3, 384, 384))
            self.assertEqual(tuple(outputs.pixel_values_local.shape), (expected[0] * expected[1], 3, 384, 384))

    def test_batch_keeps_input_order_across_shape_groups(self):
        """Images are processed in shape groups; tiles and grids must still come back in input order."""
        image_processor = self.image_processing_classes["torchvision"]()
        colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        images = [Image.new("RGB", (640, 480), colours[0]), Image.new("RGB", (300, 1200), colours[1])]
        images.append(Image.new("RGB", (640, 480), colours[2]))
        outputs = image_processor(images, return_tensors="pt")
        self.assertEqual(outputs.image_spatial_crop.tolist(), [[2, 2], [1, 4], [2, 2]])
        self.assertEqual(outputs.pixel_values_local.shape[0], 4 + 4 + 4)
        # The centre of each global view carries its own image's colour, normalised to [-1, 1].
        for global_view, colour in zip(outputs.pixel_values, colours):
            expected = torch.tensor(colour, dtype=torch.float32) / 255 * 2 - 1
            torch.testing.assert_close(global_view[:, 192, 192], expected, atol=1e-4, rtol=0)
        # First tile of the second image (rows 4..7 of the flat list) is green.
        torch.testing.assert_close(
            outputs.pixel_values_local[4][:, 192, 192], torch.tensor([-1.0, 1.0, -1.0]), atol=1e-4, rtol=0
        )

    def test_without_cropping_each_image_gets_one_tile(self):
        image_processor = self.image_processing_classes["torchvision"]()
        images = [Image.new("RGB", (640, 480)), Image.new("RGB", (2000, 500)), Image.new("RGB", (300, 1200))]
        outputs = image_processor(images, crop_to_patches=False, return_tensors="pt")
        self.assertEqual(outputs.image_spatial_crop.tolist(), [[1, 1]] * 3)
        self.assertEqual(tuple(outputs.pixel_values_local.shape), (3, 3, 384, 384))

    def test_padding_uses_the_mean_colour(self):
        image_processor = self.image_processing_classes["torchvision"]()
        # A wide black image is padded top and bottom; the padding normalises to 127/255 * 2 - 1.
        outputs = image_processor(Image.new("RGB", (768, 192), (0, 0, 0)), return_tensors="pt")
        global_view = outputs.pixel_values[0]
        expected_pad = 127 / 255 * 2 - 1
        torch.testing.assert_close(global_view[:, 0, 0], torch.full((3,), expected_pad), atol=1e-4, rtol=0)
        torch.testing.assert_close(global_view[:, 192, 192], torch.full((3,), -1.0), atol=1e-4, rtol=0)
