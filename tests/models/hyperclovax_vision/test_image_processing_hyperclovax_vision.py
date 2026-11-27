# Copyright 2026 NAVER Corp. and The HuggingFace Team. All rights reserved.
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

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTester, ImageProcessingTestMixin


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image


class HyperCLOVAXVisionImageProcessingTester(ImageProcessingTester):
    def __init__(self, **kwargs):
        kwargs.setdefault("size", {"shortest_edge": 18})
        kwargs.setdefault("crop_size", {"height": 18, "width": 18})
        kwargs.setdefault("image_grid_pinpoints", [[18, 18], [18, 36], [36, 18], [36, 36]])
        super().__init__(**kwargs)


@require_torch
@require_vision
class HyperCLOVAXVisionImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processor_tester_class = HyperCLOVAXVisionImageProcessingTester

    # Square images select the 2x2 layout: one global view and four local grids
    expected_output_image_shape = (5, 3, 18, 18)

    def test_call_pil(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True)
            for image in image_inputs:
                self.assertIsInstance(image, Image.Image)

            encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
            self.assertEqual(tuple(encoded_images.shape), (1, *self.expected_output_image_shape))

            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
            self.assertEqual(tuple(encoded_images.shape), (7, *self.expected_output_image_shape))

    def test_call_numpy(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, numpify=True)
            for image in image_inputs:
                self.assertIsInstance(image, np.ndarray)

            encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
            self.assertEqual(tuple(encoded_images.shape), (1, *self.expected_output_image_shape))

            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
            self.assertEqual(tuple(encoded_images.shape), (7, *self.expected_output_image_shape))

    def test_call_pytorch(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, torchify=True)
            for image in image_inputs:
                self.assertIsInstance(image, torch.Tensor)

            encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
            self.assertEqual(tuple(encoded_images.shape), (1, *self.expected_output_image_shape))

            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
            self.assertEqual(tuple(encoded_images.shape), (7, *self.expected_output_image_shape))

    @unittest.skip(reason="The AnyRes image processors don't treat 4 channel PIL and numpy consistently yet")
    def test_call_numpy_4_channels(self):
        pass

    def test_global_view_keeps_aspect_ratio(self):
        # A white 40x20 (height x width) image: the global view is resized to 18x9 and padded to 18x18 with the mean
        # color, so 4 columns on the left and 5 on the right hold the normalized mean color.
        image = Image.fromarray(np.full((40, 20, 3), 255, dtype=np.uint8))
        padding_value = (int(0.5 * 255) / 255 - 0.5) / 0.5
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            global_view = image_processing(image, return_tensors="pt").pixel_values[0, 0]

            torch.testing.assert_close(global_view[:, :, :4], torch.full((3, 18, 4), padding_value))
            torch.testing.assert_close(global_view[:, :, 4:13], torch.ones(3, 18, 9))
            torch.testing.assert_close(global_view[:, :, 13:], torch.full((3, 18, 5), padding_value))

    def test_local_grids_padded_with_mean_color(self):
        # A white 18x30 image selects the 18x36 layout: it is resized to 18x30 and padded with 3 columns on each side
        image = Image.fromarray(np.full((18, 30, 3), 255, dtype=np.uint8))
        padding_value = (int(0.5 * 255) / 255 - 0.5) / 0.5
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            pixel_values = image_processing(image, return_tensors="pt").pixel_values[0]
            self.assertEqual(pixel_values.shape[0], 3)

            left_grid, right_grid = pixel_values[1], pixel_values[2]
            torch.testing.assert_close(left_grid[:, :, :3], torch.full((3, 18, 3), padding_value))
            torch.testing.assert_close(left_grid[:, :, 3:], torch.ones(3, 18, 15))
            torch.testing.assert_close(right_grid[:, :, :15], torch.ones(3, 18, 15))
            torch.testing.assert_close(right_grid[:, :, 15:], torch.full((3, 18, 3), padding_value))

    def test_possible_resolutions_alias(self):
        # The original checkpoints store the grid pinpoints as `possible_resolutions`
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(possible_resolutions=[[18, 36]])
            self.assertEqual(image_processing.image_grid_pinpoints, [[18, 36]])
