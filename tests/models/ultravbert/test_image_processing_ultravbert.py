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

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTester, ImageProcessingTestMixin


if is_vision_available():
    from PIL import Image

if is_torch_available():
    import torch


class UltraVBertImageProcessingTester(ImageProcessingTester):
    def __init__(self, **kwargs):
        kwargs.setdefault("max_resolution", 40)
        kwargs.setdefault("do_image_splitting", True)
        kwargs.setdefault("size", {"longest_edge": 40})
        kwargs.setdefault("max_image_size", {"longest_edge": 20})
        kwargs.setdefault("max_image_tiles", 4)
        super().__init__(**kwargs)

    def expected_output_image_shape(self, images):
        # Inputs of 30 to 40 pixels resize to a 40-pixel longest edge: 2 x 2 tiles plus the global image.
        return 5, self.num_channels, self.max_image_size["longest_edge"], self.max_image_size["longest_edge"]

    def prepare_image_inputs(self, **kwargs):
        return [[image] for image in super().prepare_image_inputs(**kwargs)]


@require_torch
@require_vision
class UltraVBertImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processor_tester_class = UltraVBertImageProcessingTester

    def check_pixel_values_shape(self, image_inputs, **kwargs):
        expected_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            pixel_values = image_processing(image_inputs[0], return_tensors="pt", **kwargs).pixel_values
            self.assertEqual(tuple(pixel_values.shape), (1, *expected_shape))
            pixel_values = image_processing(image_inputs, return_tensors="pt", **kwargs).pixel_values
            self.assertEqual(tuple(pixel_values.shape), (len(image_inputs), *expected_shape))

    def test_call_pil(self):
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False)
        self.assertIsInstance(image_inputs[0][0], Image.Image)
        self.check_pixel_values_shape(image_inputs)

    def test_call_numpy(self):
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)
        self.assertIsInstance(image_inputs[0][0], np.ndarray)
        self.check_pixel_values_shape(image_inputs)

    def test_call_pytorch(self):
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)
        self.assertIsInstance(image_inputs[0][0], torch.Tensor)
        self.check_pixel_values_shape(image_inputs)

    def test_call_numpy_4_channels(self):
        self.image_processor_tester.num_channels = 4
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)
        self.check_pixel_values_shape(
            image_inputs,
            do_convert_rgb=False,
            input_data_format="channels_last",
            image_mean=[0.0] * 4,
            image_std=[1.0] * 4,
        )

    def test_get_number_of_image_patches(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            self.assertEqual(image_processing.get_number_of_image_patches(100, 100), (5, 2, 2))
            self.assertEqual(image_processing.get_number_of_image_patches(300, 500), (5, 2, 2))
            self.assertEqual(
                image_processing.get_number_of_image_patches(300, 500, {"do_image_splitting": False}), (0, 0, 0)
            )
            self.assertEqual(
                image_processing.get_number_of_image_patches(300, 600, {"max_image_size": {"longest_edge": 30}}),
                (3, 1, 2),
            )

    def test_max_image_tiles_lowers_the_longest_edge(self):
        page = Image.new("RGB", (1700, 2200), "white")
        for image_processing_class in self.image_processing_classes.values():
            uncapped = image_processing_class(max_image_tiles=36)
            self.assertEqual(uncapped.get_number_of_image_patches(2200, 1700), (31, 6, 5))

            capped = image_processing_class()
            self.assertEqual(capped.max_image_tiles, 27)
            self.assertEqual(capped.get_number_of_image_patches(2200, 1700), (21, 5, 4))
            outputs = capped([[page]], return_row_col_info=True, return_tensors="pt")
            self.assertEqual((outputs["rows"][0][0], outputs["cols"][0][0]), (5, 4))
            self.assertEqual(tuple(outputs["pixel_values"].shape), (1, 21, 3, 512, 512))
