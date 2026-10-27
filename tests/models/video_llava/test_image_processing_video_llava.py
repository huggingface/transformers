# Copyright 2026 HuggingFace Inc.
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


class VideoLlavaImageProcessingTester(ImageProcessingTester):
    def __init__(self, **kwargs):
        # Image processor init kwargs
        kwargs.setdefault("size", {"shortest_edge": 20})
        kwargs.setdefault("crop_size", {"height": 18, "width": 18})

        super().__init__(**kwargs)


@require_torch
@require_vision
class VideoLlavaImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processor_tester_class = VideoLlavaImageProcessingTester

    @unittest.skip("Model doesn't support backends yet!")
    def test_new_models_require_torchvision_backend(self):
        pass

    @unittest.skip("Model doesn't support backends yet!")
    def test_pil_can_load_without_torchvision(self):
        pass

    def test_call_pil(self):
        for image_processing_class in self.image_processing_classes.values():
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random PIL images
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False)
            for image in image_inputs:
                self.assertIsInstance(image, Image.Image)

            # Test not batched input
            encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values_images
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
            self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))

            # Test batched
            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values_images
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
            self.assertEqual(
                tuple(encoded_images.shape), (self.image_processor_tester.batch_size, *expected_output_image_shape)
            )

    def test_call_numpy(self):
        for image_processing_class in self.image_processing_classes.values():
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random numpy tensors
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)
            for image in image_inputs:
                self.assertIsInstance(image, np.ndarray)

            # Test not batched input
            encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values_images
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
            self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))

            # Test batched
            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values_images
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
            self.assertEqual(
                tuple(encoded_images.shape), (self.image_processor_tester.batch_size, *expected_output_image_shape)
            )

    def test_call_pytorch(self):
        for image_processing_class in self.image_processing_classes.values():
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random PyTorch tensors
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)

            for image in image_inputs:
                self.assertIsInstance(image, torch.Tensor)

            # Test not batched input
            encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values_images
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
            self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))

            # Test batched
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values_images
            self.assertEqual(
                tuple(encoded_images.shape),
                (self.image_processor_tester.batch_size, *expected_output_image_shape),
            )

    def test_call_numpy_4_channels(self):
        for image_processing_class in self.image_processing_classes.values():
            # Test that can process images which have an arbitrary number of channels
            # Initialize image_processing
            image_processor = image_processing_class(**self.image_processor_dict)

            # create random numpy tensors
            self.image_processor_tester.num_channels = 4
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)

            # Test not batched input
            encoded_images = image_processor(
                image_inputs[0],
                do_convert_rgb=False,
                return_tensors="pt",
                input_data_format="channels_last",
                image_mean=[0.0, 0.0, 0.0, 0.0],
                image_std=[1.0, 1.0, 1.0, 1.0],
            ).pixel_values_images
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
            self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))

            # Test batched
            encoded_images = image_processor(
                image_inputs,
                do_convert_rgb=False,
                return_tensors="pt",
                input_data_format="channels_last",
                image_mean=[0.0, 0.0, 0.0, 0.0],
                image_std=[1.0, 1.0, 1.0, 1.0],
            ).pixel_values_images
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
            self.assertEqual(
                tuple(encoded_images.shape), (self.image_processor_tester.batch_size, *expected_output_image_shape)
            )

    def test_legacy_processor_inputs(self):
        """
        Tests that `return_legacy_image_output` actually takes effect.
        """
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False)

            inputs = image_processing(image_inputs, return_legacy_image_output=False, return_tensors="pt")
            self.assertTrue("pixel_values" in inputs)
            self.assertTrue("pixel_values_images" not in inputs)

            inputs = image_processing(image_inputs, return_legacy_image_output=True, return_tensors="pt")
            self.assertTrue("pixel_values" not in inputs)
            self.assertTrue("pixel_values_images" in inputs)
