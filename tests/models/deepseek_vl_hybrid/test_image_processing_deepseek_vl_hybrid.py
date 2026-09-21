# Copyright 2025 HuggingFace Inc.
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

from transformers.image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_image_processing_common import (
    ImageProcessingTester,
    ImageProcessingTestMixin,
    load_coco_image,
)


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image


class DeepseekVLHybridImageProcessingTester(ImageProcessingTester):
    # Image processor init kwargs
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    high_res_image_mean = IMAGENET_STANDARD_MEAN
    high_res_image_std = IMAGENET_STANDARD_STD
    size = {"height": 18, "width": 18}
    high_res_size = {"height": 36, "width": 36}

    def expected_output_image_shape(self, images):
        max_size = max(self.size["height"], self.size["width"])
        return self.num_channels, max_size, max_size

    def expected_output_high_res_image_shape(self, images):
        max_size = max(self.high_res_size["height"], self.high_res_size["width"])
        return self.num_channels, max_size, max_size


@require_torch
@require_vision
class DeepseekVLHybridImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_tester_class = DeepseekVLHybridImageProcessingTester

    def test_call_pil_high_res(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False)
            for image in image_inputs:
                self.assertIsInstance(image, Image.Image)

            # Test not batched input
            encoded_images = image_processing(image_inputs[0], return_tensors="pt").high_res_pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_high_res_image_shape(
                [image_inputs[0]]
            )
            self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))

            # Test batched
            encoded_images = image_processing(image_inputs, return_tensors="pt").high_res_pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_high_res_image_shape(
                image_inputs
            )
            self.assertEqual(
                tuple(encoded_images.shape), (self.image_processor_tester.batch_size, *expected_output_image_shape)
            )

    def test_call_numpy_high_res(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)
            for image in image_inputs:
                self.assertIsInstance(image, np.ndarray)

            # Test not batched input
            encoded_images = image_processing(image_inputs[0], return_tensors="pt").high_res_pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_high_res_image_shape(
                [image_inputs[0]]
            )
            self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))

            # Test batched
            encoded_images = image_processing(image_inputs, return_tensors="pt").high_res_pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_high_res_image_shape(
                image_inputs
            )
            self.assertEqual(
                tuple(encoded_images.shape), (self.image_processor_tester.batch_size, *expected_output_image_shape)
            )

    def test_call_pytorch_high_res(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)

            for image in image_inputs:
                self.assertIsInstance(image, torch.Tensor)

            # Test not batched input
            encoded_images = image_processing(image_inputs[0], return_tensors="pt").high_res_pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_high_res_image_shape(
                [image_inputs[0]]
            )
            self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))

            # Test batched
            expected_output_image_shape = self.image_processor_tester.expected_output_high_res_image_shape(
                image_inputs
            )
            encoded_images = image_processing(image_inputs, return_tensors="pt").high_res_pixel_values
            self.assertEqual(
                tuple(encoded_images.shape),
                (self.image_processor_tester.batch_size, *expected_output_image_shape),
            )

    def test_backends_equivalence(self):
        """Override to also compare high_res_pixel_values."""
        if len(self.image_processing_classes) < 2:
            self.skipTest(reason="Skipping backends equivalence test as there are less than 2 backends")

        dummy_image = load_coco_image("000000039769.jpg")

        encodings = {}
        for backend_name, image_processing_class in self.image_processing_classes.items():
            image_processor = image_processing_class(**self.image_processor_dict)
            encodings[backend_name] = image_processor(dummy_image, return_tensors="pt")

        backend_names = list(encodings.keys())
        reference_backend = backend_names[0]
        for backend_name in backend_names[1:]:
            self._assert_tensors_equivalence(
                encodings[reference_backend].pixel_values, encodings[backend_name].pixel_values
            )
            self._assert_tensors_equivalence(
                encodings[reference_backend].high_res_pixel_values,
                encodings[backend_name].high_res_pixel_values,
            )

    def test_backends_equivalence_batched(self):
        """Override to also compare high_res_pixel_values (variable shape - list of tensors)."""
        if len(self.image_processing_classes) < 2:
            self.skipTest(reason="Skipping backends equivalence test as there are less than 2 backends")

        dummy_images = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)

        encodings = {}
        for backend_name, image_processing_class in self.image_processing_classes.items():
            image_processor = image_processing_class(**self.image_processor_dict)
            encodings[backend_name] = image_processor(dummy_images, return_tensors=None)

        backend_names = list(encodings.keys())
        reference_backend = "pil"
        ref_pixel_values = encodings[reference_backend].pixel_values
        ref_high_res = encodings[reference_backend].high_res_pixel_values

        for backend_name in [backend_name for backend_name in backend_names if backend_name != reference_backend]:
            for i in range(len(ref_pixel_values)):
                self._assert_tensors_equivalence(
                    torch.from_numpy(ref_pixel_values[i]), encodings[backend_name].pixel_values[i]
                )
            for i in range(len(ref_high_res)):
                self._assert_tensors_equivalence(
                    torch.from_numpy(ref_high_res[i]), encodings[backend_name].high_res_pixel_values[i]
                )

    @unittest.skip(reason="Not supported")
    def test_call_numpy_4_channels(self):
        pass
