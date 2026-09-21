# Copyright 2024 HuggingFace Inc.
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

from transformers.image_utils import OPENAI_CLIP_MEAN
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTester, ImageProcessingTestMixin


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image


class JanusImageProcessingTester(ImageProcessingTester):
    image_size = 384
    max_resolution = 200

    # Image processor init kwargs
    size = {"height": 384, "width": 384}
    # Passing the mean explicitly also selects the padding background color.
    image_mean = OPENAI_CLIP_MEAN
    do_convert_rgb = True


@require_torch
@require_vision
class JanusImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_tester_class = JanusImageProcessingTester

    def test_call_pil(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True)
            for image in image_inputs:
                self.assertIsInstance(image, Image.Image)

            # Test Non batched input
            encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
            expected_output_image_shape = (1, 3, 384, 384)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

            # Test batched
            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
            expected_output_image_shape = (7, 3, 384, 384)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

    def test_call_numpy(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, numpify=True)
            for image in image_inputs:
                self.assertIsInstance(image, np.ndarray)

            encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
            expected_output_image_shape = (1, 3, 384, 384)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
            expected_output_image_shape = (7, 3, 384, 384)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

    def test_call_pytorch(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, torchify=True)

            for image in image_inputs:
                self.assertIsInstance(image, torch.Tensor)

            encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
            expected_output_image_shape = (1, 3, 384, 384)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
            expected_output_image_shape = (7, 3, 384, 384)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

    def test_nested_input(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True)

            # Test batched as a list of images.
            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
            expected_output_image_shape = (7, 3, 384, 384)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

            # Test batched as a nested list of images, where each sublist is one batch.
            image_inputs_nested = [image_inputs[:3], image_inputs[3:]]
            encoded_images_nested = image_processing(image_inputs_nested, return_tensors="pt").pixel_values
            expected_output_image_shape = (7, 3, 384, 384)
            self.assertEqual(tuple(encoded_images_nested.shape), expected_output_image_shape)

            # Image processor should return same pixel values, independently of input format.
            self.assertTrue((encoded_images_nested == encoded_images).all())

    @require_vision
    @require_torch
    def test_backends_equivalence_postprocess(self):
        dummy_images = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)
        dummy_images = [image / 255.0 for image in dummy_images]
        encodings = {}
        for backend_name, image_processing_class in self.image_processing_classes.items():
            image_processor = image_processing_class(**self.image_processor_dict)
            encodings[backend_name] = image_processor(dummy_images, return_tensors="pt")

        # Compare all backends to the first one (reference backend)
        backend_names = list(encodings.keys())
        reference_backend = backend_names[0]
        reference_encoding = encodings[reference_backend].pixel_values
        for backend_name in backend_names[1:]:
            self._assert_tensors_equivalence(reference_encoding, encodings[backend_name].pixel_values)

    @unittest.skip(reason="Not supported")
    def test_call_numpy_4_channels(self):
        pass
