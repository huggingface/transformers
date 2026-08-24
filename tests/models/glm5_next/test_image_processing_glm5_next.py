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

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_torchvision_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

if is_torchvision_available():
    from transformers.models.glm5_next.image_processing_glm5_next import smart_resize

if is_vision_available():
    from PIL import Image


class Glm5NextImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        num_channels=3,
        min_resolution=30,
        max_resolution=80,
        do_rescale=True,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        temporal_patch_size=2,
        patch_size=14,
        merge_size=2,
        patch_expand_factor=1,  # We only expect 1s atp, if this changes the implementation also needs to change
        min_image_tokens=1,
        max_image_tokens=64,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_rescale = do_rescale
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.temporal_patch_size = temporal_patch_size
        self.patch_size = patch_size
        self.merge_size = merge_size
        self.patch_expand_factor = patch_expand_factor
        self.min_image_tokens = min_image_tokens
        self.max_image_tokens = max_image_tokens

    def prepare_image_processor_dict(self):
        return {
            "do_rescale": self.do_rescale,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "temporal_patch_size": self.temporal_patch_size,
            "patch_size": self.patch_size,
            "merge_size": self.merge_size,
            "patch_expand_factor": self.patch_expand_factor,
            "min_image_tokens": self.min_image_tokens,
            "max_image_tokens": self.max_image_tokens,
        }

    def expected_output_image_shape(self, images):
        hidden_dim = self.num_channels * self.temporal_patch_size * self.patch_size**2
        factor = self.patch_size * self.merge_size * self.patch_expand_factor
        seq_len = 0
        for image in images:
            if isinstance(image, Image.Image):
                width, height = image.size
            elif isinstance(image, np.ndarray):
                height, width = image.shape[:2]
            else:
                height, width = image.shape[-2:]
            resized_height, resized_width = smart_resize(
                self.temporal_patch_size,
                height,
                width,
                temporal_factor=self.temporal_patch_size,
                factor=factor,
                min_pixels=self.min_image_tokens,
                max_pixels=self.max_image_tokens,
            )
            seq_len += (resized_height // self.patch_size) * (resized_width // self.patch_size)
        return (
            seq_len,
            hidden_dim,
        )

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


@require_torch
@require_vision
class Glm5NextImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.image_processor_tester = Glm5NextImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            self.assertTrue(hasattr(image_processing, "image_mean"))
            self.assertTrue(hasattr(image_processing, "image_std"))
            self.assertTrue(hasattr(image_processing, "do_normalize"))
            self.assertTrue(hasattr(image_processing, "do_resize"))
            self.assertTrue(hasattr(image_processing, "min_image_tokens"))
            self.assertTrue(hasattr(image_processing, "max_image_tokens"))

    def test_image_processor_from_dict_with_kwargs(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class.from_dict(self.image_processor_dict)
            self.assertEqual(image_processor.min_image_tokens, 1)

            image_processor = image_processing_class.from_dict(self.image_processor_dict, min_image_tokens=42)
            self.assertEqual(image_processor.min_image_tokens, 42)

    # batch size is flattened
    def test_call_pil(self):
        for image_processing_class in self.image_processing_classes.values():
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)

            # Create random PIL images
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False)
            for image in image_inputs:
                self.assertIsInstance(image, Image.Image)

            # Test not batched input
            encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

            # Test batched
            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

    def test_call_numpy(self):
        for image_processing_class in self.image_processing_classes.values():
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)

            # Create random NumPy arrays
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)
            for image in image_inputs:
                self.assertIsInstance(image, np.ndarray)

            # Test not batched input
            encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

            # Test batched
            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

    def test_call_pytorch(self):
        for image_processing_class in self.image_processing_classes.values():
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)

            # Create random PyTorch tensors
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)
            for image in image_inputs:
                self.assertIsInstance(image, torch.Tensor)

            # Test not batched input
            encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

            # Test batched
            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

    def test_call_numpy_4_channels(self):
        for image_processing_class in self.image_processing_classes.values():
            # Test that images with an arbitrary number of channels can be processed
            self.image_processor_tester.num_channels = 4

            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)

            # Create random NumPy arrays
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)

            # Test not batched input
            encoded_images = image_processing(
                image_inputs[0],
                return_tensors="pt",
                input_data_format="channels_last",
                image_mean=(0.0, 0.0, 0.0, 0.0),
                image_std=(1.0, 1.0, 1.0, 1.0),
            ).pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

            # Test batched
            encoded_images = image_processing(
                image_inputs,
                return_tensors="pt",
                input_data_format="channels_last",
                image_mean=(0.0, 0.0, 0.0, 0.0),
                image_std=(1.0, 1.0, 1.0, 1.0),
            ).pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

            # Test that normalization is applied independently to each channel
            image_processing = image_processing_class(
                **self.image_processor_dict,
                do_convert_rgb=False,
                do_resize=False,
            )
            image = np.zeros((28, 28, 4), dtype=np.uint8)
            for channel, value in enumerate((255, 128, 51, 0)):
                image[..., channel] = value

            output = image_processing(
                image,
                input_data_format="channels_last",
                image_mean=(0.0, 0.0, 0.0, 0.0),
                image_std=(1.0, 2.0, 4.0, 8.0),
                return_tensors="pt",
            ).pixel_values

            patch_dim = self.image_processor_tester.temporal_patch_size * self.image_processor_tester.patch_size**2
            for channel, expected in enumerate((1.0, 128 / 255 / 2, 51 / 255 / 4, 0.0)):
                block = output[:, channel * patch_dim : (channel + 1) * patch_dim]
                torch.testing.assert_close(block, torch.full_like(block, expected))
