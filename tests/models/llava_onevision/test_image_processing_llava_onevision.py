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
import pytest

from transformers.image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD, ChannelDimension
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_torchvision_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    from transformers import LlavaOnevisionImageProcessor

    if is_torchvision_available():
        from transformers import LlavaOnevisionImageProcessorFast


class LlavaOnevisionImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        image_size=20,
        min_resolution=30,
        max_resolution=400,
        do_resize=True,
        size=None,
        do_normalize=True,
        image_mean=OPENAI_CLIP_MEAN,
        image_std=OPENAI_CLIP_STD,
        do_convert_rgb=True,
    ):
        super().__init__()
        size = size if size is not None else {"height": 20, "width": 20}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_convert_rgb = do_convert_rgb

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_convert_rgb": self.do_convert_rgb,
        }

    def expected_output_image_shape(self, images):
        return self.num_channels, self.size["height"], self.size["width"]

    # Copied from tests.models.clip.test_image_processing_clip.CLIPImageProcessingTester.prepare_image_inputs
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
class LlavaOnevisionImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = LlavaOnevisionImageProcessor if is_vision_available() else None
    fast_image_processing_class = LlavaOnevisionImageProcessorFast if is_torchvision_available() else None

    # Copied from tests.models.clip.test_image_processing_clip.CLIPImageProcessingTest.setUp with CLIP->LlavaOnevision
    def setUp(self):
        super().setUp()
        self.image_processor_tester = LlavaOnevisionImageProcessingTester(self)

    @property
    # Copied from tests.models.clip.test_image_processing_clip.CLIPImageProcessingTest.image_processor_dict
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            self.assertTrue(hasattr(image_processing, "do_resize"))
            self.assertTrue(hasattr(image_processing, "size"))
            self.assertTrue(hasattr(image_processing, "do_normalize"))
            self.assertTrue(hasattr(image_processing, "image_mean"))
            self.assertTrue(hasattr(image_processing, "image_std"))
            self.assertTrue(hasattr(image_processing, "do_convert_rgb"))
            self.assertTrue(hasattr(image_processing, "image_grid_pinpoints"))

    def test_image_processor_from_dict_with_kwargs(self):
        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class.from_dict(self.image_processor_dict)
            self.assertEqual(image_processor.size, {"height": 20, "width": 20})

            image_processor = image_processing_class.from_dict(self.image_processor_dict, size=42)
            self.assertEqual(image_processor.size, {"shortest_edge": 42})

    def test_call_pil(self):
        for image_processing_class in self.image_processor_list:
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random PIL images
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True)
            for image in image_inputs:
                self.assertIsInstance(image, Image.Image)

            # Test not batched input
            encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
            expected_output_image_shape = (1, 1522, 3, 20, 20)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

            # Test batched
            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
            expected_output_image_shape = (7, 1522, 3, 20, 20)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

    def test_call_numpy(self):
        for image_processing_class in self.image_processor_list:
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random numpy tensors
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, numpify=True)
            for image in image_inputs:
                self.assertIsInstance(image, np.ndarray)

            # Test not batched input
            encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
            expected_output_image_shape = (1, 1522, 3, 20, 20)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

            # Test batched
            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
            expected_output_image_shape = (7, 1522, 3, 20, 20)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

    def test_call_pytorch(self):
        for image_processing_class in self.image_processor_list:
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random PyTorch tensors
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, torchify=True)

            for image in image_inputs:
                self.assertIsInstance(image, torch.Tensor)

            # Test not batched input
            encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
            expected_output_image_shape = (1, 1522, 3, 20, 20)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

            # Test batched
            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
            expected_output_image_shape = (7, 1522, 3, 20, 20)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

    @unittest.skip(
        reason="LlavaOnevisionImageProcessor doesn't treat 4 channel PIL and numpy consistently yet"
    )  # FIXME raushan
    def test_call_numpy_4_channels(self):
        pass

    def test_nested_input(self):
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True)

            # Test batched as a list of images
            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
            expected_output_image_shape = (7, 1522, 3, 20, 20)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

            # Test batched as a nested list of images, where each sublist is one batch
            image_inputs_nested = [[image_input] for image_input in image_inputs]
            encoded_images_nested = image_processing(image_inputs_nested, return_tensors="pt").pixel_values
            expected_output_image_shape = (7, 1522, 3, 20, 20)
            self.assertEqual(tuple(encoded_images_nested.shape), expected_output_image_shape)

            # Image processor should return same pixel values, independently of input format
            self.assertTrue((encoded_images_nested == encoded_images).all())

    def test_multi_images(self):
        length = 384
        scale_single, scale_multi = 2, 3
        image_processor_dict = self.image_processor_tester.prepare_image_processor_dict()
        image_processor_dict["size"] = {"height": length, "width": length}  # patch size
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**image_processor_dict)

            # Test batched as a nested list of images, where each sublist is one batch
            len_image_1 = length * scale_single
            image_inputs_1 = prepare_image_inputs(
                batch_size=1,
                min_resolution=0,  # not used
                max_resolution=len_image_1,
                num_channels=3,
                equal_resolution=True,
            )
            len_image_2 = length * scale_multi
            image_inputs_2 = prepare_image_inputs(
                batch_size=7,
                min_resolution=0,  # not used
                max_resolution=len_image_2,
                num_channels=3,
                equal_resolution=True,
            )
            image_inputs = [image_inputs_1, image_inputs_2]

            # Only single image should be patchified
            expected_num_patches = scale_single**2 + 1  # +1 for base image patch
            expected_output_image_shape = (8, expected_num_patches, 3, length, length)
            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

    @unittest.skip(
        reason="LlavaOnevisionImageProcessorFast doesn't compile (infinitely) when using class transforms"
    )  # FIXME yoni
    @pytest.mark.torch_compile_test
    def test_can_compile_fast_image_processor(self):
        pass

    def test_pad_for_patching(self):
        for image_processing_class in self.image_processor_list:
            if image_processing_class == self.fast_image_processing_class:
                numpify = False
                torchify = True
                input_data_format = image_processing_class.data_format
            else:
                numpify = True
                torchify = False
                input_data_format = ChannelDimension.LAST
            image_processing = image_processing_class(**self.image_processor_dict)
            # Create odd-sized images
            image_input = self.image_processor_tester.prepare_image_inputs(
                equal_resolution=True,
                numpify=numpify,
                torchify=torchify,
            )[0]
            self.assertIn(image_input.shape, [(3, 400, 400), (400, 400, 3)])

            # Test odd-width
            image_shape = (400, 601)
            encoded_images = image_processing._pad_for_patching(image_input, image_shape, input_data_format)
            encoded_image_shape = (
                encoded_images.shape[:-1] if input_data_format == ChannelDimension.LAST else encoded_images.shape[1:]
            )
            self.assertEqual(encoded_image_shape, image_shape)

            # Test odd-height
            image_shape = (503, 400)
            encoded_images = image_processing._pad_for_patching(image_input, image_shape, input_data_format)
            encoded_image_shape = (
                encoded_images.shape[:-1] if input_data_format == ChannelDimension.LAST else encoded_images.shape[1:]
            )
            self.assertEqual(encoded_image_shape, image_shape)

    def test_call_without_padding(self):
        for image_processing_class in self.image_processor_list:
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random PyTorch tensors
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True)

            # Test not batched input
            encoded_images = image_processing(image_inputs[0], do_pad=False).pixel_values
            self.assertEqual(len(encoded_images), 1)

            # Test batched
            encoded_images = image_processing(image_inputs, do_pad=False).pixel_values
            self.assertEqual(len(encoded_images), len(image_inputs))
