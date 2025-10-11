# coding=utf-8
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

from transformers.image_utils import load_image
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_torchvision_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs
from ...test_processing_common import url_to_local_path


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    from transformers import DeepseekVLHybridImageProcessor

    if is_torchvision_available():
        from transformers import DeepseekVLHybridImageProcessorFast


class DeepseekVLHybridImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        image_size=18,
        min_resolution=30,
        max_resolution=400,
        do_resize=True,
        size=None,
        high_res_size=None,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        high_res_image_mean=[0.5, 0.5, 0.5],
        high_res_image_std=[0.5, 0.5, 0.5],
    ):
        size = size if size is not None else {"height": 18, "width": 18}
        high_res_size = high_res_size if high_res_size is not None else {"height": 36, "width": 36}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.high_res_size = high_res_size
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.high_res_image_mean = high_res_image_mean
        self.high_res_image_std = high_res_image_std

    def prepare_image_processor_dict(self):
        return {
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "high_res_image_mean": self.high_res_image_mean,
            "high_res_image_std": self.high_res_image_std,
            "do_normalize": self.do_normalize,
            "do_resize": self.do_resize,
            "size": self.size,
            "high_res_size": self.high_res_size,
        }

    def expected_output_image_shape(self, images):
        max_size = max(self.size["height"], self.size["width"])
        return self.num_channels, max_size, max_size

    def expected_output_high_res_image_shape(self, images):
        max_size = max(self.high_res_size["height"], self.high_res_size["width"])
        return self.num_channels, max_size, max_size

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
class DeepseekVLHybridImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = DeepseekVLHybridImageProcessor if is_vision_available() else None
    fast_image_processing_class = DeepseekVLHybridImageProcessorFast if is_torchvision_available() else None

    # Copied from tests.models.vit.test_image_processing_vit.ViTImageProcessingTester.setUp with ViT->DeepseekVLHybrid
    def setUp(self):
        super().setUp()
        self.image_processor_tester = DeepseekVLHybridImageProcessingTester(self)

    @property
    # Copied from tests.models.vit.test_image_processing_vit.ViTImageProcessingTester.image_processor_dict with ViT->DeepseekVLHybrid
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    # Copied from tests.models.vit.test_image_processing_vit.ViTImageProcessingTester.test_image_processor_from_dict_with_kwargs
    def test_image_processor_from_dict_with_kwargs(self):
        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class.from_dict(self.image_processor_dict)
            self.assertEqual(image_processor.size, {"height": 18, "width": 18})

            image_processor = image_processing_class.from_dict(self.image_processor_dict, size=42)
            self.assertEqual(image_processor.size, {"height": 42, "width": 42})

    def test_image_processor_properties(self):
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            self.assertTrue(hasattr(image_processing, "image_mean"))
            self.assertTrue(hasattr(image_processing, "image_std"))
            self.assertTrue(hasattr(image_processing, "high_res_image_mean"))
            self.assertTrue(hasattr(image_processing, "high_res_image_std"))
            self.assertTrue(hasattr(image_processing, "do_normalize"))
            self.assertTrue(hasattr(image_processing, "do_resize"))
            self.assertTrue(hasattr(image_processing, "size"))
            self.assertTrue(hasattr(image_processing, "high_res_size"))

    def test_call_pil_high_res(self):
        for image_processing_class in self.image_processor_list:
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random PIL images
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
        for image_processing_class in self.image_processor_list:
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random numpy tensors
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
        for image_processing_class in self.image_processor_list:
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random PyTorch tensors
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

    @require_vision
    @require_torch
    def test_slow_fast_equivalence(self):
        if not self.test_slow_image_processor or not self.test_fast_image_processor:
            self.skipTest(reason="Skipping slow/fast equivalence test")

        if self.image_processing_class is None or self.fast_image_processing_class is None:
            self.skipTest(reason="Skipping slow/fast equivalence test as one of the image processors is not defined")

        dummy_image = load_image(url_to_local_path("http://images.cocodataset.org/val2017/000000039769.jpg"))
        image_processor_slow = self.image_processing_class(**self.image_processor_dict)
        image_processor_fast = self.fast_image_processing_class(**self.image_processor_dict)

        encoding_slow = image_processor_slow(dummy_image, return_tensors="pt")
        encoding_fast = image_processor_fast(dummy_image, return_tensors="pt")
        self._assert_slow_fast_tensors_equivalence(encoding_slow.pixel_values, encoding_fast.pixel_values)
        self._assert_slow_fast_tensors_equivalence(
            encoding_slow.high_res_pixel_values, encoding_fast.high_res_pixel_values
        )

    @require_vision
    @require_torch
    def test_slow_fast_equivalence_batched(self):
        if not self.test_slow_image_processor or not self.test_fast_image_processor:
            self.skipTest(reason="Skipping slow/fast equivalence test")

        if self.image_processing_class is None or self.fast_image_processing_class is None:
            self.skipTest(reason="Skipping slow/fast equivalence test as one of the image processors is not defined")

        if hasattr(self.image_processor_tester, "do_center_crop") and self.image_processor_tester.do_center_crop:
            self.skipTest(
                reason="Skipping as do_center_crop is True and center_crop functions are not equivalent for fast and slow processors"
            )

        dummy_images = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)
        image_processor_slow = self.image_processing_class(**self.image_processor_dict)
        image_processor_fast = self.fast_image_processing_class(**self.image_processor_dict)

        encoding_slow = image_processor_slow(dummy_images, return_tensors=None)
        encoding_fast = image_processor_fast(dummy_images, return_tensors=None)

        # Overwrite as the outputs are not always all of the same shape (kept for BC)
        for i in range(len(encoding_slow.pixel_values)):
            self._assert_slow_fast_tensors_equivalence(
                torch.from_numpy(encoding_slow.pixel_values[i]), encoding_fast.pixel_values[i]
            )
        for i in range(len(encoding_slow.high_res_pixel_values)):
            self._assert_slow_fast_tensors_equivalence(
                torch.from_numpy(encoding_slow.high_res_pixel_values[i]), encoding_fast.high_res_pixel_values[i]
            )

    @unittest.skip(reason="Not supported")
    def test_call_numpy_4_channels(self):
        pass
