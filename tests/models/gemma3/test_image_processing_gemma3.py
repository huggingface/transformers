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
from transformers.utils import is_torch_available, is_torchvision_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    from transformers import Gemma3ImageProcessor

    if is_torchvision_available():
        from transformers import Gemma3ImageProcessorFast


class Gemma3ImageProcessingTester:
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
        do_normalize=True,
        image_mean=IMAGENET_STANDARD_MEAN,
        image_std=IMAGENET_STANDARD_STD,
        do_convert_rgb=True,
        do_pan_and_scan=True,
        pan_and_scan_min_crop_size=10,
        pan_and_scan_max_num_crops=2,
        pan_and_scan_min_ratio_to_activate=1.2,
    ):
        super().__init__()
        size = size if size is not None else {"height": 18, "width": 18}
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
        self.do_pan_and_scan = do_pan_and_scan
        self.pan_and_scan_min_crop_size = pan_and_scan_min_crop_size
        self.pan_and_scan_max_num_crops = pan_and_scan_max_num_crops
        self.pan_and_scan_min_ratio_to_activate = pan_and_scan_min_ratio_to_activate

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_convert_rgb": self.do_convert_rgb,
            "do_pan_and_scan": self.do_pan_and_scan,
            "pan_and_scan_min_crop_size": self.pan_and_scan_min_crop_size,
            "pan_and_scan_max_num_crops": self.pan_and_scan_max_num_crops,
            "pan_and_scan_min_ratio_to_activate": self.pan_and_scan_min_ratio_to_activate,
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
class Gemma3ImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = Gemma3ImageProcessor if is_vision_available() else None
    fast_image_processing_class = Gemma3ImageProcessorFast if is_torchvision_available() else None

    # Copied from tests.models.clip.test_image_processing_clip.CLIPImageProcessingTest.setUp with CLIP->Gemma3
    def setUp(self):
        super().setUp()
        self.image_processor_tester = Gemma3ImageProcessingTester(self)

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
            self.assertTrue(hasattr(image_processing, "do_pan_and_scan"))
            self.assertTrue(hasattr(image_processing, "pan_and_scan_min_crop_size"))
            self.assertTrue(hasattr(image_processing, "pan_and_scan_max_num_crops"))
            self.assertTrue(hasattr(image_processing, "pan_and_scan_min_ratio_to_activate"))

    def test_image_processor_from_dict_with_kwargs(self):
        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class.from_dict(self.image_processor_dict)
            self.assertEqual(image_processor.size, {"height": 18, "width": 18})

            image_processor = image_processing_class.from_dict(self.image_processor_dict, size=84)
            self.assertEqual(image_processor.size, {"height": 84, "width": 84})

    def test_without_pan_and_scan(self):
        """
        Disable do_pan_and_scan parameter.
        """
        for image_processing_class in self.image_processor_list:
            # Initialize image_processing
            image_processor = image_processing_class.from_dict(self.image_processor_dict, do_pan_and_scan=False)

            # create random PIL images
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True)
            for image in image_inputs:
                self.assertIsInstance(image, Image.Image)

            # Test not batched input
            encoded_images = image_processor(image_inputs[0], return_tensors="pt").pixel_values
            expected_output_image_shape = (1, 3, 18, 18)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

            # Test batched
            encoded_images = image_processor(image_inputs, return_tensors="pt").pixel_values
            expected_output_image_shape = (7, 3, 18, 18)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

    def test_pan_and_scan(self):
        """
        Enables Pan and Scan path by choosing the correct input image resolution. If you are changing
        image processor attributes for PaS, please update this test.
        """
        for image_processing_class in self.image_processor_list:
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random numpy tensors
            """This function prepares a list of PIL images"""
            image_inputs = [np.random.randint(255, size=(3, 300, 600), dtype=np.uint8)] * 3
            image_inputs = [Image.fromarray(np.moveaxis(x, 0, -1)) for x in image_inputs]

            # Test not batched input, 3 images because we have base image + 2 crops
            encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
            expected_output_image_shape = (3, 3, 18, 18)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

            # Test batched, 9 images because we have base image + 2 crops per each item
            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
            expected_output_image_shape = (9, 3, 18, 18)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

            # Test batched unbalanced, 9 images because we have base image + 2 crops per each item
            encoded_images = image_processing(
                [[image_inputs[0], image_inputs[1]], [image_inputs[2]]], return_tensors="pt"
            ).pixel_values
            expected_output_image_shape = (9, 3, 18, 18)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

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
            expected_output_image_shape = (1, 3, 18, 18)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

            # Test batched
            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
            expected_output_image_shape = (7, 3, 18, 18)
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
            expected_output_image_shape = (1, 3, 18, 18)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

            # Test batched
            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
            expected_output_image_shape = (7, 3, 18, 18)
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
            expected_output_image_shape = (1, 3, 18, 18)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

            # Test batched
            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
            expected_output_image_shape = (7, 3, 18, 18)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

    @unittest.skip("Gemma3 doesn't work with 4 channels due to pan and scan method")
    def test_call_numpy_4_channels(self):
        pass

    @require_vision
    @require_torch
    def test_slow_fast_equivalence_batched_pas(self):
        if not self.test_slow_image_processor or not self.test_fast_image_processor:
            self.skipTest(reason="Skipping slow/fast equivalence test")

        if self.image_processing_class is None or self.fast_image_processing_class is None:
            self.skipTest(reason="Skipping slow/fast equivalence test as one of the image processors is not defined")

        if hasattr(self.image_processor_tester, "do_center_crop") and self.image_processor_tester.do_center_crop:
            self.skipTest(
                reason="Skipping as do_center_crop is True and center_crop functions are not equivalent for fast and slow processors"
            )
        crop_config = {
            "do_pan_and_scan": True,
            "pan_and_scan_max_num_crops": 448,
            "pan_and_scan_min_crop_size": 32,
            "pan_and_scan_min_ratio_to_activate": 0.3,
        }
        image_processor_dict = self.image_processor_dict
        image_processor_dict.update(crop_config)
        dummy_images = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)
        image_processor_slow = self.image_processing_class(**image_processor_dict)
        image_processor_fast = self.fast_image_processing_class(**image_processor_dict)

        encoding_slow = image_processor_slow(dummy_images, return_tensors="pt")
        encoding_fast = image_processor_fast(dummy_images, return_tensors="pt")

        torch.testing.assert_close(encoding_slow.num_crops, encoding_fast.num_crops)
        self._assert_slow_fast_tensors_equivalence(encoding_slow.pixel_values, encoding_fast.pixel_values)
