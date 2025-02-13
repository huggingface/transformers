# coding=utf-8
# Copyright 2022 Meta Platforms authors and HuggingFace Inc.
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

import random
import unittest

import numpy as np

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

if is_vision_available():
    import PIL

    from transformers import FlavaImageProcessor
    from transformers.image_utils import PILImageResampling
    from transformers.models.flava.image_processing_flava import (
        FLAVA_CODEBOOK_MEAN,
        FLAVA_CODEBOOK_STD,
        FLAVA_IMAGE_MEAN,
        FLAVA_IMAGE_STD,
    )
else:
    FLAVA_IMAGE_MEAN = FLAVA_IMAGE_STD = FLAVA_CODEBOOK_MEAN = FLAVA_CODEBOOK_STD = None


class FlavaImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        min_resolution=30,
        max_resolution=400,
        do_resize=True,
        size=None,
        do_center_crop=True,
        crop_size=None,
        resample=None,
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        image_mean=FLAVA_IMAGE_MEAN,
        image_std=FLAVA_IMAGE_STD,
        input_size_patches=14,
        total_mask_patches=75,
        mask_group_max_patches=None,
        mask_group_min_patches=16,
        mask_group_min_aspect_ratio=0.3,
        mask_group_max_aspect_ratio=None,
        codebook_do_resize=True,
        codebook_size=None,
        codebook_resample=None,
        codebook_do_center_crop=True,
        codebook_crop_size=None,
        codebook_do_map_pixels=True,
        codebook_do_normalize=True,
        codebook_image_mean=FLAVA_CODEBOOK_MEAN,
        codebook_image_std=FLAVA_CODEBOOK_STD,
    ):
        size = size if size is not None else {"height": 224, "width": 224}
        crop_size = crop_size if crop_size is not None else {"height": 224, "width": 224}
        codebook_size = codebook_size if codebook_size is not None else {"height": 112, "width": 112}
        codebook_crop_size = codebook_crop_size if codebook_crop_size is not None else {"height": 112, "width": 112}

        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.do_resize = do_resize
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.size = size
        self.resample = resample if resample is not None else PILImageResampling.BICUBIC
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size

        self.input_size_patches = input_size_patches
        self.total_mask_patches = total_mask_patches
        self.mask_group_max_patches = mask_group_max_patches
        self.mask_group_min_patches = mask_group_min_patches
        self.mask_group_min_aspect_ratio = mask_group_min_aspect_ratio
        self.mask_group_max_aspect_ratio = mask_group_max_aspect_ratio

        self.codebook_do_resize = codebook_do_resize
        self.codebook_size = codebook_size
        self.codebook_resample = codebook_resample if codebook_resample is not None else PILImageResampling.LANCZOS
        self.codebook_do_center_crop = codebook_do_center_crop
        self.codebook_crop_size = codebook_crop_size
        self.codebook_do_map_pixels = codebook_do_map_pixels
        self.codebook_do_normalize = codebook_do_normalize
        self.codebook_image_mean = codebook_image_mean
        self.codebook_image_std = codebook_image_std

    def prepare_image_processor_dict(self):
        return {
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_normalize": self.do_normalize,
            "do_resize": self.do_resize,
            "size": self.size,
            "resample": self.resample,
            "do_rescale": self.do_rescale,
            "rescale_factor": self.rescale_factor,
            "do_center_crop": self.do_center_crop,
            "crop_size": self.crop_size,
            "input_size_patches": self.input_size_patches,
            "total_mask_patches": self.total_mask_patches,
            "mask_group_max_patches": self.mask_group_max_patches,
            "mask_group_min_patches": self.mask_group_min_patches,
            "mask_group_min_aspect_ratio": self.mask_group_min_aspect_ratio,
            "mask_group_max_aspect_ratio": self.mask_group_min_aspect_ratio,
            "codebook_do_resize": self.codebook_do_resize,
            "codebook_size": self.codebook_size,
            "codebook_resample": self.codebook_resample,
            "codebook_do_center_crop": self.codebook_do_center_crop,
            "codebook_crop_size": self.codebook_crop_size,
            "codebook_do_map_pixels": self.codebook_do_map_pixels,
            "codebook_do_normalize": self.codebook_do_normalize,
            "codebook_image_mean": self.codebook_image_mean,
            "codebook_image_std": self.codebook_image_std,
        }

    def get_expected_image_size(self):
        return (self.size["height"], self.size["width"])

    def get_expected_mask_size(self):
        return (
            (self.input_size_patches, self.input_size_patches)
            if not isinstance(self.input_size_patches, tuple)
            else self.input_size_patches
        )

    def get_expected_codebook_image_size(self):
        return (self.codebook_size["height"], self.codebook_size["width"])

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
class FlavaImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = FlavaImageProcessor if is_vision_available() else None
    maxDiff = None

    def setUp(self):
        super().setUp()
        self.image_processor_tester = FlavaImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        image_processing = self.image_processing_class(**self.image_processor_dict)
        self.assertTrue(hasattr(image_processing, "image_mean"))
        self.assertTrue(hasattr(image_processing, "image_std"))
        self.assertTrue(hasattr(image_processing, "do_normalize"))
        self.assertTrue(hasattr(image_processing, "do_resize"))
        self.assertTrue(hasattr(image_processing, "resample"))
        self.assertTrue(hasattr(image_processing, "crop_size"))
        self.assertTrue(hasattr(image_processing, "do_center_crop"))
        self.assertTrue(hasattr(image_processing, "do_rescale"))
        self.assertTrue(hasattr(image_processing, "rescale_factor"))
        self.assertTrue(hasattr(image_processing, "masking_generator"))
        self.assertTrue(hasattr(image_processing, "codebook_do_resize"))
        self.assertTrue(hasattr(image_processing, "codebook_size"))
        self.assertTrue(hasattr(image_processing, "codebook_resample"))
        self.assertTrue(hasattr(image_processing, "codebook_do_center_crop"))
        self.assertTrue(hasattr(image_processing, "codebook_crop_size"))
        self.assertTrue(hasattr(image_processing, "codebook_do_map_pixels"))
        self.assertTrue(hasattr(image_processing, "codebook_do_normalize"))
        self.assertTrue(hasattr(image_processing, "codebook_image_mean"))
        self.assertTrue(hasattr(image_processing, "codebook_image_std"))

    def test_image_processor_from_dict_with_kwargs(self):
        image_processor = self.image_processing_class.from_dict(self.image_processor_dict)
        self.assertEqual(image_processor.size, {"height": 224, "width": 224})
        self.assertEqual(image_processor.crop_size, {"height": 224, "width": 224})
        self.assertEqual(image_processor.codebook_size, {"height": 112, "width": 112})
        self.assertEqual(image_processor.codebook_crop_size, {"height": 112, "width": 112})

        image_processor = self.image_processing_class.from_dict(
            self.image_processor_dict, size=42, crop_size=84, codebook_size=33, codebook_crop_size=66
        )
        self.assertEqual(image_processor.size, {"height": 42, "width": 42})
        self.assertEqual(image_processor.crop_size, {"height": 84, "width": 84})
        self.assertEqual(image_processor.codebook_size, {"height": 33, "width": 33})
        self.assertEqual(image_processor.codebook_crop_size, {"height": 66, "width": 66})

    def test_call_pil(self):
        # Initialize image_processing
        image_processing = self.image_processing_class(**self.image_processor_dict)
        # create random PIL images
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False)
        for image in image_inputs:
            self.assertIsInstance(image, PIL.Image.Image)

        # Test not batched input
        encoded_images = image_processing(image_inputs[0], return_tensors="pt")

        # Test no bool masked pos
        self.assertFalse("bool_masked_pos" in encoded_images)

        expected_height, expected_width = self.image_processor_tester.get_expected_image_size()

        self.assertEqual(
            encoded_images.pixel_values.shape,
            (1, self.image_processor_tester.num_channels, expected_height, expected_width),
        )

        # Test batched
        encoded_images = image_processing(image_inputs, return_tensors="pt")
        expected_height, expected_width = self.image_processor_tester.get_expected_image_size()

        # Test no bool masked pos
        self.assertFalse("bool_masked_pos" in encoded_images)

        self.assertEqual(
            encoded_images.pixel_values.shape,
            (
                self.image_processor_tester.batch_size,
                self.image_processor_tester.num_channels,
                expected_height,
                expected_width,
            ),
        )

    def _test_call_framework(self, instance_class, prepare_kwargs):
        # Initialize image_processing
        image_processing = self.image_processing_class(**self.image_processor_dict)
        # create random tensors
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, **prepare_kwargs)
        for image in image_inputs:
            self.assertIsInstance(image, instance_class)

        # Test not batched input
        encoded_images = image_processing(image_inputs[0], return_tensors="pt")

        expected_height, expected_width = self.image_processor_tester.get_expected_image_size()
        self.assertEqual(
            encoded_images.pixel_values.shape,
            (1, self.image_processor_tester.num_channels, expected_height, expected_width),
        )

        encoded_images = image_processing(image_inputs, return_image_mask=True, return_tensors="pt")

        expected_height, expected_width = self.image_processor_tester.get_expected_image_size()
        self.assertEqual(
            encoded_images.pixel_values.shape,
            (
                self.image_processor_tester.batch_size,
                self.image_processor_tester.num_channels,
                expected_height,
                expected_width,
            ),
        )

        expected_height, expected_width = self.image_processor_tester.get_expected_mask_size()
        self.assertEqual(
            encoded_images.bool_masked_pos.shape,
            (
                self.image_processor_tester.batch_size,
                expected_height,
                expected_width,
            ),
        )

        # Test batched
        encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values

        expected_height, expected_width = self.image_processor_tester.get_expected_image_size()
        self.assertEqual(
            encoded_images.shape,
            (
                self.image_processor_tester.batch_size,
                self.image_processor_tester.num_channels,
                expected_height,
                expected_width,
            ),
        )

        # Test masking
        encoded_images = image_processing(image_inputs, return_image_mask=True, return_tensors="pt")

        expected_height, expected_width = self.image_processor_tester.get_expected_image_size()
        self.assertEqual(
            encoded_images.pixel_values.shape,
            (
                self.image_processor_tester.batch_size,
                self.image_processor_tester.num_channels,
                expected_height,
                expected_width,
            ),
        )

        expected_height, expected_width = self.image_processor_tester.get_expected_mask_size()
        self.assertEqual(
            encoded_images.bool_masked_pos.shape,
            (
                self.image_processor_tester.batch_size,
                expected_height,
                expected_width,
            ),
        )

    def test_call_numpy(self):
        self._test_call_framework(np.ndarray, prepare_kwargs={"numpify": True})

    def test_call_numpy_4_channels(self):
        self.image_processing_class.num_channels = 4
        self._test_call_framework(np.ndarray, prepare_kwargs={"numpify": True})
        self.image_processing_class.num_channels = 3

    def test_call_pytorch(self):
        self._test_call_framework(torch.Tensor, prepare_kwargs={"torchify": True})

    def test_masking(self):
        # Initialize image_processing
        random.seed(1234)
        image_processing = self.image_processing_class(**self.image_processor_dict)
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)

        # Test not batched input
        encoded_images = image_processing(image_inputs[0], return_image_mask=True, return_tensors="pt")
        self.assertEqual(encoded_images.bool_masked_pos.sum().item(), 75)

    def test_codebook_pixels(self):
        # Initialize image_processing
        image_processing = self.image_processing_class(**self.image_processor_dict)
        # create random PIL images
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False)
        for image in image_inputs:
            self.assertIsInstance(image, PIL.Image.Image)

        # Test not batched input
        encoded_images = image_processing(image_inputs[0], return_codebook_pixels=True, return_tensors="pt")
        expected_height, expected_width = self.image_processor_tester.get_expected_codebook_image_size()
        self.assertEqual(
            encoded_images.codebook_pixel_values.shape,
            (1, self.image_processor_tester.num_channels, expected_height, expected_width),
        )

        # Test batched
        encoded_images = image_processing(image_inputs, return_codebook_pixels=True, return_tensors="pt")
        expected_height, expected_width = self.image_processor_tester.get_expected_codebook_image_size()
        self.assertEqual(
            encoded_images.codebook_pixel_values.shape,
            (
                self.image_processor_tester.batch_size,
                self.image_processor_tester.num_channels,
                expected_height,
                expected_width,
            ),
        )
