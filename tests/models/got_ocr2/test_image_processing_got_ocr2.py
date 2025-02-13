# coding=utf-8
# Copyright 2022 HuggingFace Inc.
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

from transformers.image_utils import SizeDict
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_torchvision_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from transformers import GotOcr2ImageProcessor

    if is_torchvision_available():
        from transformers import GotOcr2ImageProcessorFast


class GotOcr2ImageProcessingTester(unittest.TestCase):
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
        do_pad=False,
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
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
        self.do_pad = do_pad
        self.do_convert_rgb = do_convert_rgb

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_convert_rgb": self.do_convert_rgb,
            "do_pad": self.do_pad,
        }

    def expected_output_image_shape(self, images):
        return self.num_channels, self.size["height"], self.size["width"]

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
class GotOcr2ProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = GotOcr2ImageProcessor if is_vision_available() else None
    fast_image_processing_class = GotOcr2ImageProcessorFast if is_torchvision_available() else None

    def setUp(self):
        super().setUp()
        self.image_processor_tester = GotOcr2ImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class(**self.image_processor_dict)
            self.assertTrue(hasattr(image_processor, "do_resize"))
            self.assertTrue(hasattr(image_processor, "size"))
            self.assertTrue(hasattr(image_processor, "do_normalize"))
            self.assertTrue(hasattr(image_processor, "image_mean"))
            self.assertTrue(hasattr(image_processor, "image_std"))
            self.assertTrue(hasattr(image_processor, "do_convert_rgb"))

    def test_slow_fast_equivalence_crop_to_patches(self):
        dummy_image = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)[0]

        image_processor_slow = self.image_processing_class(**self.image_processor_dict, crop_to_patches=True)
        image_processor_fast = self.fast_image_processing_class(**self.image_processor_dict, crop_to_patches=True)

        encoding_slow = image_processor_slow(dummy_image, return_tensors="pt")
        encoding_fast = image_processor_fast(dummy_image, return_tensors="pt")
        self.assertTrue(torch.allclose(encoding_slow.pixel_values, encoding_fast.pixel_values, atol=1e-1))
        self.assertLessEqual(
            torch.mean(torch.abs(encoding_slow.pixel_values - encoding_fast.pixel_values)).item(), 1e-3
        )

    def test_slow_fast_equivalence_batched_crop_to_patches(self):
        # Prepare image inputs so that we have two groups of images with equal resolution with a group of images with
        # different resolutions in between
        dummy_images = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, torchify=True)
        dummy_images += self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)
        dummy_images += self.image_processor_tester.prepare_image_inputs(equal_resolution=True, torchify=True)

        image_processor_slow = self.image_processing_class(**self.image_processor_dict, crop_to_patches=True)
        image_processor_fast = self.fast_image_processing_class(**self.image_processor_dict, crop_to_patches=True)

        encoding_slow = image_processor_slow(dummy_images, return_tensors="pt")
        encoding_fast = image_processor_fast(dummy_images, return_tensors="pt")

        self.assertTrue(torch.allclose(encoding_slow.pixel_values, encoding_fast.pixel_values, atol=1e-1))
        self.assertLessEqual(
            torch.mean(torch.abs(encoding_slow.pixel_values - encoding_fast.pixel_values)).item(), 1e-3
        )

    def test_crop_to_patches(self):
        # test slow image processor
        image_processor = self.image_processor_list[0](**self.image_processor_dict)
        image = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, numpify=True)[0]
        processed_images = image_processor.crop_image_to_patches(
            image,
            min_patches=1,
            max_patches=6,
            use_thumbnail=True,
            patch_size={"height": 20, "width": 20},
        )
        self.assertEqual(len(processed_images), 5)
        self.assertEqual(processed_images[0].shape[:2], (20, 20))

        # test fast image processor (process batch)
        image_processor = self.image_processor_list[1](**self.image_processor_dict)
        image = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, torchify=True)[0]
        processed_images = image_processor.crop_image_to_patches(
            image.unsqueeze(0),
            min_patches=1,
            max_patches=6,
            use_thumbnail=True,
            patch_size=SizeDict(height=20, width=20),
        )
        self.assertEqual(len(processed_images[0]), 5)
        self.assertEqual(processed_images.shape[-2:], (20, 20))
