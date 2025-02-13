# coding=utf-8
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

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torchvision_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_vision_available():
    from PIL import Image

    from transformers import Siglip2ImageProcessor


if is_torchvision_available():
    from transformers import Siglip2ImageProcessorFast


class Siglip2ImageProcessingTester:
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
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        resample=None,
        patch_size=16,
        max_num_patches=256,
    ):
        size = size if size is not None else {"height": 18, "width": 18}
        resample = resample if resample is not None else Image.Resampling.BILINEAR
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.resample = resample
        self.patch_size = patch_size
        self.max_num_patches = max_num_patches

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "do_rescale": self.do_rescale,
            "rescale_factor": self.rescale_factor,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "resample": self.resample,
            "patch_size": self.patch_size,
            "max_num_patches": self.max_num_patches,
        }

    def expected_output_image_shape(self, images):
        return self.max_num_patches, self.patch_size * self.patch_size * self.num_channels

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
# Copied from tests.models.clip.test_image_processing_clip.CLIPImageProcessingTest with CLIP->Siglip2
class Siglip2ImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = Siglip2ImageProcessor if is_vision_available() else None
    fast_image_processing_class = Siglip2ImageProcessorFast if is_torchvision_available() else None

    def setUp(self):
        super().setUp()
        self.image_processor_tester = Siglip2ImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    # Ignore copy
    def test_image_processor_properties(self):
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            self.assertTrue(hasattr(image_processing, "do_resize"))
            self.assertTrue(hasattr(image_processing, "resample"))
            self.assertTrue(hasattr(image_processing, "do_rescale"))
            self.assertTrue(hasattr(image_processing, "rescale_factor"))
            self.assertTrue(hasattr(image_processing, "do_normalize"))
            self.assertTrue(hasattr(image_processing, "image_mean"))
            self.assertTrue(hasattr(image_processing, "image_std"))
            self.assertTrue(hasattr(image_processing, "patch_size"))
            self.assertTrue(hasattr(image_processing, "max_num_patches"))

    # Ignore copy
    def test_image_processor_from_dict_with_kwargs(self):
        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class.from_dict(self.image_processor_dict)
            self.assertEqual(image_processor.max_num_patches, 256)
            self.assertEqual(image_processor.patch_size, 16)

            image_processor = self.image_processing_class.from_dict(
                self.image_processor_dict, patch_size=32, max_num_patches=512
            )
            self.assertEqual(image_processor.patch_size, 32)
            self.assertEqual(image_processor.max_num_patches, 512)

    @unittest.skip(reason="not supported")
    # Ignore copy
    def test_call_numpy_4_channels(self):
        pass
