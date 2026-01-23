# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
from transformers.testing_utils import require_torch, require_torchvision, require_vision
from transformers.utils import is_torch_available, is_torchvision_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_vision_available() and is_torchvision_available():
    from transformers import Molmo2ImageProcessor


class Molmo2ImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        image_size=378,
        min_resolution=30,
        max_resolution=400,
        do_resize=True,
        size=None,
        do_normalize=True,
        image_mean=IMAGENET_STANDARD_MEAN,
        image_std=IMAGENET_STANDARD_STD,
        do_convert_rgb=True,
        max_crops=8,
        overlap_margins=[4, 4],
        patch_size=14,
        pooling_size=[2, 2],
    ):
        size = size if size is not None else {"height": 378, "width": 378}
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
        self.max_crops = max_crops
        self.overlap_margins = overlap_margins
        self.patch_size = patch_size
        self.pooling_size = pooling_size

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_convert_rgb": self.do_convert_rgb,
            "max_crops": self.max_crops,
            "overlap_margins": self.overlap_margins,
            "patch_size": self.patch_size,
            "pooling_size": self.pooling_size,
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
@require_torchvision
class Molmo2ImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = Molmo2ImageProcessor if (is_vision_available() and is_torchvision_available()) else None

    def setUp(self):
        self.image_processor_tester = Molmo2ImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        image_processor = self.image_processing_class(**self.image_processor_dict)
        self.assertTrue(hasattr(image_processor, "do_resize"))
        self.assertTrue(hasattr(image_processor, "size"))
        self.assertTrue(hasattr(image_processor, "do_normalize"))
        self.assertTrue(hasattr(image_processor, "image_mean"))
        self.assertTrue(hasattr(image_processor, "image_std"))
        self.assertTrue(hasattr(image_processor, "do_convert_rgb"))
        self.assertTrue(hasattr(image_processor, "max_crops"))
        self.assertTrue(hasattr(image_processor, "overlap_margins"))
        self.assertTrue(hasattr(image_processor, "patch_size"))
        self.assertTrue(hasattr(image_processor, "pooling_size"))

    def test_image_processor_from_dict_with_kwargs(self):
        image_processor = self.image_processing_class.from_dict(self.image_processor_dict)
        self.assertEqual(image_processor.size, {"height": 378, "width": 378})
        self.assertEqual(image_processor.do_normalize, True)

        image_processor = self.image_processing_class.from_dict(
            self.image_processor_dict, size={"height": 400, "width": 400}, do_normalize=False
        )
        self.assertEqual(image_processor.size, {"height": 400, "width": 400})
        self.assertEqual(image_processor.do_normalize, False)
