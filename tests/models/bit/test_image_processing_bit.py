# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from transformers.utils import is_torchvision_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs



if is_vision_available():

    from transformers import BitImageProcessor

    if is_torchvision_available():
        from transformers import BitImageProcessorFast


class BitImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        image_size=18,
        min_resolution=30,
        max_resolution=400,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        size={"shortest_edge": 224},
    ):

        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.image_mean = image_mean
        self.image_std = image_std
        self.size = size

    def prepare_image_processor_dict(self):
        return {
            "image_mean": self.image_mean,
            "image_std": self.image_std,

        }

    def expected_output_image_shape(self, images):
        return self.num_channels, self.size["shortest_edge"], self.size["shortest_edge"]

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
class BitImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = BitImageProcessor if is_vision_available() else None
    fast_image_processing_class = BitImageProcessorFast if is_torchvision_available() else None

    def setUp(self):
        super().setUp()
        self.image_processor_tester = BitImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            self.assertTrue(hasattr(image_processing, "do_normalize"))
            self.assertTrue(hasattr(image_processing, "image_mean"))
            self.assertTrue(hasattr(image_processing, "image_std"))
            self.assertTrue(hasattr(image_processing, "do_resize"))
            self.assertTrue(hasattr(image_processing, "size"))

    def test_internal(self):
        self.test_slow_fast_equivalence_batched()

