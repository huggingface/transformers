# Copyright 2025 The HuggingFace Team. All rights reserved.
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
    if is_torchvision_available():
        from transformers import Phi3VImageProcessorFast


class Phi3VImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        image_size=18,
        do_resize=True,
        min_resolution=30,
        max_resolution=400,
        size=None,
        do_normalize=True,
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
        do_convert_rgb=True,
        num_crops=2,
    ):
        size = size if size is not None else {"height": 336, "width": 336}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.do_resize = do_resize
        self.image_size = image_size
        self.size = size
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_convert_rgb = do_convert_rgb
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.num_crops = num_crops

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_convert_rgb": self.do_convert_rgb,
            "num_crops": self.num_crops,
        }

    def prepare_image_inputs(self, equal_resolution=False, numpify=False, torchify=False):
        return prepare_image_inputs(
            batch_size=self.batch_size,
            num_channels=self.num_channels,
            equal_resolution=equal_resolution,
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
            numpify=numpify,
            torchify=torchify,
        )

    def expected_output_image_shape(self, images):
        height, width = self.size["height"], self.size["width"]
        return self.num_crops + 1, self.num_channels, height, width


@require_torch
@require_vision
class Phi3VImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    fast_image_processing_class = Phi3VImageProcessorFast if is_torchvision_available() else None
    test_slow_image_processor = False

    def setUp(self):
        super().setUp()
        self.image_processor_tester = Phi3VImageProcessingTester(self)

    @property
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
            self.assertTrue(hasattr(image_processing, "num_crops"))

    def test_image_processor_from_dict_with_kwargs(self):
        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class.from_dict(self.image_processor_dict)
            self.assertEqual(image_processor.size, {"height": 336, "width": 336})
            self.assertEqual(image_processor.image_mean, [0.48145466, 0.4578275, 0.40821073])

            image_processor = image_processing_class.from_dict(
                self.image_processor_dict, size=42, image_mean=[1.0, 2.0, 1.0]
            )
            self.assertEqual(image_processor.size, {"height": 42, "width": 42})
            self.assertEqual(image_processor.image_mean, [1.0, 2.0, 1.0])

    @unittest.skip(reason="Not supported; Model only supports RGB images")
    def test_call_numpy_4_channels(self):
        pass
