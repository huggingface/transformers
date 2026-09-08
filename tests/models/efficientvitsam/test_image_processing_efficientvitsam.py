# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from transformers.testing_utils import require_torchvision, require_vision
from transformers.utils import is_vision_available

from ...test_image_processing_common import ImageProcessingTester, ImageProcessingTestMixin


if is_vision_available():
    from PIL import Image


class EfficientViTSamImageProcessingTester(ImageProcessingTester):
    def __init__(self, parent):
        self.parent = parent
        self.batch_size = 2
        self.num_channels = 3
        self.min_resolution = 30
        self.max_resolution = 40
        self.do_resize = True
        self.size = {"longest_edge": 32}
        self.do_normalize = True
        self.image_mean = [0.5, 0.5, 0.5]
        self.image_std = [0.5, 0.5, 0.5]
        self.do_pad = True
        self.pad_size = {"height": 32, "width": 32}
        self.mask_size = {"longest_edge": 16}
        self.mask_pad_size = {"height": 16, "width": 16}

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_pad": self.do_pad,
            "pad_size": self.pad_size,
            "mask_size": self.mask_size,
            "mask_pad_size": self.mask_pad_size,
        }

    def expected_output_image_shape(self, images):
        return self.num_channels, self.pad_size["height"], self.pad_size["width"]


@require_vision
@require_torchvision
class EfficientViTSamImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.image_processor_tester = EfficientViTSamImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(**self.image_processor_dict)
            self.assertEqual(image_processor.size, {"longest_edge": 32})
            self.assertEqual(image_processor.pad_size, {"height": 32, "width": 32})

    def test_image_processor_call(self):
        image_processor = next(iter(self.image_processing_classes.values()))(**self.image_processor_dict)
        image = Image.new("RGB", (40, 30))
        encoding = image_processor(image, return_tensors="pt")
        self.assertEqual(tuple(encoding.pixel_values.shape), (1, 3, 32, 32))
