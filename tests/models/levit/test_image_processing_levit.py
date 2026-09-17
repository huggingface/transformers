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

from transformers.testing_utils import require_torch, require_vision

from ...test_image_processing_common import ImageProcessingTester, ImageProcessingTestMixin


class LevitImageProcessingTester(ImageProcessingTester):
    def __init__(self, **kwargs):
        kwargs.setdefault("image_mean", [0.5, 0.5, 0.5])
        kwargs.setdefault("image_std", [0.5, 0.5, 0.5])
        kwargs.setdefault("do_normalize", True)
        kwargs.setdefault("do_resize", True)
        kwargs.setdefault("do_center_crop", True)
        kwargs.setdefault("size", {"shortest_edge": 18})
        kwargs.setdefault("crop_size", {"height": 18, "width": 18})
        super().__init__(**kwargs)


@require_torch
@require_vision
class LevitImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_tester_class = LevitImageProcessingTester

    @property
    def image_processor_dict(self):
        return self.image_processing_tester.prepare_image_processor_dict()

    def test_image_processor_from_dict_with_kwargs(self):
        for image_processing_class in self.image_processor_classes.values():
            image_processor = image_processing_class.from_dict(self.image_processor_dict)
            self.assertEqual(image_processor.size, {"shortest_edge": 18})
            self.assertEqual(image_processor.crop_size, {"height": 18, "width": 18})

            image_processor = image_processing_class.from_dict(self.image_processor_dict, size=42, crop_size=84)
            self.assertEqual(image_processor.size, {"shortest_edge": 42})
            self.assertEqual(image_processor.crop_size, {"height": 84, "width": 84})
