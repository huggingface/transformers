# Copyright 2021 HuggingFace Inc.
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


class CLIPImageProcessingTester(ImageProcessingTester):
    def __init__(self, parent, **kwargs):
        kwargs.setdefault("do_resize", True)
        kwargs.setdefault("size", {"shortest_edge": 20})
        kwargs.setdefault("do_center_crop", True)
        kwargs.setdefault("crop_size", {"height": 18, "width": 18})
        kwargs.setdefault("do_normalize", True)
        kwargs.setdefault("image_mean", [0.48145466, 0.4578275, 0.40821073])
        kwargs.setdefault("image_std", [0.26862954, 0.26130258, 0.27577711])
        kwargs.setdefault("do_convert_rgb", True)
        super().__init__(parent, **kwargs)


@require_torch
@require_vision
class CLIPImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.image_processor_tester = CLIPImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            self.assertTrue(hasattr(image_processing, "do_resize"))
            self.assertTrue(hasattr(image_processing, "size"))
            self.assertTrue(hasattr(image_processing, "do_center_crop"))
            self.assertTrue(hasattr(image_processing, "crop_size"))
            self.assertTrue(hasattr(image_processing, "do_normalize"))
            self.assertTrue(hasattr(image_processing, "image_mean"))
            self.assertTrue(hasattr(image_processing, "image_std"))
            self.assertTrue(hasattr(image_processing, "do_convert_rgb"))

    def test_image_processor_from_dict_with_kwargs(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class.from_dict(self.image_processor_dict)
            self.assertEqual(image_processor.size, {"shortest_edge": 20})
            self.assertEqual(image_processor.crop_size, {"height": 18, "width": 18})

            image_processor = image_processing_class.from_dict(self.image_processor_dict, size=42, crop_size=84)
            self.assertEqual(image_processor.size, {"shortest_edge": 42})
            self.assertEqual(image_processor.crop_size, {"height": 84, "width": 84})
