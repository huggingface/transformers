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


class Llama4ImageProcessingTester(ImageProcessingTester):
    def __init__(self, parent, **kwargs):
        kwargs.setdefault("max_patches", 1)
        kwargs.setdefault("do_resize", True)
        kwargs.setdefault("size", {"height": 20, "width": 20})
        kwargs.setdefault("do_normalize", True)
        kwargs.setdefault("image_mean", [0.5, 0.5, 0.5])
        kwargs.setdefault("image_std", [0.5, 0.5, 0.5])
        kwargs.setdefault("do_convert_rgb", True)
        kwargs.setdefault("do_pad", False)
        super().__init__(parent, **kwargs)


@require_torch
@require_vision
class Llama4ImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.image_processor_tester = Llama4ImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(**self.image_processor_dict)
            self.assertTrue(hasattr(image_processor, "do_resize"))
            self.assertTrue(hasattr(image_processor, "size"))
            self.assertTrue(hasattr(image_processor, "do_normalize"))
            self.assertTrue(hasattr(image_processor, "image_mean"))
            self.assertTrue(hasattr(image_processor, "image_std"))
            self.assertTrue(hasattr(image_processor, "do_convert_rgb"))

    def test_split_tiles(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(**self.image_processor_dict)
            image = self.image_processor_tester.prepare_image_inputs(equal_resolution=True)[0]
            processed_images = image_processor(
                image,
                max_patches=16,
            )
            self.assertEqual(len(processed_images.pixel_values), 1)
            self.assertEqual(processed_images.pixel_values[0].shape[0], 17)
            self.assertEqual(processed_images.pixel_values[0].shape[-2:], (20, 20))
