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
    # Image processor init kwargs
    max_patches = 1
    do_resize = True
    size = {"height": 20, "width": 20}
    do_normalize = True
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    do_convert_rgb = True
    do_pad = False


@require_torch
@require_vision
class Llama4ImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_tester_class = Llama4ImageProcessingTester

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
