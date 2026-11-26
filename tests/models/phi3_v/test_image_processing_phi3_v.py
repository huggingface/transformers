# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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


class Phi3VImageProcessingTester(ImageProcessingTester):
    def __init__(self, **kwargs):
        kwargs.setdefault("size", {"height": 28, "width": 28})
        kwargs.setdefault("num_crops", 1)
        super().__init__(**kwargs)


@require_torch
@require_vision
class Phi3VImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processor_tester_class = Phi3VImageProcessingTester

    def test_num_crops(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(**self.image_processor_dict)
            image = self.image_processor_tester.prepare_image_inputs(batch_size=1, equal_resolution=True)[0]
            image = image.resize((50, 100))
            # Portrait 1:2 with 4 crops: the long side is resized to 2 * 28, then 2x1 crops plus the global image
            encoding = image_processor(image, num_crops=4, return_tensors="pt")
            self.assertEqual(tuple(encoding.pixel_values.shape), (3, 3, 28, 28))
            self.assertEqual(encoding.image_sizes.tolist(), [[2, 1]])

    # Overwritten: the crops of all images are packed along the first dimension of `pixel_values`
    def _check_call(self, image_inputs, **kwargs):
        num_channels, crop_size = self.image_processor_tester.num_channels, self.image_processor_tester.size["height"]
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(**self.image_processor_dict)
            for images, num_images in ((image_inputs[0], 1), (image_inputs, len(image_inputs))):
                encoding = image_processor(images, return_tensors="pt", **kwargs)
                # With `num_crops=1` every image becomes the global image and one crop
                self.assertEqual(
                    tuple(encoding.pixel_values.shape), (2 * num_images, num_channels, crop_size, crop_size)
                )
                self.assertEqual(encoding.image_sizes.tolist(), [[1, 1]] * num_images)

    def test_call_pil(self):
        self._check_call(self.image_processor_tester.prepare_image_inputs(equal_resolution=False))

    def test_call_numpy(self):
        self._check_call(self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True))

    def test_call_pytorch(self):
        self._check_call(self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True))

    def test_call_numpy_4_channels(self):
        self.image_processor_tester.num_channels = 4
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)
        self._check_call(
            image_inputs,
            do_convert_rgb=False,
            input_data_format="channels_last",
            image_mean=[0.0] * 4,
            image_std=[1.0] * 4,
        )
