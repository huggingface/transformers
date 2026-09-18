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

import numpy as np

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_vision_available

from ...test_image_processing_common import ImageProcessingTester, ImageProcessingTestMixin


if is_vision_available():
    from PIL import Image


class ViltImageProcessingTester(ImageProcessingTester):
    def __init__(self, parent, **kwargs):
        kwargs.setdefault("image_mean", [0.5, 0.5, 0.5])
        kwargs.setdefault("image_std", [0.5, 0.5, 0.5])
        kwargs.setdefault("do_normalize", True)
        kwargs.setdefault("do_resize", True)
        kwargs.setdefault("size", {"shortest_edge": 30})
        kwargs.setdefault("size_divisor", 2)
        super().__init__(parent, **kwargs)

    def get_expected_values(self, image_inputs, batched=False):
        """
        This function computes the expected height and width when providing images to ViltImageProcessor,
        assuming do_resize is set to True with a scalar size and size_divisor.
        """
        if not batched:
            size = self.size["shortest_edge"]
            image = image_inputs[0]
            if isinstance(image, Image.Image):
                w, h = image.size
            elif isinstance(image, np.ndarray):
                h, w = image.shape[0], image.shape[1]
            else:
                h, w = image.shape[1], image.shape[2]
            scale = size / min(w, h)
            if h < w:
                newh, neww = size, scale * w
            else:
                newh, neww = scale * h, size

            max_size = int((1333 / 800) * size)
            if max(newh, neww) > max_size:
                scale = max_size / max(newh, neww)
                newh = newh * scale
                neww = neww * scale

            newh, neww = int(newh + 0.5), int(neww + 0.5)
            expected_height, expected_width = (
                newh // self.size_divisor * self.size_divisor,
                neww // self.size_divisor * self.size_divisor,
            )

        else:
            expected_values = []
            for image in image_inputs:
                expected_height, expected_width = self.get_expected_values([image])
                expected_values.append((expected_height, expected_width))
            expected_height = max(expected_values, key=lambda item: item[0])[0]
            expected_width = max(expected_values, key=lambda item: item[1])[1]

        return expected_height, expected_width

    def expected_output_image_shape(self, images):
        height, width = self.get_expected_values(images, batched=True)
        return (self.num_channels, height, width)


@require_torch
@require_vision
class ViltImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.image_processor_tester = ViltImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            self.assertTrue(hasattr(image_processing, "image_mean"))
            self.assertTrue(hasattr(image_processing, "image_std"))
            self.assertTrue(hasattr(image_processing, "do_normalize"))
            self.assertTrue(hasattr(image_processing, "do_resize"))
            self.assertTrue(hasattr(image_processing, "size"))
            self.assertTrue(hasattr(image_processing, "size_divisor"))
            self.assertTrue(hasattr(image_processing, "do_pad"))
            self.assertTrue(hasattr(image_processing, "resample"))
            self.assertTrue(hasattr(image_processing, "do_rescale"))
            self.assertTrue(hasattr(image_processing, "model_input_names"))

    def test_image_processor_from_dict_with_kwargs(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class.from_dict(self.image_processor_dict)
            self.assertEqual(image_processor.size, {"shortest_edge": 30})

            image_processor = image_processing_class.from_dict(self.image_processor_dict, size=42)
            self.assertEqual(image_processor.size, {"shortest_edge": 42})
