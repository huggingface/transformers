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
# applicable limitations under the License.
"""Testing suite for the emu3 image processor."""

import unittest

import numpy as np

from transformers.testing_utils import require_vision
from transformers.utils import is_vision_available


if is_vision_available():
    from PIL import Image

    from transformers.models.emu3.image_processing_emu3 import Emu3ImageProcessor


@require_vision
class Emu3ImageProcessorTest(unittest.TestCase):
    processor_class = Emu3ImageProcessor

    def _get_image_processor(self, **kwargs):
        # Small pixel budgets keep the dynamically resized outputs fast
        kwargs.setdefault("min_pixels", 28 * 28)
        kwargs.setdefault("max_pixels", 56 * 56)
        return self.processor_class(**kwargs)

    def test_call_numpy_float(self):
        # Float numpy images (pixel values in [0, 1] with `do_rescale=False`) are a supported
        # input type of image processors, see the `do_rescale` docstring
        image_processor = self._get_image_processor(do_rescale=False)
        image = np.random.rand(32, 32, 3).astype(np.float32)

        encoded = image_processor(images=[image, image], return_tensors=None)
        self.assertEqual(encoded["pixel_values"].ndim, 4)
        self.assertEqual(encoded["pixel_values"].shape[0], 2)
        self.assertEqual(encoded["pixel_values"].shape[1], 3)

    def test_call_numpy_float_4_channels(self):
        # Float RGBA numpy images are composited onto a white background
        image_processor = self._get_image_processor(do_rescale=False)
        image = np.random.rand(32, 32, 4).astype(np.float32)

        encoded = image_processor(images=[image], return_tensors=None)
        self.assertEqual(encoded["pixel_values"].ndim, 4)
        self.assertEqual(encoded["pixel_values"].shape[1], 3)

    def test_call_numpy_uint8(self):
        image_processor = self._get_image_processor()
        image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)

        encoded = image_processor(images=[image], return_tensors=None)
        self.assertEqual(encoded["pixel_values"].ndim, 4)
        self.assertEqual(encoded["pixel_values"].shape[0], 1)
        self.assertEqual(encoded["pixel_values"].shape[1], 3)

    def test_call_pil_rgba(self):
        # PIL RGBA images are composited onto a white background
        image_processor = self._get_image_processor()
        image = Image.new("RGBA", (32, 32), (255, 0, 0, 128))

        encoded = image_processor(images=[image], return_tensors=None)
        self.assertEqual(encoded["pixel_values"].ndim, 4)
        self.assertEqual(encoded["pixel_values"].shape[0], 1)
        self.assertEqual(encoded["pixel_values"].shape[1], 3)


if __name__ == "__main__":
    unittest.main()
