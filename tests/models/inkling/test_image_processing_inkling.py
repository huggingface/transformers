# Copyright 2026 the HuggingFace Team. All rights reserved.
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

from transformers.image_utils import PILImageResampling
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

from ...test_image_processing_common import ImageProcessingTester, ImageProcessingTestMixin


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image


class InklingImageProcessingTester(ImageProcessingTester):
    # Image processor init kwargs
    size = {"height": 40, "width": 40}
    do_resize = True
    do_normalize = False


@require_torch
@require_vision
class InklingImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_tester_class = InklingImageProcessingTester

    @unittest.skip("Inkling patchification requires RGB (3-channel) images; 4-channel inputs are unsupported.")
    def test_call_numpy_4_channels(self):
        pass

    def test_image_processor_defaults(self):
        for image_processing_class in self.image_processing_classes.values():
            proc = image_processing_class()
            self.assertEqual(proc.size["height"], 40)
            self.assertEqual(proc.size["width"], 40)
            self.assertTrue(proc.do_normalize)
            self.assertTrue(proc.do_convert_rgb)
            self.assertEqual(list(proc.image_mean), list(OPENAI_CLIP_MEAN))
            self.assertEqual(list(proc.image_std), list(OPENAI_CLIP_STD))
            self.assertEqual(proc.resample, PILImageResampling.LANCZOS)

    def test_output_keys(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            image = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
            result = image_processing(image, return_tensors="pt")
            self.assertEqual(set(result.keys()), {"pixel_values", "num_patches"})

    def _check_packed_output(self, encoding, num_images):
        """Inkling packs every image's patches into one (sum(num_patches), 2, H, W, 3) tensor."""
        size = self.image_processor_tester.size
        pixel_values = encoding.pixel_values
        num_patches = encoding.num_patches
        self.assertEqual(pixel_values.dtype, torch.float32)
        self.assertEqual(pixel_values.ndim, 5)
        self.assertEqual(tuple(pixel_values.shape[1:]), (2, size["height"], size["width"], 3))
        self.assertEqual(len(num_patches), num_images)
        self.assertEqual(pixel_values.shape[0], int(num_patches.sum()))

    def test_call_pil(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False)
            for image in image_inputs:
                self.assertIsInstance(image, Image.Image)

            self._check_packed_output(image_processing(image_inputs[0], return_tensors="pt"), 1)
            self._check_packed_output(
                image_processing(image_inputs, return_tensors="pt"), self.image_processor_tester.batch_size
            )

    def test_call_numpy(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)
            for image in image_inputs:
                self.assertIsInstance(image, np.ndarray)

            self._check_packed_output(image_processing(image_inputs[0], return_tensors="pt"), 1)
            self._check_packed_output(
                image_processing(image_inputs, return_tensors="pt"), self.image_processor_tester.batch_size
            )

    def test_call_pytorch(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)
            for image in image_inputs:
                self.assertIsInstance(image, torch.Tensor)

            self._check_packed_output(image_processing(image_inputs[0], return_tensors="pt"), 1)
            self._check_packed_output(
                image_processing(image_inputs, return_tensors="pt"), self.image_processor_tester.batch_size
            )
