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

from transformers import InklingImageProcessor
from transformers.image_utils import PILImageResampling
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image


class InklingImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        min_resolution=30,
        max_resolution=400,
        do_resize=True,
        do_normalize=False,
        image_mean=None,
        image_std=None,
        do_convert_rgb=True,
        size=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else [0.0, 0.0, 0.0]
        self.image_std = image_std if image_std is not None else [1.0, 1.0, 1.0]
        self.do_convert_rgb = do_convert_rgb
        self.size = size if size is not None else {"height": 40, "width": 40}

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_convert_rgb": self.do_convert_rgb,
            "size": self.size,
        }

    def prepare_image_inputs(self, equal_resolution=False, numpify=False, torchify=False):
        return prepare_image_inputs(
            batch_size=self.batch_size,
            num_channels=self.num_channels,
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
            equal_resolution=equal_resolution,
            numpify=numpify,
            torchify=torchify,
        )


@require_torch
@require_vision
class InklingImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    def setUp(self):
        self.image_processing_classes = {"torchvision": InklingImageProcessor}
        self.image_processor_tester = InklingImageProcessingTester(self)

    @unittest.skip("Inkling patchification requires RGB (3-channel) images; 4-channel inputs are unsupported.")
    def test_call_numpy_4_channels(self):
        pass

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            self.assertTrue(hasattr(image_processing, "do_resize"))
            self.assertTrue(hasattr(image_processing, "do_normalize"))
            self.assertTrue(hasattr(image_processing, "image_mean"))
            self.assertTrue(hasattr(image_processing, "image_std"))
            self.assertTrue(hasattr(image_processing, "do_convert_rgb"))
            self.assertTrue(hasattr(image_processing, "size"))

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

    def test_image_processor_from_dict_with_kwargs(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class.from_dict(self.image_processor_dict)
            self.assertEqual(image_processor.size, {"height": 40, "width": 40})

            image_processor = image_processing_class.from_dict(
                self.image_processor_dict, size={"height": 16, "width": 16}
            )
            self.assertEqual(image_processor.size, {"height": 16, "width": 16})

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
