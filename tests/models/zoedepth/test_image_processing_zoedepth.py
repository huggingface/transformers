# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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

from transformers.file_utils import is_vision_available
from transformers.testing_utils import require_torch, require_vision

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_vision_available():
    from transformers import ZoeDepthImageProcessor


class ZoeDepthImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        image_size=18,
        min_resolution=30,
        max_resolution=400,
        do_resize=True,
        size=None,
        ensure_multiple_of=32,
        keep_aspect_ratio=False,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        do_pad=False,
    ):
        size = size if size is not None else {"height": 18, "width": 18}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.ensure_multiple_of = ensure_multiple_of
        self.keep_aspect_ratio = keep_aspect_ratio
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_pad = do_pad

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "ensure_multiple_of": self.ensure_multiple_of,
            "keep_aspect_ratio": self.keep_aspect_ratio,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_pad": self.do_pad,
        }

    def expected_output_image_shape(self, images):
        return self.num_channels, self.ensure_multiple_of, self.ensure_multiple_of

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
class ZoeDepthImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = ZoeDepthImageProcessor if is_vision_available() else None

    def setUp(self):
        super().setUp()

        self.image_processor_tester = ZoeDepthImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        image_processing = self.image_processing_class(**self.image_processor_dict)
        self.assertTrue(hasattr(image_processing, "image_mean"))
        self.assertTrue(hasattr(image_processing, "image_std"))
        self.assertTrue(hasattr(image_processing, "do_normalize"))
        self.assertTrue(hasattr(image_processing, "do_resize"))
        self.assertTrue(hasattr(image_processing, "size"))
        self.assertTrue(hasattr(image_processing, "ensure_multiple_of"))
        self.assertTrue(hasattr(image_processing, "do_rescale"))
        self.assertTrue(hasattr(image_processing, "rescale_factor"))
        self.assertTrue(hasattr(image_processing, "do_pad"))

    def test_image_processor_from_dict_with_kwargs(self):
        image_processor = self.image_processing_class.from_dict(self.image_processor_dict)
        self.assertEqual(image_processor.size, {"height": 18, "width": 18})

        image_processor = self.image_processing_class.from_dict(self.image_processor_dict, size=42)
        self.assertEqual(image_processor.size, {"height": 42, "width": 42})

    def test_ensure_multiple_of(self):
        # Test variable by turning off all other variables which affect the size, size which is not multiple of 32
        image = np.zeros((489, 640, 3))

        size = {"height": 380, "width": 513}
        multiple = 32
        image_processor = ZoeDepthImageProcessor(
            do_pad=False, ensure_multiple_of=multiple, size=size, keep_aspect_ratio=False
        )
        pixel_values = image_processor(image, return_tensors="pt").pixel_values

        self.assertEqual(list(pixel_values.shape), [1, 3, 384, 512])
        self.assertTrue(pixel_values.shape[2] % multiple == 0)
        self.assertTrue(pixel_values.shape[3] % multiple == 0)

        # Test variable by turning off all other variables which affect the size, size which is already multiple of 32
        image = np.zeros((511, 511, 3))

        height, width = 512, 512
        size = {"height": height, "width": width}
        multiple = 32
        image_processor = ZoeDepthImageProcessor(
            do_pad=False, ensure_multiple_of=multiple, size=size, keep_aspect_ratio=False
        )
        pixel_values = image_processor(image, return_tensors="pt").pixel_values

        self.assertEqual(list(pixel_values.shape), [1, 3, height, width])
        self.assertTrue(pixel_values.shape[2] % multiple == 0)
        self.assertTrue(pixel_values.shape[3] % multiple == 0)

    def test_keep_aspect_ratio(self):
        # Test `keep_aspect_ratio=True` by turning off all other variables which affect the size
        height, width = 489, 640
        image = np.zeros((height, width, 3))

        size = {"height": 512, "width": 512}
        image_processor = ZoeDepthImageProcessor(do_pad=False, keep_aspect_ratio=True, size=size, ensure_multiple_of=1)
        pixel_values = image_processor(image, return_tensors="pt").pixel_values

        # As can be seen, the image is resized to the maximum size that fits in the specified size
        self.assertEqual(list(pixel_values.shape), [1, 3, 512, 670])

        # Test `keep_aspect_ratio=False` by turning off all other variables which affect the size
        image_processor = ZoeDepthImageProcessor(
            do_pad=False, keep_aspect_ratio=False, size=size, ensure_multiple_of=1
        )
        pixel_values = image_processor(image, return_tensors="pt").pixel_values

        # As can be seen, the size is respected
        self.assertEqual(list(pixel_values.shape), [1, 3, size["height"], size["width"]])

        # Test `keep_aspect_ratio=True` with `ensure_multiple_of` set
        image = np.zeros((489, 640, 3))

        size = {"height": 511, "width": 511}
        multiple = 32
        image_processor = ZoeDepthImageProcessor(size=size, keep_aspect_ratio=True, ensure_multiple_of=multiple)

        pixel_values = image_processor(image, return_tensors="pt").pixel_values

        self.assertEqual(list(pixel_values.shape), [1, 3, 512, 672])
        self.assertTrue(pixel_values.shape[2] % multiple == 0)
        self.assertTrue(pixel_values.shape[3] % multiple == 0)
