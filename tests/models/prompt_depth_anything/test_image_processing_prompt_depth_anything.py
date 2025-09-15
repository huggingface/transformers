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

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torchvision_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_vision_available():
    from transformers import PromptDepthAnythingImageProcessor

    if is_torchvision_available():
        from transformers import PromptDepthAnythingImageProcessorFast


class PromptDepthAnythingImageProcessingTester(unittest.TestCase):
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
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    ):
        super().__init__()
        size = size if size is not None else {"height": 18, "width": 18}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std

    def prepare_image_processor_dict(self):
        return {
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_normalize": self.do_normalize,
            "do_resize": self.do_resize,
            "size": self.size,
        }

    def expected_output_image_shape(self, images):
        return self.num_channels, self.size["height"], self.size["width"]

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
class PromptDepthAnythingImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = PromptDepthAnythingImageProcessor if is_vision_available() else None
    fast_image_processing_class = PromptDepthAnythingImageProcessorFast if is_torchvision_available() else None

    def setUp(self):
        super().setUp()
        self.image_processor_tester = PromptDepthAnythingImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            self.assertTrue(hasattr(image_processing, "image_mean"))
            self.assertTrue(hasattr(image_processing, "image_std"))
            self.assertTrue(hasattr(image_processing, "do_normalize"))
            self.assertTrue(hasattr(image_processing, "do_resize"))
            self.assertTrue(hasattr(image_processing, "size"))
            self.assertTrue(hasattr(image_processing, "do_rescale"))
            self.assertTrue(hasattr(image_processing, "rescale_factor"))
            self.assertTrue(hasattr(image_processing, "do_pad"))
            self.assertTrue(hasattr(image_processing, "size_divisor"))
            self.assertTrue(hasattr(image_processing, "prompt_scale_to_meter"))

    def test_image_processor_from_dict_with_kwargs(self):
        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class.from_dict(self.image_processor_dict)
            self.assertEqual(image_processor.size, {"height": 18, "width": 18})

            image_processor = image_processing_class.from_dict(self.image_processor_dict, size=42)
            self.assertEqual(image_processor.size, {"height": 42, "width": 42})

    def test_keep_aspect_ratio(self):
        size = {"height": 512, "width": 512}
        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class(size=size, keep_aspect_ratio=True, ensure_multiple_of=32)

            image = np.zeros((489, 640, 3))

            pixel_values = image_processor(image, return_tensors="pt").pixel_values

            self.assertEqual(list(pixel_values.shape), [1, 3, 512, 672])

    def test_prompt_depth_processing(self):
        size = {"height": 756, "width": 756}
        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class(size=size, keep_aspect_ratio=True, ensure_multiple_of=32)

            image = np.zeros((756, 1008, 3))
            prompt_depth = np.random.random((192, 256))

            outputs = image_processor(image, prompt_depth=prompt_depth, return_tensors="pt")
            pixel_values = outputs.pixel_values
            prompt_depth_values = outputs.prompt_depth

            self.assertEqual(list(pixel_values.shape), [1, 3, 768, 1024])
            self.assertEqual(list(prompt_depth_values.shape), [1, 1, 192, 256])

    @require_torch
    @require_vision
    def test_slow_fast_equivalence(self):
        if not self.test_slow_image_processor or not self.test_fast_image_processor:
            self.skipTest(reason="Skipping slow/fast equivalence test")

        if self.image_processing_class is None or self.fast_image_processing_class is None:
            self.skipTest(reason="Skipping slow/fast equivalence test as one of the image processors is not defined")

        image = np.zeros((756, 1008, 3))
        prompt_depth = np.random.random((192, 256))

        size = {"height": 756, "width": 756}
        image_processor_slow = self.image_processing_class(
            size=size, keep_aspect_ratio=True, ensure_multiple_of=32, do_pad=True, size_divisor=51
        )
        image_processor_fast = self.fast_image_processing_class(
            size=size, keep_aspect_ratio=True, ensure_multiple_of=32, do_pad=True, size_divisor=51
        )

        encoding_slow = image_processor_slow(image, prompt_depth=prompt_depth, return_tensors="pt")
        encoding_fast = image_processor_fast(image, prompt_depth=prompt_depth, return_tensors="pt")

        self._assert_slow_fast_tensors_equivalence(encoding_slow.pixel_values, encoding_fast.pixel_values)
        self.assertEqual(encoding_slow.prompt_depth.dtype, encoding_fast.prompt_depth.dtype)

        self._assert_slow_fast_tensors_equivalence(encoding_slow.prompt_depth, encoding_fast.prompt_depth)

    @require_torch
    @require_vision
    def test_slow_fast_equivalence_batched(self):
        if not self.test_slow_image_processor or not self.test_fast_image_processor:
            self.skipTest(reason="Skipping slow/fast equivalence test")

        if self.image_processing_class is None or self.fast_image_processing_class is None:
            self.skipTest(reason="Skipping slow/fast equivalence test as one of the image processors is not defined")

        if hasattr(self.image_processor_tester, "do_center_crop") and self.image_processor_tester.do_center_crop:
            self.skipTest(
                reason="Skipping as do_center_crop is True and center_crop functions are not equivalent for fast and slow processors"
            )

        batch_size = self.image_processor_tester.batch_size
        images = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, torchify=True)
        prompt_depths = [np.random.random((192, 256)) for _ in range(batch_size)]

        size = {"height": 756, "width": 756}
        image_processor_slow = self.image_processing_class(size=size, keep_aspect_ratio=False, ensure_multiple_of=32)
        image_processor_fast = self.fast_image_processing_class(
            size=size, keep_aspect_ratio=False, ensure_multiple_of=32
        )

        encoding_slow = image_processor_slow(images, prompt_depth=prompt_depths, return_tensors="pt")
        encoding_fast = image_processor_fast(images, prompt_depth=prompt_depths, return_tensors="pt")

        self._assert_slow_fast_tensors_equivalence(encoding_slow.pixel_values, encoding_fast.pixel_values)
        self.assertEqual(encoding_slow.prompt_depth.dtype, encoding_fast.prompt_depth.dtype)

        self._assert_slow_fast_tensors_equivalence(encoding_slow.prompt_depth, encoding_fast.prompt_depth)
