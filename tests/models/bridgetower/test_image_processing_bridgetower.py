# Copyright 2023 The Intel Labs Team Authors, The Microsoft Research Team Authors and HuggingFace Inc. team. All rights reserved.
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
from typing import Optional, Union

import requests

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torchvision_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_vision_available():
    from PIL import Image

    from transformers import BridgeTowerImageProcessor

    if is_torchvision_available():
        from transformers import BridgeTowerImageProcessorFast


class BridgeTowerImageProcessingTester:
    def __init__(
        self,
        parent,
        do_resize: bool = True,
        size: Optional[dict[str, int]] = None,
        size_divisor: int = 32,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        do_center_crop: bool = True,
        image_mean: Optional[Union[float, list[float]]] = [0.48145466, 0.4578275, 0.40821073],
        image_std: Optional[Union[float, list[float]]] = [0.26862954, 0.26130258, 0.27577711],
        do_pad: bool = True,
        batch_size=7,
        min_resolution=30,
        max_resolution=400,
        num_channels=3,
    ):
        self.parent = parent
        self.do_resize = do_resize
        self.size = size if size is not None else {"shortest_edge": 288}
        self.size_divisor = size_divisor
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.do_center_crop = do_center_crop
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_pad = do_pad
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution

    def prepare_image_processor_dict(self):
        return {
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_normalize": self.do_normalize,
            "do_resize": self.do_resize,
            "size": self.size,
            "size_divisor": self.size_divisor,
        }

    def get_expected_values(self, image_inputs, batched=False):
        return self.size["shortest_edge"], self.size["shortest_edge"]

    def expected_output_image_shape(self, images):
        height, width = self.get_expected_values(images, batched=True)
        return self.num_channels, height, width

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
class BridgeTowerImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = BridgeTowerImageProcessor if is_vision_available() else None
    fast_image_processing_class = BridgeTowerImageProcessorFast if is_torchvision_available() else None

    def setUp(self):
        super().setUp()
        self.image_processor_tester = BridgeTowerImageProcessingTester(self)

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
            self.assertTrue(hasattr(image_processing, "size_divisor"))

    @require_vision
    @require_torch
    def test_slow_fast_equivalence(self):
        if not self.test_slow_image_processor or not self.test_fast_image_processor:
            self.skipTest(reason="Skipping slow/fast equivalence test")

        if self.image_processing_class is None or self.fast_image_processing_class is None:
            self.skipTest(reason="Skipping slow/fast equivalence test as one of the image processors is not defined")

        dummy_image = Image.open(
            requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw
        )
        image_processor_slow = self.image_processing_class(**self.image_processor_dict)
        image_processor_fast = self.fast_image_processing_class(**self.image_processor_dict)

        encoding_slow = image_processor_slow(dummy_image, return_tensors="pt")
        encoding_fast = image_processor_fast(dummy_image, return_tensors="pt")

        self._assert_slow_fast_tensors_equivalence(encoding_slow.pixel_values, encoding_fast.pixel_values)
        self._assert_slow_fast_tensors_equivalence(encoding_slow.pixel_mask.float(), encoding_fast.pixel_mask.float())

    @require_vision
    @require_torch
    def test_slow_fast_equivalence_batched(self):
        if not self.test_slow_image_processor or not self.test_fast_image_processor:
            self.skipTest(reason="Skipping slow/fast equivalence test")

        if self.image_processing_class is None or self.fast_image_processing_class is None:
            self.skipTest(reason="Skipping slow/fast equivalence test as one of the image processors is not defined")

        if hasattr(self.image_processor_tester, "do_center_crop") and self.image_processor_tester.do_center_crop:
            self.skipTest(
                reason="Skipping as do_center_crop is True and center_crop functions are not equivalent for fast and slow processors"
            )

        dummy_images = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)
        image_processor_slow = self.image_processing_class(**self.image_processor_dict)
        image_processor_fast = self.fast_image_processing_class(**self.image_processor_dict)

        encoding_slow = image_processor_slow(dummy_images, return_tensors="pt")
        encoding_fast = image_processor_fast(dummy_images, return_tensors="pt")

        self._assert_slow_fast_tensors_equivalence(encoding_slow.pixel_values, encoding_fast.pixel_values)
        self._assert_slow_fast_tensors_equivalence(encoding_slow.pixel_mask.float(), encoding_fast.pixel_mask.float())
