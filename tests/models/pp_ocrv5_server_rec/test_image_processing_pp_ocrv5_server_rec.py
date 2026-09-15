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

import math
import unittest

import numpy as np

from transformers import is_vision_available
from transformers.testing_utils import require_torch, require_vision

from ...test_image_processing_common import ImageProcessingTester, ImageProcessingTestMixin


if is_vision_available():
    from PIL import Image


class PPOCRV5ServerRecImageProcessingTester(ImageProcessingTester):
    def __init__(self, parent, do_rescale=True, rescale_factor=1 / 255, max_image_width=3200, **kwargs):
        kwargs.setdefault("min_resolution", 10)
        kwargs.setdefault("image_mean", [0.5, 0.5, 0.5])
        kwargs.setdefault("image_std", [0.5, 0.5, 0.5])
        kwargs.setdefault("do_normalize", True)
        kwargs.setdefault("do_resize", True)
        kwargs.setdefault("size", {"height": 48, "width": 320})
        kwargs.setdefault("keep_aspect_ratio", False)
        kwargs.setdefault("do_pad", False)
        super().__init__(parent, **kwargs)
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.max_image_width = max_image_width

    def get_expected_value(self, images):
        shape_list = []
        for image in images:
            if isinstance(image, Image.Image):
                width, height = image.size
            elif isinstance(image, np.ndarray):
                height, width = image.shape[0], image.shape[1]
            else:
                height, width = image.shape[1], image.shape[2]
            shape_list.append((height, width))

        max_width = -1
        max_height = -1
        for height, width in shape_list:
            # We need the width and height of the widest image in the batch
            if width > max_width:
                max_width = width
                max_height = height

        default_height, default_width = self.size["height"], self.size["width"]
        ratio = max(max_width / max_height, default_width / default_height)

        target_width = int(default_height * ratio)
        target_height = default_height

        if target_width > self.max_image_width:
            target_width = self.max_image_width
        else:
            ratio = max_width / float(max_height)
            if target_width >= math.ceil(default_height * ratio):
                target_width = int(math.ceil(default_height * ratio))

        return target_height, target_width

    def expected_output_image_shape(self, images):
        height, width = self.get_expected_value(images)
        return self.num_channels, height, width


@require_torch
@require_vision
class PPOCRV5ServerRecImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.image_processor_tester = PPOCRV5ServerRecImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    @unittest.skip(reason="PPOCRV5ServerRecImageProcessor does not support 4 channel images yet")
    def test_call_numpy_4_channels():
        pass
