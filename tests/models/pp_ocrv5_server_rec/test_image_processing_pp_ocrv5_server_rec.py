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

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_vision_available():
    from PIL import Image


class PPOCRV5ServerRecImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        image_size=18,
        min_resolution=10,
        max_resolution=400,
        do_resize=True,
        size=None,
        do_rescale=True,
        do_pad=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        max_image_width=3200,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    ):
        size = size if size is not None else {"height": 48, "width": 320}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.do_pad = do_pad
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.max_image_width = max_image_width

    def prepare_image_processor_dict(self):
        return {
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_normalize": self.do_normalize,
            "do_resize": self.do_resize,
            "size": self.size,
            "keep_aspect_ratio": False,
            "do_pad": False,
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
