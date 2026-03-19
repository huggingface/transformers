# coding = utf-8
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

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


class PPLCNetImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        num_channels=3,
        image_size=18,
        min_resolution=30,
        max_resolution=400,
        do_resize=True,
        size=None,
        do_normalize=True,
        image_mean=[0.406, 0.456, 0.485],
        image_std=[0.225, 0.224, 0.229],
        rescale_factor=0.00392156862745098,
        do_rescale=True,
        do_center_crop=True,
        crop_size=None,
        resize_short=256,
        resample=2,
    ):
        size = size if size is not None else {"height": 256, "width": 256}
        crop_size = crop_size if crop_size is not None else {"height": 224, "width": 224}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.rescale_factor = rescale_factor
        self.do_rescale = do_rescale
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.resize_short = resize_short
        self.resample = resample

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

    def prepare_image_processor_dict(self):
        return {
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_normalize": self.do_normalize,
            "do_resize": self.do_resize,
            "rescale_factor": self.rescale_factor,
            "do_rescale": self.do_rescale,
            "do_center_crop": self.do_center_crop,
            "crop_size": self.crop_size,
            "resize_short": self.resize_short,
            "resample": self.resample,
        }

    def expected_output_image_shape(self, images):
        return self.num_channels, self.crop_size["height"], self.crop_size["width"]


@require_torch
@require_vision
class PPLCNetImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.image_processor_tester = PPLCNetImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    @unittest.skip(reason="PPLCNet does not support 4 channel images yet")
    def test_call_numpy_4_channels(self):
        pass
