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

from transformers.file_utils import is_torch_available
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torchvision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_torch_available():
    pass

if is_torchvision_available():
    from transformers import PPOCRV5ServerDetImageProcessorFast


class PPOCRV5ServerDetImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        image_size=18,
        min_resolution=10,
        max_resolution=400,
        limit_side_len=960,
        limit_type="max",
        max_side_limit=4000,
        do_resize=True,
        size=None,
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    ):
        size = size if size is not None else {"height": 512, "width": 512}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.limit_side_len = limit_side_len
        self.limit_type = limit_type
        self.max_side_limit = max_side_limit
        self.do_resize = do_resize
        self.size = size
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
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


@require_torch
@require_vision
class PPOCRV5ServerDetImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    test_slow_image_processor = False
    fast_image_processing_class = PPOCRV5ServerDetImageProcessorFast if is_torchvision_available() else None

    def setUp(self):
        super().setUp()
        self.image_processor_tester = PPOCRV5ServerDetImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    @unittest.skip(
        reason="PPOCRV5ServerDet can’t stack the images into a batch because the image processor resizes them adaptively, leading to inconsistent output sizes."
    )
    def test_call_pytorch():
        pass

    @unittest.skip(
        reason="PPOCRV5ServerDet can’t stack the images into a batch because the image processor resizes them adaptively, leading to inconsistent output sizes."
    )
    def test_call_numpy():
        pass

    @unittest.skip(
        reason="PPOCRV5ServerDet can’t stack the images into a batch because the image processor resizes them adaptively, leading to inconsistent output sizes."
    )
    def test_call_pil():
        pass

    @unittest.skip(reason="PPOCRV5ServerDetImageProcessorFast does not support 4 channel images yet")
    def test_call_numpy_4_channels():
        pass
