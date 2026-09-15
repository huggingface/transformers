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

from ...test_image_processing_common import ImageProcessingTester, ImageProcessingTestMixin


class PPLCNetImageProcessingTester(ImageProcessingTester):
    def __init__(self, parent, size=None, **kwargs):
        kwargs.setdefault("batch_size", 3)
        kwargs.setdefault("image_mean", [0.406, 0.456, 0.485])
        kwargs.setdefault("image_std", [0.225, 0.224, 0.229])
        kwargs.setdefault("do_normalize", True)
        kwargs.setdefault("do_resize", True)
        kwargs.setdefault("rescale_factor", 0.00392156862745098)
        kwargs.setdefault("do_rescale", True)
        kwargs.setdefault("do_center_crop", True)
        kwargs.setdefault("crop_size", {"height": 224, "width": 224})
        kwargs.setdefault("resize_short", 256)
        kwargs.setdefault("resample", 2)
        super().__init__(parent, **kwargs)
        self.size = size if size is not None else {"height": 256, "width": 256}


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
