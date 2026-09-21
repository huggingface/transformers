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
    batch_size = 3

    # Image processor init kwargs
    size = {"height": 256, "width": 256}
    image_mean = [0.406, 0.456, 0.485]
    image_std = [0.225, 0.224, 0.229]
    do_normalize = True
    do_resize = True
    rescale_factor = 0.00392156862745098
    do_rescale = True
    do_center_crop = True
    crop_size = {"height": 224, "width": 224}
    resize_short = 256
    resample = 2


@require_torch
@require_vision
class PPLCNetImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_tester_class = PPLCNetImageProcessingTester

    @unittest.skip(reason="PPLCNet does not support 4 channel images yet")
    def test_call_numpy_4_channels(self):
        pass
