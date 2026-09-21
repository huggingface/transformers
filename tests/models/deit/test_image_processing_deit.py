# Copyright 2021 HuggingFace Inc.
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


class DeiTImageProcessingTester(ImageProcessingTester):
    # Image processor init kwargs
    do_resize = True
    size = {"height": 20, "width": 20}
    do_center_crop = True
    crop_size = {"height": 18, "width": 18}
    do_normalize = True
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]


@require_torch
@require_vision
class DeiTImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    test_cast_dtype = True

    image_processing_tester_class = DeiTImageProcessingTester
