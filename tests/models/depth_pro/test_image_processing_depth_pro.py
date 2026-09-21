# Copyright 2024 The HuggingFace Team. All rights reserved.
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


class DepthProImageProcessingTester(ImageProcessingTester):
    # Image processor init kwargs
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    do_rescale = True
    do_normalize = True
    do_resize = True
    size = {"height": 18, "width": 18}


@require_torch
@require_vision
class DepthProImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_tester_class = DepthProImageProcessingTester
