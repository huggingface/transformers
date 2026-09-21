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


class ChineseCLIPImageProcessingTester(ImageProcessingTester):
    # Image processor init kwargs
    size = {"height": 224, "width": 224}
    crop_size = {"height": 18, "width": 18}


@require_torch
@require_vision
class ChineseCLIPImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_tester_class = ChineseCLIPImageProcessingTester

    @unittest.skip(
        reason="ChineseCLIPImageProcessor doesn't treat 4 channel PIL and numpy consistently yet"
    )  # FIXME Amy
    def test_call_numpy_4_channels(self):
        pass


@require_torch
@require_vision
class ChineseCLIPImageProcessingTestFourChannels(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_tester_class = ChineseCLIPImageProcessingTester

    def setUp(self):
        super().setUp()
        self.expected_encoded_image_num_channels = 3

    @unittest.skip(
        reason="ChineseCLIPImageProcessor doesn't treat 4 channel PIL and numpy consistently yet"
    )  # FIXME Amy
    def test_call_numpy_4_channels(self):
        pass
