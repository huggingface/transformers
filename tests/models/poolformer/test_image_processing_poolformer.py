# Copyright 2022 HuggingFace Inc.
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


class PoolFormerImageProcessingTester(ImageProcessingTester):
    def __init__(self, **kwargs):
        kwargs.setdefault("size", {"shortest_edge": 30})
        kwargs.setdefault("do_resize", True)
        kwargs.setdefault("do_center_crop", True)
        kwargs.setdefault("crop_pct", 0.9)
        kwargs.setdefault("crop_size", {"height": 30, "width": 30})
        kwargs.setdefault("do_normalize", True)
        kwargs.setdefault("image_mean", [0.5, 0.5, 0.5])
        kwargs.setdefault("image_std", [0.5, 0.5, 0.5])
        super().__init__(**kwargs)


@require_torch
@require_vision
class PoolFormerImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_tester_class = PoolFormerImageProcessingTester


@require_torch
@require_vision
class PoolFormerImageProcessingNoCropPctTest(PoolFormerImageProcessingTest):
    image_processing_tester_class = PoolFormerImageProcessingTester

    def setUp(self):
        super().setUp()
        self.image_processor_tester.crop_pct = None
