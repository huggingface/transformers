# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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


class BitImageProcessingTester(ImageProcessingTester):
    def __init__(self, **kwargs):
        # Image processor init kwargs
        kwargs.setdefault("size", {"shortest_edge": 20})
        kwargs.setdefault("crop_size", {"height": 18, "width": 18})

        super().__init__(**kwargs)


@require_torch
@require_vision
class BitImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processor_tester_class = BitImageProcessingTester
