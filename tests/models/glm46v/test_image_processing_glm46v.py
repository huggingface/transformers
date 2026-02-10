# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from transformers.testing_utils import require_torch, require_torchvision, require_vision
from transformers.utils import is_torch_available, is_torchvision_available, is_vision_available


if is_vision_available():
    from transformers import Glm46VImageProcessor

if is_torch_available() and is_torchvision_available():
    from transformers import Glm46VImageProcessorFast


@require_vision
@require_torch
@require_torchvision
class Glm46VImageProcessorFastTest(unittest.TestCase):
    def test_get_number_of_image_patches_matches_slow_processor(self):
        slow_processor = Glm46VImageProcessor()
        fast_processor = Glm46VImageProcessorFast()

        test_cases = [
            (100, 100, {}, 64),
            (200, 50, {}, 56),
            (100, 100, {"patch_size": 28}, 16),
        ]

        for height, width, images_kwargs, expected in test_cases:
            with self.subTest(height=height, width=width, images_kwargs=images_kwargs):
                self.assertEqual(
                    slow_processor.get_number_of_image_patches(
                        height=height, width=width, images_kwargs=images_kwargs
                    ),
                    expected,
                )
                self.assertEqual(
                    fast_processor.get_number_of_image_patches(
                        height=height, width=width, images_kwargs=images_kwargs
                    ),
                    expected,
                )
