# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from transformers import LlavaOnevision1_5ImageProcessor, LlavaOnevision1_5ImageProcessorPil
from transformers.testing_utils import require_torchvision


class LlavaOnevision1_5ImageProcessingTest(unittest.TestCase):
    def test_pil_temporal_patch_size(self):
        self.assertEqual(LlavaOnevision1_5ImageProcessorPil().temporal_patch_size, 1)

    @require_torchvision
    def test_torchvision_temporal_patch_size(self):
        self.assertEqual(LlavaOnevision1_5ImageProcessor().temporal_patch_size, 1)
