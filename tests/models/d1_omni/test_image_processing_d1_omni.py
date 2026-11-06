# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import numpy as np
from PIL import Image

from transformers import D1OmniImageProcessor
from transformers.testing_utils import require_torch, require_torchvision, require_vision


@require_torch
@require_torchvision
@require_vision
class D1OmniImageProcessorTest(unittest.TestCase):
    def test_publisher_tiling(self):
        image = Image.fromarray(np.zeros((800, 1500, 3), dtype=np.uint8))
        outputs = D1OmniImageProcessor()(image, return_tensors="pt")
        # Publisher layout: eight square tiles followed by the resized thumbnail.
        self.assertEqual(outputs.pixel_values.shape, (9, 1024, 768))
        self.assertEqual(outputs.spatial_shapes.tolist(), [[32, 32]] * 8 + [[22, 42]])
