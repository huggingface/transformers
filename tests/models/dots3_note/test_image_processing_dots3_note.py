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

import numpy as np
from PIL import Image

from transformers import Dots3NoteImageProcessorPil, is_torch_available
from transformers.testing_utils import require_torchvision


if is_torch_available():
    import torch


@require_torchvision
class Dots3NoteImageProcessingTest(unittest.TestCase):
    def test_rgba_images_are_composited_on_white(self):
        processor = Dots3NoteImageProcessorPil(size={"shortest_edge": 16, "longest_edge": 64}, patch_size=2)
        array = np.zeros((8, 12, 4), dtype=np.uint8)
        array[..., :3] = [200, 40, 10]
        array[..., 3] = np.arange(12, dtype=np.uint8)[None] * 20
        rgba = Image.fromarray(array, "RGBA")
        white = Image.new("RGB", rgba.size, (255, 255, 255))
        white.paste(rgba, mask=rgba.getchannel("A"))

        actual = processor(rgba, return_tensors="pt")
        expected = processor(white, return_tensors="pt")

        torch.testing.assert_close(actual.pixel_values, expected.pixel_values, rtol=0, atol=0)
        torch.testing.assert_close(actual.image_grid_thw, expected.image_grid_thw, rtol=0, atol=0)
