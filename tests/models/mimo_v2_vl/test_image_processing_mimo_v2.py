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

from transformers import MiMoV2ImageProcessor
from transformers.testing_utils import require_torch, require_torchvision, require_vision


@require_torch
@require_torchvision
@require_vision
class MiMoV2ImageProcessorTest(unittest.TestCase):
    def test_spatial_merge_and_repeated_temporal_patch_order(self):
        processor = MiMoV2ImageProcessor(
            patch_size=2, merge_size=2, temporal_patch_size=2, do_resize=False, do_rescale=False, do_normalize=False
        )
        image = np.arange(3 * 4 * 8, dtype=np.float32).reshape(3, 4, 8)
        output = processor(image, input_data_format="channels_first", return_tensors="np")
        coordinates = [(0, 0), (0, 2), (2, 0), (2, 2), (0, 4), (0, 6), (2, 4), (2, 6)]
        expected = np.stack(
            [np.stack([image[:, row : row + 2, col : col + 2]] * 2, axis=1).reshape(-1) for row, col in coordinates]
        )
        np.testing.assert_array_equal(output.pixel_values, expected)
        np.testing.assert_array_equal(output.image_grid_thw, [[1, 2, 4]])
