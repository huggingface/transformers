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

from transformers import MiMoV2VideoProcessor
from transformers.testing_utils import require_torch, require_torchvision, require_vision


@require_torch
@require_torchvision
@require_vision
class MiMoV2VideoProcessorTest(unittest.TestCase):
    def test_temporal_patches_keep_frame_order_and_repeat_last_frame(self):
        processor = MiMoV2VideoProcessor(
            patch_size=2,
            merge_size=2,
            temporal_patch_size=2,
            do_resize=False,
            do_rescale=False,
            do_normalize=False,
            do_sample_frames=False,
        )
        frames = np.arange(3 * 3 * 4 * 4, dtype=np.float32).reshape(3, 3, 4, 4)
        output = processor(frames, input_data_format="channels_first", return_tensors="np")
        first_patch = output.pixel_values_videos[0].reshape(3, 2, 2, 2)
        final_patch = output.pixel_values_videos[4].reshape(3, 2, 2, 2)
        np.testing.assert_array_equal(first_patch[:, 0], frames[0, :, :2, :2])
        np.testing.assert_array_equal(first_patch[:, 1], frames[1, :, :2, :2])
        np.testing.assert_array_equal(final_patch[:, 0], frames[2, :, :2, :2])
        np.testing.assert_array_equal(final_patch[:, 1], frames[2, :, :2, :2])
        np.testing.assert_array_equal(output.video_grid_thw, [[2, 2, 2]])
