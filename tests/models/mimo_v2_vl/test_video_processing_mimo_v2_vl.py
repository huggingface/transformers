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
from parameterized import parameterized

from transformers import MiMoV2VLVideoProcessor
from transformers.testing_utils import require_torch, require_torchvision, require_vision


@require_torch
@require_torchvision
@require_vision
class MiMoV2VLVideoProcessorTest(unittest.TestCase):
    @parameterized.expand(
        [
            (0, 8, 12, {}),
            (5, 0, 12, {}),
            (5, 8, 0, {}),
            (5, 8, 12, {"temporal_compression_ratio": 0}),
            (5, 8, 12, {"patch_size": 0}),
            (5, 8, 12, {"total_max_pixels": 0}),
            (5, 8, 12, {"total_max_pixels": 0, "do_resize": False}),
        ]
    )
    def test_patch_count_rejects_invalid_dimensions(self, frames, height, width, kwargs):
        processor = MiMoV2VLVideoProcessor()
        with self.assertRaisesRegex(ValueError, "positive"):
            processor.get_num_of_video_patches(frames, height, width, kwargs)

    def test_grouping_preserves_variable_length_video_order(self):
        processor = MiMoV2VLVideoProcessor(
            patch_size=2,
            merge_size=2,
            temporal_patch_size=2,
            temporal_compression_ratio=2,
            size={"shortest_edge": 16, "longest_edge": 256},
            total_max_pixels=128,
            do_sample_frames=False,
        )
        rng = np.random.default_rng(0)
        videos = [rng.integers(0, 256, (frames, 3, 8, 12), dtype=np.uint8) for frames in [1, 5, 1]]
        kwargs = {"input_data_format": "channels_first", "return_tensors": "np"}
        actual = processor(videos, **kwargs)
        expected = [processor(video, **kwargs) for video in videos]
        np.testing.assert_array_equal(
            actual.pixel_values_videos, np.concatenate([out.pixel_values_videos for out in expected])
        )
        np.testing.assert_array_equal(actual.video_grid_thw, [[2, 4, 6], [4, 4, 6], [2, 4, 6]])

    def test_patch_count_respects_budget_and_temporal_compression(self):
        processor = MiMoV2VLVideoProcessor(
            patch_size=2,
            merge_size=2,
            temporal_patch_size=2,
            size={"shortest_edge": 16, "longest_edge": 256},
            do_sample_frames=False,
        )
        video = np.zeros((5, 3, 8, 12), dtype=np.uint8)
        kwargs = {"temporal_compression_ratio": 2, "total_max_pixels": 40}
        output = processor(video, input_data_format="channels_first", return_tensors="np", **kwargs)
        np.testing.assert_array_equal(output.video_grid_thw, [[4, 2, 2]])
        self.assertEqual(processor.get_num_of_video_patches(5, 8, 12, kwargs), 16)
        self.assertEqual(output.pixel_values_videos.shape[0], 16)

    def test_temporal_patches_keep_frame_order_and_repeat_last_frame(self):
        processor = MiMoV2VLVideoProcessor(
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
