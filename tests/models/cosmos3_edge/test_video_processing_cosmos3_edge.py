# Copyright 2026 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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

"""Tests for the Cosmos3 Edge video processor."""

import unittest

import numpy as np

from transformers import Cosmos3EdgeVideoProcessor
from transformers.testing_utils import require_torch, require_torchvision, require_vision
from transformers.video_utils import VideoMetadata


@require_torch
@require_torchvision
@require_vision
class Cosmos3EdgeVideoProcessingTest(unittest.TestCase):
    def test_temporal_patch_size_is_fixed_to_one(self):
        with self.assertRaisesRegex(ValueError, "temporal_patch_size=1"):
            Cosmos3EdgeVideoProcessor(temporal_patch_size=2)

    def test_samples_two_frames_per_second_by_default(self):
        processor = Cosmos3EdgeVideoProcessor()
        metadata = VideoMetadata(total_num_frames=8, fps=4, duration=2.0)

        indices = processor.sample_frames(metadata)

        self.assertEqual(indices.tolist(), [0, 2, 5, 7])

    def test_video_processor_preserves_one_grid_step_per_frame(self):
        processor = Cosmos3EdgeVideoProcessor(
            do_resize=False,
            do_rescale=False,
            do_normalize=False,
            patch_size=2,
            merge_size=2,
            temporal_patch_size=1,
        )
        video = np.zeros((3, 4, 8, 3), dtype=np.uint8)

        output = processor(
            video,
            video_metadata=[{"fps": 2, "total_num_frames": 3, "duration": 1.5}],
            return_tensors="pt",
        )

        self.assertEqual(output.video_grid_thw.tolist(), [[3, 2, 4]])
        self.assertEqual(tuple(output.pixel_values_videos.shape), (24, 12))
