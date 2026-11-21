# Copyright 2026 NVIDIA CORPORATION and the HuggingFace Inc. team. All rights reserved.
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

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torchvision_available, is_vision_available

from ...test_video_processing_common import VideoProcessingTestMixin, prepare_video_inputs


if is_vision_available() and is_torchvision_available():
    from transformers import AudioVisualFlamingoVideoProcessor


class AudioVisualFlamingoVideoProcessingTester:
    def __init__(self, parent, batch_size=5, num_frames=8, num_channels=3, min_resolution=30, max_resolution=80):
        self.parent = parent
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.num_channels = num_channels
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.size = {"height": 20, "width": 20}

    def prepare_video_processor_dict(self):
        return {"size": self.size, "num_frames": self.num_frames}

    def expected_output_video_shape(self, videos):
        return self.num_frames, self.num_channels, self.size["height"], self.size["width"]

    def prepare_video_inputs(self, equal_resolution=False, return_tensors="pil"):
        return prepare_video_inputs(
            batch_size=self.batch_size,
            num_frames=self.num_frames,
            num_channels=self.num_channels,
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
            equal_resolution=equal_resolution,
            return_tensors=return_tensors,
        )


@require_torch
@require_vision
class AudioVisualFlamingoVideoProcessingTest(VideoProcessingTestMixin, unittest.TestCase):
    fast_video_processing_class = AudioVisualFlamingoVideoProcessor if is_torchvision_available() else None
    input_name = "pixel_values_videos"

    def setUp(self):
        super().setUp()
        self.video_processor_tester = AudioVisualFlamingoVideoProcessingTester(self)

    @property
    def video_processor_dict(self):
        return self.video_processor_tester.prepare_video_processor_dict()

    def test_video_processor_from_dict_with_kwargs(self):
        video_processor = self.fast_video_processing_class.from_dict(self.video_processor_dict)
        self.assertEqual(video_processor.size, {"height": 20, "width": 20})

        video_processor = self.fast_video_processing_class.from_dict(self.video_processor_dict, size=42)
        self.assertEqual(video_processor.size, {"height": 42, "width": 42})

    def test_call_sample_frames(self):
        video_processor = self.fast_video_processing_class(**self.video_processor_dict)
        sample_frames_indices = {}
        for num_total_frames in (10, 3):
            inputs = video_processor(
                np.zeros((num_total_frames, 32, 32, 3), dtype=np.uint8),
                num_frames=5,
                return_tensors="pt",
                return_metadata=True,
            )
            self.assertEqual(inputs.pixel_values_videos.shape, (1, 5, 3, 20, 20))
            sample_frames_indices[num_total_frames] = inputs.video_metadata[0].frames_indices.tolist()

        # Frames are sampled evenly from the first to the last one, and repeated for short videos
        self.assertEqual(sample_frames_indices, {10: [0, 2, 4, 7, 9], 3: [0, 0, 1, 2, 2]})

        # A fixed number of frames is sampled, so that videos can be batched
        with self.assertRaisesRegex(ValueError, "use `num_frames` instead of `fps`"):
            video_processor(np.zeros((10, 32, 32, 3), dtype=np.uint8), num_frames=None, fps=2, return_tensors="pt")
