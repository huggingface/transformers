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

from transformers import LlavaOnevision2VideoProcessor
from transformers.image_utils import PILImageResampling
from transformers.testing_utils import require_torch, require_torchvision, require_vision
from transformers.video_utils import VideoMetadata

from ...test_video_processing_common import VideoProcessingTestMixin, prepare_video_inputs


class LlavaOnevision2VideoProcessingTester:
    batch_size = 3
    num_frames = 6
    num_channels = 3

    def prepare_video_inputs(self, equal_resolution=False, return_tensors="pil"):
        return prepare_video_inputs(
            self.batch_size,
            self.num_frames,
            self.num_channels,
            8,
            9,
            equal_resolution,
            return_tensors,
        )

    def expected_output_video_shape(self, videos):
        return (sum(len(video) for video in videos) * 16, 12)


@require_torch
@require_torchvision
@require_vision
class LlavaOnevision2VideoProcessorTest(VideoProcessingTestMixin, unittest.TestCase):
    fast_video_processing_class = LlavaOnevision2VideoProcessor
    video_processor_dict = {
        "patch_size": 2,
        "size": {"shortest_edge": 64, "longest_edge": 64},
        "do_sample_frames": False,
    }

    def setUp(self):
        super().setUp()
        self.video_processor_tester = LlavaOnevision2VideoProcessingTester()
        self.processor = LlavaOnevision2VideoProcessor(patch_size=2, size={"shortest_edge": 16, "longest_edge": 64})

    def assert_expected_videos_shape(self, encoded_videos, expected_output_video_shape, num_videos):
        self.assertEqual(tuple(encoded_videos.shape), expected_output_video_shape)

    def test_call_numpy_4_channels(self):
        self.video_processor_tester.num_channels = 4
        videos = self.video_processor_tester.prepare_video_inputs(return_tensors="np")
        processor = self.fast_video_processing_class(**self.video_processor_dict)
        for inputs in (videos[:1], videos):
            output = processor(inputs, return_tensors="pt", input_data_format="channels_last")
            self.assertEqual(
                tuple(output.pixel_values_videos.shape),
                self.video_processor_tester.expected_output_video_shape(inputs),
            )

    def test_video_processor_from_dict_with_kwargs(self):
        size = {"shortest_edge": 64, "longest_edge": 256}
        processor = self.fast_video_processing_class.from_dict(self.video_processor_dict, size=size)
        self.assertEqual(processor.size, size)

    def test_call_sample_frames(self):
        video = np.zeros((32, 8, 8, 3), dtype=np.uint8)
        outputs = self.processor(
            video, video_metadata={"total_num_frames": 32, "fps": 8}, num_frames=4, return_tensors="pt"
        )
        self.assertEqual(outputs["video_grid_thw"].tolist(), [[4, 4, 4]])
        self.assertEqual(outputs["patch_positions"][:, 0].unique_consecutive().tolist(), [0, 10, 21, 31])
        self.assertEqual(outputs["frame_timestamps"][0], [0, 1.25, 2.625, 3.875])

    def test_duration_sampling(self):
        for duration, expected in ((5, 8), (20, 16), (40, 32)):
            indices = self.processor.sample_frames(VideoMetadata(total_num_frames=duration * 8, fps=8), max_frames=32)
            self.assertEqual(len(indices), expected)
        with self.assertRaises(ValueError):
            self.processor.sample_frames(VideoMetadata(total_num_frames=32, fps=8), num_frames=4, fps=2)

    def test_call_does_not_mutate_defaults(self):
        video = np.zeros((8, 8, 8, 3), dtype=np.uint8)
        self.processor(video, num_frames=2)
        self.assertIsNone(self.processor.num_frames)
        self.assertEqual(self.processor(video)["video_grid_thw"].tolist(), [[8, 4, 4]])

    def test_invalid_backend(self):
        with self.assertRaises(ValueError):
            self.processor(np.zeros((2, 8, 8, 3), dtype=np.uint8), video_backend="invalid")

    def test_conflicting_video_geometry(self):
        video = np.zeros((2, 8, 8, 3), dtype=np.uint8)
        for name, value in (("patch_size", 4), ("merge_size", 1), ("temporal_patch_size", 2)):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, f"{name} must match"):
                self.processor(video, **{name: value})

    def test_invalid_sampling_arguments(self):
        metadata = VideoMetadata(total_num_frames=32, fps=8)
        for name in ("num_frames", "fps", "max_frames"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, f"{name} must be positive"):
                self.processor.sample_frames(metadata, **{name: 0})

    def test_resample_override(self):
        video = np.arange(2 * 12 * 12 * 3, dtype=np.uint8).reshape(2, 12, 12, 3)
        bicubic = self.processor(video, do_sample_frames=False, return_tensors="np")
        nearest = self.processor(
            video, do_sample_frames=False, resample=PILImageResampling.NEAREST, return_tensors="np"
        )
        np.testing.assert_array_equal(bicubic["patch_positions"], nearest["patch_positions"])
        self.assertFalse(np.allclose(bicubic["pixel_values_videos"], nearest["pixel_values_videos"]))
