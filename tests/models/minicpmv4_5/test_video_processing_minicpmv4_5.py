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

from transformers.image_utils import get_image_size
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torchvision_available, is_vision_available
from transformers.video_utils import VideoMetadata

from ...test_video_processing_common import VideoProcessingTestMixin, prepare_video_inputs


if is_vision_available():
    from PIL import Image

if is_torchvision_available():
    from transformers import MiniCPMV4_5VideoProcessor
    from transformers.models.minicpmv2_5.image_processing_minicpmv2_5 import find_best_resize


class MiniCPMV4_5VideoProcessingTester:
    def __init__(self, parent, batch_size=5, num_frames=8, num_channels=3, min_resolution=30, max_resolution=80):
        self.parent = parent
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.num_channels = num_channels
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.scale_resolution = 112
        self.patch_size = 14

    def prepare_video_processor_dict(self):
        return {"do_sample_frames": False, "scale_resolution": self.scale_resolution, "patch_size": self.patch_size}

    def expected_output_video_shape(self, videos):
        # Every frame is one (unsliced) row of patches, padded to the longest frame
        num_frames, max_patches = 0, 0
        for video in videos:
            first = np.array(video[0]) if isinstance(video[0], Image.Image) else video[0]
            height, width = find_best_resize(get_image_size(first), self.scale_resolution, self.patch_size, True)
            num_frames += len(video)
            max_patches = max(max_patches, height * width // self.patch_size**2)
        return [num_frames, self.num_channels, self.patch_size, max_patches * self.patch_size]

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
class MiniCPMV4_5VideoProcessingTest(VideoProcessingTestMixin, unittest.TestCase):
    fast_video_processing_class = MiniCPMV4_5VideoProcessor if is_torchvision_available() else None
    input_name = "pixel_values_videos"

    def setUp(self):
        super().setUp()
        self.video_processor_tester = MiniCPMV4_5VideoProcessingTester(self)

    @property
    def video_processor_dict(self):
        return self.video_processor_tester.prepare_video_processor_dict()

    def assert_expected_videos_shape(self, encoded_videos, expected_output_video_shape, num_videos):
        # The frames of all videos are concatenated along the first axis
        self.assertEqual(list(encoded_videos.shape), expected_output_video_shape)

    def test_video_processor_from_dict_with_kwargs(self):
        video_processor = self.fast_video_processing_class.from_dict(self.video_processor_dict)
        self.assertEqual(video_processor.patch_size, 14)

        video_processor = self.fast_video_processing_class.from_dict(self.video_processor_dict, patch_size=28)
        self.assertEqual(video_processor.patch_size, 28)

    @unittest.skip("NaViT expected_output_video_shape cannot infer channel dim for 4-channel images")
    def test_call_numpy_4_channels(self):
        pass

    @unittest.skip("Frames are sampled by `fps` and packing, see `test_sample_frames_and_temporal_ids`")
    def test_call_sample_frames(self):
        pass

    def test_sample_frames_and_temporal_ids(self):
        video_processor = self.fast_video_processing_class(do_sample_frames=True, scale_resolution=56)
        video = np.random.randint(255, size=(1000, 64, 64, 3), dtype=np.uint8)
        # 100 s at 10 fps: the original packing samples 500 frames at 5 fps and packs them 3 per group
        metadata = VideoMetadata(total_num_frames=1000, fps=10.0, duration=100.0)
        outputs = video_processor(video, video_metadata=[metadata], return_tensors="pt")
        self.assertEqual(outputs["pixel_values_videos"].shape[0], 500)
        self.assertListEqual(outputs["temporal_ids"][:2].tolist(), [[1, 3, 5], [7, 9, 11]])
        self.assertEqual(outputs["temporal_ids"].shape, (167, 3))
