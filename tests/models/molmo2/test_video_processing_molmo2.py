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

from transformers.image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD
from transformers.testing_utils import require_torch, require_torchvision, require_vision
from transformers.utils import is_torchvision_available, is_vision_available

from ...test_video_processing_common import VideoProcessingTestMixin


if is_vision_available() and is_torchvision_available():
    from transformers import Molmo2VideoProcessor


class Molmo2VideoProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=5,
        num_frames=8,
        num_channels=3,
        min_resolution=32,
        max_resolution=80,
        do_resize=True,
        size=None,
        do_normalize=True,
        image_mean=IMAGENET_STANDARD_MEAN,
        image_std=IMAGENET_STANDARD_STD,
        do_convert_rgb=True,
        patch_size=14,
        pooling_size=[3, 3],
        do_sample_frames=True,
        frame_sample_mode="uniform_last_frame",
        max_fps=2,
        sampling_fps=2,
    ):
        size = size if size is not None else {"height": 378, "width": 378}
        self.parent = parent
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.num_channels = num_channels
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_convert_rgb = do_convert_rgb
        self.patch_size = patch_size
        self.pooling_size = pooling_size
        self.do_sample_frames = do_sample_frames
        self.frame_sample_mode = frame_sample_mode
        self.max_fps = max_fps
        self.sampling_fps = sampling_fps

    def prepare_video_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_convert_rgb": self.do_convert_rgb,
            "patch_size": self.patch_size,
            "pooling_size": self.pooling_size,
            "do_sample_frames": self.do_sample_frames,
            "frame_sample_mode": self.frame_sample_mode,
            "max_fps": self.max_fps,
            "sampling_fps": self.sampling_fps,
        }


@require_torch
@require_vision
@require_torchvision
class Molmo2VideoProcessingTest(VideoProcessingTestMixin, unittest.TestCase):
    video_processing_class = Molmo2VideoProcessor if (is_vision_available() and is_torchvision_available()) else None

    def setUp(self):
        self.video_processor_tester = Molmo2VideoProcessingTester(self)

    @property
    def video_processor_dict(self):
        return self.video_processor_tester.prepare_video_processor_dict()

    def test_video_processor_properties(self):
        video_processor = self.video_processing_class(**self.video_processor_dict)
        self.assertTrue(hasattr(video_processor, "do_resize"))
        self.assertTrue(hasattr(video_processor, "size"))
        self.assertTrue(hasattr(video_processor, "do_normalize"))
        self.assertTrue(hasattr(video_processor, "image_mean"))
        self.assertTrue(hasattr(video_processor, "image_std"))
        self.assertTrue(hasattr(video_processor, "do_convert_rgb"))
        self.assertTrue(hasattr(video_processor, "patch_size"))
        self.assertTrue(hasattr(video_processor, "pooling_size"))
        self.assertTrue(hasattr(video_processor, "do_sample_frames"))
