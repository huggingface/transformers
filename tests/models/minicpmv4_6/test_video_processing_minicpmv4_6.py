# Copyright 2026 HuggingFace Inc.
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

import math
import unittest

import numpy as np

from transformers.image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD, get_image_size
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torchvision_available, is_vision_available

from ...test_video_processing_common import VideoProcessingTestMixin, prepare_video_inputs


if is_vision_available():
    from PIL import Image

if is_torchvision_available():
    from transformers import MiniCPMV4_6VideoProcessor


class MiniCPMV4_6VideoProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=5,
        num_frames=8,
        num_channels=3,
        min_resolution=30,
        max_resolution=80,
        merge_size=2,
        do_resize=True,
        do_normalize=True,
        image_mean=IMAGENET_STANDARD_MEAN,
        image_std=IMAGENET_STANDARD_STD,
        do_convert_rgb=True,
        max_slice_nums=5,
        scale_resolution=448,
        patch_size=28,
        slice_mode=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.num_channels = num_channels
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_convert_rgb = do_convert_rgb
        self.patch_size = patch_size
        self.merge_size = merge_size
        self.max_slice_nums = max_slice_nums
        self.scale_resolution = scale_resolution
        self.patch_size = patch_size
        self.slice_mode = slice_mode

    def prepare_video_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_convert_rgb": self.do_convert_rgb,
            "do_sample_frames": False,
            "max_slice_nums": self.max_slice_nums,
            "scale_resolution": self.scale_resolution,
            "patch_size": self.patch_size,
            "slice_mode": self.slice_mode,
        }

    def expected_output_video_shape(self, video_inputs):
        for video in video_inputs:
            if isinstance(video, list) and isinstance(video[0], Image.Image):
                video = np.stack([np.array(frame) for frame in video])
            elif hasattr(video, "shape"):
                pass
            else:
                video = np.array(video)

            height, width = get_image_size(video[0])
            aspect_ratio = width / height
            height = int(self.scale_resolution / math.sqrt(aspect_ratio))
            width = int(height * aspect_ratio)

            divisor = self.patch_size * 4
            best_height = max(round(height / divisor) * divisor, divisor)
            best_width = max(round(width / divisor) * divisor, divisor)
            patch_dim = best_height * best_width // self.patch_size

        return [self.num_frames, self.num_channels, self.patch_size, patch_dim]

    def prepare_video_inputs(self, equal_resolution=False, return_tensors="pil"):
        videos = prepare_video_inputs(
            batch_size=self.batch_size,
            num_frames=self.num_frames,
            num_channels=self.num_channels,
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
            equal_resolution=equal_resolution,
            return_tensors=return_tensors,
        )
        return videos


@require_torch
@require_vision
class MiniCPMV4_6VideoProcessingTest(VideoProcessingTestMixin, unittest.TestCase):
    fast_video_processing_class = MiniCPMV4_6VideoProcessor if is_torchvision_available() else None
    input_name = "pixel_values_videos"

    def setUp(self):
        super().setUp()
        self.video_processor_tester = MiniCPMV4_6VideoProcessingTester(self)

    @property
    def video_processor_dict(self):
        return self.video_processor_tester.prepare_video_processor_dict()

    def test_video_processor_from_dict_with_kwargs(self):
        video_processor = self.fast_video_processing_class.from_dict(self.video_processor_dict)
        self.assertEqual(video_processor.patch_size, 28)

        video_processor = self.fast_video_processing_class.from_dict(self.video_processor_dict, patch_size=36)
        self.assertEqual(video_processor.patch_size, 36)

    def test_call_sample_frames(self):
        for video_processing_class in self.video_processor_list:
            video_processor_dict = self.video_processor_dict.copy()
            video_processor = video_processing_class(**video_processor_dict, max_num_frames=20, stack_frames=1)

            video_inputs = self.video_processor_tester.prepare_video_inputs(
                equal_resolution=False,
                return_tensors="torch",
            )
            encoded_videos = video_processor(
                video_inputs[0],
                return_tensors="pt",
            )[self.input_name]
            encoded_videos_batched = video_processor(video_inputs, return_tensors="pt")[self.input_name]
            self.assertEqual(encoded_videos.shape, (8, 3, 12, 12))
            self.assertEqual(encoded_videos_batched.shape, (5, 8, 3, 12, 12))

            # Test with more frames to stack
            # video_processor = video_processing_class(**video_processor_dict, max_num_frames=20, stack_frames=4)
            # encoded_videos = video_processor(video_inputs[0], return_tensors="pt",)[self.input_name]
            # encoded_videos_batched = video_processor(video_inputs, return_tensors="pt")[self.input_name]
