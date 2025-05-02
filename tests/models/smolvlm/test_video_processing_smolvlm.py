# coding=utf-8
# Copyright 2025 HuggingFace Inc.
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

from transformers.image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_torchvision_available, is_vision_available

from ...test_video_processing_common import VideoProcessingTestMixin, prepare_video_inputs


if is_torch_available():
    import torch

if is_vision_available():
    if is_torchvision_available():
        from transformers import SmolVLMVideoProcessor
        from transformers.models.smolvlm.video_processing_smolvlm import get_resize_output_image_size


class SmolVLMVideoProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=5,
        num_frames=8,
        num_channels=3,
        min_resolution=30,
        max_resolution=80,
        do_resize=True,
        size=None,
        do_normalize=True,
        image_mean=IMAGENET_STANDARD_MEAN,
        image_std=IMAGENET_STANDARD_STD,
        do_convert_rgb=True,
    ):
        size = size if size is not None else {"longest_edge": 20}
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

    def prepare_video_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_convert_rgb": self.do_convert_rgb,
        }

    def expected_output_video_shape(self, videos):
        max_height, max_width = 0, 0
        if not isinstance(videos[0], torch.Tensor):
            videos = [torch.tensor(np.array(video)).permute(0, -1, -3, -2) for video in videos]
        for video in videos:
            height, width = get_resize_output_image_size(video, self.size["longest_edge"])
            max_height = max(height, max_height)
            max_width = max(width, max_width)
        return [self.num_frames, self.num_channels, max_height, max_width]

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
class SmolVLMVideoProcessingTest(VideoProcessingTestMixin, unittest.TestCase):
    fast_video_processing_class = SmolVLMVideoProcessor if is_torchvision_available() else None
    input_name = "pixel_values"

    def setUp(self):
        super().setUp()
        self.video_processor_tester = SmolVLMVideoProcessingTester(self)

    @property
    def video_processor_dict(self):
        return self.video_processor_tester.prepare_video_processor_dict()

    def test_video_processor_from_dict_with_kwargs(self):
        video_processor = self.fast_video_processing_class.from_dict(self.video_processor_dict)
        self.assertEqual(video_processor.size, {"longest_edge": 20})

        video_processor = self.fast_video_processing_class.from_dict(self.video_processor_dict, size=42)
        self.assertEqual(video_processor.size, {"height": 42, "width": 42})
