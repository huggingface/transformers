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

from transformers.image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_torchvision_available, is_vision_available

from ...test_video_processing_common import VideoProcessingTestMixin, prepare_video_inputs


if is_torch_available():
    pass

if is_vision_available():
    from transformers import LlavaNextVideoVideoProcessor

    if is_torchvision_available():
        from transformers import LlavaNextVideoVideoProcessorFast


class LlavaNextVideoProcessingTester(unittest.TestCase):
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
        do_center_crop=True,
        crop_size=None,
        do_normalize=True,
        image_mean=OPENAI_CLIP_MEAN,
        image_std=OPENAI_CLIP_STD,
        do_convert_rgb=True,
    ):
        size = size if size is not None else {"height": 20, "width": 20}
        crop_size = crop_size if crop_size is not None else {"height": 18, "width": 18}
        self.parent = parent
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.num_channels = num_channels
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_convert_rgb = do_convert_rgb

    def prepare_video_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "do_center_crop": self.do_center_crop,
            "crop_size": self.crop_size,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_convert_rgb": self.do_convert_rgb,
        }

    def expected_output_video_shape(self, images):
        return self.num_frames, self.num_channels, self.crop_size["height"], self.crop_size["width"]

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
class LlavaNextVideoProcessingTest(VideoProcessingTestMixin, unittest.TestCase):
    video_processing_class = LlavaNextVideoVideoProcessor if is_vision_available() else None
    fast_video_processing_class = LlavaNextVideoVideoProcessorFast if is_torchvision_available() else None

    def setUp(self):
        super().setUp()
        self.video_processor_tester = LlavaNextVideoProcessingTester(self)

    @property
    def video_processor_dict(self):
        return self.video_processor_tester.prepare_video_processor_dict()

    def test_video_processor_properties(self):
        video_processing = self.video_processing_class(**self.video_processor_dict)
        self.assertTrue(hasattr(video_processing, "do_resize"))
        self.assertTrue(hasattr(video_processing, "size"))
        self.assertTrue(hasattr(video_processing, "do_center_crop"))
        self.assertTrue(hasattr(video_processing, "center_crop"))
        self.assertTrue(hasattr(video_processing, "do_normalize"))
        self.assertTrue(hasattr(video_processing, "image_mean"))
        self.assertTrue(hasattr(video_processing, "image_std"))
        self.assertTrue(hasattr(video_processing, "do_convert_rgb"))

    def test_video_processor_from_dict_with_kwargs(self):
        video_processor = self.video_processing_class.from_dict(self.video_processor_dict)
        self.assertEqual(video_processor.size, {"height": 20, "width": 20})
        self.assertEqual(video_processor.crop_size, {"height": 18, "width": 18})

        video_processor = self.video_processing_class.from_dict(self.video_processor_dict, size=42, crop_size=84)
        self.assertEqual(video_processor.size, {"shortest_edge": 42})
        self.assertEqual(video_processor.crop_size, {"height": 84, "width": 84})
