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

from transformers.image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torchvision_available, is_vision_available

from ...test_video_processing_common import VideoProcessingTestMixin, prepare_video_inputs


if is_vision_available():
    if is_torchvision_available():
        from transformers import SmolVLMVideoProcessor


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
        self.max_image_size = size
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
            "max_image_size": self.max_image_size,
        }

    def expected_output_video_shape(self, videos):
        return [
            self.num_frames,
            self.num_channels,
            self.max_image_size["longest_edge"],
            self.max_image_size["longest_edge"],
        ]

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

    # overwrite, SmolVLM requires to have metadata no matter how we sample
    def test_call_sample_frames(self):
        for video_processing_class in self.video_processor_list:
            video_processing = video_processing_class(**self.video_processor_dict)

            prev_num_frames = self.video_processor_tester.num_frames
            self.video_processor_tester.num_frames = 8
            video_inputs = self.video_processor_tester.prepare_video_inputs(
                equal_resolution=False,
                return_tensors="torch",
            )

            # Force set sampling to False. No sampling is expected even when `num_frames` exists
            video_processing.do_sample_frames = False

            encoded_videos = video_processing(video_inputs[0], return_tensors="pt", num_frames=3)[self.input_name]
            encoded_videos_batched = video_processing(video_inputs, return_tensors="pt", num_frames=3)[self.input_name]
            self.assertEqual(encoded_videos.shape[1], 8)
            self.assertEqual(encoded_videos_batched.shape[1], 8)

            # Set sampling to True. Video frames should be sampled with `num_frames` in the output
            video_processing.do_sample_frames = True
            metadata = [[{"duration": 2.0, "total_num_frames": 8, "fps": 4}]]
            batched_metadata = metadata * len(video_inputs)

            # Sample with `fps` requires metadata to infer number of frames from total duration
            with self.assertRaises(ValueError):
                encoded_videos = video_processing(video_inputs[0], return_tensors="pt", num_frames=6, fps=3)[
                    self.input_name
                ]
                encoded_videos_batched = video_processing(video_inputs, return_tensors="pt", num_frames=6, fps=3)[
                    self.input_name
                ]

            encoded_videos = video_processing(
                video_inputs[0], return_tensors="pt", num_frames=6, fps=3, video_metadata=metadata
            )[self.input_name]
            encoded_videos_batched = video_processing(
                video_inputs, return_tensors="pt", num_frames=6, fps=3, video_metadata=batched_metadata
            )[self.input_name]
            self.assertEqual(encoded_videos.shape[1], 6)
            self.assertEqual(encoded_videos_batched.shape[1], 6)

            # We should raise error when asked to sample more frames than there are in input video
            with self.assertRaises(ValueError):
                encoded_videos = video_processing(video_inputs[0], return_tensors="pt", fps=10, num_frames=20)[
                    self.input_name
                ]
                encoded_videos_batched = video_processing(video_inputs, return_tensors="pt", fps=10, num_frames=20)[
                    self.input_name
                ]

            # Assign back the actual num frames in tester
            self.video_processor_tester.num_frames = prev_num_frames
