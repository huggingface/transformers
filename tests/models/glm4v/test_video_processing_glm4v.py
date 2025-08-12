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
    from PIL import Image

if is_vision_available():
    if is_torchvision_available():
        from transformers import Glm4vVideoProcessor
        from transformers.models.glm4v.video_processing_glm4v import smart_resize


class Glm4vVideoProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=5,
        num_frames=8,
        num_channels=3,
        min_resolution=30,
        max_resolution=80,
        temporal_patch_size=2,
        patch_size=14,
        merge_size=2,
        do_resize=True,
        size=None,
        do_normalize=True,
        image_mean=IMAGENET_STANDARD_MEAN,
        image_std=IMAGENET_STANDARD_STD,
        do_convert_rgb=True,
    ):
        size = size if size is not None else {"longest_edge": 20, "shortest_edge": 10}
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
        self.temporal_patch_size = temporal_patch_size
        self.patch_size = patch_size
        self.merge_size = merge_size

    def prepare_video_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_convert_rgb": self.do_convert_rgb,
            "do_sample_frames": True,
        }

    def prepare_video_metadata(self, videos):
        video_metadata = []
        for video in videos:
            if isinstance(video, list):
                num_frames = len(video)
            elif hasattr(video, "shape"):
                if len(video.shape) == 4:  # (T, H, W, C)
                    num_frames = video.shape[0]
                else:
                    num_frames = 1
            else:
                num_frames = self.num_frames

            metadata = {
                "fps": 2,
                "duration": num_frames / 2,
                "total_num_frames": num_frames,
            }
            video_metadata.append(metadata)
        return video_metadata

    def expected_output_video_shape(self, videos):
        grid_t = self.num_frames // self.temporal_patch_size
        hidden_dim = self.num_channels * self.temporal_patch_size * self.patch_size * self.patch_size
        seq_len = 0
        for video in videos:
            if isinstance(video, list) and isinstance(video[0], Image.Image):
                video = np.stack([np.array(frame) for frame in video])
            elif hasattr(video, "shape"):
                pass
            else:
                video = np.array(video)

            if hasattr(video, "shape") and len(video.shape) >= 3:
                if len(video.shape) == 4:
                    t, height, width = video.shape[:3]
                elif len(video.shape) == 3:
                    height, width = video.shape[:2]
                    t = 1
                else:
                    t, height, width = self.num_frames, self.min_resolution, self.min_resolution
            else:
                t, height, width = self.num_frames, self.min_resolution, self.min_resolution

            resized_height, resized_width = smart_resize(
                t,
                height,
                width,
                factor=self.patch_size * self.merge_size,
                min_pixels=self.size["shortest_edge"],
                max_pixels=self.size["longest_edge"],
            )
            grid_h, grid_w = resized_height // self.patch_size, resized_width // self.patch_size
            seq_len += grid_t * grid_h * grid_w
        return [seq_len, hidden_dim]

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
class Glm4vVideoProcessingTest(VideoProcessingTestMixin, unittest.TestCase):
    fast_video_processing_class = Glm4vVideoProcessor if is_torchvision_available() else None
    input_name = "pixel_values_videos"

    def setUp(self):
        super().setUp()
        self.video_processor_tester = Glm4vVideoProcessingTester(self)

    @property
    def video_processor_dict(self):
        return self.video_processor_tester.prepare_video_processor_dict()

    def test_video_processor_from_dict_with_kwargs(self):
        video_processor = self.fast_video_processing_class.from_dict(self.video_processor_dict)
        self.assertEqual(video_processor.size, {"longest_edge": 20, "shortest_edge": 10})

        video_processor = self.fast_video_processing_class.from_dict(
            self.video_processor_dict, size={"longest_edge": 42, "shortest_edge": 42}
        )
        self.assertEqual(video_processor.size, {"longest_edge": 42, "shortest_edge": 42})

    def test_call_pil(self):
        for video_processing_class in self.video_processor_list:
            video_processing = video_processing_class(**self.video_processor_dict)
            video_inputs = self.video_processor_tester.prepare_video_inputs(
                equal_resolution=False, return_tensors="pil"
            )

            for video in video_inputs:
                self.assertIsInstance(video[0], Image.Image)

            video_metadata = self.video_processor_tester.prepare_video_metadata(video_inputs)
            encoded_videos = video_processing(
                video_inputs[0], video_metadata=[video_metadata[0]], return_tensors="pt"
            )[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape([video_inputs[0]])
            self.assertEqual(list(encoded_videos.shape), expected_output_video_shape)
            encoded_videos = video_processing(video_inputs, video_metadata=video_metadata, return_tensors="pt")[
                self.input_name
            ]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape(video_inputs)
            self.assertEqual(list(encoded_videos.shape), expected_output_video_shape)

    def test_call_numpy(self):
        for video_processing_class in self.video_processor_list:
            video_processing = video_processing_class(**self.video_processor_dict)
            video_inputs = self.video_processor_tester.prepare_video_inputs(
                equal_resolution=False, return_tensors="np"
            )

            video_metadata = self.video_processor_tester.prepare_video_metadata(video_inputs)
            encoded_videos = video_processing(
                video_inputs[0], video_metadata=[video_metadata[0]], return_tensors="pt"
            )[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape([video_inputs[0]])
            self.assertEqual(list(encoded_videos.shape), expected_output_video_shape)

            encoded_videos = video_processing(video_inputs, video_metadata=video_metadata, return_tensors="pt")[
                self.input_name
            ]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape(video_inputs)
            self.assertEqual(list(encoded_videos.shape), expected_output_video_shape)

    def test_call_pytorch(self):
        for video_processing_class in self.video_processor_list:
            video_processing = video_processing_class(**self.video_processor_dict)
            video_inputs = self.video_processor_tester.prepare_video_inputs(
                equal_resolution=False, return_tensors="pt"
            )
            video_metadata = self.video_processor_tester.prepare_video_metadata(video_inputs)
            encoded_videos = video_processing(
                video_inputs[0], video_metadata=[video_metadata[0]], return_tensors="pt"
            )[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape([video_inputs[0]])
            self.assertEqual(list(encoded_videos.shape), expected_output_video_shape)
            encoded_videos = video_processing(video_inputs, video_metadata=video_metadata, return_tensors="pt")[
                self.input_name
            ]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape(video_inputs)
            self.assertEqual(list(encoded_videos.shape), expected_output_video_shape)

    @unittest.skip("Skip for now, the test needs adjustment for GLM-4.1V")
    def test_call_numpy_4_channels(self):
        for video_processing_class in self.video_processor_list:
            # Test that can process videos which have an arbitrary number of channels
            # Initialize video_processing
            video_processor = video_processing_class(**self.video_processor_dict)

            # create random numpy tensors
            self.video_processor_tester.num_channels = 4
            video_inputs = self.video_processor_tester.prepare_video_inputs(
                equal_resolution=False, return_tensors="np"
            )

            # Test not batched input
            encoded_videos = video_processor(
                video_inputs[0],
                return_tensors="pt",
                input_data_format="channels_last",
                image_mean=0,
                image_std=1,
            )[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape([video_inputs[0]])
            self.assertEqual(list(encoded_videos.shape), expected_output_video_shape)

            # Test batched
            encoded_videos = video_processor(
                video_inputs,
                return_tensors="pt",
                input_data_format="channels_last",
                image_mean=0,
                image_std=1,
            )[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape(video_inputs)
            self.assertEqual(list(encoded_videos.shape), expected_output_video_shape)

    def test_nested_input(self):
        """Tests that the processor can work with nested list where each video is a list of arrays"""
        for video_processing_class in self.video_processor_list:
            video_processing = video_processing_class(**self.video_processor_dict)
            video_inputs = self.video_processor_tester.prepare_video_inputs(
                equal_resolution=False, return_tensors="np"
            )

            video_inputs_nested = [list(video) for video in video_inputs]
            video_metadata = self.video_processor_tester.prepare_video_metadata(video_inputs)

            # Test not batched input
            encoded_videos = video_processing(
                video_inputs_nested[0], video_metadata=[video_metadata[0]], return_tensors="pt"
            )[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape([video_inputs[0]])
            self.assertEqual(list(encoded_videos.shape), expected_output_video_shape)

            # Test batched
            encoded_videos = video_processing(video_inputs_nested, video_metadata=video_metadata, return_tensors="pt")[
                self.input_name
            ]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape(video_inputs)
            self.assertEqual(list(encoded_videos.shape), expected_output_video_shape)

    def test_call_sample_frames(self):
        for video_processing_class in self.video_processor_list:
            video_processor_dict = self.video_processor_dict.copy()
            video_processing = video_processing_class(**video_processor_dict)

            prev_num_frames = self.video_processor_tester.num_frames
            self.video_processor_tester.num_frames = 8
            prev_min_resolution = getattr(self.video_processor_tester, "min_resolution", None)
            prev_max_resolution = getattr(self.video_processor_tester, "max_resolution", None)
            self.video_processor_tester.min_resolution = 56
            self.video_processor_tester.max_resolution = 112

            video_inputs = self.video_processor_tester.prepare_video_inputs(
                equal_resolution=False,
                return_tensors="torch",
            )

            metadata = [[{"total_num_frames": 8, "fps": 4}]]
            batched_metadata = metadata * len(video_inputs)

            encoded_videos = video_processing(video_inputs[0], return_tensors="pt", video_metadata=metadata)[
                self.input_name
            ]
            encoded_videos_batched = video_processing(
                video_inputs, return_tensors="pt", video_metadata=batched_metadata
            )[self.input_name]

            self.assertIsNotNone(encoded_videos)
            self.assertIsNotNone(encoded_videos_batched)
            self.assertEqual(len(encoded_videos.shape), 2)
            self.assertEqual(len(encoded_videos_batched.shape), 2)

            with self.assertRaises(ValueError):
                video_processing(video_inputs[0], return_tensors="pt")[self.input_name]

            self.video_processor_tester.num_frames = prev_num_frames
            if prev_min_resolution is not None:
                self.video_processor_tester.min_resolution = prev_min_resolution
            if prev_max_resolution is not None:
                self.video_processor_tester.max_resolution = prev_max_resolution
