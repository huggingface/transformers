# Copyright 2026 the HuggingFace Team. All rights reserved.
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


if is_torchvision_available():
    from transformers import Glm5NextVideoProcessor
    from transformers.models.glm5_next.video_processing_glm5_next import smart_resize

if is_vision_available():
    from PIL import Image


class Glm5NextVideoProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        num_frames=8,
        num_channels=3,
        min_resolution=30,
        max_resolution=80,
        do_resize=True,
        do_rescale=True,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        do_convert_rgb=True,
        temporal_patch_size=2,
        patch_size=14,
        merge_size=2,
        patch_expand_factor=1,
        min_image_tokens=1,
        max_image_tokens=64,
        fps=2,
        max_frames=16,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.num_channels = num_channels
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.do_rescale = do_rescale
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_convert_rgb = do_convert_rgb
        self.temporal_patch_size = temporal_patch_size
        self.patch_size = patch_size
        self.merge_size = merge_size
        self.patch_expand_factor = patch_expand_factor
        self.min_image_tokens = min_image_tokens
        self.max_image_tokens = max_image_tokens
        self.fps = fps
        self.max_frames = max_frames

    def prepare_video_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "do_rescale": self.do_rescale,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_convert_rgb": self.do_convert_rgb,
            "temporal_patch_size": self.temporal_patch_size,
            "patch_size": self.patch_size,
            "merge_size": self.merge_size,
            "patch_expand_factor": self.patch_expand_factor,
            "min_image_tokens": self.min_image_tokens,
            "max_image_tokens": self.max_image_tokens,
            "fps": self.fps,
            "max_frames": self.max_frames,
            "do_sample_frames": True,
        }

    def prepare_video_metadata(self, videos):
        video_metadata = []
        for video in videos:
            if isinstance(video, list):
                num_frames = len(video)
            else:
                num_frames = video.shape[0]

            metadata = {
                "fps": 2,
                "duration": num_frames / 2,
                "total_num_frames": num_frames,
            }
            video_metadata.append(metadata)

        return video_metadata

    def expected_output_video_shape(self, videos):
        grid_t = self.num_frames // self.temporal_patch_size
        hidden_dim = self.num_channels * self.temporal_patch_size * self.patch_size**2
        factor = self.patch_size * self.merge_size * self.patch_expand_factor
        seq_len = 0
        for video in videos:
            if isinstance(video, list):
                frame = video[0]
                num_frames = len(video)
                if isinstance(frame, Image.Image):
                    height, width = frame.size[1], frame.size[0]
                elif isinstance(frame, np.ndarray):
                    height, width = frame.shape[:2]
                else:
                    height, width = frame.shape[-2:]
            elif isinstance(video, np.ndarray):
                num_frames, height, width = video.shape[:3]
            else:
                num_frames = video.shape[0]
                height, width = video.shape[-2:]
            resized_height, resized_width = smart_resize(
                num_frames,
                height,
                width,
                temporal_factor=self.temporal_patch_size,
                factor=factor,
                min_pixels=self.min_image_tokens,
                max_pixels=self.max_image_tokens,
            )

            grid_h = resized_height // self.patch_size
            grid_w = resized_width // self.patch_size
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
class Glm5NextVideoProcessingTest(VideoProcessingTestMixin, unittest.TestCase):
    fast_video_processing_class = Glm5NextVideoProcessor if is_torchvision_available() else None
    input_name = "pixel_values_videos"

    def setUp(self):
        super().setUp()
        self.video_processor_tester = Glm5NextVideoProcessingTester(self)

    @property
    def video_processor_dict(self):
        return self.video_processor_tester.prepare_video_processor_dict()

    def test_video_processor_from_dict_with_kwargs(self):
        for video_processing_class in self.video_processor_list:
            video_processor = video_processing_class.from_dict(self.video_processor_dict)
            self.assertEqual(video_processor.min_image_tokens, 1)

            video_processor = video_processing_class.from_dict(
                self.video_processor_dict,
                min_image_tokens=42,
            )
            self.assertEqual(video_processor.min_image_tokens, 42)

    def test_call_pil(self):
        for video_processing_class in self.video_processor_list:
            video_processing = video_processing_class(**self.video_processor_dict)
            video_inputs = self.video_processor_tester.prepare_video_inputs(
                equal_resolution=False,
                return_tensors="pil",
            )

            for video in video_inputs:
                self.assertIsInstance(video[0], Image.Image)

            video_metadata = self.video_processor_tester.prepare_video_metadata(video_inputs)

            encoded_videos = video_processing(
                video_inputs[0],
                video_metadata=[video_metadata[0]],
                return_tensors="pt",
            )[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape([video_inputs[0]])
            self.assertEqual(list(encoded_videos.shape), expected_output_video_shape)

            encoded_videos = video_processing(
                video_inputs,
                video_metadata=video_metadata,
                return_tensors="pt",
            )[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape(video_inputs)
            self.assertEqual(list(encoded_videos.shape), expected_output_video_shape)

    def test_call_numpy(self):
        for video_processing_class in self.video_processor_list:
            video_processing = video_processing_class(**self.video_processor_dict)
            video_inputs = self.video_processor_tester.prepare_video_inputs(
                equal_resolution=False,
                return_tensors="np",
            )

            video_metadata = self.video_processor_tester.prepare_video_metadata(video_inputs)

            encoded_videos = video_processing(
                video_inputs[0],
                video_metadata=[video_metadata[0]],
                return_tensors="pt",
            )[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape([video_inputs[0]])
            self.assertEqual(list(encoded_videos.shape), expected_output_video_shape)

            encoded_videos = video_processing(
                video_inputs,
                video_metadata=video_metadata,
                return_tensors="pt",
            )[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape(video_inputs)
            self.assertEqual(list(encoded_videos.shape), expected_output_video_shape)

    def test_call_pytorch(self):
        for video_processing_class in self.video_processor_list:
            video_processing = video_processing_class(**self.video_processor_dict)
            video_inputs = self.video_processor_tester.prepare_video_inputs(
                equal_resolution=False,
                return_tensors="pt",
            )

            video_metadata = self.video_processor_tester.prepare_video_metadata(video_inputs)

            encoded_videos = video_processing(
                video_inputs[0],
                video_metadata=[video_metadata[0]],
                return_tensors="pt",
            )[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape([video_inputs[0]])
            self.assertEqual(list(encoded_videos.shape), expected_output_video_shape)

            encoded_videos = video_processing(
                video_inputs,
                video_metadata=video_metadata,
                return_tensors="pt",
            )[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape(video_inputs)
            self.assertEqual(list(encoded_videos.shape), expected_output_video_shape)

    def test_call_numpy_4_channels(self):
        for video_processing_class in self.video_processor_list:
            # Test that can process videos which have an arbitrary number of channels
            video_processor = video_processing_class(**self.video_processor_dict)

            self.video_processor_tester.num_channels = 4
            video_inputs = self.video_processor_tester.prepare_video_inputs(
                equal_resolution=False,
                return_tensors="np",
            )
            video_metadata = self.video_processor_tester.prepare_video_metadata(video_inputs)

            # Test not batched input
            encoded_videos = video_processor(
                video_inputs[0],
                video_metadata=[video_metadata[0]],
                return_tensors="pt",
                do_convert_rgb=False,
                input_data_format="channels_last",
                image_mean=(0.0, 0.0, 0.0, 0.0),
                image_std=(1.0, 1.0, 1.0, 1.0),
            )[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape([video_inputs[0]])
            self.assertEqual(list(encoded_videos.shape), expected_output_video_shape)

            # Test batched
            encoded_videos = video_processor(
                video_inputs,
                video_metadata=video_metadata,
                return_tensors="pt",
                do_convert_rgb=False,
                input_data_format="channels_last",
                image_mean=(0.0, 0.0, 0.0, 0.0),
                image_std=(1.0, 1.0, 1.0, 1.0),
            )[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape(video_inputs)
            self.assertEqual(list(encoded_videos.shape), expected_output_video_shape)

    def test_nested_input(self):
        """Tests that the processor can work with nested list where each video is a list of arrays."""
        for video_processing_class in self.video_processor_list:
            video_processing = video_processing_class(**self.video_processor_dict)
            video_inputs = self.video_processor_tester.prepare_video_inputs(
                equal_resolution=False,
                return_tensors="np",
            )

            video_inputs_nested = [list(video) for video in video_inputs]
            video_metadata = self.video_processor_tester.prepare_video_metadata(video_inputs)

            # Test not batched input
            encoded_videos = video_processing(
                video_inputs_nested[0],
                video_metadata=[video_metadata[0]],
                return_tensors="pt",
            )[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape([video_inputs[0]])
            self.assertEqual(list(encoded_videos.shape), expected_output_video_shape)

            # Test batched
            encoded_videos = video_processing(
                video_inputs_nested,
                video_metadata=video_metadata,
                return_tensors="pt",
            )[self.input_name]
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

            encoded_videos = video_processing(
                video_inputs[0],
                return_tensors="pt",
                video_metadata=metadata,
            )[self.input_name]
            encoded_videos_batched = video_processing(
                video_inputs,
                return_tensors="pt",
                video_metadata=batched_metadata,
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
