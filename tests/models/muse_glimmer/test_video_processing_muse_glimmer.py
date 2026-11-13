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

import unittest

import numpy as np

from transformers.image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_torchvision_available, is_vision_available
from transformers.video_utils import VideoMetadata

from ...test_video_processing_common import VideoProcessingTestMixin, prepare_video_inputs


if is_torch_available():
    import torch
    from PIL import Image

if is_vision_available():
    if is_torchvision_available():
        from transformers import MuseGlimmerVideoProcessor
        from transformers.models.muse_glimmer.video_processing_muse_glimmer import smart_resize


class MuseGlimmerVideoProcessingTester:
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
        max_video_frame_tokens=40,
        do_resize=True,
        do_normalize=True,
        image_mean=IMAGENET_STANDARD_MEAN,
        image_std=IMAGENET_STANDARD_STD,
        do_convert_rgb=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.num_channels = num_channels
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.temporal_patch_size = temporal_patch_size
        self.patch_size = patch_size
        self.merge_size = merge_size
        self.max_video_frame_tokens = max_video_frame_tokens
        self.do_resize = do_resize
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_convert_rgb = do_convert_rgb

    def prepare_video_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_convert_rgb": self.do_convert_rgb,
            "temporal_patch_size": self.temporal_patch_size,
            "patch_size": self.patch_size,
            "merge_size": self.merge_size,
            "max_video_frame_tokens": self.max_video_frame_tokens,
            "do_sample_frames": False,
        }

    def expected_output_video_shape(self, videos):
        hidden_dim = self.num_channels * self.temporal_patch_size * self.patch_size * self.patch_size
        seq_len = 0
        for video in videos:
            if isinstance(video, list) and isinstance(video[0], Image.Image):
                video = np.stack([np.array(frame) for frame in video])
            elif not hasattr(video, "shape"):
                video = np.array(video)

            # PIL and numpy videos are (frames, height, width, channels), torch videos are channels first.
            num_frames = video.shape[0]
            height, width = video.shape[1:3] if video.shape[-1] == self.num_channels else video.shape[-2:]
            resized_height, resized_width = smart_resize(
                height,
                width,
                patch_size=self.patch_size * self.merge_size,
                max_tokens=self.max_video_frame_tokens,
            )
            grid_t = -(-num_frames // self.temporal_patch_size)
            grid_h, grid_w = resized_height // self.patch_size, resized_width // self.patch_size
            seq_len += grid_t * grid_h * grid_w
        return (seq_len, hidden_dim)

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
class MuseGlimmerVideoProcessingTest(VideoProcessingTestMixin, unittest.TestCase):
    fast_video_processing_class = MuseGlimmerVideoProcessor if is_torchvision_available() else None
    input_name = "pixel_values_videos"

    def setUp(self):
        super().setUp()
        self.video_processor_tester = MuseGlimmerVideoProcessingTester(self)

    @property
    def video_processor_dict(self):
        return self.video_processor_tester.prepare_video_processor_dict()

    def test_video_processor_from_dict_with_kwargs(self):
        # Overwritten -- resizing is driven by `max_video_frame_tokens`, there is no `size` dict.
        video_processor = self.fast_video_processing_class.from_dict(self.video_processor_dict)
        self.assertEqual(video_processor.max_video_frame_tokens, 40)
        self.assertEqual(video_processor.patch_size, 14)

        video_processor = self.fast_video_processing_class.from_dict(
            self.video_processor_dict, max_video_frame_tokens=64
        )
        self.assertEqual(video_processor.max_video_frame_tokens, 64)

    # batch size is flattened
    def test_call_pil(self):
        for video_processing_class in self.video_processor_list:
            # Initialize video_processing
            video_processing = video_processing_class(**self.video_processor_dict)
            # create random PIL videos
            video_inputs = self.video_processor_tester.prepare_video_inputs(
                equal_resolution=False, return_tensors="pil"
            )
            for video in video_inputs:
                self.assertIsInstance(video[0], Image.Image)

            # Test not batched input
            encoded_videos = video_processing(video_inputs[0], return_tensors="pt")[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape([video_inputs[0]])
            self.assertEqual(tuple(encoded_videos.shape), expected_output_video_shape)

            # Test batched
            encoded_videos = video_processing(video_inputs, return_tensors="pt")[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape(video_inputs)
            self.assertEqual(tuple(encoded_videos.shape), expected_output_video_shape)

    def test_call_numpy(self):
        for video_processing_class in self.video_processor_list:
            # Initialize video_processing
            video_processing = video_processing_class(**self.video_processor_dict)
            # create random numpy tensors
            video_inputs = self.video_processor_tester.prepare_video_inputs(
                equal_resolution=False, return_tensors="np"
            )
            for video in video_inputs:
                self.assertIsInstance(video, np.ndarray)

            # Test not batched input
            encoded_videos = video_processing(video_inputs[0], return_tensors="pt")[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape([video_inputs[0]])
            self.assertEqual(tuple(encoded_videos.shape), expected_output_video_shape)

            # Test batched
            encoded_videos = video_processing(video_inputs, return_tensors="pt")[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape(video_inputs)
            self.assertEqual(tuple(encoded_videos.shape), expected_output_video_shape)

    def test_call_pytorch(self):
        for video_processing_class in self.video_processor_list:
            # Initialize video_processing
            video_processing = video_processing_class(**self.video_processor_dict)
            # create random PyTorch tensors
            video_inputs = self.video_processor_tester.prepare_video_inputs(
                equal_resolution=False, return_tensors="torch"
            )
            for video in video_inputs:
                self.assertIsInstance(video, torch.Tensor)

            # Test not batched input
            encoded_videos = video_processing(video_inputs[0], return_tensors="pt")[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape([video_inputs[0]])
            self.assertEqual(tuple(encoded_videos.shape), expected_output_video_shape)

            # Test batched
            encoded_videos = video_processing(video_inputs, return_tensors="pt")[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape(video_inputs)
            self.assertEqual(tuple(encoded_videos.shape), expected_output_video_shape)

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
                do_convert_rgb=False,
                image_mean=(0.0, 0.0, 0.0, 0.0),
                image_std=(1.0, 1.0, 1.0, 1.0),
            )[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape([video_inputs[0]])
            self.assertEqual(tuple(encoded_videos.shape), expected_output_video_shape)

            # Test batched
            encoded_videos = video_processor(
                video_inputs,
                return_tensors="pt",
                input_data_format="channels_last",
                do_convert_rgb=False,
                image_mean=(0.0, 0.0, 0.0, 0.0),
                image_std=(1.0, 1.0, 1.0, 1.0),
            )[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape(video_inputs)
            self.assertEqual(tuple(encoded_videos.shape), expected_output_video_shape)

    def test_nested_input(self):
        for video_processing_class in self.video_processor_list:
            video_processing = video_processing_class(**self.video_processor_dict)
            video_inputs = self.video_processor_tester.prepare_video_inputs(
                equal_resolution=False, return_tensors="np"
            )
            video_inputs_nested = [list(video) for video in video_inputs]

            encoded_videos = video_processing(video_inputs_nested[0], return_tensors="pt")[self.input_name]
            expected_shape = self.video_processor_tester.expected_output_video_shape([video_inputs[0]])
            self.assertEqual(tuple(encoded_videos.shape), expected_shape)

            encoded_videos = video_processing(video_inputs_nested, return_tensors="pt")[self.input_name]
            expected_shape = self.video_processor_tester.expected_output_video_shape(video_inputs)
            self.assertEqual(tuple(encoded_videos.shape), expected_shape)

    def test_call_sample_frames(self):
        # Overwritten -- the sampled frame count is the smallest of the `fps` budget and `num_frames`,
        # then rounded down to a multiple of `temporal_patch_size`.
        temporal_patch_size = self.video_processor_tester.temporal_patch_size
        video_inputs = self.video_processor_tester.prepare_video_inputs(equal_resolution=True, return_tensors="torch")
        # 8 frames recorded at 4 fps, so a 2 fps budget keeps 4 of them.
        metadata = [[{"duration": 2.0, "total_num_frames": 8, "fps": 4}]]

        for video_processing_class in self.video_processor_list:
            video_processing = video_processing_class(**self.video_processor_dict)
            video_processing.do_sample_frames = True

            for requested_frames, requested_fps, expected_frames in [
                (3, 2.0, 2),
                (4, 2.0, 4),
                (8, 2.0, 4),
                (8, 4.0, 8),
                (8, 1.0, 2),
            ]:
                encoded = video_processing(
                    video_inputs[0],
                    return_tensors="pt",
                    num_frames=requested_frames,
                    fps=requested_fps,
                    video_metadata=metadata,
                )
                grid_t = encoded["video_grid_thw"][0][0].item()
                self.assertEqual(grid_t, expected_frames // temporal_patch_size)

    def test_sample_frames_without_fps_defaults_to_24(self):
        video_processor = self.fast_video_processing_class(**self.video_processor_dict)
        metadata = VideoMetadata(total_num_frames=48, fps=None, duration=None)

        indices = video_processor.sample_frames(metadata=metadata, temporal_patch_size=2, num_frames=96, fps=2.0)

        self.assertEqual(metadata.fps, 24)
        self.assertEqual(len(indices), 4)
