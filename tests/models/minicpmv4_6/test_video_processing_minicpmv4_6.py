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
from transformers.utils import is_torch_available, is_torchvision_available, is_vision_available
from transformers.video_utils import VideoMetadata

from ...test_video_processing_common import VideoProcessingTestMixin, prepare_video_inputs


if is_torch_available():
    import torch

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
        do_resize=True,
        do_normalize=True,
        image_mean=IMAGENET_STANDARD_MEAN,
        image_std=IMAGENET_STANDARD_STD,
        do_convert_rgb=True,
        max_slice_nums=5,
        scale_resolution=448,
        patch_size=14,
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
        self.max_slice_nums = max_slice_nums
        self.scale_resolution = scale_resolution
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
        """Return the expected NaViT-packed shape [C, P, total_L] for encoded_videos[0]."""
        total_L = 0
        for video in video_inputs:
            if isinstance(video, list) and isinstance(video[0], Image.Image):
                frames = np.stack([np.array(frame) for frame in video])
            elif hasattr(video, "shape"):
                frames = video
            else:
                frames = np.array(video)

            height, width = get_image_size(frames[0])
            num_frames = len(frames)

            aspect_ratio = width / height
            new_height = int(self.scale_resolution / math.sqrt(aspect_ratio))
            new_width = int(new_height * aspect_ratio)

            divisor = self.patch_size * 4
            best_height = max(round(new_height / divisor) * divisor, divisor)
            best_width = max(round(new_width / divisor) * divisor, divisor)

            total_L += num_frames * (best_height * best_width // self.patch_size)

        return [self.num_channels, self.patch_size, total_L]

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
        self.assertEqual(video_processor.patch_size, 14)

        video_processor = self.fast_video_processing_class.from_dict(self.video_processor_dict, patch_size=36)
        self.assertEqual(video_processor.patch_size, 36)

    def test_call_pil(self):
        for video_processing_class in self.video_processor_list:
            video_processing = video_processing_class(**self.video_processor_dict)
            video_inputs = self.video_processor_tester.prepare_video_inputs(equal_resolution=False)

            for video in video_inputs:
                self.assertIsInstance(video[0], Image.Image)

            # Test not batched input
            encoded_videos = video_processing(video_inputs[0], return_tensors="pt")[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape([video_inputs[0]])
            self.assertEqual(len(encoded_videos), 1)
            self.assertListEqual(list(encoded_videos[0].shape), expected_output_video_shape)

            # Test batched
            encoded_videos = video_processing(video_inputs, return_tensors="pt")[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape(video_inputs)
            self.assertEqual(len(encoded_videos), 1)
            self.assertListEqual(list(encoded_videos[0].shape), expected_output_video_shape)

    def test_call_numpy(self):
        for video_processing_class in self.video_processor_list:
            video_processing = video_processing_class(**self.video_processor_dict)
            video_inputs = self.video_processor_tester.prepare_video_inputs(
                equal_resolution=False, return_tensors="np"
            )
            for video in video_inputs:
                self.assertIsInstance(video, np.ndarray)

            # Test not batched input
            encoded_videos = video_processing(video_inputs[0], return_tensors="pt")[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape([video_inputs[0]])
            self.assertEqual(len(encoded_videos), 1)
            self.assertListEqual(list(encoded_videos[0].shape), expected_output_video_shape)

            # Test batched
            encoded_videos = video_processing(video_inputs, return_tensors="pt")[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape(video_inputs)
            self.assertEqual(len(encoded_videos), 1)
            self.assertListEqual(list(encoded_videos[0].shape), expected_output_video_shape)

    def test_call_pytorch(self):
        for video_processing_class in self.video_processor_list:
            video_processing = video_processing_class(**self.video_processor_dict)
            video_inputs = self.video_processor_tester.prepare_video_inputs(
                equal_resolution=False, return_tensors="torch"
            )

            for video in video_inputs:
                self.assertIsInstance(video, torch.Tensor)

            # Test not batched input
            encoded_videos = video_processing(video_inputs[0], return_tensors="pt")[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape([video_inputs[0]])
            self.assertEqual(len(encoded_videos), 1)
            self.assertListEqual(list(encoded_videos[0].shape), expected_output_video_shape)

            # Test batched
            encoded_videos = video_processing(video_inputs, return_tensors="pt")[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape(video_inputs)
            self.assertEqual(len(encoded_videos), 1)
            self.assertListEqual(list(encoded_videos[0].shape), expected_output_video_shape)

    def test_nested_input(self):
        """NaViT packing: dim 0 is always 1 regardless of batch size."""
        for video_processing_class in self.video_processor_list:
            video_processing = video_processing_class(**self.video_processor_dict)
            video_inputs = self.video_processor_tester.prepare_video_inputs(
                equal_resolution=False, return_tensors="np"
            )
            video_inputs = [list(video) for video in video_inputs]

            # Test not batched input
            encoded_videos = video_processing(video_inputs[0], return_tensors="pt")[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape([video_inputs[0]])
            self.assertEqual(tuple(encoded_videos.shape), (1, *expected_output_video_shape))

            # Test batched
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape(video_inputs)
            encoded_videos = video_processing(video_inputs, return_tensors="pt")[self.input_name]
            self.assertEqual(tuple(encoded_videos.shape), (1, *expected_output_video_shape))

    @unittest.skip("NaViT expected_output_video_shape cannot infer channel dim for 4-channel images")
    def test_call_numpy_4_channels(self):
        pass

    def test_call_sample_frames(self):
        for video_processing_class in self.video_processor_list:
            video_processor_dict = self.video_processor_dict.copy()
            video_inputs = self.video_processor_tester.prepare_video_inputs(
                equal_resolution=False,
                return_tensors="torch",
            )
            video_metadata = [
                VideoMetadata(total_num_frames=len(video), duration=4.0, fps=2.0) for video in video_inputs
            ]
            sampled_video_processor_dict = {**video_processor_dict, "do_sample_frames": True}

            # stack_frames=1: sample one main frame per second from complete metadata.
            video_processor = video_processing_class(**sampled_video_processor_dict, max_num_frames=20, stack_frames=1)

            encoded_videos = video_processor(
                video_inputs[0],
                video_metadata=[video_metadata[0]],
                return_tensors="pt",
            )[self.input_name]
            expected_shape = self.video_processor_tester.expected_output_video_shape([video_inputs[0][[0, 2, 4, 6]]])
            self.assertEqual(len(encoded_videos), 1)
            self.assertListEqual(list(encoded_videos[0].shape), expected_shape)

            encoded_videos_batched = video_processor(video_inputs, video_metadata=video_metadata, return_tensors="pt")[
                self.input_name
            ]
            expected_shape_batched = self.video_processor_tester.expected_output_video_shape(
                [video[[0, 2, 4, 6]] for video in video_inputs]
            )
            self.assertEqual(len(encoded_videos_batched), 1)
            self.assertListEqual(list(encoded_videos_batched[0].shape), expected_shape_batched)

            # stack_frames=2, duration=4.0: tensor layout = [4 main | 4 sub]
            # Each second gets 1 sub-frame composited (single frame → same size as main)
            # → 8 visual units interleaved: [main_0, comp_0, ..., main_3, comp_3]
            video_processor_sf2 = video_processing_class(
                **sampled_video_processor_dict, max_num_frames=20, stack_frames=2
            )

            encoded_videos_sf2 = video_processor_sf2(
                video_inputs[0],
                video_metadata=[video_metadata[0]],
                return_tensors="pt",
            )[self.input_name]
            self.assertEqual(len(encoded_videos_sf2), 1)
            self.assertListEqual(
                list(encoded_videos_sf2[0].shape),
                self.video_processor_tester.expected_output_video_shape([video_inputs[0]]),
            )
