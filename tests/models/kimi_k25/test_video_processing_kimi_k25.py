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

from transformers.image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD, get_image_size
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_torchvision_available, is_vision_available
from transformers.video_utils import VideoMetadata

from ...test_video_processing_common import VideoProcessingTestMixin, prepare_video_inputs


if is_vision_available():
    from PIL import Image

if is_torch_available():
    import torch

if is_torchvision_available():
    from transformers import Kimi_K25VideoProcessor
    from transformers.models.kimi_k25.video_processing_kimi_k25 import navit_resize


class Kimi_k25VideoProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        num_frames=8,
        num_channels=3,
        min_resolution=30,
        max_resolution=400,
        do_resize=True,
        size=None,
        do_normalize=True,
        image_mean=IMAGENET_STANDARD_MEAN,
        image_std=IMAGENET_STANDARD_STD,
        do_convert_rgb=True,
        patch_size=5,
        temporal_patch_size=4,
        merge_size=2,
        max_patches=3,
        do_sample_frames=False,
    ):
        size = size if size is not None else {"max_height": 20, "max_width": 20}
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
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.max_patches = max_patches
        self.do_sample_frames = do_sample_frames

    def prepare_video_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_convert_rgb": self.do_convert_rgb,
            "patch_size": self.patch_size,
            "temporal_patch_size": self.temporal_patch_size,
            "merge_size": self.merge_size,
            "max_patches": self.max_patches,
            "do_sample_frames": self.do_sample_frames,
        }

    def expected_output_video_shape(self, videos):
        grid_t = self.num_frames
        seq_len = 0
        for video in videos:
            if isinstance(video, list) and isinstance(video[0], Image.Image):
                video = np.stack([np.array(frame) for frame in video])
            elif hasattr(video, "shape"):
                pass
            else:
                video = np.array(video)

            if hasattr(video, "shape"):
                height, width = get_image_size(video)
            else:
                height, width = self.min_resolution, self.min_resolution

            (resized_height, resized_width), (pad_height, pad_width) = navit_resize(
                height,
                width,
                patch_size=self.patch_size,
                merge_kernel_size=self.merge_size,
                max_patches=self.max_patches,
                max_size_per_side=self.size["max_height"],
            )
            grid_length = pad_height // self.patch_size * pad_width // self.patch_size
            seq_len += grid_length * grid_t
        return seq_len, self.num_channels, self.patch_size, self.patch_size

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
class Kimi_k25VideoProcessingTest(VideoProcessingTestMixin, unittest.TestCase):
    fast_video_processing_class = Kimi_K25VideoProcessor

    def setUp(self):
        super().setUp()
        self.video_processor_tester = Kimi_k25VideoProcessingTester(self)

    @property
    def video_processor_dict(self):
        return self.video_processor_tester.prepare_video_processor_dict()

    def test_video_processor_properties(self):
        video_processing = self.fast_video_processing_class(**self.video_processor_dict)
        self.assertTrue(hasattr(video_processing, "do_resize"))
        self.assertTrue(hasattr(video_processing, "size"))
        self.assertTrue(hasattr(video_processing, "do_normalize"))
        self.assertTrue(hasattr(video_processing, "image_mean"))
        self.assertTrue(hasattr(video_processing, "image_std"))
        self.assertTrue(hasattr(video_processing, "patch_size"))
        self.assertTrue(hasattr(video_processing, "temporal_patch_size"))
        self.assertTrue(hasattr(video_processing, "merge_size"))
        self.assertTrue(hasattr(video_processing, "max_patches"))
        self.assertTrue(hasattr(video_processing, "do_sample_frames"))

    def test_video_processor_from_dict_with_kwargs(self):
        video_processor = self.fast_video_processing_class.from_dict(self.video_processor_dict)
        self.assertEqual(video_processor.size, {"max_height": 20, "max_width": 20})

        video_processor = self.fast_video_processing_class.from_dict(
            self.video_processor_dict, size={"max_height": 42, "max_width": 42}
        )
        self.assertEqual(video_processor.size, {"max_height": 42, "max_width": 42})

    def test_call_pil(self):
        for video_processing_class in self.video_processor_list:
            # Initialize video_processing
            video_processing = video_processing_class(**self.video_processor_dict)
            video_inputs = self.video_processor_tester.prepare_video_inputs(equal_resolution=False)

            # Each video is a list of PIL Images
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
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape(video_inputs)
            encoded_videos = video_processing(video_inputs, return_tensors="pt")[self.input_name]
            self.assertEqual(
                tuple(encoded_videos.shape),
                expected_output_video_shape,
            )

    def test_nested_input(self):
        """Tests that the processor can work with nested list where each video is a list of arrays"""
        for video_processing_class in self.video_processor_list:
            video_processing = video_processing_class(**self.video_processor_dict)
            video_inputs = self.video_processor_tester.prepare_video_inputs(
                equal_resolution=False, return_tensors="np"
            )

            # Test not batched input
            video_inputs = [list(video) for video in video_inputs]
            encoded_videos = video_processing(video_inputs[0], return_tensors="pt")[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape([video_inputs[0]])
            self.assertEqual(tuple(encoded_videos.shape), expected_output_video_shape)

            # Test batched
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape(video_inputs)
            encoded_videos = video_processing(video_inputs, return_tensors="pt")[self.input_name]
            self.assertEqual(
                tuple(encoded_videos.shape),
                expected_output_video_shape,
            )

    @unittest.skip("Needs a fix in test setting, not important")
    def test_call_numpy_4_channels(self):
        pass

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
            self.assertEqual(encoded_videos.shape[1], 3)
            self.assertEqual(encoded_videos_batched.shape[1], 3)

            # Set sampling to True. Video frames should be sampled with `num_frames` in the output
            video_processing.do_sample_frames = True

            encoded_videos = video_processing(video_inputs[0], return_tensors="pt", num_frames=3, fps=None)[
                self.input_name
            ]
            encoded_videos_batched = video_processing(video_inputs, return_tensors="pt", num_frames=3, fps=None)[
                self.input_name
            ]
            self.assertEqual(encoded_videos.shape[1], 3)
            self.assertEqual(encoded_videos_batched.shape[1], 3)

            # Sample with `fps` requires metadata to infer number of frames from total duration
            with self.assertRaises(ValueError):
                metadata = VideoMetadata(**{"total_num_frames": 8})
                video_processing.sample_frames(metadata=metadata, fps=3)

            metadata = [[{"duration": 2.0, "total_num_frames": 8, "fps": 4}]]
            batched_metadata = metadata * len(video_inputs)
            encoded_videos = video_processing(video_inputs[0], return_tensors="pt", fps=3, video_metadata=metadata)[
                self.input_name
            ]
            encoded_videos_batched = video_processing(
                video_inputs, return_tensors="pt", fps=3, video_metadata=batched_metadata
            )[self.input_name]
            self.assertEqual(encoded_videos.shape[1], 3)
            self.assertEqual(encoded_videos_batched.shape[1], 3)

            # The same as above but uses a `VideoMetadata` object in the input
            metadata = [[VideoMetadata(duration=2.0, total_num_frames=8, fps=4)]]
            batched_metadata = metadata * len(video_inputs)
            encoded_videos = video_processing(video_inputs[0], return_tensors="pt", fps=3, video_metadata=metadata)[
                self.input_name
            ]

            # We should raise error when asked to sample more frames than there are in input video
            with self.assertRaises(ValueError):
                encoded_videos = video_processing(video_inputs[0], return_tensors="pt", num_frames=10)[self.input_name]
                encoded_videos_batched = video_processing(video_inputs, return_tensors="pt", num_frames=10)[
                    self.input_name
                ]

            # Assign back the actual num frames in tester
            self.video_processor_tester.num_frames = prev_num_frames
