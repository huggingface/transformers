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

from transformers.image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_torchvision_available, is_vision_available

from ...test_video_processing_common import VideoProcessingTestMixin, prepare_video_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    from transformers.image_utils import get_image_size
    from transformers.models.minimax_m3_vl.video_processing_minimax_m3_vl import smart_resize

    if is_torchvision_available():
        from transformers import MiniMaxM3VLVideoProcessor


class MiniMaxM3VLVideoProcessingTester:
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
        image_mean=OPENAI_CLIP_MEAN,
        image_std=OPENAI_CLIP_STD,
        do_convert_rgb=True,
        temporal_patch_size=2,
        patch_size=14,
        min_pixels=20 * 20,
        max_pixels=100 * 100,
        merge_size=2,
    ):
        size = size if size is not None else {"height": 672, "width": 672}
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
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.merge_size = merge_size

    def prepare_video_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_convert_rgb": self.do_convert_rgb,
            "temporal_patch_size": self.temporal_patch_size,
            "patch_size": self.patch_size,
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
            "merge_size": self.merge_size,
        }

    @require_vision
    def expected_output_video_shape(self, videos, num_frames=None):
        num_frames = num_frames if num_frames is not None else self.num_frames
        grid_t = num_frames // self.temporal_patch_size
        hidden_dim = self.num_channels * self.temporal_patch_size * self.patch_size * self.patch_size
        seq_len = 0
        for video in videos:
            if isinstance(video[0], Image.Image):
                video = np.stack([np.array(frame) for frame in video])
            height, width = get_image_size(video)
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=self.patch_size * self.merge_size,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
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
class MiniMaxM3VLVideoProcessingTest(VideoProcessingTestMixin, unittest.TestCase):
    fast_video_processing_class = MiniMaxM3VLVideoProcessor if is_torchvision_available() else None

    def setUp(self):
        super().setUp()
        self.video_processor_tester = MiniMaxM3VLVideoProcessingTester(self)

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
        self.assertTrue(hasattr(video_processing, "do_convert_rgb"))

    def test_video_processor_from_dict_with_kwargs(self):
        video_processor = self.fast_video_processing_class.from_dict(self.video_processor_dict)
        self.assertEqual(video_processor.size, {"height": 672, "width": 672})

        video_processor = self.fast_video_processing_class.from_dict(
            self.video_processor_dict, size={"height": 42, "width": 42}
        )
        self.assertEqual(video_processor.size, {"height": 42, "width": 42})

    def test_get_num_patches_without_videos(self):
        video_processing = self.fast_video_processing_class(**self.video_processor_dict)
        num_patches = video_processing.get_num_of_video_patches(num_frames=8, height=100, width=100, videos_kwargs={})
        self.assertEqual(num_patches, 144)

        num_patches = video_processing.get_num_of_video_patches(num_frames=7, height=200, width=50, videos_kwargs={})
        self.assertEqual(num_patches, 112)

        num_patches = video_processing.get_num_of_video_patches(
            num_frames=8, height=480, width=640, videos_kwargs={"min_pixels": 400, "max_pixels": 5000}
        )
        self.assertEqual(num_patches, 64)
        num_patches = video_processing.get_num_of_video_patches(
            num_frames=8, height=480, width=640, videos_kwargs={"max_pixels": 5000}
        )
        self.assertEqual(num_patches, 64)
        num_patches = video_processing.get_num_of_video_patches(
            num_frames=8, height=480, width=640, videos_kwargs={"min_pixels": 1000000, "max_pixels": 47040000}
        )
        self.assertEqual(num_patches, 20832)

    def test_call_pil(self):
        for video_processing_class in self.video_processor_list:
            # Initialize video_processing
            video_processing = video_processing_class(**self.video_processor_dict)
            video_inputs = self.video_processor_tester.prepare_video_inputs(
                equal_resolution=False, return_tensors="pil"
            )

            # Each video is a list of PIL Images
            for video in video_inputs:
                self.assertIsInstance(video[0], Image.Image)

            # Test not batched input
            encoded_videos = video_processing(video_inputs[0], return_tensors="pt")[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape([video_inputs[0]])
            self.assertEqual(list(encoded_videos.shape), expected_output_video_shape)

            # Test batched
            encoded_videos = video_processing(video_inputs, return_tensors="pt")[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape(video_inputs)
            self.assertEqual(list(encoded_videos.shape), expected_output_video_shape)

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
            self.assertEqual(list(encoded_videos.shape), expected_output_video_shape)

            # Test batched
            encoded_videos = video_processing(video_inputs, return_tensors="pt")[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape(video_inputs)
            self.assertEqual(list(encoded_videos.shape), expected_output_video_shape)

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
            self.assertEqual(list(encoded_videos.shape), expected_output_video_shape)

            # Test batched
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape(video_inputs)
            encoded_videos = video_processing(video_inputs, return_tensors="pt")[self.input_name]
            self.assertEqual(
                list(encoded_videos.shape),
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
            video_inputs_nested = [list(video) for video in video_inputs]
            encoded_videos = video_processing(video_inputs_nested[0], return_tensors="pt")[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape([video_inputs[0]])
            self.assertEqual(list(encoded_videos.shape), expected_output_video_shape)

            # Test batched
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape(video_inputs)
            encoded_videos = video_processing(video_inputs_nested, return_tensors="pt")[self.input_name]
            self.assertEqual(list(encoded_videos.shape), expected_output_video_shape)

    @unittest.skip("Skip for now, the test needs adjustment for MiniMaxM3VL")
    def test_call_numpy_4_channels(self):
        pass

    def test_num_frames_equal_temporal_patch_size_plus_two(self):
        for video_processing_class in self.video_processor_list:
            video_processor_dict = self.video_processor_dict.copy()
            video_processor_dict["do_sample_frames"] = False
            temporal_patch_size = 3
            video_processor_dict["temporal_patch_size"] = temporal_patch_size
            video_processing = video_processing_class(**video_processor_dict)

            n, w, h = 5, 28, 28
            video_inputs = [(np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)) for _ in range(n)]

            video_processed = video_processing(video_inputs, return_tensors="pt")
            encoded_videos = video_processed[self.input_name]
            self.assertEqual(list(encoded_videos.shape), [8, temporal_patch_size * 3 * 14 * 14])

            video_grid_thw = video_processed["video_grid_thw"]
            self.assertEqual(video_grid_thw.tolist(), [[2, 2, 2]])

    def test_call_sample_frames(self):
        for video_processing_class in self.video_processor_list:
            video_processing = video_processing_class(**self.video_processor_dict)

            video_inputs = self.video_processor_tester.prepare_video_inputs(
                equal_resolution=True,
                return_tensors="torch",
            )

            # Force set sampling to False. No sampling is expected even when `num_frames` exists
            video_processing.do_sample_frames = False

            video_processed = video_processing(video_inputs[0], return_tensors="pt", num_frames=4)
            self.assertEqual(video_processed["video_grid_thw"].tolist(), [[4, 6, 6]])

            # Sample with `fps` requires metadata to infer number of frames from total duration
            video_processing.do_sample_frames = True
            metadata = [[{"duration": 2.0, "total_num_frames": 8, "fps": 4}]]

            video_processed = video_processing(video_inputs[0], return_tensors="pt", fps=3, video_metadata=metadata)
            self.assertEqual(video_processed["video_grid_thw"].tolist(), [[3, 6, 6]])

            video_processed = video_processing(video_inputs[0], return_tensors="pt", fps=2, video_metadata=metadata)
            self.assertEqual(video_processed["video_grid_thw"].tolist(), [[2, 6, 6]])
