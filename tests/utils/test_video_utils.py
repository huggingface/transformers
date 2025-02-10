# coding=utf-8
# Copyright 2021 HuggingFace Inc.
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
from huggingface_hub import hf_hub_download

from transformers import is_torch_available, is_vision_available
from transformers.testing_utils import (
    require_av,
    require_cv2,
    require_decord,
    require_torch,
    require_torchvision,
    require_vision,
)
from transformers.video_utils import make_batched_videos


if is_torch_available():
    import torch

if is_vision_available():
    import PIL

    from transformers import BaseVideoProcessor
    from transformers.video_utils import load_video


def get_random_video(height, width):
    random_frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return np.array(([random_frame] * 8))


@require_vision
class BaseVideoProcessorTester(unittest.TestCase):
    """
    Tests that the `transforms` can be applied to a 4-dim array directly, i.e. to a whole video.
    """

    def test_make_batched_videos_pil(self):
        # Test a single image is converted to a list of 1 video with 1 frame
        video = get_random_video(16, 32)
        pil_image = PIL.Image.fromarray(video[0])
        videos_list = make_batched_videos(pil_image)
        self.assertIsInstance(videos_list, list)
        self.assertIsInstance(videos_list[0], np.ndarray)
        self.assertEqual(videos_list[0].shape, (1, 16, 32, 3))
        self.assertTrue(np.array_equal(videos_list[0][0], np.array(pil_image)))

        # Test a list of videos is converted to a list of 1 video
        video = get_random_video(16, 32)
        video = [PIL.Image.fromarray(frame) for frame in video]
        videos_list = make_batched_videos(video)
        self.assertIsInstance(videos_list, list)
        self.assertIsInstance(videos_list[0], np.ndarray)
        self.assertEqual(videos_list[0].shape, (8, 16, 32, 3))
        self.assertTrue(np.array_equal(videos_list[0], video))

        # Test a nested list of videos is not modified
        video = get_random_video(16, 32)
        video = [PIL.Image.fromarray(frame) for frame in video]
        videos = [video, video]
        videos_list = make_batched_videos(videos)
        self.assertIsInstance(videos_list, list)
        self.assertIsInstance(videos_list[0], np.ndarray)
        self.assertEqual(videos_list[0].shape, (8, 16, 32, 3))
        self.assertTrue(np.array_equal(videos_list[0], video))

    def test_make_batched_videos_numpy(self):
        # Test a single image is converted to a list of 1 video with 1 frame
        video = get_random_video(16, 32)[0]
        videos_list = make_batched_videos(video)
        self.assertIsInstance(videos_list, list)
        self.assertIsInstance(videos_list[0], np.ndarray)
        self.assertEqual(videos_list[0].shape, (1, 16, 32, 3))
        self.assertTrue(np.array_equal(videos_list[0][0], video))

        # Test a 4d array of videos is converted to a a list of 1 video
        video = get_random_video(16, 32)
        videos_list = make_batched_videos(video)
        self.assertIsInstance(videos_list, list)
        self.assertIsInstance(videos_list[0], np.ndarray)
        self.assertEqual(videos_list[0].shape, (8, 16, 32, 3))
        self.assertTrue(np.array_equal(videos_list[0], video))

        # Test a list of videos is converted to a list of videos
        video = get_random_video(16, 32)
        videos = [video, video]
        videos_list = make_batched_videos(videos)
        self.assertIsInstance(videos_list, list)
        self.assertIsInstance(videos_list[0], np.ndarray)
        self.assertEqual(videos_list[0].shape, (8, 16, 32, 3))
        self.assertTrue(np.array_equal(videos_list[0], video))

    @require_torch
    def test_make_batched_videos_torch(self):
        # Test a single image is converted to a list of 1 video with 1 frame
        video = get_random_video(16, 32)[0]
        torch_video = torch.from_numpy(video)
        videos_list = make_batched_videos(torch_video)
        self.assertIsInstance(videos_list, list)
        self.assertIsInstance(videos_list[0], np.ndarray)
        self.assertEqual(videos_list[0].shape, (1, 16, 32, 3))
        self.assertTrue(np.array_equal(videos_list[0][0], video))

        # Test a 4d array of videos is converted to a a list of 1 video
        video = get_random_video(16, 32)
        torch_video = torch.from_numpy(video)
        videos_list = make_batched_videos(torch_video)
        self.assertIsInstance(videos_list, list)
        self.assertIsInstance(videos_list[0], torch.Tensor)
        self.assertEqual(videos_list[0].shape, (8, 16, 32, 3))
        self.assertTrue(np.array_equal(videos_list[0], video))

        # Test a list of videos is converted to a list of videos
        video = get_random_video(16, 32)
        torch_video = torch.from_numpy(video)
        videos = [torch_video, torch_video]
        videos_list = make_batched_videos(videos)
        self.assertIsInstance(videos_list, list)
        self.assertIsInstance(videos_list[0], torch.Tensor)
        self.assertEqual(videos_list[0].shape, (8, 16, 32, 3))
        self.assertTrue(np.array_equal(videos_list[0], video))

    def test_resize(self):
        video_processor = BaseVideoProcessor()
        video = get_random_video(16, 32)

        # Size can be an int or a tuple of ints.
        resized_video = video_processor.resize(video, size=(8, 8))
        self.assertIsInstance(resized_video, np.ndarray)
        self.assertEqual(resized_video.shape, (8, 8, 8, 3))

    def test_normalize(self):
        video_processor = BaseVideoProcessor()
        array = np.random.random((4, 16, 32, 3))
        mean = [0.1, 0.5, 0.9]
        std = [0.2, 0.4, 0.6]

        # mean and std can be passed as lists or NumPy arrays.
        expected = (array - np.array(mean)) / np.array(std)
        normalized_array = video_processor.normalize(array, mean, std)
        self.assertTrue(np.array_equal(normalized_array, expected))

        # Normalize will detect automatically if channel first or channel last is used.
        array = np.random.random((4, 3, 16, 32))
        expected = (array - np.array(mean)[:, None, None]) / np.array(std)[:, None, None]
        normalized_array = video_processor.normalize(array, mean, std)
        self.assertTrue(np.array_equal(normalized_array, expected))

    def test_center_crop(self):
        video_processor = BaseVideoProcessor()
        video = get_random_video(16, 32)

        # Test various crop sizes: bigger on all dimensions, on one of the dimensions only and on both dimensions.
        crop_sizes = [8, (8, 64), 20, (32, 64)]
        for size in crop_sizes:
            cropped_video = video_processor.center_crop(video, size)
            self.assertIsInstance(cropped_video, np.ndarray)

            expected_size = (size, size) if isinstance(size, int) else size
            self.assertEqual(cropped_video.shape, (8, *expected_size, 3))

    def test_convert_to_rgb(self):
        video_processor = BaseVideoProcessor()
        video = get_random_video(20, 20)

        rgb_video = video_processor.convert_to_rgb(video[..., :1])
        self.assertEqual(rgb_video.shape, (8, 20, 20, 3))

        rgb_video = video_processor.convert_to_rgb(np.concatenate([video, video[..., :1]], axis=-1))
        self.assertEqual(rgb_video.shape, (8, 20, 20, 3))


@require_vision
@require_av
class LoadVideoTester(unittest.TestCase):
    def test_load_video_url(self):
        video = load_video(
            "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4"
        )
        self.assertEqual(video.shape, (243, 360, 640, 3))  # 243 frames is the whole video, no sampling applied

    def test_load_video_local(self):
        video_file_path = hf_hub_download(
            repo_id="raushan-testing-hf/videos-test", filename="sample_demo_1.mp4", repo_type="dataset"
        )
        video = load_video(video_file_path)
        self.assertEqual(video.shape, (243, 360, 640, 3))  # 243 frames is the whole video, no sampling applied

    # @requires_yt_dlp
    # def test_load_video_youtube(self):
    #     video = load_video("https://www.youtube.com/watch?v=QC8iQqtG0hg")
    #     self.assertEqual(video.shape, (243, 360, 640, 3)) # 243 frames is the whole video, no sampling applied

    @require_decord
    @require_torchvision
    @require_cv2
    def test_load_video_backend_url(self):
        video = load_video(
            "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4",
            backend="decord",
        )
        self.assertEqual(video.shape, (243, 360, 640, 3))

        # Can't use certain backends with url
        with self.assertRaises(ValueError):
            video = load_video(
                "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4",
                backend="opencv",
            )
        with self.assertRaises(ValueError):
            video = load_video(
                "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4",
                backend="torchvision",
            )

    @require_decord
    @require_torchvision
    @require_cv2
    def test_load_video_backend_local(self):
        video_file_path = hf_hub_download(
            repo_id="raushan-testing-hf/videos-test", filename="sample_demo_1.mp4", repo_type="dataset"
        )
        video = load_video(video_file_path, backend="decord")
        self.assertEqual(video.shape, (243, 360, 640, 3))

        video = load_video(video_file_path, backend="opencv")
        self.assertEqual(video.shape, (243, 360, 640, 3))

        video = load_video(video_file_path, backend="torchvision")
        self.assertEqual(video.shape, (243, 360, 640, 3))

    def test_load_video_num_frames(self):
        video = load_video(
            "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4",
            num_frames=16,
        )
        self.assertEqual(video.shape, (16, 360, 640, 3))

        video = load_video(
            "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4",
            num_frames=22,
        )
        self.assertEqual(video.shape, (22, 360, 640, 3))

    def test_load_video_fps(self):
        video = load_video(
            "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4", fps=1
        )
        self.assertEqual(video.shape, (9, 360, 640, 3))

        video = load_video(
            "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4", fps=2
        )
        self.assertEqual(video.shape, (19, 360, 640, 3))

        # `num_frames` is mutually exclusive with `video_fps`
        with self.assertRaises(ValueError):
            video = load_video(
                "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4",
                fps=1,
                num_frames=10,
            )
