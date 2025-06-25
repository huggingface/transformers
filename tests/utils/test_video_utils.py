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
from huggingface_hub import hf_hub_download

from transformers import is_torch_available, is_vision_available
from transformers.image_processing_utils import get_size_dict
from transformers.image_utils import SizeDict
from transformers.processing_utils import VideosKwargs
from transformers.testing_utils import (
    require_av,
    require_cv2,
    require_decord,
    require_torch,
    require_torchcodec,
    require_torchvision,
    require_vision,
)
from transformers.video_utils import group_videos_by_shape, make_batched_videos, reorder_videos


if is_torch_available():
    import torch

if is_vision_available():
    import PIL

    from transformers import BaseVideoProcessor
    from transformers.video_utils import VideoMetadata, load_video


def get_random_video(height, width, num_frames=8, return_torch=False):
    random_frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    video = np.array([random_frame] * num_frames)
    if return_torch:
        # move channel first
        return torch.from_numpy(video).permute(0, 3, 1, 2)
    return video


@require_vision
@require_torchvision
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
        pil_video = [PIL.Image.fromarray(frame) for frame in video]
        videos_list = make_batched_videos(pil_video)
        self.assertIsInstance(videos_list, list)
        self.assertIsInstance(videos_list[0], np.ndarray)
        self.assertEqual(videos_list[0].shape, (8, 16, 32, 3))
        self.assertTrue(np.array_equal(videos_list[0], video))

        # Test a nested list of videos is not modified
        video = get_random_video(16, 32)
        pil_video = [PIL.Image.fromarray(frame) for frame in video]
        videos = [pil_video, pil_video]
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
        video_processor = BaseVideoProcessor(model_init_kwargs=VideosKwargs)
        video = get_random_video(16, 32, return_torch=True)

        # Size can be an int or a tuple of ints.
        size_dict = SizeDict(**get_size_dict((8, 8), param_name="size"))
        resized_video = video_processor.resize(video, size=size_dict)
        self.assertIsInstance(resized_video, torch.Tensor)
        self.assertEqual(resized_video.shape, (8, 3, 8, 8))

    def test_normalize(self):
        video_processor = BaseVideoProcessor(model_init_kwargs=VideosKwargs)
        array = torch.randn(4, 3, 16, 32)
        mean = [0.1, 0.5, 0.9]
        std = [0.2, 0.4, 0.6]

        # mean and std can be passed as lists or NumPy arrays.
        expected = (array - torch.tensor(mean)[:, None, None]) / torch.tensor(std)[:, None, None]
        normalized_array = video_processor.normalize(array, mean, std)
        torch.testing.assert_close(normalized_array, expected)

    def test_center_crop(self):
        video_processor = BaseVideoProcessor(model_init_kwargs=VideosKwargs)
        video = get_random_video(16, 32, return_torch=True)

        # Test various crop sizes: bigger on all dimensions, on one of the dimensions only and on both dimensions.
        crop_sizes = [8, (8, 64), 20, (32, 64)]
        for size in crop_sizes:
            size_dict = SizeDict(**get_size_dict(size, default_to_square=True, param_name="crop_size"))
            cropped_video = video_processor.center_crop(video, size_dict)
            self.assertIsInstance(cropped_video, torch.Tensor)

            expected_size = (size, size) if isinstance(size, int) else size
            self.assertEqual(cropped_video.shape, (8, 3, *expected_size))

    def test_convert_to_rgb(self):
        video_processor = BaseVideoProcessor(model_init_kwargs=VideosKwargs)
        video = get_random_video(20, 20, return_torch=True)

        rgb_video = video_processor.convert_to_rgb(video[:, :1])
        self.assertEqual(rgb_video.shape, (8, 3, 20, 20))

        rgb_video = video_processor.convert_to_rgb(torch.cat([video, video[:, :1]], dim=1))
        self.assertEqual(rgb_video.shape, (8, 3, 20, 20))

    def test_group_and_reorder_videos(self):
        """Tests that videos can be grouped by frame size and number of frames"""
        video_1 = get_random_video(20, 20, num_frames=3, return_torch=True)
        video_2 = get_random_video(20, 20, num_frames=5, return_torch=True)

        # Group two videos of same size but different number of frames
        grouped_videos, grouped_videos_index = group_videos_by_shape([video_1, video_2])
        self.assertEqual(len(grouped_videos), 2)

        regrouped_videos = reorder_videos(grouped_videos, grouped_videos_index)
        self.assertTrue(len(regrouped_videos), 2)
        self.assertEqual(video_1.shape, regrouped_videos[0].shape)

        # Group two videos of different size but same number of frames
        video_3 = get_random_video(15, 20, num_frames=3, return_torch=True)
        grouped_videos, grouped_videos_index = group_videos_by_shape([video_1, video_3])
        self.assertEqual(len(grouped_videos), 2)

        regrouped_videos = reorder_videos(grouped_videos, grouped_videos_index)
        self.assertTrue(len(regrouped_videos), 2)
        self.assertEqual(video_1.shape, regrouped_videos[0].shape)

        # Group all three videos where some have same size or same frame count
        # But since none have frames and sizes identical, we'll have 3 groups
        grouped_videos, grouped_videos_index = group_videos_by_shape([video_1, video_2, video_3])
        self.assertEqual(len(grouped_videos), 3)

        regrouped_videos = reorder_videos(grouped_videos, grouped_videos_index)
        self.assertTrue(len(regrouped_videos), 3)
        self.assertEqual(video_1.shape, regrouped_videos[0].shape)

        # Group if we had some videos with identical shapes
        grouped_videos, grouped_videos_index = group_videos_by_shape([video_1, video_1, video_3])
        self.assertEqual(len(grouped_videos), 2)

        regrouped_videos = reorder_videos(grouped_videos, grouped_videos_index)
        self.assertTrue(len(regrouped_videos), 2)
        self.assertEqual(video_1.shape, regrouped_videos[0].shape)

        # Group if we had all videos with identical shapes
        grouped_videos, grouped_videos_index = group_videos_by_shape([video_1, video_1, video_1])
        self.assertEqual(len(grouped_videos), 1)

        regrouped_videos = reorder_videos(grouped_videos, grouped_videos_index)
        self.assertTrue(len(regrouped_videos), 1)
        self.assertEqual(video_1.shape, regrouped_videos[0].shape)


@require_vision
@require_av
class LoadVideoTester(unittest.TestCase):
    def test_load_video_url(self):
        video, _ = load_video(
            "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4",
        )
        self.assertEqual(video.shape, (243, 360, 640, 3))  # 243 frames is the whole video, no sampling applied

    def test_load_video_local(self):
        video_file_path = hf_hub_download(
            repo_id="raushan-testing-hf/videos-test", filename="sample_demo_1.mp4", repo_type="dataset"
        )
        video, _ = load_video(video_file_path)
        self.assertEqual(video.shape, (243, 360, 640, 3))  # 243 frames is the whole video, no sampling applied

    # FIXME: @raushan, yt-dlp downloading works for for some reason it cannot redirect to out buffer?
    # @requires_yt_dlp
    # def test_load_video_youtube(self):
    #     video = load_video("https://www.youtube.com/watch?v=QC8iQqtG0hg")
    #     self.assertEqual(video.shape, (243, 360, 640, 3)) # 243 frames is the whole video, no sampling applied

    @require_decord
    @require_torchvision
    @require_torchcodec
    @require_cv2
    def test_load_video_backend_url(self):
        video, _ = load_video(
            "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4",
            backend="decord",
        )
        self.assertEqual(video.shape, (243, 360, 640, 3))

        video, _ = load_video(
            "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4",
            backend="torchcodec",
        )
        self.assertEqual(video.shape, (243, 360, 640, 3))

        # Can't use certain backends with url
        with self.assertRaises(ValueError):
            video, _ = load_video(
                "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4",
                backend="opencv",
            )
        with self.assertRaises(ValueError):
            video, _ = load_video(
                "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4",
                backend="torchvision",
            )

    @require_decord
    @require_torchvision
    @require_torchcodec
    @require_cv2
    def test_load_video_backend_local(self):
        video_file_path = hf_hub_download(
            repo_id="raushan-testing-hf/videos-test", filename="sample_demo_1.mp4", repo_type="dataset"
        )
        video, metadata = load_video(video_file_path, backend="decord")
        self.assertEqual(video.shape, (243, 360, 640, 3))
        self.assertIsInstance(metadata, VideoMetadata)

        video, metadata = load_video(video_file_path, backend="opencv")
        self.assertEqual(video.shape, (243, 360, 640, 3))
        self.assertIsInstance(metadata, VideoMetadata)

        video, metadata = load_video(video_file_path, backend="torchvision")
        self.assertEqual(video.shape, (243, 360, 640, 3))
        self.assertIsInstance(metadata, VideoMetadata)

        video, metadata = load_video(video_file_path, backend="torchcodec")
        self.assertEqual(video.shape, (243, 360, 640, 3))
        self.assertIsInstance(metadata, VideoMetadata)

    def test_load_video_num_frames(self):
        video, _ = load_video(
            "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4",
            num_frames=16,
        )
        self.assertEqual(video.shape, (16, 360, 640, 3))

        video, _ = load_video(
            "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4",
            num_frames=22,
        )
        self.assertEqual(video.shape, (22, 360, 640, 3))

    def test_load_video_fps(self):
        video, _ = load_video(
            "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4", fps=1
        )
        self.assertEqual(video.shape, (9, 360, 640, 3))

        video, _ = load_video(
            "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4", fps=2
        )
        self.assertEqual(video.shape, (19, 360, 640, 3))

        # `num_frames` is mutually exclusive with `video_fps`
        with self.assertRaises(ValueError):
            video, _ = load_video(
                "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4",
                fps=1,
                num_frames=10,
            )
