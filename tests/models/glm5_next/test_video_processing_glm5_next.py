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
from transformers.utils import is_torch_available, is_torchvision_available, is_vision_available

from ...test_video_processing_common import VideoProcessingTestMixin, prepare_video_inputs


if is_torch_available():
    import torch

if is_torchvision_available():
    from transformers import Glm5NextVideoProcessor
    from transformers.models.glm5_next.video_processing_glm5_next import smart_resize
    from transformers.video_utils import VideoMetadata

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
        do_rescale=True,
        do_normalize=True,
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
        temporal_patch_size=2,
        patch_size=2,
        merge_size=2,
        patch_expand_factor=2,
        min_image_tokens=1,
        max_image_tokens=64,
        fps=2,
        max_frame_count_dynamic=16,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.num_channels = num_channels
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_rescale = do_rescale
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.temporal_patch_size = temporal_patch_size
        self.patch_size = patch_size
        self.merge_size = merge_size
        self.patch_expand_factor = patch_expand_factor
        self.min_image_tokens = min_image_tokens
        self.max_image_tokens = max_image_tokens
        self.fps = fps
        self.max_frame_count_dynamic = max_frame_count_dynamic

    def prepare_video_processor_dict(self):
        return {
            "do_rescale": self.do_rescale,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "temporal_patch_size": self.temporal_patch_size,
            "patch_size": self.patch_size,
            "merge_size": self.merge_size,
            "patch_expand_factor": self.patch_expand_factor,
            "min_image_tokens": self.min_image_tokens,
            "max_image_tokens": self.max_image_tokens,
            "fps": self.fps,
            "max_frame_count_dynamic": self.max_frame_count_dynamic,
        }

    # fps=2 with duration=num_frames/2 makes the sampler keep every frame
    def prepare_video_metadata(self, videos):
        return [
            {"fps": 2, "duration": len(video) / 2, "total_num_frames": len(video)}
            if isinstance(video, list)
            else {"fps": 2, "duration": video.shape[0] / 2, "total_num_frames": video.shape[0]}
            for video in videos
        ]

    def expected_output_video_shape(self, videos):
        grid_t = self.num_frames // self.temporal_patch_size
        hidden_dim = self.num_channels * self.temporal_patch_size * self.patch_size**2
        pixels_per_token = self.temporal_patch_size * (self.patch_size * self.merge_size) ** 2
        min_pixels = self.min_image_tokens * pixels_per_token
        max_pixels = self.max_image_tokens * pixels_per_token
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
                height_factor=factor,
                width_factor=factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            seq_len += grid_t * (resized_height // self.patch_size) * (resized_width // self.patch_size)
        return [seq_len, hidden_dim]

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
class Glm5NextVideoProcessingTest(VideoProcessingTestMixin, unittest.TestCase):
    fast_video_processing_class = Glm5NextVideoProcessor if is_torchvision_available() else None
    input_name = "pixel_values_videos"

    def setUp(self):
        super().setUp()
        self.video_processor_tester = Glm5NextVideoProcessingTester(self)

    @property
    def video_processor_dict(self):
        return self.video_processor_tester.prepare_video_processor_dict()

    # videos are flattened into (seq_len, hidden_dim), so batched calls concatenate instead of stacking
    def _test_call(self, video_inputs, **preprocess_kwargs):
        video_processing = self.fast_video_processing_class(**self.video_processor_dict)
        video_metadata = self.video_processor_tester.prepare_video_metadata(video_inputs)
        encoded_videos = video_processing(
            video_inputs[0], video_metadata=[video_metadata[0]], return_tensors="pt", **preprocess_kwargs
        )[self.input_name]
        self.assertEqual(
            list(encoded_videos.shape),
            self.video_processor_tester.expected_output_video_shape([video_inputs[0]]),
        )
        encoded_videos = video_processing(
            video_inputs, video_metadata=video_metadata, return_tensors="pt", **preprocess_kwargs
        )[self.input_name]
        self.assertEqual(
            list(encoded_videos.shape),
            self.video_processor_tester.expected_output_video_shape(video_inputs),
        )

    def test_call_pil(self):
        self._test_call(self.video_processor_tester.prepare_video_inputs(equal_resolution=False, return_tensors="pil"))

    def test_call_numpy(self):
        self._test_call(self.video_processor_tester.prepare_video_inputs(equal_resolution=False, return_tensors="np"))

    def test_call_pytorch(self):
        self._test_call(self.video_processor_tester.prepare_video_inputs(equal_resolution=False, return_tensors="pt"))

    def test_nested_input(self):
        video_inputs = self.video_processor_tester.prepare_video_inputs(equal_resolution=False, return_tensors="np")
        self._test_call([list(video) for video in video_inputs])

    def test_call_numpy_4_channels(self):
        self.video_processor_tester.num_channels = 4
        video_inputs = self.video_processor_tester.prepare_video_inputs(equal_resolution=False, return_tensors="np")
        self._test_call(
            video_inputs,
            do_convert_rgb=False,
            input_data_format="channels_last",
            image_mean=(0.0, 0.0, 0.0, 0.0),
            image_std=(1.0, 1.0, 1.0, 1.0),
        )

        # per-channel normalization: each channel block in the patch dim must be scaled by its own std
        tester = self.video_processor_tester
        video_processor = self.fast_video_processing_class(
            **self.video_processor_dict, do_convert_rgb=False, do_resize=False
        )
        frame = np.zeros((16, 16, 4), dtype=np.uint8)
        for channel, value in enumerate((255, 128, 51, 0)):
            frame[..., channel] = value
        video = np.stack([frame] * 4)
        output = video_processor(
            video,
            do_sample_frames=False,
            input_data_format="channels_last",
            image_mean=(0.0, 0.0, 0.0, 0.0),
            image_std=(1.0, 2.0, 4.0, 8.0),
            return_tensors="pt",
        )[self.input_name]
        patch_dim = tester.temporal_patch_size * tester.patch_size**2
        for channel, expected in enumerate((1.0, 128 / 255 / 2, 51 / 255 / 4, 0.0)):
            block = output[:, channel * patch_dim : (channel + 1) * patch_dim]
            torch.testing.assert_close(block, torch.full_like(block, expected))

    def test_video_processor_from_dict_with_kwargs(self):
        video_processor = self.fast_video_processing_class.from_dict(
            self.video_processor_dict, size={"longest_edge": 42}
        )
        self.assertEqual(video_processor.size, {"longest_edge": 42})

    # GLM-5-Next samples by fps over the metadata duration
    def test_sample_frames_reference_cases(self):
        processor = Glm5NextVideoProcessor(**{**self.video_processor_dict, "fps": 1})
        cases = [
            (VideoMetadata(total_num_frames=3, fps=1, duration=3), None, None, [0, 1, 2, 2]),
            (VideoMetadata(total_num_frames=20, fps=4, duration=5), None, None, [0, 4, 8, 12, 16, 16]),
            (VideoMetadata(total_num_frames=20, fps=4, duration=5), 2, None, list(range(0, 20, 2))),
            (VideoMetadata(total_num_frames=120, fps=24, duration=5), 3, 8, [0, 17, 34, 51, 68, 85, 102, 119]),
        ]
        for metadata, target_fps, max_frames, expected in cases:
            self.assertEqual(
                processor.sample_frames(metadata, fps=target_fps, max_frames=max_frames).tolist(),
                expected,
            )

    def test_sample_frames_respects_duration_cap(self):
        processor = Glm5NextVideoProcessor(**{**self.video_processor_dict, "fps": 1, "max_duration": 5})
        metadata = VideoMetadata(total_num_frames=1000, fps=10, duration=100)
        self.assertEqual(processor.sample_frames(metadata).tolist(), [0, 249, 499, 749, 999, 999])

    # 3 frames are padded with a repeated frame up to a multiple of temporal_patch_size
    def test_odd_num_frames_is_padded_to_temporal_patch_size(self):
        processor = Glm5NextVideoProcessor(**self.video_processor_dict, do_sample_frames=False)
        video = torch.arange(3 * 3 * 60 * 100, dtype=torch.uint8).reshape(3, 3, 60, 100)
        output = processor(video, return_tensors="pt")
        self.assertEqual(output.video_grid_thw[0, 0].item(), 2)

    # halving the sampled window halves the number of frames that reach the encoder
    def test_call_sample_frames(self):
        video_processing = self.fast_video_processing_class(**self.video_processor_dict)
        video_inputs = self.video_processor_tester.prepare_video_inputs(equal_resolution=False, return_tensors="np")
        metadata = [
            {"fps": m["fps"], "duration": m["duration"] / 2, "total_num_frames": m["total_num_frames"]}
            for m in self.video_processor_tester.prepare_video_metadata(video_inputs)
        ]
        output = video_processing(video_inputs, video_metadata=metadata, return_tensors="pt")
        expected_grid_t = (
            self.video_processor_tester.num_frames // 2 // self.video_processor_tester.temporal_patch_size
        )
        self.assertEqual(output["video_grid_thw"][:, 0].tolist(), [expected_grid_t] * len(video_inputs))

    def test_call_without_metadata_raises(self):
        video_processing = self.fast_video_processing_class(**self.video_processor_dict)
        video_inputs = self.video_processor_tester.prepare_video_inputs(equal_resolution=False, return_tensors="np")
        with self.assertRaises(ValueError):
            video_processing(video_inputs[0], return_tensors="pt")

    def test_per_call_overrides(self):
        kwargs = self.video_processor_dict
        processor = Glm5NextVideoProcessor(**kwargs, do_sample_frames=False)
        video = torch.arange(4 * 3 * 37 * 53, dtype=torch.uint8).reshape(4, 3, 37, 53)
        overrides = {"max_image_tokens": 8, "patch_expand_factor": 1}
        actual = processor(video, return_tensors="pt", **overrides)
        expected = Glm5NextVideoProcessor(**(kwargs | overrides), do_sample_frames=False)(video, return_tensors="pt")
        self.assertTrue(torch.equal(actual.pixel_values_videos, expected.pixel_values_videos))
        self.assertTrue(torch.equal(actual.video_grid_thw, expected.video_grid_thw))
