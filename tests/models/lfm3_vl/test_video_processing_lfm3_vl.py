# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Testing suite for the LFM3-VL video processor."""

import tempfile
import unittest

import numpy as np

from transformers.testing_utils import require_torch, require_torchvision, require_vision
from transformers.utils import is_torch_available, is_torchvision_available, is_vision_available

from ...test_video_processing_common import VideoProcessingTestMixin, prepare_video_inputs


if is_torch_available():
    import torch

if is_vision_available() and is_torchvision_available():
    from PIL import Image

    from transformers import AutoVideoProcessor, Lfm3VlVideoProcessor
    from transformers.video_utils import VideoMetadata


class Lfm3VlVideoProcessingTester:
    def __init__(self, parent, batch_size=3, num_frames=8, num_channels=3, min_resolution=32, max_resolution=80):
        self.parent = parent
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.num_channels = num_channels
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution

    def prepare_video_processor_dict(self):
        return {"size": {"shortest_edge": 32 * 32, "longest_edge": 256 * 256 * 4}, "do_sample_frames": True}

    def prepare_video_metadata(self, videos):
        return [{"fps": 2, "total_num_frames": len(video), "duration": len(video) / 2} for video in videos]

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


@require_vision
@require_torch
@require_torchvision
class Lfm3VlVideoProcessingTest(VideoProcessingTestMixin, unittest.TestCase):
    fast_video_processing_class = Lfm3VlVideoProcessor if is_torchvision_available() else None
    input_name = "pixel_values_videos"

    def setUp(self):
        super().setUp()
        self.video_processor_tester = Lfm3VlVideoProcessingTester(self)
        self.video = torch.from_numpy(np.random.randint(0, 255, size=(8, 3, 64, 96), dtype=np.uint8))

    @property
    def video_processor_dict(self):
        return self.video_processor_tester.prepare_video_processor_dict()

    def _check_call(self, video_inputs):
        video_processing = self.fast_video_processing_class(**self.video_processor_dict)
        patch_dim = 3 * video_processing.temporal_patch_size * video_processing.patch_size**2
        metadata = self.video_processor_tester.prepare_video_metadata(video_inputs)
        for videos, video_metadata in ((video_inputs[0], [metadata[0]]), (video_inputs, metadata)):
            outputs = video_processing(videos, video_metadata=video_metadata, return_tensors="pt")
            num_videos = 1 if videos is video_inputs[0] else self.video_processor_tester.batch_size
            self.assertEqual(outputs["video_grid_thw"].shape, (num_videos, 3))
            self.assertEqual(
                outputs[self.input_name].shape, (int(outputs["video_grid_thw"].prod(-1).sum()), patch_dim)
            )
            # 8 frames at the clip's own 2 fps are kept whole and paired along time
            for grid in outputs["video_grid_thw"].tolist():
                self.assertEqual(
                    grid[0], self.video_processor_tester.num_frames // video_processing.temporal_patch_size
                )

    def test_call_pil(self):
        video_inputs = self.video_processor_tester.prepare_video_inputs(return_tensors="pil")
        for video in video_inputs:
            self.assertIsInstance(video[0], Image.Image)
        self._check_call(video_inputs)

    def test_call_numpy(self):
        video_inputs = self.video_processor_tester.prepare_video_inputs(return_tensors="np")
        for video in video_inputs:
            self.assertIsInstance(video, np.ndarray)
        self._check_call(video_inputs)

    def test_call_pytorch(self):
        video_inputs = self.video_processor_tester.prepare_video_inputs(return_tensors="torch")
        for video in video_inputs:
            self.assertIsInstance(video, torch.Tensor)
        self._check_call(video_inputs)

    def test_nested_input(self):
        video_inputs = self.video_processor_tester.prepare_video_inputs(return_tensors="np")
        self._check_call([list(video) for video in video_inputs])

    def test_call_sample_frames(self):
        video_processing = self.fast_video_processing_class(**self.video_processor_dict)
        video_inputs = self.video_processor_tester.prepare_video_inputs(return_tensors="torch")
        metadata = [{"total_num_frames": 8, "fps": 4, "duration": 2.0}] * len(video_inputs)
        # explicit frame count, the default 2 fps on a 4 fps clip, an explicit fps, no sampling
        for kwargs, expected_t in (
            ({"num_frames": 3}, 2),
            ({}, 2),
            ({"fps": 3}, 3),
            ({"do_sample_frames": False}, 4),
        ):
            with self.subTest(kwargs=kwargs):
                grids = video_processing(video_inputs, video_metadata=metadata, return_tensors="pt", **kwargs)[
                    "video_grid_thw"
                ]
                self.assertEqual(grids[:, 0].tolist(), [expected_t] * len(video_inputs))

    @unittest.skip("the Qwen3-VL patchifier does not treat 4-channel inputs consistently")
    def test_call_numpy_4_channels(self):
        pass

    def test_video_processor_from_dict_with_kwargs(self):
        video_processor = self.fast_video_processing_class.from_dict(self.video_processor_dict)
        self.assertEqual(video_processor.size, self.video_processor_dict["size"])
        video_processor = self.fast_video_processing_class.from_dict(
            self.video_processor_dict, size={"shortest_edge": 42, "longest_edge": 4242}
        )
        self.assertEqual(video_processor.size, {"shortest_edge": 42, "longest_edge": 4242})

    def test_defaults_match_the_released_checkpoints(self):
        video_processor = Lfm3VlVideoProcessor()
        self.assertEqual(video_processor.patch_size, 16)
        self.assertEqual(video_processor.temporal_patch_size, 2)
        self.assertEqual(video_processor.merge_size, 2)
        self.assertEqual(video_processor.size, {"shortest_edge": 4_096, "longest_edge": 25_165_824})
        self.assertEqual((video_processor.fps, video_processor.min_frames, video_processor.max_frames), (2.0, 4, 768))
        self.assertEqual(list(video_processor.image_mean), [0.5, 0.5, 0.5])
        self.assertEqual(list(video_processor.image_std), [0.5, 0.5, 0.5])
        self.assertFalse(video_processor.cap_pixels_per_frame)
        self.assertEqual(video_processor.model_input_names, ["pixel_values_videos", "video_grid_thw"])

    def test_patch_grid_pairs_frames(self):
        video_processor = Lfm3VlVideoProcessor(**self.video_processor_dict)
        inputs = video_processor(self.video, do_sample_frames=False, return_tensors="pt")
        # 8 frames of 64x96 pixels: frames are paired along time, the 4x6 patch grid is kept
        self.assertEqual(inputs["video_grid_thw"].tolist(), [[4, 4, 6]])
        self.assertEqual(inputs["pixel_values_videos"].shape, (4 * 24, 3 * 2 * 16 * 16))

    def test_sample_frames_at_two_fps(self):
        video_processor = Lfm3VlVideoProcessor(**self.video_processor_dict)
        metadata = VideoMetadata(total_num_frames=240, fps=24.0, duration=10.0)
        indices = video_processor.sample_frames(metadata)
        self.assertEqual(len(indices), 20)
        self.assertEqual(len(video_processor.sample_frames(metadata, num_frames=6)), 6)
        # never fewer than `min_frames`
        short = VideoMetadata(total_num_frames=24, fps=24.0, duration=1.0)
        self.assertGreaterEqual(len(video_processor.sample_frames(short)), video_processor.min_frames)

    def test_save_load_through_auto_video_processor(self):
        video_processor = Lfm3VlVideoProcessor(**self.video_processor_dict)
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_processor.save_pretrained(tmp_dir)
            reloaded = AutoVideoProcessor.from_pretrained(tmp_dir)
        self.assertIsInstance(reloaded, Lfm3VlVideoProcessor)
        self.assertEqual(reloaded.to_dict(), video_processor.to_dict())
