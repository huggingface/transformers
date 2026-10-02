# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
from parameterized import parameterized

from transformers.testing_utils import require_torch, require_torchvision, require_vision, torch_device
from transformers.utils import is_torch_available, is_torchvision_available, is_vision_available
from transformers.video_utils import VideoMetadata

from ...test_video_processing_common import VideoProcessingTestMixin, prepare_video_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    if is_torchvision_available():
        from transformers import HYV4VLVideoProcessor


class HYV4VLVideoProcessingTester:
    def __init__(self, parent):
        self.parent = parent
        self.batch_size = 3
        self.num_frames = 3
        self.num_channels = 3

    def prepare_video_processor_dict(self):
        return {
            "device": torch_device,
            "size": {"shortest_edge": 1024, "longest_edge": 32768},
            "max_image_size": {"longest_edge": 8192},
            "patch_size": 16,
            "merge_size": 2,
            "temporal_patch_size": 1,
            "do_sample_frames": False,
            "do_resize": True,
            "do_rescale": True,
            "do_normalize": True,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
        }

    def prepare_video_inputs(self, equal_resolution=False, return_tensors="pil"):
        return prepare_video_inputs(
            batch_size=self.batch_size,
            num_frames=self.num_frames,
            num_channels=self.num_channels,
            min_resolution=32,
            max_resolution=64,
            equal_resolution=equal_resolution,
            return_tensors=return_tensors,
        )


@require_torch
@require_vision
@require_torchvision
class HYV4VLVideoProcessingTest(VideoProcessingTestMixin, unittest.TestCase):
    fast_video_processing_class = HYV4VLVideoProcessor
    test_cast_dtype = True

    def setUp(self):
        super().setUp()
        self.video_processor_tester = HYV4VLVideoProcessingTester(self)

    @property
    def video_processor_dict(self):
        return self.video_processor_tester.prepare_video_processor_dict()

    def test_defaults_and_model_input_names(self):
        processor = self.fast_video_processing_class()
        self.assertEqual(processor.size, {"shortest_edge": 4096, "longest_edge": 33554432})
        self.assertEqual(processor.max_image_size, {"longest_edge": 1024**2})
        self.assertEqual((processor.patch_size, processor.merge_size, processor.temporal_patch_size), (16, 2, 1))
        self.assertEqual((processor.min_frames, processor.max_frames, processor.fps), (2, 1024, 2))
        self.assertFalse(processor.do_sample_frames)
        self.assertIsNone(processor.num_frames)
        self.assertEqual(tuple(processor.image_mean), (0.5,) * 3)
        self.assertEqual(tuple(processor.image_std), (0.5,) * 3)
        self.assertEqual(processor.model_input_names, ["pixel_values_videos", "video_grid_thw"])

    def test_video_processor_from_dict_with_kwargs(self):
        processor = self.fast_video_processing_class.from_dict(
            self.video_processor_dict, size={"shortest_edge": 2048, "longest_edge": 16384}, max_frames=7
        )
        self.assertEqual(processor.size, {"shortest_edge": 2048, "longest_edge": 16384})
        self.assertEqual(processor.max_frames, 7)

    def test_checkpoint_pixel_budget_aliases(self):
        processor = self.fast_video_processing_class(min_pixels=1024, max_pixels=16384, video_max_image_size=64)
        self.assertEqual(processor.size, {"shortest_edge": 1024, "longest_edge": 16384})
        self.assertEqual(processor.max_image_size, {"longest_edge": 4096})
        self.assertEqual(self.fast_video_processing_class().size, {"shortest_edge": 4096, "longest_edge": 33554432})

    def _check_input_type(self, return_tensors, nested=False):
        processor = self.fast_video_processing_class(**self.video_processor_dict)
        videos = self.video_processor_tester.prepare_video_inputs(return_tensors=return_tensors)
        if nested:
            videos = [list(video) for video in videos]
        batch = processor(videos, return_tensors="pt")
        singles = [processor(video, return_tensors="pt") for video in videos]
        torch.testing.assert_close(
            batch.pixel_values_videos, torch.cat([output.pixel_values_videos for output in singles])
        )
        torch.testing.assert_close(batch.video_grid_thw, torch.cat([output.video_grid_thw for output in singles]))
        self.assertEqual(batch.pixel_values_videos.device.type, torch.device(torch_device).type)
        self.assertEqual(batch.video_grid_thw.dtype, torch.long)
        self.assertEqual(batch.video_grid_thw.shape, (len(videos), 3))
        self.assertEqual(batch.pixel_values_videos.shape, (batch.video_grid_thw.prod(-1).sum().item(), 3 * 16 * 16))
        self.assertTrue((batch.video_grid_thw[:, 0] == self.video_processor_tester.num_frames).all())

    def test_call_pil(self):
        self._check_input_type("pil")

    def test_call_numpy(self):
        self._check_input_type("np")

    def test_call_pytorch(self):
        self._check_input_type("torch")

    def test_nested_input(self):
        self._check_input_type("np", nested=True)

    def test_call_numpy_4_channels(self):
        processor = self.fast_video_processing_class(**self.video_processor_dict)
        frames = [Image.fromarray(np.full((32, 64, 4), 255, dtype=np.uint8)) for _ in range(3)]
        output = processor(frames, input_data_format="channels_last", return_tensors="pt")
        self.assertEqual(output.pixel_values_videos.shape, (24, 768))
        self.assertEqual(output.video_grid_thw.tolist(), [[3, 2, 4]])

    def test_whole_clip_budget(self):
        processor = self.fast_video_processing_class(
            size={"shortest_edge": 1024, "longest_edge": 16384},
            max_image_size={"longest_edge": 65536},
            device=torch_device,
        )
        grids = []
        for frames in (2, 8):
            output = processor(np.zeros((frames, 128, 128, 3), dtype=np.uint8), return_tensors="pt")
            t, h, w = output.video_grid_thw[0].tolist()
            self.assertEqual(t, frames)
            self.assertLessEqual(t * h * w * 16**2, 16384)
            grids.append((h, w))
        self.assertLess(grids[1][0] * grids[1][1], grids[0][0] * grids[0][1])

    def test_per_frame_area_cap_not_edge_cap(self):
        processor = self.fast_video_processing_class(
            size={"shortest_edge": 1024, "longest_edge": 65536},
            max_image_size={"longest_edge": 4096},
            device=torch_device,
        )
        output = processor(np.zeros((2, 64, 256, 3), dtype=np.uint8), return_tensors="pt")
        self.assertEqual(output.video_grid_thw.tolist(), [[2, 2, 8]])
        self.assertEqual(output.pixel_values_videos.shape, (32, 768))
        self.assertGreater(output.video_grid_thw[0, 2].item() * 16, 64)

    def test_sample_frames_fps_and_mutual_exclusion(self):
        processor = self.fast_video_processing_class()
        metadata = VideoMetadata(total_num_frames=12, fps=6)
        self.assertEqual(processor.sample_frames(metadata, fps=2).tolist(), [0, 4, 7, 11])
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            processor.sample_frames(metadata, num_frames=3, fps=2)
        # fps-derived counts are bounded by the processor's frame limits (checkpoint: 2..1024).
        bounded = self.fast_video_processing_class(min_frames=2, max_frames=3)
        self.assertEqual(len(bounded.sample_frames(VideoMetadata(total_num_frames=12, fps=1), fps=1)), 3)
        self.assertEqual(len(bounded.sample_frames(VideoMetadata(total_num_frames=12, fps=24), fps=1)), 2)

    def test_call_sample_frames(self):
        processor = self.fast_video_processing_class(**self.video_processor_dict)
        video = np.stack([np.full((32, 64, 3), i, dtype=np.uint8) for i in range(8)])
        for enabled, expected in ((False, 8), (True, 3)):
            with self.subTest(do_sample_frames=enabled):
                output = processor(
                    video, do_sample_frames=enabled, num_frames=3, fps=None, return_tensors="pt", return_metadata=True
                )
                self.assertEqual(output.video_grid_thw[0, 0].item(), expected)
                indices = [0, 4, 7] if enabled else list(range(8))
                self.assertEqual(list(output.video_metadata[0].frames_indices), indices)
                reference = processor(video[indices], return_tensors="pt")
                torch.testing.assert_close(output.pixel_values_videos, reference.pixel_values_videos)
        output = processor(
            video,
            do_sample_frames=True,
            fps=2,
            video_metadata=[{"total_num_frames": 8, "fps": 4}],
            return_tensors="pt",
        )
        self.assertEqual(output.video_grid_thw[0, 0].item(), 4)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            processor(video, do_sample_frames=True, fps=2, num_frames=3)

    def test_temporal_padding_repeats_last_frame(self):
        processor = self.fast_video_processing_class(**self.video_processor_dict)
        video = np.stack([np.full((32, 64, 3), i * 100, dtype=np.uint8) for i in range(3)])
        output = processor(video, temporal_patch_size=2, return_tensors="pt")
        padded = processor(np.concatenate([video, video[-1:]]), temporal_patch_size=2, return_tensors="pt")
        self.assertEqual(output.video_grid_thw.tolist(), [[2, 2, 4]])
        self.assertEqual(output.pixel_values_videos.shape, (16, 1536))
        torch.testing.assert_close(output.pixel_values_videos, padded.pixel_values_videos)

    @parameterized.expand(
        [
            ("default", {}),
            ("size", {"size": {"shortest_edge": 1024, "longest_edge": 3072}}),
            ("merge", {"merge_size": 1}),
            ("temporal", {"temporal_patch_size": 2}),
            ("cap", {"max_image_size": {"longest_edge": 2048}}),
        ]
    )
    def test_patch_estimate_matches_real_grid(self, _, kwargs):
        processor = self.fast_video_processing_class(**self.video_processor_dict)
        video = np.zeros((3, 64, 128, 3), dtype=np.uint8)
        output = processor(video, **kwargs, return_tensors="pt")
        estimate = processor.get_num_of_video_patches(3, 64, 128, kwargs)
        self.assertEqual(estimate, output.pixel_values_videos.shape[0])
        self.assertEqual(estimate, output.video_grid_thw.prod().item())

    def test_rejects_invalid_size_and_aspect_ratio(self):
        processor = self.fast_video_processing_class(**self.video_processor_dict)
        with self.assertRaisesRegex(ValueError, "shortest_edge.*longest_edge"):
            processor(np.zeros((2, 32, 64, 3), dtype=np.uint8), size={"height": 32, "width": 64})
        with self.assertRaisesRegex(ValueError, "aspect ratio"):
            processor(np.zeros((2, 1, 256, 3), dtype=np.uint8), input_data_format="channels_last")

    def test_cast_dtype_device(self):
        processor = self.fast_video_processing_class(**self.video_processor_dict)
        videos = self.video_processor_tester.prepare_video_inputs()
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                output = processor(videos, return_tensors="pt").to(dtype=dtype, device=torch_device)
                self.assertEqual(output.pixel_values_videos.dtype, dtype)
                self.assertEqual(output.pixel_values_videos.device.type, torch.device(torch_device).type)
                self.assertEqual(output.video_grid_thw.dtype, torch.long)
