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

from transformers.image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD
from transformers.testing_utils import require_torchvision, torch_device
from transformers.utils import is_torch_available, is_torchvision_available, is_vision_available
from transformers.video_utils import VideoMetadata

from ...test_video_processing_common import VideoProcessingTestMixin, prepare_video_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    from transformers.image_utils import PILImageResampling

    if is_torchvision_available():
        from transformers import HYV4VLVideoProcessor


class HYV4VLVideoProcessingTester:
    def __init__(self, parent):
        self.parent = parent
        self.batch_size = 3
        self.num_frames = 3
        self.num_channels = 3
        self.size = {"shortest_edge": 1024, "longest_edge": 32768}
        self.max_image_size = {"longest_edge": 8192}
        self.patch_size = 16
        self.merge_size = 2
        self.temporal_patch_size = 1
        self.image_mean = IMAGENET_STANDARD_MEAN
        self.image_std = IMAGENET_STANDARD_STD

    def prepare_video_processor_dict(self):
        return {
            "device": torch_device,
            "size": self.size.copy(),
            "max_image_size": self.max_image_size.copy(),
            "patch_size": self.patch_size,
            "merge_size": self.merge_size,
            "temporal_patch_size": self.temporal_patch_size,
            "do_sample_frames": False,
            "do_resize": True,
            "do_rescale": True,
            "do_normalize": True,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
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

    def test_processor_attributes_and_model_input_names(self):
        processor = self.fast_video_processing_class(**self.video_processor_dict)
        tester = self.video_processor_tester
        for name in ("size", "max_image_size", "patch_size", "merge_size", "temporal_patch_size"):
            self.assertEqual(getattr(processor, name), getattr(tester, name))
        defaults = self.fast_video_processing_class()
        for name in ("image_mean", "image_std"):
            self.assertEqual(tuple(getattr(processor, name)), tuple(getattr(tester, name)))
            self.assertEqual(tuple(getattr(defaults, name)), tuple(getattr(tester, name)))
        self.assertEqual(processor.resample, PILImageResampling.BICUBIC)
        self.assertFalse(processor.do_sample_frames)
        self.assertEqual(processor.model_input_names, ["pixel_values_videos", "video_grid_thw"])
        # HY4-VL budgets pixels per frame and per clip; the removed per-pixel knobs are not valid kwargs anymore
        for name in ("cap_pixels_per_frame", "max_video_tokens"):
            self.assertNotIn(name, processor.valid_kwargs.__annotations__)
            self.assertFalse(hasattr(processor, name))
        # fps-derived frame counts are bounded by the processor's frame limits (checkpoint: 2..1024)
        bounded = self.fast_video_processing_class(min_frames=2, max_frames=3)
        self.assertEqual(len(bounded.sample_frames(VideoMetadata(total_num_frames=12, fps=1), fps=1)), 3)
        self.assertEqual(len(bounded.sample_frames(VideoMetadata(total_num_frames=12, fps=24), fps=1)), 2)

    def _check_input_type(self, return_tensors, nested=False):
        # Necessary: HY4-VL returns flattened patches of shape (total_patches, patch_dim), so the common
        # `assert_expected_videos_shape` (which expects a leading video/frame axis) does not apply
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

    def test_video_processor_from_dict_with_kwargs(self):
        # Necessary: HY4-VL has no crop size and keeps `size` as a pixel budget, so the common crop
        # assertions do not apply
        processor = self.fast_video_processing_class.from_dict(
            self.video_processor_dict, size={"shortest_edge": 2048, "longest_edge": 16384}, max_frames=7
        )
        self.assertEqual(processor.size, {"shortest_edge": 2048, "longest_edge": 16384})
        self.assertEqual(processor.max_frames, 7)

    def test_cast_dtype_device(self):
        # Necessary: the processor follows `torch_device`, unlike the common mixin which expects cpu output
        processor = self.fast_video_processing_class(**self.video_processor_dict)
        videos = self.video_processor_tester.prepare_video_inputs()
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                output = processor(videos, return_tensors="pt").to(dtype=dtype, device=torch_device)
                self.assertEqual(output.pixel_values_videos.dtype, dtype)
                self.assertEqual(output.pixel_values_videos.device.type, torch.device(torch_device).type)
                self.assertEqual(output.video_grid_thw.dtype, torch.long)

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

    def test_call_sample_frames(self):
        # Necessary: the common test asserts a frame axis, while HY4-VL reports the sampled count in the grid
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

    def test_whole_clip_budget(self):
        # The whole clip shares one pixel budget, so longer clips get fewer pixels per frame
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
        # `max_image_size` caps the area of one frame, not its longest edge
        processor = self.fast_video_processing_class(
            size={"shortest_edge": 1024, "longest_edge": 65536},
            max_image_size={"longest_edge": 4096},
            device=torch_device,
        )
        output = processor(np.zeros((2, 64, 256, 3), dtype=np.uint8), return_tensors="pt")
        self.assertEqual(output.video_grid_thw.tolist(), [[2, 2, 8]])
        self.assertEqual(output.pixel_values_videos.shape, (32, 768))
        self.assertGreater(output.video_grid_thw[0, 2].item() * 16, 64)

    @parameterized.expand(
        [
            ("default", {}),
            ("no_resize", {"do_resize": False}),
            ("patch", {"patch_size": 8}),
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
        self.assertIsInstance(estimate, int)
        patches, grid_t, grid_h, grid_w = processor.get_num_of_video_patches(3, 64, 128, kwargs, return_grid=True)
        self.assertEqual(patches, estimate)
        self.assertEqual((grid_t, grid_h, grid_w), tuple(output.video_grid_thw[0].tolist()))
        self.assertEqual(estimate, output.pixel_values_videos.shape[0])
        self.assertEqual(estimate, output.video_grid_thw.prod().item())

    @parameterized.expand(
        [
            ("below_patch", 1024, 512),
            ("minimum_exceeds_cap", 4096, 2048),
            ("no_aligned_area", 1025, 2047),
        ]
    )
    def test_rejects_infeasible_minimum_and_cap(self, _, minimum, cap):
        processor = self.fast_video_processing_class(**self.video_processor_dict)
        kwargs = {
            "size": {"shortest_edge": minimum, "longest_edge": processor.size["longest_edge"]},
            "max_image_size": {"longest_edge": cap},
        }
        video = torch.zeros((2, 3, 32, 64), dtype=torch.uint8, device=torch_device)
        with self.assertRaisesRegex(ValueError, "pixel budget"):
            processor(video, input_data_format="channels_first", return_tensors="pt", **kwargs)
        with self.assertRaisesRegex(ValueError, "pixel budget"):
            processor.get_num_of_video_patches(2, 32, 64, kwargs)

    def test_rejects_invalid_size_and_aspect_ratio(self):
        processor = self.fast_video_processing_class(**self.video_processor_dict)
        with self.assertRaisesRegex(ValueError, "shortest_edge.*longest_edge"):
            processor(np.zeros((2, 32, 64, 3), dtype=np.uint8), size={"height": 32, "width": 64})
        with self.assertRaisesRegex(ValueError, "aspect ratio"):
            processor(np.zeros((2, 1, 256, 3), dtype=np.uint8), input_data_format="channels_last")
