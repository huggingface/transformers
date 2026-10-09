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

import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from parameterized import parameterized

from transformers.image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD
from transformers.testing_utils import require_torch, require_torchvision, require_vision, torch_device
from transformers.utils import is_torch_available, is_torchvision_available, is_vision_available
from transformers.video_utils import VideoMetadata

from ...test_video_processing_common import VideoProcessingTestMixin, prepare_video_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    from transformers.image_utils import PILImageResampling, SizeDict

    if is_torchvision_available():
        from transformers import AutoVideoProcessor, HYV4VLVideoProcessor
        from transformers.video_processing_utils import BaseVideoProcessor


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
        for name in ("cap_pixels_per_frame", "max_video_tokens"):
            self.assertNotIn(name, processor.valid_kwargs.__annotations__)
            self.assertFalse(hasattr(processor, name))

    def test_video_processor_from_dict_with_kwargs(self):
        processor = self.fast_video_processing_class.from_dict(
            self.video_processor_dict, size={"shortest_edge": 2048, "longest_edge": 16384}, max_frames=7
        )
        self.assertEqual(processor.size, {"shortest_edge": 2048, "longest_edge": 16384})
        self.assertEqual(processor.max_frames, 7)

    def test_canonical_pixel_budget_roundtrip(self):
        processor = self.fast_video_processing_class(**self.video_processor_dict, resample=PILImageResampling.BILINEAR)
        video = torch.randint(0, 256, (3, 3, 64, 128), dtype=torch.uint8, device=torch_device)
        with tempfile.TemporaryDirectory() as directory:
            processor.save_pretrained(directory)
            restored = AutoVideoProcessor.from_pretrained(directory, local_files_only=True, device=torch_device)
        self.assertIs(type(restored), self.fast_video_processing_class)
        self.assertEqual(restored.size, processor.size)
        self.assertEqual(restored.max_image_size, processor.max_image_size)
        self.assertEqual(restored.resample, PILImageResampling.BILINEAR)
        for name in ("min_pixels", "max_pixels", "video_max_image_size", "cap_pixels_per_frame", "max_video_tokens"):
            self.assertNotIn(name, restored.to_dict())
        expected = processor(video, input_data_format="channels_first", return_tensors="pt")
        actual = restored(video, input_data_format="channels_first", return_tensors="pt")
        torch.testing.assert_close(actual.pixel_values_videos, expected.pixel_values_videos, atol=0, rtol=0)
        torch.testing.assert_close(actual.video_grid_thw, expected.video_grid_thw)

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

    def test_rgb_conversion_is_handled_before_preprocess(self):
        processor = self.fast_video_processing_class(**self.video_processor_dict)
        video = torch.randint(0, 256, (3, 3, 32, 64), device=torch_device, dtype=torch.uint8)
        with patch.object(processor, "convert_to_rgb", wraps=processor.convert_to_rgb) as convert:
            output = processor(video, input_data_format="channels_first", do_convert_rgb=True, return_tensors="pt")
        self.assertEqual(convert.call_count, 1)
        self.assertEqual(output.video_grid_thw.tolist(), [[3, 2, 4]])
        self.assertTrue(output.pixel_values_videos.isfinite().all())

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

    @parameterized.expand(
        [
            ("round_half_to_even", 3, 80, 112, 4, 8),
            ("downscale", 2, 128, 256, 4, 8),
            ("upscale", 3, 32, 32, 4, 4),
        ]
    )
    def test_regular_resize_rounding(self, _, frames, height, width, grid_h, grid_w):
        settings = {**self.video_processor_dict, "size": {"shortest_edge": 4096, "longest_edge": 32768}}
        processor = self.fast_video_processing_class(**settings)
        video = torch.zeros((frames, 3, height, width), dtype=torch.uint8, device=torch_device)
        output = processor(video, input_data_format="channels_first", return_tensors="pt")
        self.assertEqual(output.video_grid_thw.tolist(), [[frames, grid_h, grid_w]])
        self.assertEqual(
            processor.get_num_of_video_patches(frames, height, width, return_grid=True)[1:],
            (frames, grid_h, grid_w),
        )

    def test_call_time_area_cap_overrides_instance_without_mutating_it(self):
        processor = self.fast_video_processing_class(**self.video_processor_dict)
        video = torch.zeros((3, 3, 64, 128), dtype=torch.uint8, device=torch_device)
        original_cap = processor.max_image_size.copy()
        default = processor(video, input_data_format="channels_first", return_tensors="pt")
        kwargs = {"max_image_size": {"longest_edge": 2048}}
        capped = processor(video, input_data_format="channels_first", return_tensors="pt", **kwargs)
        self.assertEqual(default.video_grid_thw.tolist(), [[3, 4, 8]])
        self.assertEqual(capped.video_grid_thw.tolist(), [[3, 2, 4]])
        self.assertEqual(
            processor.get_num_of_video_patches(3, 64, 128, kwargs, return_grid=True)[1:],
            tuple(capped.video_grid_thw[0].tolist()),
        )
        self.assertEqual(processor.get_num_of_video_patches(3, 64, 128, kwargs), capped.pixel_values_videos.shape[0])
        self.assertEqual(processor.max_image_size, original_cap)
        repeated = processor(video, input_data_format="channels_first", return_tensors="pt")
        torch.testing.assert_close(repeated.pixel_values_videos, default.pixel_values_videos, atol=0, rtol=0)

    @parameterized.expand([("landscape", 64, 8192), ("portrait", 8192, 64)])
    def test_extreme_aspect_ratio_respects_clip_budget(self, _, height, width):
        processor = self.fast_video_processing_class(device=torch_device)
        _, grid_t, grid_h, grid_w = processor.get_num_of_video_patches(1024, height, width, return_grid=True)
        patch_area = processor.patch_size**2
        self.assertLessEqual(grid_t * grid_h * grid_w * patch_area, processor.size["longest_edge"])
        self.assertEqual((grid_h, grid_w), (2, 64) if height < width else (64, 2))
        # The same per-frame budget on two frames exercises real resizing without a 1024-frame allocation.
        size = {
            "shortest_edge": processor.size["shortest_edge"],
            "longest_edge": processor.size["longest_edge"] // 512,
        }
        video = torch.zeros((2, 3, height, width), dtype=torch.uint8, device=torch_device)
        actual = processor(video, size=size, input_data_format="channels_first", return_tensors="pt")
        self.assertEqual(actual.video_grid_thw.tolist(), [[2, grid_h, grid_w]])
        self.assertLessEqual(actual.video_grid_thw.prod().item() * patch_area, size["longest_edge"])

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

    def test_resample_argument_controls_pixels(self):
        processor = self.fast_video_processing_class(**self.video_processor_dict)
        video = torch.randint(0, 256, (3, 3, 97, 193), dtype=torch.uint8, device=torch_device)
        kwargs = {
            "do_rescale": False,
            "do_normalize": False,
            "input_data_format": "channels_first",
            "return_tensors": "pt",
        }
        default = processor(video, **kwargs)
        for resample in (PILImageResampling.BILINEAR, PILImageResampling.NEAREST):
            with self.subTest(resample=resample):
                actual = processor(video, resample=resample, **kwargs)
                _, grid_h, grid_w = actual.video_grid_thw[0].tolist()
                resized = BaseVideoProcessor.resize(
                    processor,
                    image=video.unsqueeze(0),
                    size=SizeDict(height=grid_h * processor.patch_size, width=grid_w * processor.patch_size),
                    resample=resample,
                )
                expected, *_ = processor.patchify(
                    resized, processor.patch_size, processor.merge_size, processor.temporal_patch_size
                )
                torch.testing.assert_close(actual.pixel_values_videos, expected[0], atol=0, rtol=0)
                self.assertFalse(torch.equal(actual.pixel_values_videos, default.pixel_values_videos))
        self.assertEqual(processor.resample, PILImageResampling.BICUBIC)

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

    def test_row_major_patch_order(self):
        processor = self.fast_video_processing_class(**self.video_processor_dict)
        video = torch.arange(3 * 3 * 32 * 64, dtype=torch.float32, device=torch_device).reshape(3, 3, 32, 64)
        output = processor(
            video,
            do_resize=False,
            do_rescale=False,
            do_normalize=False,
            input_data_format="channels_first",
            return_tensors="pt",
        )
        expected = video.reshape(3, 3, 2, 16, 4, 16).permute(0, 2, 4, 1, 3, 5).reshape(24, 768)
        torch.testing.assert_close(output.pixel_values_videos, expected, atol=0, rtol=0)

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
