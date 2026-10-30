# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torchvision_available, is_vision_available


if is_vision_available() and is_torchvision_available():
    from transformers import NemotronH_Omni_Reasoning_V3VideoProcessor


NORM_MEAN = [0.48145466, 0.4578275, 0.40821073]
NORM_STD = [0.26862954, 0.26130258, 0.27577711]


@require_torch
@require_vision
class NemotronHOmniVideoProcessingTest(unittest.TestCase):
    """Frames are resized to one aspect-preserving tile near `video_target_num_patches`.

    The budgets here are tiny so the expected patch grids can be worked out by hand.
    """

    def _processor(self, **kwargs):
        defaults = {
            "norm_mean": NORM_MEAN,
            "norm_std": NORM_STD,
            "patch_size": 16,
            "downsample_ratio": 0.5,
            "video_target_num_patches": 16,
        }
        defaults.update(kwargs)
        return NemotronH_Omni_Reasoning_V3VideoProcessor(**defaults)

    def test_square_frames_get_a_square_patch_grid(self):
        processor = self._processor()
        # 16 patches, aspect 1:1 -> 4x4 grid, each patch 16px -> 64x64 pixels
        self.assertEqual(processor._compute_target_patches(64, 64), (4, 4))

    def test_patch_grid_is_a_multiple_of_the_downsample_factor(self):
        processor = self._processor()
        divisor = processor._downsample_factor
        for height, width in [(64, 64), (90, 160), (200, 50)]:
            wp, hp = processor._compute_target_patches(height, width)
            self.assertEqual(wp % divisor, 0, f"width patches {wp} not divisible by {divisor}")
            self.assertEqual(hp % divisor, 0, f"height patches {hp} not divisible by {divisor}")

    def test_aspect_ratio_is_preserved_in_patch_grid(self):
        processor = self._processor(video_target_num_patches=64)
        # a wide frame must not come back taller than it is wide
        wp, hp = processor._compute_target_patches(64, 256)
        self.assertGreater(wp, hp)

    def test_square_tile_when_aspect_ratio_not_maintained(self):
        processor = self._processor(video_maintain_aspect_ratio=False)
        wp, hp = processor._compute_target_patches(64, 256)
        self.assertEqual(wp, hp)

    def test_call_returns_normalized_tubelets(self):
        import torch

        processor = self._processor()
        num_frames = 3
        video = torch.randint(0, 256, (num_frames, 3, 64, 64), dtype=torch.uint8)

        out = processor(videos=[video], return_tensors="pt")

        wp, hp = processor._compute_target_patches(64, 64)
        expected_size = (hp * processor.patch_size, wp * processor.patch_size)
        # 3 frames pack into 2 tubelets of `temporal_patch_size` frames stacked along the channel dim
        self.assertEqual(tuple(out["pixel_values_videos"].shape), (1, 2, 2 * 3, *expected_size))
        self.assertEqual([int(n) for n in out["num_frames"]], [num_frames])
        torch.testing.assert_close(out["pixel_values_videos"][0, 1, 3:], out["pixel_values_videos"][0, 1, :3])
        # normalization actually ran, so values are no longer in [0, 255]
        self.assertLess(out["pixel_values_videos"].abs().max().item(), 10.0)

    def test_frames_are_resized_to_the_patch_grid(self):
        import torch

        processor = self._processor()
        num_frames = 4
        videos = [torch.randint(0, 256, (num_frames, 3, 72, 128), dtype=torch.uint8) for _ in range(2)]

        out = processor(videos=videos, return_tensors="pt")

        wp, hp = processor._compute_target_patches(72, 128)
        expected_size = (hp * processor.patch_size, wp * processor.patch_size)
        self.assertNotEqual(expected_size, (72, 128))
        self.assertEqual(tuple(out["pixel_values_videos"].shape), (2, num_frames // 2, 2 * 3, *expected_size))

    def test_videos_that_cannot_be_stacked_are_rejected(self):
        import torch

        processor = self._processor()
        for videos in (
            # different frame sizes after resizing
            [
                torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8),
                torch.randint(0, 256, (2, 3, 72, 128), dtype=torch.uint8),
            ],
            # different numbers of tubelets
            [
                torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8),
                torch.randint(0, 256, (4, 3, 64, 64), dtype=torch.uint8),
            ],
        ):
            with self.assertRaises(ValueError):
                processor(videos=videos, return_tensors="pt")
