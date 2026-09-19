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

from transformers import Dots3NoteVideoProcessor, is_torch_available
from transformers.testing_utils import require_torchvision


if is_torch_available():
    import torch


@require_torchvision
class Dots3NoteVideoProcessingTest(unittest.TestCase):
    def test_preprocess(self):
        processor = Dots3NoteVideoProcessor(patch_size=2, merge_size=2)
        for total_frames, do_sample_frames, expected_frames in (
            (2, False, 2),
            (230, True, 7),
            (237, True, 7),
            (359, True, 11),
        ):
            with self.subTest(total_frames=total_frames):
                outputs = processor(
                    torch.zeros(total_frames, 3, 4, 4),
                    video_metadata=[{"total_num_frames": total_frames, "fps": 30}],
                    do_resize=False,
                    do_sample_frames=do_sample_frames,
                    fps=1,
                    return_tensors="pt",
                )
                self.assertEqual(outputs.video_grid_thw.tolist(), [[expected_frames, 2, 2]])
                self.assertEqual(outputs.pixel_values_videos.shape, (expected_frames * 4, 12))
