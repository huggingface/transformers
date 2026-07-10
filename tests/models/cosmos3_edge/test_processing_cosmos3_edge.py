# Copyright 2026 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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
"""Focused processor tests for Cosmos3 Edge packed vision inputs."""

import unittest
from types import SimpleNamespace

import numpy as np

from transformers import Cosmos3EdgeImageProcessor, Cosmos3EdgeProcessor, Cosmos3EdgeVideoProcessor
from transformers.models.cosmos3_edge.processing_cosmos3_edge import Cosmos3EdgeProcessorKwargs
from transformers.testing_utils import require_torch, require_torchvision, require_vision


@require_torch
@require_vision
@require_torchvision
class Cosmos3EdgeVisionProcessorTest(unittest.TestCase):
    def test_image_processor_emits_packed_patches_and_thw_grid(self):
        processor = Cosmos3EdgeImageProcessor(
            do_resize=False,
            do_rescale=False,
            do_normalize=False,
            patch_size=2,
            merge_size=2,
        )
        image = np.zeros((4, 8, 3), dtype=np.uint8)

        processed = processor(image, return_tensors="pt")

        # 4 x 8 pixels with 2 x 2 patches produces a 2 x 4 patch grid.
        self.assertEqual(tuple(processed["pixel_values"].shape), (8, 12))
        self.assertEqual(processed["image_grid_thw"].tolist(), [[1, 2, 4]])

    def test_image_processor_uses_projector_block_major_patch_order(self):
        processor = Cosmos3EdgeImageProcessor(
            do_resize=False,
            do_rescale=False,
            do_normalize=False,
            patch_size=2,
            merge_size=2,
        )
        image = np.zeros((4, 8, 3), dtype=np.uint8)
        for height_index in range(2):
            for width_index in range(4):
                patch_index = height_index * 4 + width_index
                image[height_index * 2 : (height_index + 1) * 2, width_index * 2 : (width_index + 1) * 2] = patch_index

        processed = processor(image, return_tensors="pt")

        # The 2×2 groups must be contiguous for the checkpoint projector: the
        # first group is (0, 0), (0, 1), (1, 0), (1, 1), followed by the next group.
        self.assertEqual(processed["pixel_values"][:, 0].tolist(), [0, 1, 4, 5, 2, 3, 6, 7])

    def test_video_processor_emits_packed_patches_and_thw_grid(self):
        processor = Cosmos3EdgeVideoProcessor(
            do_resize=False,
            do_rescale=False,
            do_normalize=False,
            patch_size=2,
            merge_size=2,
            temporal_patch_size=1,
        )
        video = np.zeros((2, 4, 8, 3), dtype=np.uint8)
        metadata = [{"fps": 2, "total_num_frames": 2, "duration": 1.0}]

        processed = processor(video, video_metadata=metadata, return_tensors="pt")

        # Two 4 x 8 frames yield two 2 x 4 patch grids. Temporal patches stay
        # unmerged because Edge encodes one timestamped vision span per frame.
        self.assertEqual(tuple(processed["pixel_values_videos"].shape), (16, 12))
        self.assertEqual(processed["video_grid_thw"].tolist(), [[2, 2, 4]])

    def test_video_processor_uses_projector_block_major_patch_order_per_frame(self):
        processor = Cosmos3EdgeVideoProcessor(
            do_resize=False,
            do_rescale=False,
            do_normalize=False,
            patch_size=2,
            merge_size=2,
            temporal_patch_size=1,
        )
        video = np.zeros((2, 4, 8, 3), dtype=np.uint8)
        for frame_index in range(2):
            for height_index in range(2):
                for width_index in range(4):
                    patch_index = frame_index * 10 + height_index * 4 + width_index
                    video[
                        frame_index,
                        height_index * 2 : (height_index + 1) * 2,
                        width_index * 2 : (width_index + 1) * 2,
                    ] = patch_index

        processed = processor(
            video,
            video_metadata=[{"fps": 2, "total_num_frames": 2, "duration": 1.0}],
            return_tensors="pt",
        )

        self.assertEqual(
            processed["pixel_values_videos"][:, 0].tolist(),
            [0, 1, 4, 5, 2, 3, 6, 7, 10, 11, 14, 15, 12, 13, 16, 17],
        )

    def test_public_processor_name_is_cosmos_specific(self):
        self.assertEqual(Cosmos3EdgeProcessor.__name__, "Cosmos3EdgeProcessor")

    def test_processor_returns_multimodal_token_types_by_default(self):
        self.assertTrue(Cosmos3EdgeProcessorKwargs._defaults["text_kwargs"]["return_mm_token_type_ids"])

    def test_video_placeholder_uses_one_timestamped_vision_span_per_frame(self):
        # This isolates placeholder expansion from tokenizer loading. The checkpoint
        # records raw frames (temporal_patch_size=1), so each frame needs its own
        # timestamp and vision wrapper instead of one wrapper around the full video.
        processor = object.__new__(Cosmos3EdgeProcessor)
        processor.video_token = "<|video_pad|>"
        processor.vision_start_token = "<|vision_start|>"
        processor.vision_end_token = "<|vision_end|>"
        processor.video_processor = SimpleNamespace(merge_size=2, temporal_patch_size=1)
        video_inputs = {
            "video_grid_thw": np.asarray([[2, 2, 4]]),
            "video_metadata": [{"fps": 2, "frames_indices": [0, 2]}],
        }

        replacement = processor.replace_video_token(video_inputs, video_idx=0)

        frame_span = "<|vision_start|><|video_pad|><|video_pad|><|vision_end|>"
        self.assertEqual(replacement, f"<0.0 seconds>{frame_span}<1.0 seconds>{frame_span}")

    def test_video_replacement_consumes_the_template_vision_wrapper_as_one_unit(self):
        processor = object.__new__(Cosmos3EdgeProcessor)
        processor.image_token = "<|image_pad|>"
        processor.video_token = "<|video_pad|>"
        processor.vision_start_token = "<|vision_start|>"
        processor.vision_end_token = "<|vision_end|>"

        frame_span = "<|vision_start|><|video_pad|><|video_pad|><|vision_end|>"
        replacement = f"<0.0 seconds>{frame_span}<1.0 seconds>{frame_span}"
        template_text = "before<|vision_start|><|video_pad|><|vision_end|>after"
        text, replacement_offsets = processor.get_text_with_replacements(
            [template_text], videos_replacements=[replacement]
        )

        self.assertEqual(text, [f"before{replacement}after"])
        self.assertEqual(replacement_offsets[0][0]["text"], "<|vision_start|><|video_pad|><|vision_end|>")
