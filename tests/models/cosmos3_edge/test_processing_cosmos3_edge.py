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

from transformers import (
    Cosmos3EdgeImageProcessor,
    Cosmos3EdgeImageProcessorPil,
    Cosmos3EdgeProcessor,
    Cosmos3EdgeVideoProcessor,
)
from transformers.testing_utils import (
    require_torch,
    require_torchvision,
    require_vision,
)
from transformers.utils import (
    is_vision_available,
)
from transformers.video_utils import VideoMetadata

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from PIL import Image


@require_torch
@require_vision
@require_torchvision
class Cosmos3EdgeProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Cosmos3EdgeProcessor
    tiny_model_id = "hf-internal-testing/tiny-processor-cosmos3-edge"

    @property
    def video_sampling_expectations(self):
        return [
            {"num_frames": 2, "fps": None, "expected_dim": 0, "output_length": 240},
            {"num_frames": None, "fps": 1, "expected_dim": 0, "output_length": 192},
            {"do_sample_frames": False, "fps": 10, "expected_dim": 0, "output_length": 176},
            {"do_sample_frames": False, "expected_dim": 0, "output_length": 176},
            {"expected_dim": 0, "output_length": 176},
        ]

    def prepare_images_inputs(self, batch_size: int | None = None, nested: bool = False):
        """Create small 64x96 inputs aligned to patch_size * merge_size (32).

        The fixed size keeps the processor tests lightweight and valid for patch
        merging; it is unrelated to testing per-image keyword arguments.
        """
        image = Image.fromarray(np.random.randint(255, size=(64, 96, 3), dtype=np.uint8))
        if batch_size is None:
            return image
        if nested:
            return [[image] for _ in range(batch_size)]
        return [image] * batch_size

    def prepare_videos_inputs(self, batch_size: int | None = None):
        """Create four 64x96 frames aligned to patch_size * merge_size (32).

        The fixed shape keeps frame-wise packing tests lightweight and valid; it
        is unrelated to testing per-video keyword arguments.
        """
        video = np.random.randint(255, size=(4, 64, 96, 3), dtype=np.uint8)
        if batch_size is None:
            return video
        return [video] * batch_size

    def test_image_processor_uses_projector_block_major_patch_order(self):
        """Protect the checkpoint's block-major patches and HWC values within each patch."""
        image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
        expected_patches = [
            [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17],
            [6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23],
            [24, 25, 26, 27, 28, 29, 36, 37, 38, 39, 40, 41],
            [30, 31, 32, 33, 34, 35, 42, 43, 44, 45, 46, 47],
        ]

        for image_processor_class in (Cosmos3EdgeImageProcessor, Cosmos3EdgeImageProcessorPil):
            processor = image_processor_class(
                do_resize=False,
                do_rescale=False,
                do_normalize=False,
                patch_size=2,
                merge_size=2,
            )
            processed = processor(image, return_tensors="pt")
            self.assertEqual(processed["pixel_values"].tolist(), expected_patches)

    def test_video_processor_uses_projector_block_major_patch_order_per_frame(self):
        """Protect projector block-major ordering independently within every frame."""
        processor = Cosmos3EdgeVideoProcessor(
            do_resize=False,
            do_rescale=False,
            do_normalize=False,
            patch_size=2,
            merge_size=2,
            temporal_patch_size=1,
        )
        video = np.arange(2 * 4 * 4 * 3, dtype=np.uint8).reshape(2, 4, 4, 3)
        first_frame_patches = [
            [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17],
            [6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23],
            [24, 25, 26, 27, 28, 29, 36, 37, 38, 39, 40, 41],
            [30, 31, 32, 33, 34, 35, 42, 43, 44, 45, 46, 47],
        ]
        expected_patches = first_frame_patches + [[value + 48 for value in patch] for patch in first_frame_patches]

        processed = processor(
            video,
            video_metadata=[{"fps": 2, "total_num_frames": 2, "duration": 1.0}],
            return_tensors="pt",
        )

        self.assertEqual(processed["pixel_values_videos"].tolist(), expected_patches)

    def test_processor_returns_multimodal_token_types_by_default(self):
        """Check the Edge default while allowing an explicit tokenizer override."""
        processor = object.__new__(Cosmos3EdgeProcessor)
        processor.tokenizer = SimpleNamespace()
        merged_kwargs = processor._merge_kwargs(
            Cosmos3EdgeProcessor.valid_processor_kwargs,
            tokenizer_init_kwargs={"return_mm_token_type_ids": True},
        )
        overridden_kwargs = processor._merge_kwargs(
            Cosmos3EdgeProcessor.valid_processor_kwargs,
            tokenizer_init_kwargs={"return_mm_token_type_ids": True},
            text_kwargs={"return_mm_token_type_ids": False},
        )

        self.assertTrue(merged_kwargs["text_kwargs"]["return_mm_token_type_ids"])
        self.assertFalse(overridden_kwargs["text_kwargs"]["return_mm_token_type_ids"])

    def test_video_placeholder_uses_one_timestamped_vision_span_per_frame(self):
        """Require one timestamped vision wrapper for each unmerged video frame."""
        processor = object.__new__(Cosmos3EdgeProcessor)
        processor.video_token = "<|video_pad|>"
        processor.vision_start_token = "<|vision_start|>"
        processor.vision_end_token = "<|vision_end|>"
        processor.video_processor = SimpleNamespace(merge_size=2, temporal_patch_size=1)
        video_inputs = {
            "video_grid_thw": np.asarray([[2, 2, 4]]),
            "video_metadata": [
                VideoMetadata(
                    total_num_frames=3,
                    fps=2,
                    duration=1.5,
                    frames_indices=[0, 2],
                )
            ],
        }

        replacement = processor.replace_video_token(video_inputs, video_idx=0)

        frame_span = "<|vision_start|><|video_pad|><|video_pad|><|vision_end|>"
        self.assertEqual(replacement, f"<0.0 seconds>{frame_span}<1.0 seconds>{frame_span}")

    def test_video_replacement_consumes_the_template_vision_wrapper_as_one_unit(self):
        """Ensure frame spans replace the full template wrapper without nested markers."""
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

    @unittest.skip("Model needs real tokenizer and isn't worth testing, as it's used in diffusers pipe")
    def test_replacement_offsets(self):
        pass
