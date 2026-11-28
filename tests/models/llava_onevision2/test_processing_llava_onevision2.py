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
from tokenizers import Tokenizer, models, pre_tokenizers

from transformers import (
    LlavaOnevision2ImageProcessorPil,
    LlavaOnevision2Processor,
    LlavaOnevision2VideoProcessor,
    TokenizersBackend,
)
from transformers.testing_utils import require_torch, require_torchvision, require_vision

from ...test_processing_common import ProcessorTesterMixin


@require_torch
@require_torchvision
@require_vision
class LlavaOnevision2ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = LlavaOnevision2Processor
    videos_input_name = "pixel_values"

    @classmethod
    def _setup_tokenizer(cls):
        special_tokens = ["[UNK]", "[PAD]", "<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>"]
        tokens = special_tokens + ["lower", "newer", "upper", "older", "longer", "string"]
        tokenizer = Tokenizer(models.WordLevel({token: i for i, token in enumerate(tokens)}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        return TokenizersBackend(
            tokenizer_object=tokenizer,
            unk_token="[UNK]",
            pad_token="[PAD]",
            additional_special_tokens=special_tokens[2:],
        )

    @classmethod
    def _setup_image_processor(cls):
        return LlavaOnevision2ImageProcessorPil(patch_size=2, size={"shortest_edge": 16, "longest_edge": 64})

    @classmethod
    def _setup_video_processor(cls):
        return LlavaOnevision2VideoProcessor(patch_size=2, size={"shortest_edge": 16, "longest_edge": 64})

    def setUp(self):
        super().setUp()
        self.processor = self.get_processor()
        self.image = np.zeros((8, 8, 3), dtype=np.uint8)
        self.image_prompt = "<|vision_start|><|image_pad|><|vision_end|>Describe."

    def test_image_tokens_match_patches(self):
        outputs = self.processor(images=self.image, text=self.image_prompt, return_tensors="pt")
        self.assertEqual(outputs["pixel_values"].shape, (16, 12))
        self.assertEqual((outputs["input_ids"] == self.processor.image_token_id).sum(), 4)
        self.assertEqual(outputs["patch_positions"].shape, (16, 3))

    def test_multiple_images(self):
        outputs = self.processor(images=[self.image, self.image], text=self.image_prompt * 2, return_tensors="pt")
        self.assertEqual(outputs["image_grid_thw"].shape, (2, 3))
        self.assertEqual((outputs["input_ids"] == self.processor.image_token_id).sum(), 8)

    def test_images_without_text(self):
        outputs = self.processor(images=self.image, return_tensors="pt")
        self.assertEqual(outputs["pixel_values"].shape, (16, 12))
        self.assertNotIn("input_ids", outputs)

    def test_conflicting_image_geometry(self):
        for name, value in (("patch_size", 4), ("merge_size", 1), ("temporal_patch_size", 2)):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, f"{name} must match"):
                self.processor(images=self.image, text=self.image_prompt, images_kwargs={name: value})

    def test_truncated_visual_tokens(self):
        with self.assertRaises(ValueError):
            self.processor(images=self.image, text=self.image_prompt, truncation=True, max_length=2)

    def test_batch_with_unequal_visual_counts(self):
        outputs = self.processor(
            images=[self.image] * 3, text=[self.image_prompt, self.image_prompt * 2], padding=True, return_tensors="pt"
        )
        self.assertEqual(outputs["input_ids"].shape[0], 2)
        self.assertEqual(outputs["image_grid_thw"].shape[0], 3)

    def test_visual_input_count_validation(self):
        with self.assertRaises((ValueError, StopIteration)):
            self.processor(images=[self.image] * 2, text=self.image_prompt)
        with self.assertRaises((ValueError, StopIteration)):
            self.processor(images=self.image, text=self.image_prompt * 2)
