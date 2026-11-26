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

import numpy as np
from tokenizers import Tokenizer, models

from transformers import (
    AutoProcessor,
    LlavaOnevision2ImageProcessorPil,
    LlavaOnevision2Processor,
    LlavaOnevision2VideoProcessor,
    TokenizersBackend,
)
from transformers.testing_utils import require_torch, require_torchvision, require_vision


def make_processor():
    tokens = ["[UNK]", "[PAD]", "<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>"]
    tokenizer = TokenizersBackend(
        tokenizer_object=Tokenizer(models.WordLevel({token: i for i, token in enumerate(tokens)}, unk_token="[UNK]")),
        unk_token="[UNK]",
        pad_token="[PAD]",
        additional_special_tokens=tokens[2:],
    )
    image_processor = LlavaOnevision2ImageProcessorPil(patch_size=2, size={"shortest_edge": 16, "longest_edge": 64})
    video_processor = LlavaOnevision2VideoProcessor(patch_size=2, size={"shortest_edge": 16, "longest_edge": 64})
    return LlavaOnevision2Processor(image_processor, tokenizer, video_processor)


@require_torch
@require_torchvision
@require_vision
class LlavaOnevision2ProcessorTest(unittest.TestCase):
    def setUp(self):
        self.processor = make_processor()
        self.image = np.zeros((8, 8, 3), dtype=np.uint8)
        self.image_prompt = "<|vision_start|><|image_pad|><|vision_end|>Describe."
        self.video_prompt = "<|vision_start|><|video_pad|><|vision_end|>Describe."

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

    def test_mixed_image_video_order(self):
        import torch

        video = np.full((2, 8, 8, 3), 255, dtype=np.uint8)
        options = {"do_sample_frames": False}
        for image_first in (True, False):
            prompt = self.image_prompt + self.video_prompt if image_first else self.video_prompt + self.image_prompt
            outputs = self.processor(
                images=self.image, videos=video, text=prompt, videos_kwargs=options, return_tensors="pt"
            )
            image = self.processor(images=self.image, text=self.image_prompt, return_tensors="pt")
            offset = 0 if image_first else 32
            torch.testing.assert_close(outputs["pixel_values"][offset : offset + 16], image["pixel_values"])
            self.assertEqual(outputs["image_grid_thw"].shape, (3, 3))
            self.assertEqual((outputs["input_ids"] == self.processor.image_token_id).sum(), 12)
            self.assertFalse((outputs["input_ids"] == self.processor.video_token_id).any())

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

    def test_save_load(self):
        with tempfile.TemporaryDirectory() as directory:
            self.processor.save_pretrained(directory)
            loaded = AutoProcessor.from_pretrained(directory, trust_remote_code=False)
            self.assertIsInstance(loaded, LlavaOnevision2Processor)
            self.assertEqual(loaded.video_processor.size, self.processor.video_processor.size)
            self.assertEqual(loaded.image_processor.temporal_patch_size, 1)

    def test_numpy_outputs(self):
        outputs = self.processor(images=self.image, text=self.image_prompt, return_tensors="np")
        self.assertIsInstance(outputs["input_ids"], np.ndarray)
        self.assertIsInstance(outputs["patch_positions"], np.ndarray)
