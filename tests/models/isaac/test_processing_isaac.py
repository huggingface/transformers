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

"""Testing suite for the Isaac processor."""

import os
import unittest
from pathlib import Path

import numpy as np
import pytest
from huggingface_hub import is_offline_mode

from transformers.models.isaac.processing_isaac import IsaacProcessor
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image
else:
    Image = None


def _make_dummy_image(size=(32, 32), color=(255, 0, 0)):
    if Image is None:
        raise RuntimeError("PIL.Image is not available in this environment.")
    return Image.new("RGB", size, color=color)


BASE_MODEL_ID = os.environ.get("ISAAC_TEST_MODEL_ID", "PerceptronAI/Isaac-0.1-Base")
BASE_MODEL_REVISION = os.environ.get("ISAAC_TEST_MODEL_REVISION", "refs/pr/3") or None
LOCAL_CHECKPOINT = os.environ.get("ISAAC_TEST_MODEL_PATH")


def _checkpoint_or_skip(model_id=BASE_MODEL_ID):
    if LOCAL_CHECKPOINT:
        resolved = Path(LOCAL_CHECKPOINT).expanduser()
        if not resolved.exists():
            pytest.skip(f"Local checkpoint path {resolved} does not exist.")
        return str(resolved)
    if is_offline_mode():
        pytest.skip("Offline mode: set ISAAC_TEST_MODEL_PATH to a local checkpoint to run these tests.")
    return model_id


@require_torch
@require_vision
class IsaacProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = IsaacProcessor
    model_id = BASE_MODEL_ID
    images_input_name = "pixel_values"

    @classmethod
    def _setup_from_pretrained(cls, model_id, **kwargs):
        checkpoint = _checkpoint_or_skip(model_id)
        return super()._setup_from_pretrained(
            checkpoint,
            revision=BASE_MODEL_REVISION,
            patch_size=4,
            max_num_patches=4,
            **kwargs,
        )

    @classmethod
    def _setup_test_attributes(cls, processor):
        cls.image_token = processor.image_token
        cls.pad_token_id = processor.tokenizer.pad_token_id
        cls.image_pad_token_id = processor.image_token_id

    def prepare_image_inputs(self, batch_size: int | None = None, nested: bool = False):
        if batch_size is None:
            return _make_dummy_image(size=(16, 16))
        images = [_make_dummy_image(size=(16, 16), color=(50 * (i + 1), 0, 0)) for i in range(batch_size)]
        if nested:
            return [[image] for image in images]
        return images

    @unittest.skip("Isaac chat templates emit <image> placeholders but the processor consumes image pad tokens")
    def test_apply_chat_template_image_0(self):
        pass

    @unittest.skip("Isaac chat templates emit <image> placeholders but the processor consumes image pad tokens")
    def test_apply_chat_template_image_1(self):
        pass

    def test_apply_chat_template_image_placeholder_expands_to_image_pad_tokens(self):
        processor = self.get_processor()
        image = _make_dummy_image(size=(16, 16))
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this."},
                        {"type": "image", "image": image},
                    ],
                }
            ]
        ]

        formatted_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        self.assertEqual(len(formatted_prompt), 1)
        self.assertIn("<image>", formatted_prompt[0])

        out_dict = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        self.assertTrue(
            all(
                key in out_dict
                for key in [
                    "input_ids",
                    "attention_mask",
                    "pixel_values",
                    "image_grid_thw",
                    "image_metadata",
                    "mm_token_type_ids",
                ]
            )
        )

        expected_num_image_tokens = processor._get_num_multimodal_tokens(image_sizes=[(image.height, image.width)])[
            "num_image_tokens"
        ][0]
        actual_num_image_tokens = int(out_dict["input_ids"][0].eq(processor.image_token_id).sum().item())

        self.assertEqual(actual_num_image_tokens, expected_num_image_tokens)
        self.assertEqual(int(out_dict["mm_token_type_ids"][0].sum().item()), expected_num_image_tokens)
        self.assertEqual(int(out_dict["image_metadata"][0, 0, 1].item()), expected_num_image_tokens)
        self.assertTrue(
            torch.all(out_dict["mm_token_type_ids"][0][out_dict["input_ids"][0].eq(processor.image_token_id)] == 1)
        )

    def test_get_num_multimodal_tokens_matches_processor_call(self):
        processor = self.get_processor()

        image_sizes = [(100, 100), (300, 100), (500, 30), (213, 167)]
        image_inputs = [np.random.randint(255, size=(h, w, 3), dtype=np.uint8) for h, w in image_sizes]

        text = [f"This is an image {self.image_token}"] * len(image_inputs)
        inputs = processor(
            text=text,
            images=[[image] for image in image_inputs],
            padding=True,
            return_mm_token_type_ids=True,
            return_tensors="pt",
        )

        num_image_tokens_from_call = inputs.mm_token_type_ids.sum(-1).tolist()
        num_image_tokens_from_helper = processor._get_num_multimodal_tokens(image_sizes=image_sizes)
        self.assertListEqual(num_image_tokens_from_call, num_image_tokens_from_helper["num_image_tokens"])
