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

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from transformers import PreTrainedTokenizerFast
from transformers.testing_utils import require_torch, require_torchvision, require_vision
from transformers.utils import is_torchvision_available, is_vision_available

from ...test_processing_common import ProcessorTesterMixin


HunYuanVLImageProcessor = None
HunYuanVLProcessor = None

if is_vision_available():
    from PIL import Image

    from transformers.models.hunyuan_vl.processing_hunyuan_vl import HunYuanVLProcessor

    if is_torchvision_available():
        from transformers.models.hunyuan_vl.image_processing_hunyuan_vl import HunYuanVLImageProcessor


@require_vision
@require_torch
@require_torchvision
class HunYuanVLProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = HunYuanVLProcessor

    @classmethod
    def _setup_test_attributes(cls, processor):
        cls.image_token = processor.image_token
        cls.image_start_token = processor.image_start_token
        cls.image_end_token = processor.image_end_token

    def prepare_text_inputs(self, batch_size: int | None = None, modalities: str | list | None = None):
        if isinstance(modalities, str):
            modalities = [modalities]

        special_token_to_add = ""
        if modalities is not None and "image" in modalities:
            special_token_to_add += f"{self.image_start_token}{self.image_token}{self.image_end_token}"

        if batch_size is None:
            return f"lower newer {special_token_to_add}"

        if batch_size < 1:
            raise ValueError("batch_size must be greater than 0")

        if batch_size == 1:
            return [f"lower newer {special_token_to_add}"]
        return [f"lower newer {special_token_to_add}", f" {special_token_to_add} upper older longer string"] + [
            f"lower newer {special_token_to_add}"
        ] * (batch_size - 2)

    @classmethod
    def _setup_tokenizer(cls):
        vocab = {
            "<unk>": 0,
            "<pad>": 1,
            "<bos>": 2,
            "<eos>": 3,
            "<image_start>": 4,
            "<image>": 5,
            "<image_end>": 6,
            "hello": 7,
            "<placeholder>": 8,
            "<new_tail>": 9,
            "lower": 10,
            "newer": 11,
            "upper": 12,
            "older": 13,
            "longer": 14,
            "string": 15,
            "Describe": 16,
            "this.": 17,
        }
        tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()
        fast_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token="<unk>",
            pad_token="<pad>",
            bos_token="<bos>",
            eos_token="<eos>",
            extra_special_tokens={
                "image_start_token": "<image_start>",
                "image_token": "<image>",
                "image_end_token": "<image_end>",
            },
        )
        fast_tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% for content in message['content'] %}"
            "{% if content['type'] == 'image' %}<image_start><image><image_end>"
            "{% elif content['type'] == 'text' %}{{ content['text'] }}"
            "{% endif %}"
            "{% endfor %}"
            "{% endfor %}"
        )
        return fast_tokenizer

    @classmethod
    def _setup_image_processor(cls):
        return HunYuanVLImageProcessor(
            min_pixels=32 * 32,
            max_pixels=32 * 32,
            patch_size=16,
            temporal_patch_size=1,
            merge_size=1,
        )

    def test_processor_outputs_image_only_inputs(self):
        processor = self.get_processor()
        image = Image.new("RGB", (32, 32), color="white")

        inputs = processor(
            text=["<image_start><image><image_end> hello"], images=[image], padding=True, return_tensors="pt"
        )

        self.assertSetEqual(
            set(inputs.keys()),
            {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"},
        )
        self.assertGreater(inputs["pixel_values"].shape[0], 0)
        self.assertEqual(inputs["image_grid_thw"].shape[-1], 3)

    def test_get_num_multimodal_tokens(self):
        processor = self.get_processor()
        output = processor._get_num_multimodal_tokens(image_sizes=[(32, 32)])

        self.assertEqual(len(output["num_image_tokens"]), 1)
        self.assertEqual(len(output["num_image_patches"]), 1)
        self.assertGreater(output["num_image_tokens"][0], 0)

    def test_processor_uses_named_special_token_ids(self):
        processor = self.get_processor()
        image = Image.new("RGB", (32, 32), color="white")

        inputs = processor(
            text=["<image_start><image><image_end> hello"], images=[image], padding=True, return_tensors="pt"
        )

        input_ids = inputs["input_ids"][0].tolist()
        self.assertEqual(processor.image_token_id, processor.tokenizer.image_token_id)
        self.assertEqual(processor.image_start_token_id, processor.tokenizer.image_start_token_id)
        self.assertEqual(processor.image_end_token_id, processor.tokenizer.image_end_token_id)
        self.assertIn(processor.image_token_id, input_ids)
        self.assertNotIn(processor.tokenizer.convert_tokens_to_ids("<new_tail>"), input_ids)

    def test_processor_expands_wrapped_image_tokens(self):
        processor = self.get_processor()
        image = Image.new("RGB", (32, 32), color="white")

        inputs = processor(
            text=["<image_start><image><image_end> hello"], images=[image], padding=True, return_tensors="pt"
        )
        input_ids = inputs["input_ids"][0].tolist()
        _, grid_h, grid_w = (int(value) for value in inputs["image_grid_thw"][0])
        _, _, expected_image_tokens = processor._get_image_token_count(grid_h, grid_w)

        self.assertEqual(input_ids.count(processor.image_start_token_id), 1)
        self.assertEqual(input_ids.count(processor.image_token_id), expected_image_tokens)
        self.assertEqual(input_ids.count(processor.image_end_token_id), 1)

    def test_processor_rejects_bare_image_tokens(self):
        processor = self.get_processor()
        image = Image.new("RGB", (32, 32), color="white")

        with self.assertRaisesRegex(ValueError, "image placeholders must be formatted"):
            processor(text=["<image> hello"], images=[image], padding=True, return_tensors="pt")

    def test_apply_chat_template_keeps_wrapped_image_tokens_single_wrapped(self):
        processor = self.get_processor()
        image = Image.new("RGB", (32, 32), color="white")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "hello"},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        )
        input_ids = inputs["input_ids"][0].tolist()
        _, grid_h, grid_w = (int(value) for value in inputs["image_grid_thw"][0])
        _, _, expected_image_tokens = processor._get_image_token_count(grid_h, grid_w)

        self.assertEqual(input_ids.count(processor.image_start_token_id), 1)
        self.assertEqual(input_ids.count(processor.image_token_id), expected_image_tokens)
        self.assertEqual(input_ids.count(processor.image_end_token_id), 1)

    def test_model_input_names(self):
        processor = self.get_processor()
        self.assertSetEqual(
            set(processor.model_input_names),
            {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"},
        )
