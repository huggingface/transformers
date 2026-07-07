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

from transformers import PreTrainedTokenizerFast
from transformers.testing_utils import require_tokenizers, require_torch, require_torchvision, require_vision
from transformers.utils import (
    is_tokenizers_available,
    is_torch_available,
    is_torchvision_available,
    is_vision_available,
)

from ...test_processing_common import ProcessorTesterMixin


if is_torch_available():
    import torch

if is_tokenizers_available():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

if is_vision_available():
    from PIL import Image

    from transformers.models.hunyuan_vl.processing_hunyuan_vl import HunYuanVLProcessor

    if is_torchvision_available():
        from transformers.models.hunyuan_vl.image_processing_hunyuan_vl import HunYuanVLImageProcessor


@require_vision
@require_torch
@require_torchvision
@require_tokenizers
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
            {"input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids"},
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

        patch_h = grid_h // processor.image_processor.merge_size // processor.image_processor.spatial_patch_size
        patch_w = grid_w // processor.image_processor.merge_size // processor.image_processor.spatial_patch_size
        expected_image_tokens = patch_h * (patch_w + 1) + (2 if processor.cat_extra_token else 0)

        self.assertEqual(input_ids.count(processor.image_start_token_id), 1)
        self.assertEqual(input_ids.count(processor.image_token_id), expected_image_tokens)
        self.assertEqual(input_ids.count(processor.image_end_token_id), 1)

    def test_get_num_multimodal_tokens_matches_processor_call(self):
        "Tests that the helper used internally in vLLM works correctly"

        processor = self.get_processor()

        if not hasattr(processor, "_get_num_multimodal_tokens"):
            self.skipTest("Processor doesn't support `_get_num_multimodal_tokens` yet")

        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

        image_sizes = [(100, 100), (300, 100), (500, 30), (213, 167)]
        image_inputs = []
        for h, w in image_sizes:
            image_inputs.append(np.random.randint(255, size=(h, w, 3), dtype=np.uint8))

        image_token = f"{self.image_start_token}{self.image_token}{self.image_end_token}"
        text = [f"This is an image {image_token}"] * len(image_inputs)
        inputs = processor(
            text=text, images=image_inputs, padding=True, return_mm_token_type_ids=True, return_tensors="pt"
        )

        if "mm_token_type_ids" not in inputs:
            self.skipTest("Processor doesn't support `mm_token_type_ids`")

        num_image_tokens_from_call = inputs.mm_token_type_ids.sum(-1).tolist()
        num_image_tokens_from_helper = processor._get_num_multimodal_tokens(image_sizes=image_sizes)
        self.assertListEqual(num_image_tokens_from_call, num_image_tokens_from_helper["num_image_tokens"])

        # Test with two images per single text
        text = [f"These are two images {image_token}{image_token}"] * len(image_inputs)
        inputs = processor(
            text=text,
            images=image_inputs * 2,
            padding=True,
            return_mm_token_type_ids=True,
            return_tensors="pt",
        )

        num_image_tokens_from_call = inputs.mm_token_type_ids.sum(-1).tolist()
        num_image_tokens_from_helper = processor._get_num_multimodal_tokens(image_sizes=image_sizes * 2)
        self.assertEqual(sum(num_image_tokens_from_call), sum(num_image_tokens_from_helper["num_image_tokens"]))

    def _test_apply_chat_template(
        self,
        modality: str,
        batch_size: int,
        return_tensors: str,
        input_name: str,
        processor_name: str,
        input_data: list[str],
    ):
        processor = self.get_processor()
        if processor_name not in self.processor_class.get_attributes():
            self.skipTest(f"{processor_name} attribute not present in {self.processor_class}")

        batch_messages = [
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Describe this."}],
                },
            ]
        ] * batch_size

        # Test that jinja can be applied
        formatted_prompt = processor.apply_chat_template(batch_messages, add_generation_prompt=True, tokenize=False)
        self.assertEqual(len(formatted_prompt), batch_size)

        # Test that tokenizing with template and directly with `self.tokenizer` gives same output
        formatted_prompt_tokenized = processor.apply_chat_template(
            batch_messages, add_generation_prompt=True, tokenize=True, return_tensors=return_tensors
        )
        add_special_tokens = True
        if processor.tokenizer.bos_token is not None and formatted_prompt[0].startswith(processor.tokenizer.bos_token):
            add_special_tokens = False
        tok_output = processor.tokenizer(
            formatted_prompt, return_tensors=return_tensors, add_special_tokens=add_special_tokens
        )
        expected_output = tok_output.input_ids
        self.assertListEqual(expected_output.tolist(), formatted_prompt_tokenized.tolist())

        # Test that kwargs passed to processor's `__call__` are actually used
        tokenized_prompt_100 = processor.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=True,
            padding="max_length",
            truncation=True,
            return_tensors=return_tensors,
            max_length=100,
        )
        self.assertEqual(len(tokenized_prompt_100[0]), 100)

        # Test that `return_dict=True` returns text related inputs in the dict
        out_dict_text = processor.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors=return_tensors,
        )
        self.assertTrue(all(key in out_dict_text for key in ["input_ids", "attention_mask"]))
        self.assertEqual(len(out_dict_text["input_ids"]), batch_size)
        self.assertEqual(len(out_dict_text["attention_mask"]), batch_size)

        # Test that with modality URLs and `return_dict=True`, we get modality inputs in the dict
        for idx, url in enumerate(input_data[:batch_size]):
            batch_messages[idx][0]["content"] = [batch_messages[idx][0]["content"][0], {"type": modality, "url": url}]

        out_dict = processor.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors=return_tensors,
        )
        input_name = getattr(self, input_name)
        self.assertTrue(input_name in out_dict)
        self.assertEqual(len(out_dict["input_ids"]), batch_size)
        self.assertEqual(len(out_dict["attention_mask"]), batch_size)
        self.assertEqual(len(out_dict[input_name]), batch_size * 2)

        return_tensor_to_type = {"pt": torch.Tensor, "np": np.ndarray, None: list}
        for k in out_dict:
            self.assertIsInstance(out_dict[k], return_tensor_to_type[return_tensors])
