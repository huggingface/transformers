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
"""Testing suite for the QianfanOCR processor."""

import copy
import unittest

from parameterized import parameterized

from transformers import QianfanOCRProcessor
from transformers.testing_utils import require_torch, require_vision, slow
from transformers.utils import is_torch_available

from ...test_processing_common import MODALITY_INPUT_DATA, ProcessorTesterMixin


if is_torch_available():
    import torch


@slow
@require_vision
class QianfanOCRProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = QianfanOCRProcessor
    model_id = "bairongz/QianfanOCR"
    # QianfanOCR has no video support; images and pixel values share the same tensor key
    videos_input_name = "pixel_values"

    @classmethod
    def _setup_test_attributes(cls, processor):
        cls.image_token = processor.image_placeholder_token

    @unittest.skip("QianfanOCR does not support video processing")
    def test_video_processor_defaults(self):
        pass

    @unittest.skip("QianfanOCR does not support video processing")
    def test_process_interleaved_images_videos(self):
        pass

    def test_model_input_names(self):
        processor = self.get_processor()

        text = self.prepare_text_inputs(modalities=["image"])
        image_input = self.prepare_image_inputs()
        inputs = processor(text=text, images=image_input, return_tensors="pt")

        self.assertSetEqual(set(inputs.keys()), set(processor.model_input_names))

    @staticmethod
    def prepare_processor_dict():
        return {"image_seq_length": 2}

    @require_torch
    def _test_apply_chat_template(
        self,
        modality: str,
        batch_size: int,
        return_tensors: str,
        input_name: str,
        processor_name: str,
        input_data: list,
    ):
        processor = self.get_processor()
        if processor.chat_template is None:
            self.skipTest("Processor has no chat template")

        if processor_name not in self.processor_class.get_attributes():
            self.skipTest(f"{processor_name} attribute not present in {self.processor_class}")

        batch_messages = [
            copy.deepcopy(
                [
                    {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                    {"role": "user", "content": [{"type": "text", "text": "Describe this."}]},
                ]
            )
            for _ in range(batch_size)
        ]

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
            return_tensors=return_tensors,
            processor_kwargs={"max_length": 100, "padding": "max_length", "truncation": True},
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

        # Test that with image URLs and `return_dict=True`, we get pixel_values in the dict
        for idx, url in enumerate(input_data[:batch_size]):
            batch_messages[idx][1]["content"] = [batch_messages[idx][1]["content"][0], {"type": modality, "url": url}]

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

        # QianfanOCR uses dynamic patching: pixel_values shape is [total_patches, C, H, W],
        # not [batch_size, C, H, W]. Count image occurrences across messages to verify.
        num_images = sum(
            1
            for message_thread in batch_messages
            for message in message_thread
            for content in message.get("content", [])
            if content.get("type") == "image"
        )
        num_patches_per_image = len(out_dict[input_name]) // num_images
        self.assertEqual(len(out_dict[input_name]), num_images * num_patches_per_image)
        for k in out_dict:
            self.assertIsInstance(out_dict[k], torch.Tensor)

        # Test continue from final message
        assistant_message = {
            "role": "assistant",
            "content": [{"type": "text", "text": "It is the sound of"}],
        }
        for batch_idx in range(batch_size):
            batch_messages[batch_idx] = batch_messages[batch_idx] + [assistant_message]
        continue_prompt = processor.apply_chat_template(batch_messages, continue_final_message=True, tokenize=False)
        for prompt in continue_prompt:
            self.assertTrue(prompt.endswith("It is the sound of"))  # no `eos` token at the end

    @parameterized.expand([(1, "pt"), (2, "pt")])
    def test_apply_chat_template_image(self, batch_size: int, return_tensors: str):
        self._test_apply_chat_template(
            "image", batch_size, return_tensors, "images_input_name", "image_processor", MODALITY_INPUT_DATA["images"]
        )

    @require_torch
    def test_get_num_vision_tokens(self):
        """Tests general functionality of the helper used internally in vLLM."""
        processor = self.get_processor()

        output = processor._get_num_multimodal_tokens(image_sizes=[(100, 100), (300, 100), (500, 30)])
        self.assertIn("num_image_tokens", output)
        self.assertEqual(len(output["num_image_tokens"]), 3)

        self.assertIn("num_image_patches", output)
        self.assertEqual(len(output["num_image_patches"]), 3)
