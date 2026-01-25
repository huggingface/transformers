# Copyright 2025 The HuggingFace Team. All rights reserved.
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
from PIL import Image

from transformers.testing_utils import require_av, require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import GlmImageProcessor

if is_torch_available():
    import torch


@require_vision
@require_torch
class GlmImageProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = GlmImageProcessor
    model_id = "zai-org/GLM-Image"

    @classmethod
    def _setup_test_attributes(cls, processor):
        cls.image_token = processor.image_token

    @classmethod
    def _setup_from_pretrained(cls, model_id, **kwargs):
        return super()._setup_from_pretrained(
            model_id,
            subfolder="processor",
            do_sample_frames=False,
            patch_size=4,
            size={"shortest_edge": 12 * 12, "longest_edge": 18 * 18},
            **kwargs,
        )

    def prepare_image_inputs(self, batch_size: int | None = None, nested: bool = False):
        """Override to create images with valid aspect ratio (< 4) for GLM-Image."""
        # GLM-Image requires aspect ratio < 4, so use near-square images
        image_inputs = [Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))]
        if batch_size is None:
            return image_inputs
        if nested:
            return [image_inputs] * batch_size
        return image_inputs * batch_size

    @require_torch
    @require_av
    def _test_apply_chat_template(
        self,
        modality: str,
        batch_size: int,
        return_tensors: str,
        input_name: str,
        processor_name: str,
        input_data: list[str],
    ):
        # Skip image modality tests for GLM-Image because the processor expands image tokens
        # based on image size, making the tokenized output differ from direct tokenizer call
        if modality == "image":
            self.skipTest(
                "GLM-Image processor expands image tokens based on image size, "
                "making tokenized output differ from direct tokenizer call"
            )

        processor = self.get_processor()
        if processor.chat_template is None:
            self.skipTest("Processor has no chat template")

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
            fps=2
            if isinstance(input_data[0], str)
            else None,  # by default no more than 2 frames per second, otherwise too slow
        )
        input_name = getattr(self, input_name)
        self.assertTrue(input_name in out_dict)
        self.assertEqual(len(out_dict["input_ids"]), batch_size)
        self.assertEqual(len(out_dict["attention_mask"]), batch_size)

        mm_len = batch_size * 4
        self.assertEqual(len(out_dict[input_name]), mm_len)

        return_tensor_to_type = {"pt": torch.Tensor, "np": np.ndarray, None: list}
        for k in out_dict:
            self.assertIsInstance(out_dict[k], return_tensor_to_type[return_tensors])

    def test_model_input_names(self):
        processor = self.get_processor()

        text = self.prepare_text_inputs(modalities=["image"])
        image_input = self.prepare_image_inputs()
        inputs_dict = {"text": text, "images": image_input}
        inputs = processor(**inputs_dict, return_tensors="pt", do_sample_frames=False)

        self.assertSetEqual(set(inputs.keys()), set(processor.model_input_names))

    @unittest.skip(
        reason="GLM-Image processor adds image placeholder tokens which makes sequence length depend on image size"
    )
    def test_kwargs_overrides_default_tokenizer_kwargs(self):
        pass

    @unittest.skip(
        reason="GLM-Image processor adds image placeholder tokens which makes sequence length depend on image size"
    )
    def test_structured_kwargs_nested(self):
        pass

    @unittest.skip(
        reason="GLM-Image processor adds image placeholder tokens which makes sequence length depend on image size"
    )
    def test_structured_kwargs_nested_from_dict(self):
        pass

    @unittest.skip(
        reason="GLM-Image processor adds image placeholder tokens which makes sequence length depend on image size"
    )
    def test_unstructured_kwargs(self):
        pass

    @unittest.skip(
        reason="GLM-Image processor adds image placeholder tokens which makes sequence length depend on image size"
    )
    def test_unstructured_kwargs_batched(self):
        pass

    @unittest.skip(
        reason="GLM-Image processor adds image placeholder tokens which makes sequence length depend on image size"
    )
    def test_tokenizer_defaults(self):
        pass

    @unittest.skip(
        reason="GLM-Image processor adds image placeholder tokens which makes sequence length depend on image size"
    )
    def test_tokenizer_defaults_preserved_by_kwargs(self):
        pass
