# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch emu3 model."""

import tempfile
import unittest

import numpy as np

from transformers import Emu3Processor, GPT2TokenizerFast
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import Emu3ImageProcessor


class Emu3ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Emu3Processor

    @classmethod
    def setUpClass(cls):
        cls.tmpdirname = tempfile.mkdtemp()
        image_processor = Emu3ImageProcessor(min_pixels=28 * 28, max_pixels=56 * 56)
        extra_special_tokens = extra_special_tokens = {
            "image_token": "<image>",
            "boi_token": "<|image start|>",
            "eoi_token": "<|image end|>",
            "image_wrapper_token": "<|image token|>",
            "eof_token": "<|extra_201|>",
        }
        tokenizer = GPT2TokenizerFast.from_pretrained(
            "openai-community/gpt2", extra_special_tokens=extra_special_tokens
        )
        tokenizer.pad_token_id = 0
        tokenizer.sep_token_id = 1
        processor = cls.processor_class(
            image_processor=image_processor, tokenizer=tokenizer, chat_template="dummy_template"
        )
        processor.save_pretrained(cls.tmpdirname)
        cls.image_token = processor.image_token

    @staticmethod
    def prepare_processor_dict():
        return {
            "chat_template": "{% for message in messages %}{% if message['role'] != 'system' %}{{ message['role'].upper() + ': '}}{% endif %}{# Render all images first #}{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}{{ '<image>' }}{% endfor %}{# Render all text next #}{% if message['role'] != 'assistant' %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ content['text'] + ' '}}{% endfor %}{% else %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{% generation %}{{ content['text'] + ' '}}{% endgeneration %}{% endfor %}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}",
        }  # fmt: skip

    def test_processor_for_generation(self):
        processor_components = self.prepare_components()
        processor = self.processor_class(**processor_components)

        # we don't need an image as input because the model will generate one
        input_str = "lower newer"
        image_input = self.prepare_image_inputs()
        inputs = processor(text=input_str, return_for_image_generation=True, return_tensors="pt")
        self.assertListEqual(list(inputs.keys()), ["input_ids", "attention_mask", "image_sizes"])
        self.assertEqual(inputs[self.text_input_name].shape[-1], 8)

        # when `return_for_image_generation` is set, we raise an error that image should not be provided
        with self.assertRaises(ValueError):
            inputs = processor(
                text=input_str, images=image_input, return_for_image_generation=True, return_tensors="pt"
            )

    def test_processor_postprocess(self):
        processor_components = self.prepare_components()
        processor = self.processor_class(**processor_components)

        input_str = "lower newer"
        orig_image_input = self.prepare_image_inputs()
        orig_image = np.array(orig_image_input).transpose(2, 0, 1)

        inputs = processor(text=input_str, images=orig_image, do_resize=False, return_tensors="np")
        normalized_image_input = inputs.pixel_values
        unnormalized_images = processor.postprocess(normalized_image_input, return_tensors="np")["pixel_values"]

        # For an image where pixels go from 0 to 255 the diff can be 1 due to some numerical precision errors when scaling and unscaling
        self.assertTrue(np.abs(orig_image - unnormalized_images).max() >= 1)

    # Copied from tests.models.llava.test_processing_llava.LlavaProcessorTest.test_get_num_vision_tokens
    def test_get_num_vision_tokens(self):
        "Tests general functionality of the helper used internally in vLLM"

        processor = self.get_processor()

        output = processor._get_num_multimodal_tokens(image_sizes=[(100, 100), (300, 100), (500, 30)])
        self.assertTrue("num_image_tokens" in output)
        self.assertEqual(len(output["num_image_tokens"]), 3)

        self.assertTrue("num_image_patches" in output)
        self.assertEqual(len(output["num_image_patches"]), 3)
