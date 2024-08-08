# Copyright 2024 The HuggingFace Team. All rights reserved.
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
import json
import tempfile
import unittest

import torch

from transformers import AutoProcessor, LlamaTokenizerFast, LlavaNextProcessor
from transformers.testing_utils import require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import CLIPImageProcessor


@require_vision
class LlavaNextProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = LlavaNextProcessor

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()

        image_processor = CLIPImageProcessor()
        tokenizer = LlamaTokenizerFast.from_pretrained("huggyllama/llama-7b")
        processor_kwargs = self.prepare_processor_dict()
        processor = LlavaNextProcessor(image_processor, tokenizer, **processor_kwargs)
        processor.save_pretrained(self.tmpdirname)

    def get_tokenizer(self, **kwargs):
        return LlavaNextProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_image_processor(self, **kwargs):
        return LlavaNextProcessor.from_pretrained(self.tmpdirname, **kwargs).image_processor

    def prepare_processor_dict(self):
        return {"chat_template": "dummy_template"}

    @unittest.skip(
        "Skip because the model has no processor kwargs except for chat template and"
        "chat template is saved as a separate file. Stop skipping this test when the processor"
        "has new kwargs saved in config file."
    )
    def test_processor_to_json_string(self):
        pass

    # Copied from tests.models.llava.test_processor_llava.LlavaProcessorTest.test_chat_template_is_saved
    def test_chat_template_is_saved(self):
        processor_loaded = self.processor_class.from_pretrained(self.tmpdirname)
        processor_dict_loaded = json.loads(processor_loaded.to_json_string())
        # chat templates aren't serialized to json in processors
        self.assertFalse("chat_template" in processor_dict_loaded.keys())

        # they have to be saved as separate file and loaded back from that file
        # so we check if the same template is loaded
        processor_dict = self.prepare_processor_dict()
        self.assertTrue(processor_loaded.chat_template == processor_dict.get("chat_template", None))

    def test_chat_template(self):
        processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")
        expected_prompt = "USER: <image>\nWhat is shown in this image? ASSISTANT:"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]

        formatted_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        self.assertEqual(expected_prompt, formatted_prompt)

    def test_image_token_filling(self):
        processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")
        processor.patch_size = 14
        processor.vision_feature_select_strategy = "default"
        # Important to check with non square image
        image = torch.randint(0, 2, (3, 500, 316))
        expected_image_tokens = 1526
        image_token_index = 32000

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        inputs = processor(
            text=[processor.apply_chat_template(messages)],
            images=[image],
            return_tensors="pt",
        )
        image_tokens = (inputs["input_ids"] == image_token_index).sum().item()
        self.assertEqual(expected_image_tokens, image_tokens)

    # @require_vision
    # @require_torch
    # def test_tokenizer_defaults_preserved_by_kwargs(self):
    #     if "image_processor" not in self.processor_class.attributes:
    #         self.skipTest(f"image_processor attribute not present in {self.processor_class}")
    #     image_processor = self.get_component("image_processor")
    #     tokenizer = self.get_component("tokenizer", max_length=117, padding="max_length")
    #     if not tokenizer.pad_token:
    #         tokenizer.pad_token = "[TEST_PAD]"

    #     processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
    #     self.skip_processor_without_typed_kwargs(processor)
    #     input_str = "lower newer"
    #     image_input = self.prepare_image_inputs()

    #     inputs = processor(text=input_str, images=image_input, return_tensors="pt")
    #     self.assertEqual(len(inputs["input_ids"][0]), 117)

    # @require_torch
    # @require_vision
    # def test_image_processor_defaults_preserved_by_image_kwargs(self):
    #     if "image_processor" not in self.processor_class.attributes:
    #         self.skipTest(f"image_processor attribute not present in {self.processor_class}")
    #     image_processor = self.get_component("image_processor", crop_size=(234, 234))
    #     tokenizer = self.get_component("tokenizer", max_length=117)

    #     processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
    #     self.skip_processor_without_typed_kwargs(processor)

    #     input_str = "lower newer"
    #     image_input = self.prepare_image_inputs()

    #     inputs = processor(text=input_str, images=image_input)
    #     self.assertEqual(len(inputs["pixel_values"][0][0][0]), 234)

    # @require_vision
    # @require_torch
    # def test_kwargs_overrides_default_tokenizer_kwargs(self):
    #     if "image_processor" not in self.processor_class.attributes:
    #         self.skipTest(f"image_processor attribute not present in {self.processor_class}")
    #     image_processor = self.get_component("image_processor")
    #     tokenizer = self.get_component("tokenizer", max_length=117)
    #     if not tokenizer.pad_token:
    #         tokenizer.pad_token = "[TEST_PAD]"

    #     processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
    #     self.skip_processor_without_typed_kwargs(processor)
    #     input_str = "lower newer"
    #     image_input = self.prepare_image_inputs()

    #     inputs = processor(
    #         text=input_str, images=image_input, return_tensors="pt", max_length=112, padding="max_length"
    #     )
    #     self.assertEqual(len(inputs["input_ids"][0]), 112)

    # @require_torch
    # @require_vision
    # def test_kwargs_overrides_default_image_processor_kwargs(self):
    #     if "image_processor" not in self.processor_class.attributes:
    #         self.skipTest(f"image_processor attribute not present in {self.processor_class}")
    #     image_processor = self.get_component("image_processor", crop_size=(234, 234))
    #     tokenizer = self.get_component("tokenizer", max_length=117)

    #     processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
    #     self.skip_processor_without_typed_kwargs(processor)

    #     input_str = "lower newer"
    #     image_input = self.prepare_image_inputs()

    #     inputs = processor(text=input_str, images=image_input, crop_size=[224, 224])
    #     self.assertEqual(len(inputs["pixel_values"][0][0][0]), 224)

    # @require_torch
    # @require_vision
    # def test_unstructured_kwargs(self):
    #     if "image_processor" not in self.processor_class.attributes:
    #         self.skipTest(f"image_processor attribute not present in {self.processor_class}")
    #     image_processor = self.get_component("image_processor")
    #     tokenizer = self.get_component("tokenizer")
    #     if not tokenizer.pad_token:
    #         tokenizer.pad_token = "[TEST_PAD]"
    #     processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
    #     self.skip_processor_without_typed_kwargs(processor)

    #     input_str = "lower newer"
    #     image_input = self.prepare_image_inputs()
    #     inputs = processor(
    #         text=input_str,
    #         images=image_input,
    #         return_tensors="pt",
    #         crop_size={"height": 214, "width": 214},
    #         padding="max_length",
    #         max_length=76,
    #     )

    #     self.assertEqual(inputs["pixel_values"].shape[3], 214)
    #     self.assertEqual(len(inputs["input_ids"][0]), 76)

    # @require_torch
    # @require_vision
    # def test_unstructured_kwargs_batched(self):
    #     if "image_processor" not in self.processor_class.attributes:
    #         self.skipTest(f"image_processor attribute not present in {self.processor_class}")
    #     image_processor = self.get_component("image_processor")
    #     tokenizer = self.get_component("tokenizer")
    #     if not tokenizer.pad_token:
    #         tokenizer.pad_token = "[TEST_PAD]"
    #     processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
    #     self.skip_processor_without_typed_kwargs(processor)

    #     input_str = ["lower newer", "upper older longer string"]
    #     image_input = self.prepare_image_inputs() * 2
    #     inputs = processor(
    #         text=input_str,
    #         images=image_input,
    #         return_tensors="pt",
    #         crop_size={"height": 214, "width": 214},
    #         padding="longest",
    #         max_length=76,
    #     )
    #     print("pixel_values shape", inputs["pixel_values"].shape)
    #     self.assertEqual(inputs["pixel_values"].shape[3], 214)

    #     self.assertEqual(len(inputs["input_ids"][0]), 5)

    # @require_torch
    # @require_vision
    # def test_structured_kwargs_nested(self):
    #     if "image_processor" not in self.processor_class.attributes:
    #         self.skipTest(f"image_processor attribute not present in {self.processor_class}")
    #     image_processor = self.get_component("image_processor")
    #     tokenizer = self.get_component("tokenizer")
    #     if not tokenizer.pad_token:
    #         tokenizer.pad_token = "[TEST_PAD]"
    #     processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
    #     self.skip_processor_without_typed_kwargs(processor)

    #     input_str = "lower newer"
    #     image_input = self.prepare_image_inputs()

    #     # Define the kwargs for each modality
    #     all_kwargs = {
    #         "common_kwargs": {"return_tensors": "pt"},
    #         "images_kwargs": {"crop_size": {"height": 214, "width": 214}},
    #         "text_kwargs": {"padding": "max_length", "max_length": 76},
    #     }

    #     inputs = processor(text=input_str, images=image_input, **all_kwargs)
    #     self.skip_processor_without_typed_kwargs(processor)

    #     self.assertEqual(inputs["pixel_values"].shape[3], 214)

    #     self.assertEqual(len(inputs["input_ids"][0]), 76)

    # @require_torch
    # @require_vision
    # def test_structured_kwargs_nested_from_dict(self):
    #     if "image_processor" not in self.processor_class.attributes:
    #         self.skipTest(f"image_processor attribute not present in {self.processor_class}")

    #     image_processor = self.get_component("image_processor")
    #     tokenizer = self.get_component("tokenizer")
    #     if not tokenizer.pad_token:
    #         tokenizer.pad_token = "[TEST_PAD]"

    #     processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
    #     self.skip_processor_without_typed_kwargs(processor)
    #     input_str = "lower newer"
    #     image_input = self.prepare_image_inputs()

    #     # Define the kwargs for each modality
    #     all_kwargs = {
    #         "common_kwargs": {"return_tensors": "pt"},
    #         "images_kwargs": {"crop_size": {"height": 214, "width": 214}},
    #         "text_kwargs": {"padding": "max_length", "max_length": 76},
    #     }

    #     inputs = processor(text=input_str, images=image_input, **all_kwargs)
    #     self.assertEqual(inputs["pixel_values"].shape[3], 214)

    #     self.assertEqual(len(inputs["input_ids"][0]), 76)
