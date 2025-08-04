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

import shutil
import tempfile
import unittest
from typing import Optional

from transformers import Gemma3Processor, GemmaTokenizer
from transformers.testing_utils import get_tests_dir, require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import Gemma3ImageProcessor

SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")


@require_vision
class Gemma3ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Gemma3Processor

    @classmethod
    def setUpClass(cls):
        cls.tmpdirname = tempfile.mkdtemp()
        gemma3_image_processor_kwargs = {
            "do_pan_and_scan": True,
            "pan_and_scan_min_crop_size": 256,
            "pan_and_scan_max_num_crops": 4,
            "pan_and_scan_min_ratio_to_activate": 1.2,
        }
        image_processor = Gemma3ImageProcessor.from_pretrained(
            "google/siglip-so400m-patch14-384", **gemma3_image_processor_kwargs
        )

        extra_special_tokens = {
            "image_token": "<image_soft_token>",
            "boi_token": "<start_of_image>",
            "eoi_token": "<end_of_image>",
        }
        tokenizer = GemmaTokenizer(SAMPLE_VOCAB, keep_accents=True, extra_special_tokens=extra_special_tokens)
        processor_kwargs = cls.prepare_processor_dict()
        processor = Gemma3Processor(image_processor=image_processor, tokenizer=tokenizer, **processor_kwargs)
        processor.save_pretrained(cls.tmpdirname)
        cls.image_token = processor.boi_token

    # Copied from tests.models.llava.test_processing_llava.LlavaProcessorTest.test_get_num_vision_tokens
    def test_get_num_vision_tokens(self):
        "Tests general functionality of the helper used internally in vLLM"

        processor = self.get_processor()

        output = processor._get_num_multimodal_tokens(image_sizes=[(100, 100), (300, 100), (500, 30)])
        self.assertTrue("num_image_tokens" in output)
        self.assertEqual(len(output["num_image_tokens"]), 3)

        self.assertTrue("num_image_patches" in output)
        self.assertEqual(len(output["num_image_patches"]), 3)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    # TODO: raushan or arthur: add the real chat template
    @staticmethod
    def prepare_processor_dict():
        return {
            "chat_template": "{{ bos_token }}\n{%- if messages[0]['role'] == 'system' -%}\n    {%- set first_user_prefix = messages[0]['content'][0]['text'] + '\n\n' -%}\n    {%- set loop_messages = messages[1:] -%}\n{%- else -%}\n    {%- set first_user_prefix = \"\" -%}\n    {%- set loop_messages = messages -%}\n{%- endif -%}\n{%- for message in loop_messages -%}\n    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}\n        {{ raise_exception(\"Conversation roles must alternate user/assistant/user/assistant/...\") }}\n    {%- endif -%}\n    {%- if (message['role'] == 'assistant') -%}\n        {%- set role = \"model\" -%}\n    {%- else -%}\n        {%- set role = message['role'] -%}\n    {%- endif -%}\n    {{ '<start_of_turn>' + role + '\n' + (first_user_prefix if loop.first else \"\") }}\n    {%- if message['content'] is string -%}\n        {{ message['content'] | trim }}\n    {%- elif message['content'] is iterable -%}\n        {%- for item in message['content'] -%}\n            {%- if item['type'] == 'image' -%}\n                {{ '<start_of_image>' }}\n            {%- elif item['type'] == 'text' -%}\n                {{ item['text'] | trim }}\n            {%- endif -%}\n        {%- endfor -%}\n    {%- else -%}\n        {{ raise_exception(\"Invalid content type\") }}\n    {%- endif -%}\n    {{ '<end_of_turn>\n' }}\n{%- endfor -%}\n{%- if add_generation_prompt -%}\n    {{'<start_of_turn>model\n'}}\n{%- endif -%}\n",            "image_seq_length": 3,
        }  # fmt: skip

    # Override as Gemma3 needs images to be an explicitly nested batch
    def prepare_image_inputs(self, batch_size: Optional[int] = None):
        """This function prepares a list of PIL images for testing"""
        images = super().prepare_image_inputs(batch_size)
        if isinstance(images, (list, tuple)):
            images = [[image] for image in images]
        return images

    def test_text_with_image_tokens(self):
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer")

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        text_multi_images = f"{processor.boi_token}{processor.boi_token}Dummy text!"
        text_single_image = f"{processor.boi_token}Dummy text!"
        text_no_image = "Dummy text!"

        image = self.prepare_image_inputs()

        # If text has no image tokens, image should be `None`
        with self.assertRaises(ValueError):
            _ = processor(text=text_no_image, images=image, return_tensors="np")

        # We can't be sure what is users intention: if user wants one image per text OR two images for first text and no image for second text
        with self.assertRaises(ValueError):
            _ = processor(text=[text_single_image, text_single_image], images=[image, image], return_tensors="np")

        # The users is expected to be explicit about which image belong to which text by nesting the images list
        out_multiimages = processor(text=text_multi_images, images=[image, image], return_tensors="np")
        out_batch_oneimage = processor(
            text=[text_single_image, text_single_image], images=[[image], [image]], return_tensors="np"
        )
        self.assertListEqual(
            out_batch_oneimage[self.images_input_name].tolist(), out_multiimages[self.images_input_name].tolist()
        )

    def test_pan_and_scan(self):
        processor_components = self.prepare_components()
        processor_kwargs = self.prepare_processor_dict()
        processor = self.processor_class(**processor_components, **processor_kwargs)

        input_str = self.prepare_text_inputs(modality="image")
        image_input = self.prepare_image_inputs()
        inputs = processor(
            text=input_str,
            images=image_input,
            return_tensors="np",
            do_pan_and_scan=True,
            image_seq_length=2,
            pan_and_scan_min_crop_size=10,
        )

        # base image + 4 crops
        self.assertEqual(len(inputs[self.images_input_name]), 5)
        self.assertEqual(len(inputs[self.text_input_name][0]), 67)

    def test_special_mm_token_truncation(self):
        """Tests that special vision tokens do not get truncated when `truncation=True` is set."""

        processor = self.get_processor()

        input_str = self.prepare_text_inputs(batch_size=2, modality="image")
        image_input = self.prepare_image_inputs(batch_size=2)
        _ = processor(
            text=input_str,
            images=image_input,
            return_tensors="pt",
            truncation=None,
            padding=True,
        )

        with self.assertRaises(ValueError):
            _ = processor(
                text=input_str,
                images=image_input,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=5,
            )
