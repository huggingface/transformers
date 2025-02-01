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
import shutil
import tempfile
import unittest

import pytest

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import (
        AutoProcessor,
        InternVL2_5ImageProcessor,
        InternVL2_5Processor,
        Qwen2Tokenizer,
        Qwen2TokenizerFast,
    )


@require_vision
@require_torch
class InternVL2_5ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = InternVL2_5Processor

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
        image_processor = InternVL2_5ImageProcessor.from_pretrained("thisisiron/InternVL2_5-1B")
        tokenizer = Qwen2TokenizerFast.from_pretrained("thisisiron/InternVL2_5-1B")

        processor = InternVL2_5Processor(
            image_processor=image_processor,
            tokenizer=tokenizer,
        )
        processor.save_pretrained(self.tmpdirname)

    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_image_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).image_processor

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def test_chat_template(self):
        processor = InternVL2_5Processor.from_pretrained("thisisiron/InternVL2_5-1B")
        expected_prompt = "<|im_start|>user\n<image>\nWhat is shown in this image?<|im_end|>\n<|im_start|>assistant\n"

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

    def test_chat_template_type2(self):
        processor = InternVL2_5Processor.from_pretrained("thisisiron/InternVL2_5-1B")
        expected_prompt = "<|im_start|>user\n<image>\nWhat is shown in this image?<|im_end|>\n<|im_start|>assistant\n"

        messages = [
            {"role": "user", "content": "<image>\nWhat is shown in this image?"},
        ]

        formatted_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        self.assertEqual(expected_prompt, formatted_prompt)

    def test_save_load_pretrained_default(self):
        tokenizer = self.get_tokenizer()
        image_processor = self.get_image_processor()

        processor = InternVL2_5Processor(tokenizer=tokenizer, image_processor=image_processor)
        processor.save_pretrained(self.tmpdirname)
        processor = InternVL2_5Processor.from_pretrained(self.tmpdirname, use_fast=False)

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer.get_vocab())
        self.assertEqual(processor.image_processor.to_json_string(), image_processor.to_json_string())
        self.assertIsInstance(processor.tokenizer, Qwen2Tokenizer)
        self.assertIsInstance(processor.image_processor, InternVL2_5ImageProcessor)

    def test_image_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = InternVL2_5Processor(tokenizer=tokenizer, image_processor=image_processor)

        image_input = self.prepare_image_inputs()

        processed_image_input = image_processor(image_input, return_tensors="np")
        processed_input = processor(images=image_input, text="<image>\ndummy", return_tensors="np")

        for key in processed_image_input.keys():
            self.assertAlmostEqual(processed_image_input[key].sum(), processed_input[key].sum(), delta=1e-2)

        with pytest.raises(ValueError):
            processor(images=image_input, text=None)  # test if it raises when no text is passed

    def test_multi_image_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = InternVL2_5Processor(tokenizer=tokenizer, image_processor=image_processor)

        image_inputs = self.prepare_image_inputs(3)

        # test if it raises when number of images does not match placeholders
        with pytest.raises(ValueError):
            processor(images=image_inputs, text="<image>\n<image>\n", return_tensors="np")

        processed_image_input = image_processor(image_inputs, return_tensors="np")
        processed_input = processor(images=image_inputs, text="<image>\n<image>\n<image>\n", return_tensors="np")

        for key in processed_image_input.keys():
            self.assertAlmostEqual(processed_image_input[key].sum(), processed_input[key].sum(), delta=1e-2)
