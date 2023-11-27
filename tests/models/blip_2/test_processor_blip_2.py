# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import pytest

from transformers import Blip2Processor, BlipImageProcessor, GPT2Tokenizer, GPT2TokenizerFast
from transformers.testing_utils import require_vision

from ...test_processing_common import ProcessorTesterMixin


@require_vision
class Blip2ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    tokenizer_class = GPT2Tokenizer
    fast_tokenizer_class = GPT2TokenizerFast
    image_processor_class = BlipImageProcessor
    processor_class = Blip2Processor

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()

        image_processor = BlipImageProcessor()
        tokenizer = GPT2Tokenizer.from_pretrained("hf-internal-testing/tiny-random-GPT2Model")

        processor = Blip2Processor(image_processor, tokenizer)

        processor.save_pretrained(self.tmpdirname)

    def test_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input)

        self.assertListEqual(list(inputs.keys()), ["pixel_values", "input_ids", "attention_mask"])

        # test if it raises when no input is passed
        with pytest.raises(ValueError):
            processor()

    def test_model_input_names(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = Blip2Processor(tokenizer=tokenizer, image_processor=image_processor)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input)

        # For now the processor supports only ['pixel_values', 'input_ids', 'attention_mask']
        self.assertListEqual(list(inputs.keys()), ["pixel_values", "input_ids", "attention_mask"])
