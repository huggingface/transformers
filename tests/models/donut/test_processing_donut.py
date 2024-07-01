# coding=utf-8
# Copyright 2022 HuggingFace Inc.
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

from transformers import DonutImageProcessor, DonutProcessor, XLMRobertaTokenizerFast
from transformers.testing_utils import (
    require_torch,
    require_vision,
)

from ...test_processing_common import ProcessorTesterMixin


class DonutProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    from_pretrained_id = "naver-clova-ix/donut-base"
    processor_class = DonutProcessor

    def setUp(self):
        self.processor = DonutProcessor.from_pretrained(self.from_pretrained_id)
        self.tmpdirname = tempfile.mkdtemp()

        image_processor = DonutImageProcessor()
        tokenizer = XLMRobertaTokenizerFast.from_pretrained(self.from_pretrained_id)

        processor = DonutProcessor(image_processor, tokenizer)

        processor.save_pretrained(self.tmpdirname)

    def test_token2json(self):
        expected_json = {
            "name": "John Doe",
            "age": "99",
            "city": "Atlanta",
            "state": "GA",
            "zip": "30301",
            "phone": "123-4567",
            "nicknames": [{"nickname": "Johnny"}, {"nickname": "JD"}],
            "multiline": "text\nwith\nnewlines",
            "empty": "",
        }

        sequence = (
            "<s_name>John Doe</s_name><s_age>99</s_age><s_city>Atlanta</s_city>"
            "<s_state>GA</s_state><s_zip>30301</s_zip><s_phone>123-4567</s_phone>"
            "<s_nicknames><s_nickname>Johnny</s_nickname>"
            "<sep/><s_nickname>JD</s_nickname></s_nicknames>"
            "<s_multiline>text\nwith\nnewlines</s_multiline>"
            "<s_empty></s_empty>"
        )
        actual_json = self.processor.token2json(sequence)

        self.assertDictEqual(actual_json, expected_json)

    @require_torch
    @require_vision
    def test_unstructured_kwargs_batched(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer")
        if not tokenizer.pad_token:
            tokenizer.pad_token = "[TEST_PAD]"
        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = ["lower newer", "upper older longer string"]
        image_input = self.prepare_image_inputs() * 2
        inputs = processor(
            text=input_str,
            images=image_input,
            return_tensors="pt",
            crop_size={"height": 214, "width": 214},
            size={"height": 214, "width": 214},
            padding="longest",
            max_length=76,
        )
        self.assertEqual(inputs["pixel_values"].shape[2], 214)

        self.assertEqual(len(inputs["input_ids"][0]), 7)
