# coding=utf-8
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


import json
import tempfile
import unittest

from transformers import CLIPTokenizerFast, ProcessorMixin
from transformers.models.auto.processing_auto import processor_class_from_name
from transformers.testing_utils import (
    check_json_file_has_correct_format,
    require_tokenizers,
    require_torch,
    require_vision,
)
from transformers.utils import is_vision_available


if is_vision_available():
    from transformers import CLIPImageProcessor


@require_torch
class ProcessorTesterMixin:
    processor_class = None

    def prepare_processor_dict(self):
        return {}

    def get_component(self, attribute, **kwargs):
        assert attribute in self.processor_class.attributes
        component_class_name = getattr(self.processor_class, f"{attribute}_class")
        if isinstance(component_class_name, tuple):
            component_class_name = component_class_name[0]

        component_class = processor_class_from_name(component_class_name)
        component = component_class.from_pretrained(self.tmpdirname, **kwargs)  # noqa

        return component

    def prepare_components(self):
        components = {}
        for attribute in self.processor_class.attributes:
            component = self.get_component(attribute)
            components[attribute] = component

        return components

    def get_processor(self):
        components = self.prepare_components()
        processor = self.processor_class(**components, **self.prepare_processor_dict())
        return processor

    def test_processor_to_json_string(self):
        processor = self.get_processor()
        obj = json.loads(processor.to_json_string())
        for key, value in self.prepare_processor_dict().items():
            self.assertEqual(obj[key], value)
            self.assertEqual(getattr(processor, key, None), value)

    def test_processor_from_and_save_pretrained(self):
        processor_first = self.get_processor()

        with tempfile.TemporaryDirectory() as tmpdirname:
            saved_files = processor_first.save_pretrained(tmpdirname)
            if len(saved_files) > 0:
                check_json_file_has_correct_format(saved_files[0])
                processor_second = self.processor_class.from_pretrained(tmpdirname)

                self.assertEqual(processor_second.to_dict(), processor_first.to_dict())


class MyProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "CLIPImageProcessor"
    tokenizer_class = ("CLIPTokenizer", "CLIPTokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, processor_attr_1=1, processor_attr_2=True):
        super().__init__(image_processor, tokenizer)

        self.processor_attr_1 = processor_attr_1
        self.processor_attr_2 = processor_attr_2


@require_tokenizers
@require_vision
class ProcessorTest(unittest.TestCase):
    processor_class = MyProcessor

    def prepare_processor_dict(self):
        return {"processor_attr_1": 1, "processor_attr_2": False}

    def get_processor(self):
        image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-large-patch14")
        processor = MyProcessor(image_processor, tokenizer, **self.prepare_processor_dict())

        return processor

    def test_processor_to_json_string(self):
        processor = self.get_processor()
        obj = json.loads(processor.to_json_string())
        for key, value in self.prepare_processor_dict().items():
            self.assertEqual(obj[key], value)
            self.assertEqual(getattr(processor, key, None), value)

    def test_processor_from_and_save_pretrained(self):
        processor_first = self.get_processor()

        with tempfile.TemporaryDirectory() as tmpdirname:
            saved_file = processor_first.save_pretrained(tmpdirname)[0]
            check_json_file_has_correct_format(saved_file)
            processor_second = self.processor_class.from_pretrained(tmpdirname)

        self.assertEqual(processor_second.to_dict(), processor_first.to_dict())
