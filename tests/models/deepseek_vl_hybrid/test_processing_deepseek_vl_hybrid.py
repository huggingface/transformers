# coding=utf-8
# Copyright 2025 HuggingFace Inc.
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

from transformers import DeepseekVLHybridProcessor, LlamaTokenizer
from transformers.testing_utils import get_tests_dir
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import DeepseekVLHybridImageProcessor


SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")


class DeepseekVLHybridProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = DeepseekVLHybridProcessor

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
        image_processor = DeepseekVLHybridImageProcessor()
        tokenizer = LlamaTokenizer(
            vocab_file=SAMPLE_VOCAB,
            extra_special_tokens={
                "pad_token": "<｜end▁of▁sentence｜>",
                "image_token": "<image_placeholder>",
            },
        )
        processor_kwargs = self.prepare_processor_dict()
        processor = self.processor_class(
            image_processor=image_processor,
            tokenizer=tokenizer,
            **processor_kwargs,
        )
        processor.save_pretrained(self.tmpdirname)

    @staticmethod
    def prepare_processor_dict():
        return {
            "chat_template": "{% set seps = ['\n\n', '<\uff5cend\u2581of\u2581sentence\uff5c>'] %}{% set i = 0 %}You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.\n\n{% for message in messages %}{% if message['role']|lower == 'user' %}User: {% elif message['role']|lower == 'assistant' %}Assistant:{% if not (loop.last and not add_generation_prompt and message['content'][0]['type']=='text' and message['content'][0]['text']=='') %} {% endif %}{% else %}{{ message['role'].capitalize() }}: {% endif %}{% for content in message['content'] %}{% if content['type'] == 'image' %}<image_placeholder>{% elif content['type'] == 'text' %}{% set text = content['text'] %}{% if loop.first %}{% set text = text.lstrip() %}{% endif %}{% if loop.last %}{% set text = text.rstrip() %}{% endif %}{% if not loop.first and message['content'][loop.index0-1]['type'] == 'text' %}{{ ' ' + text }}{% else %}{{ text }}{% endif %}{% endif %}{% endfor %}{% if not loop.last or add_generation_prompt %}{% if message['role']|lower == 'user' %}{{ seps[0] }}{% else %}{{ seps[1] }}{% endif %}{% endif %}{% endfor %}{% if add_generation_prompt %}Assistant:{% endif %}",
            "num_image_tokens": 576,
        }  # fmt: skip
