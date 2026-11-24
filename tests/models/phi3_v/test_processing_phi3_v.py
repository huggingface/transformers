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

import unittest

from transformers import AutoTokenizer
from transformers.testing_utils import require_torch, require_torchvision, require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import Phi3VProcessor


@require_vision
@require_torch
@require_torchvision
class Phi3VProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Phi3VProcessor
    # One crop of 336x336 and the global image: 12 * 13 + 1 + 12 * 13 = 313 image tokens
    images_text_kwargs_max_length = 400
    images_text_kwargs_override_max_length = 396
    images_unstructured_max_length = 407

    @classmethod
    def _setup_tokenizer(cls):
        return AutoTokenizer.from_pretrained("microsoft/Phi-3.5-vision-instruct")

    @classmethod
    def _setup_image_processor(cls):
        return cls._get_component_class_from_processor("image_processor")(num_crops=1)

    @classmethod
    def prepare_processor_dict(cls):
        return {
            "chat_template": "{% for message in messages %}{{ '<|' + message['role'] + '|>\\n' }}{% if message['content'] is string %}{{ message['content'] }}{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' %}{{ '<|image|>\\n' }}{% elif content['type'] == 'text' %}{{ content['text'] }}{% endif %}{% endfor %}{% endif %}{{ '<|end|>\\n' }}{% endfor %}{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}{{ '<|assistant|>\\n' }}{% endif %}"
        }  # fmt: skip

    def test_image_tokens_end_with_bos(self):
        # The original processor tokenized the text after each image separately, so it started with BOS
        processor = self.get_processor()
        inputs = processor(text=f"<|user|>\n{processor.image_token}\nHi", images=self.prepare_images_inputs())
        input_ids = inputs["input_ids"][0]
        num_image_tokens = input_ids.count(processor.image_token_id)
        self.assertEqual(num_image_tokens, 313)
        start = input_ids.index(processor.image_token_id)
        self.assertEqual(input_ids[start + num_image_tokens], processor.tokenizer.bos_token_id)
