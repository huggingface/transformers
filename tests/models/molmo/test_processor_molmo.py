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
import json
import shutil
import tempfile
import unittest

from transformers import AutoProcessor, MolmoProcessor
from transformers.testing_utils import require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    pass


CHAT_TEMPLATE = (
    "{{ bos_token or '' }}"
    "{% for message in messages %}"
    "{%- if (loop.index % 2 == 1 and message['role'] != 'user') or (loop.index % 2 == 0 and message['role'].lower() != 'assistant') -%}"
    "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
    "{%- endif -%}"
    "{% if message['content'] is not string %}"
    "{% for content in message['content'] %}"
    "{% if content['type'] == 'image' %}"
    "{{ '<image> ' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% endif %}"
    "{{ message['role'].capitalize() + ': ' }}"
    "{% if message['content'] is string %}"
    "{{ message['content'] + ' ' }}"
    "{% else %}"
    "{% for content in message['content'] %}"
    "{% if content['type'] == 'text' %}"
    "{{ content['text'] + ' ' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ 'Assistant:' }}"
    "{% endif %}"
)


@require_vision
class MolmoProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = MolmoProcessor

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
        self.processor = MolmoProcessor.from_pretrained("Molbap/molmo-hf-7B-D")
        self.processor.save_pretrained(self.tmpdirname)

    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_image_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).image_processor

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)
        del self.processor

    @staticmethod
    def prepare_processor_dict():
        return {
            "chat_template": CHAT_TEMPLATE,
        }

    @unittest.skip(
        "Skip because the model has no processor kwargs except for chat template and"
        "chat template is saved as a separate file. Stop skipping this test when the processor"
        "has new kwargs saved in config file."
    )
    def test_processor_to_json_string(self):
        pass

    def test_chat_template_is_saved(self):
        processor_loaded = self.processor_class.from_pretrained(self.tmpdirname)
        processor_dict_loaded = json.loads(processor_loaded.to_json_string())
        # chat templates aren't serialized to json in processors
        self.assertFalse("chat_template" in processor_dict_loaded.keys())

        # they have to be saved as separate file and loaded back from that file
        # so we check if the same template is loaded
        processor_dict = self.prepare_processor_dict()
        self.assertTrue(processor_loaded.chat_template == processor_dict.get("chat_template", None))

    def test_nested_input(self):
        processor_components = self.prepare_components()
        processor_components["image_processor"] = self.get_component("image_processor")
        processor_components["tokenizer"] = self.get_component("tokenizer")

        processor = self.processor_class(**processor_components)

        input_str = self.prepare_text_inputs()
        image_input = self.prepare_image_inputs()

        # Test batched as a nested list of images, where each sublist is one batch
        image_inputs_nested = [[image_input] * 3, [image_input] * 3]
        text = [input_str] * 6
        inputs_nested = processor(text=text, images=image_inputs_nested, return_tensors="np")

        # Test batched as a flat list of images
        image_inputs_flat = [image_input] * 6
        inputs_flat = processor(text=text, images=image_inputs_flat, return_tensors="np")

        # Image processor should return same pixel values, independently of input format
        self.assertTrue((inputs_nested.pixel_values == inputs_flat.pixel_values).all())

    def test_chat_template(self):
        expected_prompt = "<|endoftext|><image> User: What is shown in this image? Assistant:"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]

        formatted_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        self.assertEqual(expected_prompt, formatted_prompt)
