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

from transformers import AutoProcessor, LlamaTokenizerFast, Phi3VProcessor
from transformers.testing_utils import require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import Phi3VImageProcessorFast

# fmt: off
CHAT_TEMPLATE = """{% for message in messages %}
<|{{ message['role'] }}|>
{% for entry in message['content'] %}
    {% if entry['type'] == 'image' %}
<|image|>
    {% elif entry['type'] == 'text' %}
{{ entry['text'] }}{%- endif %}{% endfor -%}<|end|>
{% endfor -%}
{% if add_generation_prompt and messages[-1]['role'] != 'assistant' -%}
<|assistant|>
{% endif -%}
"""
# fmt: on


@require_vision
class Phi3VProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Phi3VProcessor

    @classmethod
    def setUpClass(cls):
        cls.tmpdirname = tempfile.mkdtemp()

        image_processor = Phi3VImageProcessorFast(do_center_crop=False)
        tokenizer = LlamaTokenizerFast.from_pretrained("huggyllama/llama-7b")
        tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
        processor_kwargs = cls.prepare_processor_dict()
        processor = Phi3VProcessor(image_processor, tokenizer, **processor_kwargs)
        processor.save_pretrained(cls.tmpdirname)
        cls.image_token = processor.image_token

    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_image_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).image_processor

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    @staticmethod
    def prepare_processor_dict():
        return {"chat_template": CHAT_TEMPLATE}

    def test_chat_template_is_saved(self):
        processor_loaded = self.processor_class.from_pretrained(self.tmpdirname)
        processor_dict_loaded = json.loads(processor_loaded.to_json_string())
        # chat templates aren't serialized to json in processors
        self.assertFalse("chat_template" in processor_dict_loaded)

        # they have to be saved as separate file and loaded back from that file
        # so we check if the same template is loaded
        processor_dict = self.prepare_processor_dict()
        self.assertTrue(processor_loaded.chat_template == processor_dict.get("chat_template", None))

    @unittest.skip("Not possible as processor can't create an assistant mask.")
    def test_apply_chat_template_assistant_mask(self):
        pass

    def test_unstructured_kwargs_batched(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_kwargs = self.prepare_processor_dict()
        processor = self.processor_class(**processor_components, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        # The <|image|> token used in comman test is a special image token hence replace with dummy image token.
        input_str = ["lower newer |image|", " |image| upper older longer string"]
        image_input = self.prepare_image_inputs(batch_size=2)
        inputs = processor(
            text=input_str,
            images=image_input,
            return_tensors="pt",
            do_rescale=True,
            rescale_factor=-1.0,
            padding="longest",
            max_length=76,
        )

        self.assertLessEqual(inputs[self.images_input_name][0][0].mean(), 0)
        self.assertTrue(
            len(inputs[self.text_input_name][0]) == len(inputs[self.text_input_name][1])
            and len(inputs[self.text_input_name][1]) < 76
        )
