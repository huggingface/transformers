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

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    PerceptionLMProcessor,
)
from transformers.testing_utils import require_read_token, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import PerceptionLMImageProcessorFast, PerceptionLMVideoProcessor

if is_torch_available():
    import torch


TEST_MODEL_PATH = "facebook/Perception-LM-1B"


@require_vision
@require_read_token
@unittest.skip("Fequires read token and we didn't requests access yet. FIXME @ydshieh when you are back :)")
class PerceptionLMProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = PerceptionLMProcessor

    @classmethod
    def setUpClass(cls):
        cls.tmpdirname = tempfile.mkdtemp()

        image_processor = PerceptionLMImageProcessorFast(
            tile_size=448, max_num_tiles=4, vision_input_type="thumb+tile"
        )
        video_processor = PerceptionLMVideoProcessor()
        tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_PATH)
        tokenizer.add_special_tokens({"additional_special_tokens": ["<|image|>", "<|video|>"]})
        processor_kwargs = cls.prepare_processor_dict()
        processor = PerceptionLMProcessor(
            image_processor=image_processor, video_processor=video_processor, tokenizer=tokenizer, **processor_kwargs
        )
        processor.save_pretrained(cls.tmpdirname)
        cls.image_token_id = processor.image_token_id
        cls.video_token_id = processor.video_token_id

    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_image_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).image_processor

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    @staticmethod
    def prepare_processor_dict():
        return {
            "chat_template": CHAT_TEMPLATE,
            "patch_size": 14,
            "pooling_ratio": 2,
        }  # fmt: skip

    def test_chat_template_is_saved(self):
        processor_loaded = self.processor_class.from_pretrained(self.tmpdirname)
        processor_dict_loaded = json.loads(processor_loaded.to_json_string())
        # chat templates aren't serialized to json in processors
        self.assertFalse("chat_template" in processor_dict_loaded)

        # they have to be saved as separate file and loaded back from that file
        # so we check if the same template is loaded
        processor_dict = self.prepare_processor_dict()
        self.assertTrue(processor_loaded.chat_template == processor_dict.get("chat_template", None))

    def test_image_token_filling(self):
        processor = self.processor_class.from_pretrained(self.tmpdirname)
        # Important to check with non square image
        image = torch.randn((1, 3, 450, 500))
        #  5 tiles (thumbnail tile + 4 tiles)
        #  448/patch_size/pooling_ratio = 16 => 16*16 tokens per tile
        expected_image_tokens = 16 * 16 * 5
        image_token_index = processor.image_token_id

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


CHAT_TEMPLATE = (
    "{{- bos_token }}"
    "{%- if messages[0]['role'] == 'system' -%}"
    "    {%- set system_message = messages[0]['content']|trim %}\n"
    "    {%- set messages = messages[1:] %}\n"
    "{%- else %}"
    "    {%- set system_message = 'You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.' %}"
    "{%- endif %}"
    "{{- '<|start_header_id|>system<|end_header_id|>\\n\\n' }}"
    "{{- system_message }}"
    "{{- '<|eot_id|>' }}"
    "{%- for message in messages %}"
    "{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' }}"
    "{%- for content in message['content'] | selectattr('type', 'equalto', 'image') %}"
    "{{ '<|image|>' }}"
    "{%- endfor %}"
    "{%- for content in message['content'] | selectattr('type', 'equalto', 'video') %}"
    "{{ '<|video|>' }}"
    "{%- endfor %}"
    "{%- for content in message['content'] | selectattr('type', 'equalto', 'text') %}"
    "{{- content['text'] | trim }}"
    "{%- endfor %}"
    "{{'<|eot_id|>' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
    "{%- endif %}"
)
