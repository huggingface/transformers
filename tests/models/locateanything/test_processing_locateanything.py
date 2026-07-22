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

from parameterized import parameterized

from transformers import AutoTokenizer, LocateAnythingProcessor
from transformers.testing_utils import require_torch, require_vision

from ...test_processing_common import ProcessorTesterMixin


CHAT_TEMPLATE = (
    "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}"
    "{% for message in messages %}{% if loop.first and message['role'] != 'system' %}"
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}"
    "<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n"
    "{% else %}{% for content in message['content'] %}"
    "{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}"
    "{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}<image {{ image_count.value }}>{% endif %}"
    "<image-{{ image_count.value }}>{% elif content['type'] == 'video' or 'video' in content %}"
    "{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}<video {{ video_count.value }}>{% endif %}"
    "<video-{{ video_count.value }}>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}"
    "<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

MODEL_REVISION = "c32291ca5e996f5a7a485845b4f57a233936bba0"


@require_torch
@require_vision
class LocateAnythingProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = LocateAnythingProcessor

    @staticmethod
    def prepare_processor_dict():
        return {"chat_template": CHAT_TEMPLATE}

    @classmethod
    def _setup_image_processor(cls):
        image_processor_class = cls._get_component_class_from_processor("image_processor")
        return image_processor_class(patch_size=14)

    @classmethod
    def _setup_tokenizer(cls):
        return AutoTokenizer.from_pretrained("nvidia/LocateAnything-3B", revision=MODEL_REVISION)

    @classmethod
    def _setup_test_attributes(cls, processor):
        cls.image_token = processor.image_token

    # Packed patches have no per-batch leading dim, so check the template -> text -> processor path instead.
    @parameterized.expand([(1, "pt"), (2, "pt")])
    def test_apply_chat_template_image(self, batch_size: int, return_tensors: str):
        processor = self.get_processor()
        image = self.prepare_image_inputs()
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Locate the cat."}]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        self.assertIn("<image-1>", prompt)
        inputs = processor(images=image, text=[prompt], return_tensors="pt")
        self.assertEqual(inputs["image_grid_thw"].shape[0], 1)
        self.assertEqual(inputs["pixel_values"].shape[0], int(inputs["image_grid_thw"].prod(dim=-1).sum()))
        self.assertGreater(int((inputs["input_ids"][0] == processor.image_token_id).sum()), 0)

    def test_image_placeholder_expansion(self):
        processor = self.get_processor()
        image = self.prepare_image_inputs()
        inputs = processor(images=image, text=["<image-1>Locate the person."], return_tensors="pt")
        num_patches = inputs["image_grid_thw"][0].prod().item()
        merge_area = processor.image_processor.merge_kernel_size[0] * processor.image_processor.merge_kernel_size[1]
        expected_tokens = num_patches // merge_area
        n_image_tokens = int((inputs["input_ids"][0] == processor.image_token_id).sum())
        self.assertEqual(n_image_tokens, expected_tokens)
        decoded = processor.decode(inputs["input_ids"][0])
        self.assertIn("<image 1>", decoded)
        self.assertIn("<img>", decoded)
        self.assertIn("</img>", decoded)
