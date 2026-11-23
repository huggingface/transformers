# Copyright 2026 NAVER Corp. and The HuggingFace Team. All rights reserved.
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

from transformers import HyperCLOVAXVisionProcessor
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available

from ...test_processing_common import ProcessorTesterMixin


if is_torch_available():
    import torch


@require_vision
@require_torch
class HyperCLOVAXVisionProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = HyperCLOVAXVisionProcessor
    # Every image takes at least 81 tokens for its global view
    images_unstructured_max_length = 120

    @classmethod
    def _setup_tokenizer(cls):
        tokenizer_class = cls._get_component_class_from_processor("tokenizer")
        tokenizer = tokenizer_class.from_pretrained("hf-internal-testing/tiny-random-GPT2Model")
        tokenizer.add_special_tokens({"additional_special_tokens": ["<|dummy3|>"]})
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @staticmethod
    def prepare_processor_dict():
        return {
            "chat_template": "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' }}{% for content in message['content'] %}{% if content['type'] == 'image' %}{{ '<|dummy3|>' }}{% elif content['type'] == 'text' %}{{ content['text'] }}{% endif %}{% endfor %}{{ '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",
            "projector_num_queries": 81,
        }  # fmt: skip

    def test_get_num_vision_tokens(self):
        "Tests general functionality of the helper used internally in vLLM"
        processor = self.get_processor()

        output = processor._get_num_multimodal_tokens(image_sizes=[(100, 100), (300, 100), (500, 30)])
        self.assertTrue("num_image_tokens" in output)
        self.assertEqual(len(output["num_image_tokens"]), 3)

        self.assertTrue("num_image_patches" in output)
        self.assertEqual(len(output["num_image_patches"]), 3)

    def test_image_token_filling(self):
        processor = self.get_processor()
        # A 480x640 image selects the 756x756 layout (2x2 grids of 9x9 tokens), which is unpadded to 14x18 tokens,
        # plus one newline per row and 81 tokens for the global view: 14 * 18 + 14 + 81 = 347
        image = torch.randint(0, 256, (3, 480, 640), dtype=torch.uint8)
        expected_image_tokens = 347

        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": "What is shown in this image?"}],
            }
        ]
        inputs = processor(text=[processor.apply_chat_template(messages)], images=[image], return_tensors="pt")
        image_tokens = (inputs["input_ids"] == processor.image_token_id).sum().item()
        self.assertEqual(image_tokens, expected_image_tokens)
        self.assertEqual(
            processor._get_num_multimodal_tokens(image_sizes=[(480, 640)])["num_image_tokens"][0],
            expected_image_tokens,
        )
