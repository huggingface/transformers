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

import numpy as np

from transformers.testing_utils import require_torch, require_torchvision, require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import MiniCPMV2_5ImageProcessor, MiniCPMV2_6Processor, PreTrainedTokenizerFast


@require_vision
@require_torch
@require_torchvision
class MiniCPMV2_6ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = MiniCPMV2_6Processor

    @staticmethod
    def prepare_processor_dict():
        # Renders images like the converted checkpoints' chat templates: `<image_id>N</image_id><unk>`
        chat_template = (
            "{%- set ns = namespace(image_count=0) -%}"
            "{%- for message in messages -%}"
            "{%- if message['content'] is string -%}{%- set content = message['content'] -%}"
            "{%- else -%}{%- set parts = namespace(items=[]) -%}"
            "{%- for item in message['content'] -%}"
            "{%- if item['type'] == 'image' -%}"
            "{%- set parts.items = parts.items + ['<image_id>' ~ ns.image_count ~ '</image_id><unk>'] -%}"
            "{%- set ns.image_count = ns.image_count + 1 -%}"
            "{%- elif item['type'] == 'text' -%}{%- set parts.items = parts.items + [item['text']] -%}{%- endif -%}"
            "{%- endfor -%}{%- set content = parts.items | join('\n') -%}{%- endif -%}"
            "{{- '<|im_start|>' + message['role'] + '\n' + content + '<|im_end|>\n' -}}"
            "{%- endfor -%}"
            "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\n' -}}{%- endif -%}"
        )
        return {"chat_template": chat_template, "num_image_tokens": 4}

    @classmethod
    def _setup_tokenizer(cls):
        return PreTrainedTokenizerFast.from_pretrained("openbmb/MiniCPM-V-4", pad_token="</s>")

    @classmethod
    def _setup_image_processor(cls):
        # One overview image and no slices keeps the tests fast.
        return MiniCPMV2_5ImageProcessor(scale_resolution=64, max_slice_nums=1)

    def test_sliced_image_placeholder(self):
        processor = self.get_processor()
        processor.image_processor.max_slice_nums = 4
        # 200 x 200 px is ~10x scale_resolution**2 and square -> 2 rows x 2 columns of slices
        image = np.random.randint(255, size=(3, 200, 200), dtype=np.uint8)
        inputs = processor(text=processor.image_token, images=[image], return_tensors="pt")

        self.assertEqual(inputs["pixel_values"].shape[0], 5)  # overview + 4 slices
        image_tokens = processor.image_token * 4
        one_slice = f"<slice>{image_tokens}</slice>"
        expected = processor.tokenizer(f"<image>{image_tokens}</image>{one_slice * 2}\n{one_slice * 2}")
        self.assertListEqual(inputs["input_ids"][0].tolist(), expected["input_ids"])

        num_tokens = processor._get_num_multimodal_tokens(image_sizes=[(200, 200)])
        self.assertListEqual(num_tokens["num_image_patches"], [5])
        self.assertListEqual(num_tokens["num_image_tokens"], [20])
