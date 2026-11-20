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
from transformers.utils import is_torch_available, is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import MiniCPMV2_5ImageProcessor, MiniCPMV2_5Processor, PreTrainedTokenizerFast

if is_torch_available():
    import torch


@require_vision
@require_torch
@require_torchvision
class MiniCPMV2_5ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = MiniCPMV2_5Processor
    # Every image becomes at least <image> + 96 x <unk> + </image>, so text limits must leave room for it.
    images_unstructured_max_length = 150

    @staticmethod
    def prepare_processor_dict():
        # The converted checkpoint's chat template: each image item becomes one `<unk>` marker.
        chat_template = (
            "{%- for message in messages -%}"
            "{%- if message['content'] is string -%}{%- set content = message['content'] -%}"
            "{%- else -%}{%- set ns = namespace(parts=[]) -%}"
            "{%- for item in message['content'] -%}"
            "{%- if item['type'] == 'image' -%}{%- set ns.parts = ns.parts + ['<unk>'] -%}"
            "{%- elif item['type'] == 'text' -%}{%- set ns.parts = ns.parts + [item['text']] -%}{%- endif -%}"
            "{%- endfor -%}{%- set content = ns.parts | join('\n') -%}{%- endif -%}"
            "{%- set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + content | trim + '<|eot_id|>' -%}"
            "{%- if loop.index0 == 0 -%}{%- set content = bos_token + content -%}{%- endif -%}"
            "{{- content -}}"
            "{%- endfor -%}"
            "{%- if add_generation_prompt -%}{{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}{%- endif -%}"
        )
        return {"chat_template": chat_template}

    @classmethod
    def _setup_tokenizer(cls):
        return PreTrainedTokenizerFast.from_pretrained("openbmb/MiniCPM-Llama3-V-2_5")

    @classmethod
    def _setup_image_processor(cls):
        # One overview image and no slices keeps the tests fast.
        return MiniCPMV2_5ImageProcessor(scale_resolution=64, max_slice_nums=1)

    def test_image_processing(self):
        processor = self.get_processor()
        text = self.prepare_text_inputs(modalities=["image"])
        inputs = processor(text=text, images=self.prepare_images_inputs(), return_tensors="pt")

        self.assertListEqual(
            sorted(inputs.keys()),
            ["attention_mask", "input_ids", "patch_position_ids", "pixel_attention_mask", "pixel_values"],
        )
        self.assertIsInstance(inputs["pixel_values"], torch.Tensor)
        self.assertEqual(inputs["pixel_values"].shape[0], 1)  # one image, not sliced
        # One unsliced image -> exactly one block of 96 image tokens between <image> and </image>.
        self.assertEqual(int((inputs["input_ids"] == processor.image_token_id).sum()), 96)

    def test_sliced_image_placeholder(self):
        processor = self.get_processor()
        processor.image_processor.max_slice_nums = 9
        processor.image_processor.scale_resolution = 100
        # 100 x 200 px is twice scale_resolution**2 and twice as wide as tall -> 1 row x 2 columns of slices
        image = np.random.randint(255, size=(3, 100, 200), dtype=np.uint8)
        inputs = processor(text=f"{processor.image_token}\nDescribe.", images=[image], return_tensors="pt")

        num_slices = inputs["pixel_values"].shape[0]
        self.assertEqual(num_slices, 3)  # overview + 2 slices
        self.assertEqual(int((inputs["input_ids"] == processor.image_token_id).sum()), 96 * num_slices)
        tokens = processor.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())
        self.assertEqual(tokens.count("<slice>"), 1)
        self.assertEqual(tokens.count("<image>"), num_slices)

    def test_text_only_processing(self):
        processor = self.get_processor()
        inputs = processor(text="Hello, how are you?", return_tensors="pt")
        self.assertEqual(inputs["input_ids"].ndim, 2)
        self.assertNotIn("pixel_values", inputs)
