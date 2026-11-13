# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
import unittest

import numpy as np

from transformers.testing_utils import require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import (
        Ovis2Processor,
    )


@require_vision
class Ovis2ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Ovis2Processor
    # Tiny processor created with make_tiny_processor.py from "thisisiron/Ovis2-1B-hf"
    tiny_model_id = "hf-internal-testing/tiny-processor-ovis2"

    @classmethod
    def _setup_image_processor(cls):
        image_processor_class = cls._get_component_class_from_processor("image_processor")
        # max_patches=1 ensures each image produces exactly 1 tile, so len(pixel_values)==batch_size
        return image_processor_class(max_patches=1)

    @staticmethod
    def prepare_processor_dict():
        return {
            "chat_template": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n'}}{% if message['content'] is string %}{{ message['content'] }}{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' %}{{ '<image>\n' }}{% elif content['type'] == 'text' %}{{ content['text'] }}{% endif %}{% endfor %}{% endif %}{{'<|im_end|>\n'}}{% endfor %}{% if add_generation_prompt %}{{'<|im_start|>assistant\n' }}{% endif %}",
            "image_seq_length": 4,
            "image_token": "<image>",
        }  # fmt: skip

    def test_processor_to_json_string(self):
        processor = self.get_processor()
        obj = json.loads(processor.to_json_string())
        for key, value in self.prepare_processor_dict().items():
            # chat_tempalate are tested as a separate test because they are saved in separate files
            if key != "chat_template":
                self.assertEqual(obj[key], value)
                self.assertEqual(getattr(processor, key, None), value)

    def test_chat_template_is_saved(self):
        processor_loaded = self.processor_class.from_pretrained(self.tmpdirname)
        processor_dict_loaded = json.loads(processor_loaded.to_json_string())
        # chat templates aren't serialized to json in processors
        self.assertFalse("chat_template" in processor_dict_loaded)

        # they have to be saved as separate file and loaded back from that file
        # so we check if the same template is loaded
        processor_dict = self.prepare_processor_dict()
        self.assertTrue(processor_loaded.chat_template == processor_dict.get("chat_template", None))

    def test_get_num_multimodal_tokens_matches_processor_call(self):
        # `<image>` isn't in the expanded text, so count the `<IMG_ATOM>` tokens in each replacement
        processor = self.get_processor()
        image_sizes = [(100, 100), (300, 100), (500, 30), (213, 167)]
        images = [np.random.randint(255, size=(h, w, 3), dtype=np.uint8) for h, w in image_sizes]
        text = [f"This is an image {processor.image_token}"] * len(images)
        tiling_kwargs = {"crop_to_patches": True, "max_patches": 4}
        inputs = processor(
            text=text,
            images=images,
            **tiling_kwargs,
            padding=True,
            return_tensors="pt",
            return_text_replacement_offsets=True,
        )

        replacements = [r["replacement"] for offsets in inputs["text_replacement_offsets"] for r in offsets]
        num_image_tokens_from_call = [r.count("<IMG_ATOM>") for r in replacements]
        num_image_tokens_from_helper = processor._get_num_multimodal_tokens(image_sizes=image_sizes, **tiling_kwargs)
        self.assertListEqual(num_image_tokens_from_call, num_image_tokens_from_helper["num_image_tokens"])
