# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_vision_available


if is_vision_available():
    from PIL import Image

    from transformers import AutoTokenizer, DeepseekVLV2ImageProcessor, DeepseekVLV2Processor
    from transformers.models.deepseek_vl_v2.convert_deepseek_vl_v2_weights_to_hf import CHAT_TEMPLATE


@require_torch
@require_vision
class DeepseekVLV2ProcessorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
        tokenizer.add_special_tokens({"additional_special_tokens": ["<image>", "<|User|>", "<|Assistant|>"]})
        cls.processor = DeepseekVLV2Processor(
            image_processor=DeepseekVLV2ImageProcessor(), tokenizer=tokenizer, chat_template=CHAT_TEMPLATE
        )

    def test_image_token_count_follows_the_tile_grid(self):
        # 640x480 tiles as 2x2: 14 * 15 global + 1 separator + (2 * 14) * (2 * 14 + 1) local = 1023 image tokens.
        inputs = self.processor(text="<image>\nDescribe.", images=[Image.new("RGB", (640, 480))], return_tensors="pt")
        self.assertEqual(inputs["image_spatial_crop"].tolist(), [[2, 2]])
        image_tokens = int((inputs["input_ids"] == self.processor.image_token_id).sum())
        self.assertEqual(image_tokens, 14 * 15 + 1 + 28 * 29)

    def test_chat_template_matches_the_original_conversation_format(self):
        messages = [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is this?"}]},
            {"role": "assistant", "content": "A dog."},
            {"role": "user", "content": "And the text?"},
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        bos, eos = self.processor.tokenizer.bos_token, self.processor.tokenizer.eos_token
        expected = (
            f"{bos}Be brief.\n\n<|User|>: <image>\nWhat is this?\n\n<|Assistant|>: A dog.{eos}"
            "<|User|>: And the text?\n\n<|Assistant|>:"
        )
        self.assertEqual(prompt, expected)
