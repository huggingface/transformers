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

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from PIL import Image

    from transformers import AutoTokenizer, DeepseekVLV2ImageProcessor, DeepseekVLV2Processor
    from transformers.models.deepseek_vl_v2.convert_deepseek_vl_v2_weights_to_hf import CHAT_TEMPLATE


@require_torch
@require_vision
class DeepseekVLV2ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = DeepseekVLV2Processor

    @classmethod
    def _setup_tokenizer(cls):
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
        tokenizer.add_special_tokens({"additional_special_tokens": ["<image>", "<|User|>", "<|Assistant|>"]})
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @classmethod
    def _setup_image_processor(cls):
        # 28 px tiles give one image token per row (28 / 14 patches, merged 2x2), keeping prompts short.
        return DeepseekVLV2ImageProcessor(
            size={"height": 28, "width": 28}, candidate_resolutions=[[28, 28], [56, 28], [28, 56], [56, 56]]
        )

    @staticmethod
    def prepare_processor_dict():
        return {"chat_template": CHAT_TEMPLATE}

    def test_image_token_count_follows_the_tile_grid(self):
        processor = self.get_processor()
        processor.image_processor.size = {"height": 384, "width": 384}
        processor.image_processor.candidate_resolutions = DeepseekVLV2ImageProcessor.candidate_resolutions
        # 640x480 tiles as 2x2: 14 * 15 global + 1 separator + (2 * 14) * (2 * 14 + 1) local = 1023 image tokens.
        inputs = processor(text="<image>\nDescribe.", images=[Image.new("RGB", (640, 480))], return_tensors="pt")
        self.assertEqual(inputs["image_spatial_crop"].tolist(), [[2, 2]])
        image_tokens = int((inputs["input_ids"] == processor.image_token_id).sum())
        self.assertEqual(image_tokens, 14 * 15 + 1 + 28 * 29)

    def test_tiles_only_prompts_with_at_most_two_images(self):
        """The original model drops tiling for a conversation with more than two images."""
        processor = self.get_processor()
        wide = Image.new("RGB", (100, 50))
        texts = ["<image> one", "<image><image><image> three"]
        inputs = processor(text=texts, images=[wide] * 4, padding=True, return_tensors="pt")
        self.assertEqual(inputs["image_spatial_crop"].tolist(), [[2, 1], [1, 1], [1, 1], [1, 1]])
        # Global 1 * 2 + separator 1 + local rows: 1 * (2 + 1) when tiled 2x1, 1 * (1 + 1) when not.
        image_tokens = (inputs["input_ids"] == processor.image_token_id).sum(-1).tolist()
        self.assertEqual(image_tokens, [3 + 3, 3 * (3 + 2)])

    def test_explicit_crop_to_patches_wins(self):
        processor = self.get_processor()
        wide = Image.new("RGB", (100, 50))
        inputs = processor(text="<image><image><image>", images=[wide] * 3, crop_to_patches=True, return_tensors="pt")
        self.assertEqual(inputs["image_spatial_crop"].tolist(), [[2, 1]] * 3)

    def test_chat_template_matches_the_original_conversation_format(self):
        processor = self.get_processor()
        messages = [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is this?"}]},
            {"role": "assistant", "content": "A dog."},
            {"role": "user", "content": "And the text?"},
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        bos, eos = processor.tokenizer.bos_token, processor.tokenizer.eos_token
        expected = (
            f"{bos}Be brief.\n\n<|User|>: <image>\nWhat is this?\n\n<|Assistant|>: A dog.{eos}"
            "<|User|>: And the text?\n\n<|Assistant|>:"
        )
        self.assertEqual(prompt, expected)
