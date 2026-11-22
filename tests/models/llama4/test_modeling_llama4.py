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
"""Testing suite for the PyTorch Llama4 model."""

import unittest

from transformers import is_torch_available
from transformers.testing_utils import (
    Expectations,
    require_torch,
    require_torch_large_accelerator,
    slow,
    torch_device,
)

from ...test_memory_cleanup_mixin import MemoryCleanupMixin
from ...test_processing_common import url_to_local_path


if is_torch_available():
    import torch

    from transformers import (
        Llama4Config,
        Llama4ForConditionalGeneration,
        Llama4Processor,
    )


@require_torch
class Llama4ForConditionalGenerationTinyTest(unittest.TestCase):
    def get_tiny_model(self):
        text_config = {
            "vocab_size": 99,
            "hidden_size": 32,
            "intermediate_size": 37,
            "intermediate_size_mlp": 48,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "num_local_experts": 4,
            "num_experts_per_tok": 1,
            "interleave_moe_layer_step": 2,
            "no_rope_layers": [1, 0],
            "attention_chunk_size": 4,
            "max_position_embeddings": 64,
        }
        vision_config = {
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "intermediate_size": 128,
            "image_size": 28,
            "patch_size": 14,
            "vision_output_dim": 32,
            "projector_input_dim": 32,
            "projector_output_dim": 32,
        }
        config = Llama4Config(text_config=text_config, vision_config=vision_config, image_token_index=98)
        return Llama4ForConditionalGeneration(config).to(torch_device).eval()

    def test_forward_without_images(self):
        model = self.get_tiny_model()
        input_ids = torch.tensor([[1, 5, 6, 7]], device=torch_device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
        self.assertEqual(outputs.logits.shape, (1, 4, 99))
        self.assertIsNone(outputs.image_outputs)

    def test_forward_with_images_returns_image_outputs(self):
        model = self.get_tiny_model()
        pixel_values = torch.randn(1, 3, 28, 28, device=torch_device)
        # 28x28 image, 14x14 patches, pixel shuffle ratio 0.5: one image token
        input_ids = torch.tensor([[1, 98, 5, 6]], device=torch_device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, pixel_values=pixel_values)
        self.assertEqual(outputs.logits.shape, (1, 4, 99))
        self.assertEqual(outputs.image_outputs.last_hidden_state.shape, (1, 1, 32))


@slow
@require_torch_large_accelerator
class Llama4IntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    model_id = "meta-llama/Llama-4-Scout-17B-16E"

    @classmethod
    def setUpClass(cls):
        cls.model = Llama4ForConditionalGeneration.from_pretrained(
            "meta-llama/Llama-4-Scout-17B-16E",
            device_map="auto",
            dtype=torch.float32,
            attn_implementation="eager",
        )

    def setUp(self):
        super().setUp()
        self.processor = Llama4Processor.from_pretrained("meta-llama/Llama-4-Scout-17B-16E", padding_side="left")

        url = "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png"
        self.messages_1 = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": url},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]

        self.messages_2 = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": url_to_local_path(
                            "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png"
                        ),
                    },
                    {
                        "type": "image",
                        "url": url_to_local_path(
                            "https://huggingface.co/datasets/hf-internal-testing/fixtures_image_utils/resolve/main/australia.jpg"
                        ),
                    },
                    {"type": "text", "text": "Are these images identical?"},
                ],
            },
        ]

    def test_model_17b_16e_fp32(self):
        EXPECTED_TEXTS = Expectations(
            {
                ("xpu", 3): ['system\n\nYou are a helpful assistant.user\n\nWhat is shown in this image?assistant\n\nThe image shows a cow standing on a beach with a blue sky and a body of water in the background. The cow is brown with a white face'],
                ("cuda", None): ['system\n\nYou are a helpful assistant.user\n\nWhat is shown in this image?assistant\n\nThe image shows a cow standing on a beach, with a blue sky and a body of water in the background. The cow is brown with a white'],
            }
        )  # fmt: skip
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()

        inputs = self.processor.apply_chat_template(
            self.messages_1, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(device=torch_device, dtype=self.model.dtype)
        output = self.model.generate(**inputs, max_new_tokens=30, do_sample=False)
        output_text = self.processor.batch_decode(output, skip_special_tokens=True)

        print(output_text)
        self.assertEqual(output_text, EXPECTED_TEXT)

    def test_model_17b_16e_batch(self):
        inputs = self.processor.apply_chat_template(
            [self.messages_1, self.messages_2],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            add_generation_prompt=True,
        ).to(device=torch_device, dtype=torch.float32)

        output = self.model.generate(**inputs, max_new_tokens=30, do_sample=False)
        output_text = self.processor.batch_decode(output, skip_special_tokens=True)

        EXPECTED_TEXTS = [
            'system\n\nYou are a helpful assistant.user\n\nWhat is shown in this image?assistant\n\nThe image shows a cow standing on a beach, with a blue sky and a body of water in the background. The cow is brown with a white',
            'system\n\nYou are a helpful assistant.user\n\nAre these images identical?assistant\n\nNo, these images are not identical. The first image shows a cow standing on a beach with a blue sky and a white cloud in the background.'
        ]  # fmt: skip
        self.assertEqual(output_text, EXPECTED_TEXTS)
