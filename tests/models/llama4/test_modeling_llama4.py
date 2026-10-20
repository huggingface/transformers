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

import tempfile
import unittest

from parameterized import parameterized

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
    from safetensors.torch import load_file

    from transformers import (
        Llama4ForCausalLM,
        Llama4ForConditionalGeneration,
        Llama4Processor,
        Llama4TextConfig,
    )


@require_torch
class Llama4TextMoeTest(unittest.TestCase):
    def get_config(self, top_k=1):
        return Llama4TextConfig(
            vocab_size=97,
            hidden_size=32,
            intermediate_size=48,
            intermediate_size_mlp=48,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            num_local_experts=4,
            num_experts_per_tok=top_k,
            max_position_embeddings=64,
            attention_chunk_size=16,
        )

    @parameterized.expand([1, 2])
    def test_input_router_weights(self, top_k):
        config = self.get_config(top_k)
        model = Llama4ForCausalLM(config).to(torch_device).eval()
        moe = model.model.layers[0].feed_forward
        hidden_states = torch.randn(6, config.hidden_size, device=torch_device)
        with torch.no_grad():
            output, logits = moe(hidden_states)
            # Independent reference: the original expert-major BMM computes every expert,
            # with zero inputs for experts outside the top-k selection.
            scores = torch.zeros_like(logits)
            values, indices = logits.topk(top_k, dim=-1)
            scores.scatter_(1, indices, values.sigmoid())
            routed = hidden_states.unsqueeze(0) * scores.t().unsqueeze(-1)
            gate, up = torch.bmm(routed, moe.experts.gate_up_proj.transpose(1, 2)).chunk(2, dim=-1)
            routed = torch.bmm(moe.experts.act_fn(gate) * up, moe.experts.down_proj.transpose(1, 2))
            expected = moe.shared_expert(hidden_states) + routed.sum(dim=0)
        torch.testing.assert_close(output, expected)

    def test_legacy_expert_checkpoint_layout(self):
        model = Llama4ForCausalLM(self.get_config(top_k=2)).eval()
        input_ids = torch.tensor([[1, 4, 7]])
        with tempfile.TemporaryDirectory() as checkpoint:
            model.save_pretrained(checkpoint)
            state = load_file(f"{checkpoint}/model.safetensors")
            prefix = "model.layers.0.feed_forward.experts."
            torch.testing.assert_close(
                state[prefix + "gate_up_proj"], model.state_dict()[prefix + "gate_up_proj"].transpose(1, 2)
            )
            torch.testing.assert_close(
                state[prefix + "down_proj"], model.state_dict()[prefix + "down_proj"].transpose(1, 2)
            )
            loaded = Llama4ForCausalLM.from_pretrained(checkpoint).eval()
        with torch.no_grad():
            expected = model(input_ids, output_router_logits=True)
            actual = loaded(input_ids, output_router_logits=True)
        torch.testing.assert_close(actual.logits, expected.logits)
        self.assertEqual(len(actual.router_logits), model.config.num_hidden_layers)
        for actual_router, expected_router in zip(actual.router_logits, expected.router_logits):
            self.assertEqual(actual_router.shape, (input_ids.numel(), model.config.num_local_experts))
            torch.testing.assert_close(actual_router, expected_router)

    @parameterized.expand([1, 2])
    def test_sequential_experts_match(self, top_k):
        """Compressed-tensors and bitsandbytes swap the experts for `SequentialLlama4TextExperts`, one MLP per expert."""
        from transformers.quantizers.base import SequentialLlama4TextExperts

        config = self.get_config(top_k)
        moe = Llama4ForCausalLM(config).model.layers[0].feed_forward.eval()
        hidden_states = torch.randn(6, config.hidden_size)
        sequential = SequentialLlama4TextExperts(config)
        gate_up, down = moe.experts.gate_up_proj, moe.experts.down_proj
        with torch.no_grad():
            expected = moe(hidden_states)[0]
            for idx, mlp in enumerate(sequential):
                mlp.gate_proj.weight.copy_(gate_up[idx, : config.intermediate_size])
                mlp.up_proj.weight.copy_(gate_up[idx, config.intermediate_size :])
                mlp.down_proj.weight.copy_(down[idx])
            moe.experts = sequential
            torch.testing.assert_close(moe(hidden_states)[0], expected)


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
