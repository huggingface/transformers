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
    cleanup,
    require_torch,
    require_torch_large_accelerator,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        Llama4ForCausalLM,
        Llama4ForConditionalGeneration,
        Llama4Processor,
        Llama4TextConfig,
        Llama4TextModel,
    )


# ---------------------------------------------------------------------------
# Tiny model tester for unit / TP / EP tests (no GPU, no real weights)
# ---------------------------------------------------------------------------


class Llama4TextModelTester(CausalLMModelTester):
    if is_torch_available():
        config_class = Llama4TextConfig
        base_model_class = Llama4TextModel
        causal_lm_class = Llama4ForCausalLM

    def __init__(self, *args, **kwargs):
        # Llama4-specific tiny defaults — keep the model small enough for CPU tests
        kwargs.setdefault("hidden_size", 32)
        kwargs.setdefault("intermediate_size", 32)
        kwargs.setdefault("num_hidden_layers", 2)
        kwargs.setdefault("num_attention_heads", 2)
        kwargs.setdefault("num_key_value_heads", 2)
        kwargs.setdefault("head_dim", 16)
        kwargs.setdefault("vocab_size", 64)
        # MoE: every layer is a MoE layer with 4 tiny experts
        kwargs.setdefault("num_local_experts", 4)
        kwargs.setdefault("num_experts_per_tok", 1)
        kwargs.setdefault("interleave_moe_layer_step", 1)
        kwargs.setdefault("moe_intermediate_size", 16)
        super().__init__(*args, **kwargs)


@require_torch
class Llama4TextModelTest(CausalLMModelTest, unittest.TestCase):
    """
    Unit tests for Llama4TextModel / Llama4ForCausalLM, including tensor parallel
    and expert parallel coverage via :class:`~tests.test_tensor_parallel_mixin.TensorParallelTesterMixin`.

    The TP/EP tests (``test_tp_forward``, ``test_ep_forward``, …) exercise the fix for
    ``MoeTensorParalellExperts._prepare_input_fn`` which previously crashed on Llama4
    because it assumed a 3-argument ``(hidden_states, top_k_index, top_k_weights)``
    calling convention, while ``Llama4TextMoe`` calls ``experts(routed_in)`` — a single
    pre-weighted tensor.
    """

    model_tester_class = Llama4TextModelTester

    @unittest.skip("tp_generation_quantized requires torchao; skipped for Llama4 text-only tests")
    def test_tp_generation_quantized(self):
        pass


@slow
@require_torch_large_accelerator
class Llama4IntegrationTest(unittest.TestCase):
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
                        "url": "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png",
                    },
                    {
                        "type": "image",
                        "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/australia.jpg",
                    },
                    {"type": "text", "text": "Are these images identical?"},
                ],
            },
        ]

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

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
