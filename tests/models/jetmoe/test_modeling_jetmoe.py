# Copyright 2024 JetMoe AI and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch JetMoe model."""

import unittest

import pytest

from transformers import AutoTokenizer, is_torch_available
from transformers.testing_utils import (
    cleanup,
    require_flash_attn,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        JetMoeForCausalLM,
        JetMoeModel,
    )


class JetMoeModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = JetMoeModel

    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=False,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_key_value_heads=2,
        kv_channels=8,
        intermediate_size=37,
        hidden_act="silu",
        num_local_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        pad_token_id=0,
        scope=None,
    ):
        super().__init__(parent)
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.kv_channels = kv_channels
        self.num_attention_heads = num_key_value_heads * num_experts_per_tok
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.pad_token_id = pad_token_id
        self.scope = scope


@require_torch
class JetMoeModelTest(CausalLMModelTest, unittest.TestCase):
    test_mismatched_shapes = False
    test_cpu_offload = False
    test_disk_offload_bin = False
    test_disk_offload_safetensors = False
    model_tester_class = JetMoeModelTester

    @require_flash_attn
    @require_torch_accelerator
    @pytest.mark.flash_attn_test
    @slow
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        self.skipTest(reason="JetMoe flash attention does not support right padding")

    @unittest.skip(reason="JetMoe has no separate base model without a head.")
    def test_model_base_model_prefix(self):
        pass


@require_torch
class JetMoeIntegrationTest(unittest.TestCase):
    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_model_8b_logits(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]
        model = JetMoeForCausalLM.from_pretrained("jetmoe/jetmoe-8b", device_map="auto", torch_dtype=torch.bfloat16)
        input_ids = torch.tensor([input_ids]).to(model.model.embed_tokens.weight.device)
        with torch.no_grad():
            out = model(input_ids).logits.float().cpu()
        # Expected mean on dim = -1
        EXPECTED_MEAN = torch.tensor([[0.1943, -2.7299, -1.3466, -1.9385, -1.7457, -1.7472, -1.8647, -1.8547]])
        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, rtol=1e-2, atol=1e-2)
        # slicing logits[0, 0, 0:30]
        EXPECTED_SLICE = torch.tensor([-3.4844, 6.0625, 5.8750, -1.6875, -4.7812, -4.7812, -4.7812, -4.7812, -4.7812, -4.7812, -4.7812, -4.7812, 3.8750, 9.3750, -4.7812, -4.7812, -4.7812, -4.7812, -4.7812, -4.7812, -4.7812, -4.7812, -4.7812, -4.7812, -4.7812, -4.7812, -4.7812, -4.7812, -4.7812, -4.7812])  # fmt: skip
        torch.testing.assert_close(out[0, 0, :30], EXPECTED_SLICE, rtol=1e-4, atol=1e-4)

    @slow
    def test_model_8b_generation(self):
        EXPECTED_TEXT_COMPLETION = """My favourite condiment is ....\nI love ketchup. I love"""
        prompt = "My favourite condiment is "
        tokenizer = AutoTokenizer.from_pretrained("jetmoe/jetmoe-8b", use_fast=False)
        model = JetMoeForCausalLM.from_pretrained("jetmoe/jetmoe-8b", device_map="auto", torch_dtype=torch.bfloat16)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.model.embed_tokens.weight.device)

        # greedy generation outputs
        generated_ids = model.generate(input_ids, max_new_tokens=10, temperature=0)
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)

    @slow
    def test_model_8b_batched_generation(self):
        EXPECTED_TEXT_COMPLETION = [
            """My favourite condiment is ....\nI love ketchup. I love""",
            """My favourite 2018 Christmas present was a new pair""",
        ]
        prompt = [
            "My favourite condiment is ",
            "My favourite ",
        ]
        tokenizer = AutoTokenizer.from_pretrained("jetmoe/jetmoe-8b", use_fast=False)
        model = JetMoeForCausalLM.from_pretrained("jetmoe/jetmoe-8b", device_map="auto", torch_dtype=torch.bfloat16)
        input_ids = tokenizer(prompt, return_tensors="pt", padding=True).to(model.model.embed_tokens.weight.device)

        # greedy generation outputs
        generated_ids = model.generate(**input_ids, max_new_tokens=10, temperature=0)
        text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)
