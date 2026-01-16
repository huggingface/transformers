# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from transformers import is_torch_available

# Direct import to bypass lazy-loading issues on local dev environment
from transformers.models.mimo_v2_flash import MiMoV2FlashConfig
from transformers.testing_utils import (
    ConfigTester,
    ids_tensor,
    random_attention_mask,
    require_torch,
    torch_device,
)

if is_torch_available():
    import torch

    from transformers.models.mimo_v2_flash import (
        MiMoV2FlashForCausalLM,
        MiMoV2FlashModel,
    )


@require_torch
class MiMoV2FlashModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        v_head_dim=12,  # Asymmetric V dim
        intermediate_size=37,
        hidden_act="silu",
        max_position_embeddings=512,
        initializer_range=0.02,
        pad_token_id=0,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.pad_token_id = pad_token_id

        # MoE params
        self.n_routed_experts = 4
        self.num_experts_per_tok = 2
        self.moe_intermediate_size = 16
        self.moe_layer_freq = [0, 1]  # 1st layer dense, 2nd layer MoE
        self.layer_types = ["full_attention", "sliding_attention"]

    def prepare_config_and_inputs(self):
        input_ids = torch.randint(
            0, self.vocab_size, (self.batch_size, self.seq_length), device=torch_device
        )

        input_mask = None
        if self.use_input_mask:
            input_mask = torch.randint(
                0, 2, (self.batch_size, self.seq_length), device=torch_device
            )
        labels = None
        if self.use_labels:
            labels = torch.randint(
                0,
                self.vocab_size,
                (self.batch_size, self.seq_length),
                device=torch_device,
            )

        config = self.get_config()

        return config, input_ids, input_mask, labels

    def get_config(self):
        return MiMoV2FlashConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            v_head_dim=self.v_head_dim,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
            n_routed_experts=self.n_routed_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            moe_intermediate_size=self.moe_intermediate_size,
            moe_layer_freq=self.moe_layer_freq,
            layer_types=self.layer_types,
        )

    def create_and_check_model(self, config, input_ids, input_mask, labels):
        model = MiMoV2FlashModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, return_dict=True)
        result = model(input_ids, return_dict=True)
        self.parent.assertEqual(
            result.last_hidden_state.shape,
            (self.batch_size, self.seq_length, self.hidden_size),
        )

    def create_and_check_for_causal_lm(
        self, config, input_ids, input_mask, token_labels
    ):
        model = MiMoV2FlashForCausalLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids, attention_mask=input_mask, labels=token_labels, return_dict=True
        )
        self.parent.assertEqual(
            result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size)
        )


@require_torch
class MiMoV2FlashModelTest(unittest.TestCase):
    all_model_classes = (
        (MiMoV2FlashModel, MiMoV2FlashForCausalLM) if is_torch_available() else ()
    )
    test_head_masking = False
    test_pruning = False

    def setUp(self):
        self.model_tester = MiMoV2FlashModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=MiMoV2FlashConfig, hidden_size=37
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_causal_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_causal_lm(*config_and_inputs)
