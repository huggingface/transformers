# Copyright 2025 Meituan and the HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch LongcatFlash model."""

import unittest

from transformers import LongcatFlashConfig, is_torch_available
from transformers.testing_utils import require_torch, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor


if is_torch_available():
    import torch

    from transformers import LongcatFlashForCausalLM, LongcatFlashModel


class LongcatFlashModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=32,
        hidden_size=32,
        ffn_hidden_size=64,
        expert_ffn_hidden_size=16,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        kv_lora_rank=8,
        q_lora_rank=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        qk_nope_head_dim=16,
        n_routed_experts=4,
        zero_expert_num=2,
        moe_topk=2,
        routed_scaling_factor=1.0,
        norm_topk_prob=False,
        router_bias=False,
        hidden_act="silu",
        max_position_embeddings=128,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        pad_token_id=0,
        mla_scale_q_lora=True,
        mla_scale_kv_lora=True,
        type_sequence_label_size=2,
        num_labels=3,
        num_choices=4,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.expert_ffn_hidden_size = expert_ffn_hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.n_routed_experts = n_routed_experts
        self.zero_expert_num = zero_expert_num
        self.moe_topk = moe_topk
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.router_bias = router_bias
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pad_token_id = pad_token_id
        self.attention_method = attention_method
        self.mla_scale_q_lora = mla_scale_q_lora
        self.mla_scale_kv_lora = mla_scale_kv_lora
        self.type_sequence_label_size = type_sequence_label_size
        self.num_labels = num_labels
        self.num_choices = num_choices

    def get_config(self):
        return LongcatFlashConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            ffn_hidden_size=self.ffn_hidden_size,
            expert_ffn_hidden_size=self.expert_ffn_hidden_size,
            num_layers=self.num_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            kv_lora_rank=self.kv_lora_rank,
            q_lora_rank=self.q_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            qk_nope_head_dim=self.qk_nope_head_dim,
            n_routed_experts=self.n_routed_experts,
            zero_expert_num=self.zero_expert_num,
            moe_topk=self.moe_topk,
            routed_scaling_factor=self.routed_scaling_factor,
            norm_topk_prob=self.norm_topk_prob,
            router_bias=self.router_bias,
            hidden_act=self.hidden_act,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            rms_norm_eps=self.rms_norm_eps,
            pad_token_id=self.pad_token_id,
            attention_method=self.attention_method,
            mla_scale_q_lora=self.mla_scale_q_lora,
            mla_scale_kv_lora=self.mla_scale_kv_lora,
        )

    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = LongcatFlashModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_causal_lm(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        model = LongcatFlashForCausalLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = torch.tril(torch.ones(self.batch_size, self.seq_length)).to(torch_device)

        token_type_ids = None

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()

        return (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels = config_and_inputs

        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class LongcatFlashModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (LongcatFlashModel, LongcatFlashForCausalLM) if is_torch_available() else ()
    all_generative_model_classes = (LongcatFlashForCausalLM,) if is_torch_available() else ()

    test_headmasking = False
    test_pruning = False

    def setUp(self):
        self.model_tester = LongcatFlashModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=LongcatFlashConfig, hidden_size=37, num_attention_heads=3
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_causal_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_causal_lm(*config_and_inputs)

    @unittest.skip("LongcatFlash buffers include complex numbers, which breaks this test")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip("LongcatFlash buffers include complex numbers, which breaks this test")  
    def test_save_load_fast_init_to_base(self):
        pass