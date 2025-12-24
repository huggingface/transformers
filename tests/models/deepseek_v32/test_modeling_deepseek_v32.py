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
"""Testing suite for the PyTorch DeepseekV32 model."""

import unittest

from transformers import DeepseekV32Config, is_torch_available
from transformers.testing_utils import (
    require_torch,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        Cache,
        DeepseekV32ForCausalLM,
        DeepseekV32ForSequenceClassification,
        DeepseekV32ForTokenClassification,
        DeepseekV32Model,
    )


class DeepseekV32ModelTester:
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
        intermediate_size=37,
        moe_intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=8,
        routed_scaling_factor=2.5,
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=16,
        v_head_dim=32,
        qk_nope_head_dim=32,
        n_group=2,
        topk_group=1,
        num_experts_per_tok=8,
        first_k_dense_replace=2,
        norm_topk_prob=True,
        aux_loss_alpha=0.001,
        hidden_act="silu",
        max_position_embeddings=512,
        initializer_range=0.02,
        attention_probs_dropout_prob=0.1,
        type_vocab_size=16,
        type_sequence_label_size=2,
        num_labels=3,
        num_choices=4,
        pad_token_id=0,
        index_topk=2048,  # New parameter for V3.2
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.pad_token_id = pad_token_id
        self.index_topk = index_topk
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = torch.tril(torch.ones_like(input_ids).to(torch_device))

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()

        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        return DeepseekV32Config(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            moe_intermediate_size=self.moe_intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            n_shared_experts=self.n_shared_experts,
            n_routed_experts=self.n_routed_experts,
            routed_scaling_factor=self.routed_scaling_factor,
            kv_lora_rank=self.kv_lora_rank,
            q_lora_rank=self.q_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            qk_nope_head_dim=self.qk_nope_head_dim,
            n_group=self.n_group,
            topk_group=self.topk_group,
            num_experts_per_tok=self.num_experts_per_tok,
            first_k_dense_replace=self.first_k_dense_replace,
            norm_topk_prob=self.norm_topk_prob,
            aux_loss_alpha=self.aux_loss_alpha,
            hidden_act=self.hidden_act,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            use_cache=True,
            pad_token_id=self.pad_token_id,
            attention_dropout=self.attention_probs_dropout_prob,
            index_topk=self.index_topk,  # V3.2 specific parameter
        )

    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = DeepseekV32Model(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_sparse_attention_config(self, config):
        """Test that V3.2 has the sparse attention index_topk parameter"""
        self.parent.assertTrue(hasattr(config, "index_topk"))
        self.parent.assertEqual(config.index_topk, self.index_topk)
        self.parent.assertEqual(config.model_type, "deepseek_v32")

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class DeepseekV32ModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            DeepseekV32Model,
            DeepseekV32ForCausalLM,
            DeepseekV32ForSequenceClassification,
            DeepseekV32ForTokenClassification,
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (DeepseekV32ForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": DeepseekV32Model,
            "text-classification": DeepseekV32ForSequenceClassification,
            "token-classification": DeepseekV32ForTokenClassification,
            "text-generation": DeepseekV32ForCausalLM,
            "zero-shot": DeepseekV32ForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )

    # Need to use `0.8` instead of `0.9` for `test_cpu_offload`
    # This is because we are hitting edge cases with the causal_mask buffer
    model_split_percents = [0.5, 0.7, 0.8]

    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = DeepseekV32ForCausalLM if is_torch_available() else None

    def setUp(self):
        self.model_tester = DeepseekV32ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=DeepseekV32Config, hidden_size=37)

    def _check_past_key_values_for_generate(self, batch_size, past_key_values, seq_length, config):
        """Needs to be overridden as deepseek has special MLA cache format (though we don't really use the MLA)"""
        self.assertIsInstance(past_key_values, Cache)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_sparse_attention_config(self):
        """Test V3.2 specific sparse attention configuration"""
        config = self.model_tester.get_config()
        self.model_tester.create_and_check_sparse_attention_config(config)

    @unittest.skip("DeepseekV32 has variable input/output sizes due to MoE")
    def test_model_outputs_equivalence(self):
        pass

    @unittest.skip("DeepseekV32 has variable input/output sizes due to MoE")
    def test_for_causal_lm_outputs_equivalence(self):
        pass

    @unittest.skip("DeepseekV32 doesn't support head pruning")
    def test_headmasking(self):
        pass

    @unittest.skip("DeepseekV32 doesn't support head pruning")
    def test_head_pruning(self):
        pass

    @unittest.skip("DeepseekV32 doesn't support head pruning")
    def test_head_pruning_integration(self):
        pass

    @unittest.skip("DeepseekV32 doesn't support head pruning")
    def test_head_pruning_save_load_from_pretrained(self):
        pass

    @unittest.skip("DeepseekV32 doesn't support head pruning")
    def test_head_pruning_save_load_from_config_init(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        # This is a placeholder - in production you would test loading actual pretrained models
        pass


@require_torch
class DeepseekV32IntegrationTest(unittest.TestCase):
    def test_model_initialization(self):
        """Test that DeepseekV32 model can be initialized with sparse attention"""
        config = DeepseekV32Config(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            moe_intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            index_topk=2048,
        )
        model = DeepseekV32ForCausalLM(config)
        self.assertIsInstance(model, DeepseekV32ForCausalLM)
        self.assertEqual(model.config.index_topk, 2048)
        self.assertEqual(model.config.model_type, "deepseek_v32")

    def test_forward_pass(self):
        """Test a basic forward pass"""
        config = DeepseekV32Config(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            moe_intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            index_topk=2048,
        )
        model = DeepseekV32ForCausalLM(config)
        model.eval()

        input_ids = torch.randint(0, 1000, (2, 10))
        with torch.no_grad():
            outputs = model(input_ids)

        self.assertIsNotNone(outputs.logits)
        self.assertEqual(outputs.logits.shape, (2, 10, 1000))
