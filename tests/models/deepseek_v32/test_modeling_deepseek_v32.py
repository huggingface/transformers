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

import pytest
from packaging import version
from parameterized import parameterized

from transformers import is_torch_available
from transformers.testing_utils import (
    cleanup,
    require_torch,
    require_torch_accelerator,
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
        DeepseekV32Config,
        DeepseekV32ForCausalLM,
        DeepseekV32ForSequenceClassification,
        DeepseekV32ForTokenClassification,
        DeepseekV32Model,
    )


class DeepseekV32ModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=16,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=False,
        use_labels=True,
        vocab_size=99,
        hidden_size=64,  # Must be divisible by num_attention_heads
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=8,
        routed_scaling_factor=2.5,
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=16,  # Must be power of 2 for Hadamard
        v_head_dim=16,
        qk_nope_head_dim=16,
        n_group=2,
        topk_group=1,
        num_experts_per_tok=4,
        first_k_dense_replace=1,
        norm_topk_prob=True,
        hidden_act="silu",
        max_position_embeddings=512,
        initializer_range=0.02,
        attention_probs_dropout_prob=0.0,
        type_vocab_size=16,
        type_sequence_label_size=2,
        num_labels=3,
        num_choices=4,
        pad_token_id=0,
        scope=None,
        # DeepSeek V3.2 specific
        index_n_heads=4,
        index_head_dim=32,  # Must be power of 2 for Hadamard
        index_topk=8,
        use_sparse_attention=True,
        detach_indexer_input=False,
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
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.pad_token_id = pad_token_id
        self.scope = scope
        # V3.2 specific
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.use_sparse_attention = use_sparse_attention
        self.detach_indexer_input = detach_indexer_input

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
            hidden_act=self.hidden_act,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            attention_dropout=self.attention_probs_dropout_prob,
            pad_token_id=self.pad_token_id,
            # V3.2 specific
            index_n_heads=self.index_n_heads,
            index_head_dim=self.index_head_dim,
            index_topk=self.index_topk,
            use_sparse_attention=self.use_sparse_attention,
            detach_indexer_input=self.detach_indexer_input,
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

    def create_and_check_model_as_decoder(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        config.add_cross_attention = True
        model = DeepseekV32Model(config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        result = model(
            input_ids,
            attention_mask=input_mask,
            encoder_hidden_states=encoder_hidden_states,
        )
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
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        model = DeepseekV32ForCausalLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_decoder_model_past_large_inputs(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        config.is_decoder = True
        config.add_cross_attention = True
        model = DeepseekV32ForCausalLM(config=config)
        model.to(torch_device)
        model.eval()

        # first forward pass
        outputs = model(
            input_ids,
            attention_mask=input_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_mask = ids_tensor((self.batch_size, 3), vocab_size=2)

        # append to next input_ids and
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        next_attention_mask = torch.cat([input_mask, next_mask], dim=-1)

        output_from_no_past = model(
            next_input_ids,
            attention_mask=next_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_hidden_states=True,
        )["hidden_states"][0]
        output_from_past = model(
            next_tokens,
            attention_mask=next_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=True,
        )["hidden_states"][0]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

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
        }
        if is_torch_available()
        else {}
    )
    fx_compatible = False
    test_torchscript = False
    test_pruning = False
    test_head_masking = False
    test_disk_offload_safetensors = False
    test_disk_offload_bin = False

    def setUp(self):
        self.model_tester = DeepseekV32ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=DeepseekV32Config, hidden_size=64)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="DeepseekV32 buffers include complex numbers, which breaks this test")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="DeepseekV32 uses MQA so num_key_value_heads is not the same as num_attention_heads")
    def test_generate_with_static_cache(self):
        pass

    @unittest.skip(reason="DeepseekV32 uses a custom attention mechanism")
    def test_sdpa_equivalence(self):
        pass

    @unittest.skip(reason="DeepseekV32 uses GQA")
    def test_eager_matches_sdpa_generate(self):
        pass

    @parameterized.expand([("float16",), ("bfloat16",), ("float32",)])
    @require_torch_accelerator
    def test_model_dtype(self, torch_dtype_str):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        config = config_and_inputs[0]
        torch_dtype = getattr(torch, torch_dtype_str)
        model = DeepseekV32Model(config=config).to(torch_device, dtype=torch_dtype)
        model.eval()

        input_ids = config_and_inputs[1].to(torch_device)
        attention_mask = config_and_inputs[3].to(torch_device) if config_and_inputs[3] is not None else None

        with torch.no_grad():
            output = model(input_ids, attention_mask=attention_mask)

        self.assertEqual(output.last_hidden_state.dtype, torch_dtype)


@require_torch
class DeepseekV32ForwardBackwardTest(unittest.TestCase):
    """Test forward and backward passes for DeepSeek V3.2."""

    def get_tiny_config(self):
        """Get a tiny config for fast testing."""
        return DeepseekV32Config(
            vocab_size=100,
            hidden_size=64,
            intermediate_size=128,
            moe_intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            n_shared_experts=1,
            n_routed_experts=4,
            routed_scaling_factor=2.5,
            kv_lora_rank=16,
            q_lora_rank=32,
            qk_rope_head_dim=16,
            v_head_dim=16,
            qk_nope_head_dim=16,
            n_group=2,
            topk_group=1,
            num_experts_per_tok=2,
            first_k_dense_replace=1,
            max_position_embeddings=128,
            # V3.2 specific
            index_n_heads=4,
            index_head_dim=32,
            index_topk=8,
            use_sparse_attention=True,
        )

    def test_forward_pass(self):
        """Test that forward pass works."""
        config = self.get_tiny_config()
        model = DeepseekV32ForCausalLM(config)
        model.to(torch_device)
        model.eval()

        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=torch_device)

        with torch.no_grad():
            outputs = model(input_ids)

        self.assertEqual(outputs.logits.shape, (batch_size, seq_len, config.vocab_size))
        self.assertFalse(torch.isnan(outputs.logits).any())
        self.assertFalse(torch.isinf(outputs.logits).any())

    def test_backward_pass(self):
        """Test that backward pass works and gradients flow."""
        config = self.get_tiny_config()
        model = DeepseekV32ForCausalLM(config)
        model.to(torch_device)
        model.train()

        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=torch_device)
        labels = input_ids.clone()

        outputs = model(input_ids, labels=labels)
        loss = outputs.loss

        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))

        loss.backward()

        # Check that gradients are computed for key parameters
        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        self.assertTrue(has_grad, "No gradients were computed")

    def test_indexer_gradients(self):
        """Test that gradients flow through the indexer."""
        config = self.get_tiny_config()
        config.use_sparse_attention = True
        model = DeepseekV32ForCausalLM(config)
        model.to(torch_device)
        model.train()

        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=torch_device)
        labels = input_ids.clone()

        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()

        # Check indexer parameters have gradients
        indexer_has_grad = False
        for name, param in model.named_parameters():
            if "indexer" in name and param.grad is not None:
                if param.grad.abs().sum() > 0:
                    indexer_has_grad = True
                    break

        self.assertTrue(indexer_has_grad, "Indexer parameters have no gradients")

    def test_loss_decreases(self):
        """Test that loss decreases over training steps."""
        config = self.get_tiny_config()
        model = DeepseekV32ForCausalLM(config)
        model.to(torch_device)
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        batch_size, seq_len = 4, 32
        # Use fixed data for consistent training
        torch.manual_seed(42)
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=torch_device)
        labels = input_ids.clone()

        losses = []
        num_steps = 10

        for step in range(num_steps):
            optimizer.zero_grad()
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        # Check that loss decreased
        self.assertLess(losses[-1], losses[0], f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}")
        print(f"Loss decreased from {losses[0]:.4f} to {losses[-1]:.4f}")

    def test_sparse_vs_dense_attention(self):
        """Test both sparse and dense attention modes work."""
        config = self.get_tiny_config()

        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=torch_device)

        # Test dense attention
        config.use_sparse_attention = False
        model_dense = DeepseekV32ForCausalLM(config)
        model_dense.to(torch_device)
        model_dense.eval()

        with torch.no_grad():
            outputs_dense = model_dense(input_ids)

        self.assertFalse(torch.isnan(outputs_dense.logits).any())

        # Test sparse attention
        config.use_sparse_attention = True
        model_sparse = DeepseekV32ForCausalLM(config)
        model_sparse.to(torch_device)
        model_sparse.eval()

        with torch.no_grad():
            outputs_sparse = model_sparse(input_ids)

        self.assertFalse(torch.isnan(outputs_sparse.logits).any())


@require_torch
class DeepseekV32MetaDeviceTest(unittest.TestCase):
    """Test meta device initialization support for memory-efficient large model loading.

    These tests verify that the model can be created on meta device (0 bytes memory)
    and then properly initialized for use with FSDP/DeepSpeed.
    """

    def get_tiny_config(self):
        """Get a tiny config for fast testing."""
        return DeepseekV32Config(
            vocab_size=100,
            hidden_size=64,
            intermediate_size=128,
            moe_intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            n_shared_experts=1,
            n_routed_experts=4,
            routed_scaling_factor=2.5,
            kv_lora_rank=16,
            q_lora_rank=32,
            qk_rope_head_dim=16,
            v_head_dim=16,
            qk_nope_head_dim=16,
            n_group=2,
            topk_group=1,
            num_experts_per_tok=2,
            first_k_dense_replace=1,
            max_position_embeddings=128,
            # V3.2 specific
            index_n_heads=4,
            index_head_dim=32,
            index_topk=8,
            use_sparse_attention=True,
        )

    def test_meta_device_initialization(self):
        """Test that model can be created on meta device with zero memory."""
        config = self.get_tiny_config()

        # Create model on meta device
        with torch.device("meta"):
            model = DeepseekV32ForCausalLM(config)

        # Verify all parameters are on meta device
        for name, param in model.named_parameters():
            self.assertEqual(
                param.device.type,
                "meta",
                f"Parameter {name} is on {param.device}, expected meta device",
            )

        # Verify all buffers are on meta device
        for name, buffer in model.named_buffers():
            self.assertEqual(
                buffer.device.type,
                "meta",
                f"Buffer {name} is on {buffer.device}, expected meta device",
            )

    def test_meta_to_empty_to_device(self):
        """Test the full meta device workflow: meta -> empty -> device."""
        config = self.get_tiny_config()

        # Step 1: Create on meta device
        with torch.device("meta"):
            model = DeepseekV32ForCausalLM(config)

        # Step 2: Convert to empty (allocate memory but don't initialize)
        model = model.to_empty(device="cpu")

        # Verify parameters are now on CPU
        for name, param in model.named_parameters():
            self.assertEqual(
                param.device.type,
                "cpu",
                f"Parameter {name} is on {param.device}, expected cpu",
            )

        # Step 3: Initialize weights (simulating what post_init/FSDP would do)
        # Must apply _init_weights to each submodule, as transformers does
        for module in model.modules():
            model._init_weights(module)

        # Step 4: Verify model can do forward pass
        model.eval()
        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            outputs = model(input_ids)

        self.assertEqual(outputs.logits.shape, (batch_size, seq_len, config.vocab_size))
        self.assertFalse(torch.isnan(outputs.logits).any())

    def test_rmsnorm_meta_device(self):
        """Test RMSNorm respects meta device context."""
        from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32RMSNorm

        hidden_size = 64

        # Create on meta device
        with torch.device("meta"):
            norm = DeepseekV32RMSNorm(hidden_size)

        self.assertEqual(norm.weight.device.type, "meta")

        # Convert to CPU and initialize
        norm = norm.to_empty(device="cpu")
        torch.nn.init.ones_(norm.weight)

        # Test forward pass
        x = torch.randn(2, 8, hidden_size)
        output = norm(x)
        self.assertEqual(output.shape, x.shape)
        self.assertFalse(torch.isnan(output).any())

    def test_topk_router_meta_device(self):
        """Test TopkRouter respects meta device context."""
        from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32TopkRouter

        config = self.get_tiny_config()

        # Create on meta device
        with torch.device("meta"):
            router = DeepseekV32TopkRouter(config)

        self.assertEqual(router.weight.device.type, "meta")
        self.assertEqual(router.e_score_correction_bias.device.type, "meta")

        # Convert to CPU and initialize
        router = router.to_empty(device="cpu")
        torch.nn.init.normal_(router.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(router.e_score_correction_bias)

        # Test forward pass
        x = torch.randn(2, 8, config.hidden_size)
        output = router(x)
        self.assertEqual(output.shape[1], config.n_routed_experts)
        self.assertFalse(torch.isnan(output).any())

    def test_naive_moe_meta_device(self):
        """Test NaiveMoe (3D expert tensors) respects meta device context."""
        from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32NaiveMoe

        config = self.get_tiny_config()

        # Create on meta device
        with torch.device("meta"):
            moe = DeepseekV32NaiveMoe(config)

        self.assertEqual(moe.gate_up_proj.device.type, "meta")
        self.assertEqual(moe.down_proj.device.type, "meta")

        # Verify shapes are correct even on meta device
        expected_gate_up_shape = (config.n_routed_experts, 2 * config.intermediate_size, config.hidden_size)
        expected_down_shape = (config.n_routed_experts, config.hidden_size, config.intermediate_size)
        self.assertEqual(moe.gate_up_proj.shape, expected_gate_up_shape)
        self.assertEqual(moe.down_proj.shape, expected_down_shape)


if __name__ == "__main__":
    unittest.main()
