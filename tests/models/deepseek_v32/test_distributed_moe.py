import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from transformers import DeepseekV32Config, DeepseekV32Model, DeepseekV32ForCausalLM


def get_moe_test_config(**kwargs):
    """Get a config suitable for MoE testing."""
    defaults = dict(
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        first_k_dense_replace=1,  # Layer 1, 2, 3 will be MoE
        vocab_size=100,
        use_sparse_attention=False,  # Disable for faster tests
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_rope_head_dim=16,
        v_head_dim=16,
        qk_nope_head_dim=16,
    )
    defaults.update(kwargs)
    return DeepseekV32Config(**defaults)


class DeepseekV32MoEBasicTest(unittest.TestCase):
    """Test basic MoE functionality."""

    def test_moe_layer_structure(self):
        """Test that MoE layers are created correctly."""
        config = get_moe_test_config()
        model = DeepseekV32Model(config)

        # Layer 0 should be dense (first_k_dense_replace=1)
        self.assertFalse(hasattr(model.layers[0].mlp, 'experts'))

        # Layers 1, 2, 3 should be MoE
        for i in [1, 2, 3]:
            mlp = model.layers[i].mlp
            self.assertTrue(hasattr(mlp, 'experts'), f"Layer {i} should have experts")
            self.assertTrue(hasattr(mlp, 'shared_experts'), f"Layer {i} should have shared_experts")
            self.assertTrue(hasattr(mlp, 'gate'), f"Layer {i} should have gate")

    def test_moe_expert_count(self):
        """Test that expert count is correct."""
        config = get_moe_test_config(n_routed_experts=16)
        model = DeepseekV32Model(config)

        moe_layer = model.layers[1].mlp
        self.assertEqual(moe_layer.experts.num_experts, 16)
        # With 3D tensors, check the first dimension equals num_experts
        self.assertEqual(moe_layer.experts.gate_up_proj.shape[0], 16)
        self.assertEqual(moe_layer.experts.down_proj.shape[0], 16)

    def test_moe_forward_pass(self):
        """Test MoE forward pass produces valid output."""
        config = get_moe_test_config()
        model = DeepseekV32Model(config)
        model.eval()

        input_ids = torch.LongTensor([[1, 2, 3, 4, 5]])

        with torch.no_grad():
            outputs = model(input_ids)

        self.assertEqual(outputs.last_hidden_state.shape, (1, 5, config.hidden_size))
        self.assertFalse(torch.isnan(outputs.last_hidden_state).any())
        self.assertFalse(torch.isinf(outputs.last_hidden_state).any())

    def test_moe_backward_pass(self):
        """Test MoE backward pass computes gradients."""
        config = get_moe_test_config()
        model = DeepseekV32ForCausalLM(config)
        model.train()

        input_ids = torch.LongTensor([[1, 2, 3, 4, 5]])
        labels = torch.LongTensor([[2, 3, 4, 5, 6]])

        outputs = model(input_ids, labels=labels)
        outputs.loss.backward()

        # Check that MoE expert weights have gradients
        moe_layer = model.model.layers[1].mlp
        # With 3D tensors, check gradients on the parameter tensors
        self.assertIsNotNone(moe_layer.experts.gate_up_proj.grad)
        self.assertIsNotNone(moe_layer.experts.down_proj.grad)
        self.assertTrue(moe_layer.experts.gate_up_proj.grad.abs().sum() > 0)


class DeepseekV32MoEGradientCheckpointingTest(unittest.TestCase):
    """Test MoE with gradient checkpointing."""

    def test_gradient_checkpointing_forward_backward(self):
        """Test gradient checkpointing works with MoE."""
        config = get_moe_test_config()
        model = DeepseekV32ForCausalLM(config)
        model.gradient_checkpointing_enable()
        model.train()

        input_ids = torch.LongTensor([[1, 2, 3, 4, 5]])
        labels = torch.LongTensor([[2, 3, 4, 5, 6]])

        outputs = model(input_ids, labels=labels)
        self.assertFalse(torch.isnan(outputs.loss))

        # Backward should work without errors
        outputs.loss.backward()

        # Verify gradients exist
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        self.assertTrue(has_grad, "No gradients computed with gradient checkpointing")

    def test_gradient_checkpointing_determinism(self):
        """Test gradient checkpointing produces deterministic results."""
        config = get_moe_test_config()

        torch.manual_seed(42)
        model1 = DeepseekV32ForCausalLM(config)
        model1.gradient_checkpointing_enable()
        model1.train()

        torch.manual_seed(42)
        model2 = DeepseekV32ForCausalLM(config)
        model2.gradient_checkpointing_enable()
        model2.train()

        input_ids = torch.LongTensor([[1, 2, 3, 4, 5]])
        labels = torch.LongTensor([[2, 3, 4, 5, 6]])

        outputs1 = model1(input_ids, labels=labels)
        outputs2 = model2(input_ids, labels=labels)

        self.assertTrue(torch.allclose(outputs1.loss, outputs2.loss))


class DeepseekV32FSDPCompatibilityTest(unittest.TestCase):
    """Test FSDP/ZeRO-3 compatibility features."""

    def test_expert_uses_3d_tensors(self):
        """Test that experts are stored as 3D tensors (standard HuggingFace MoE pattern)."""
        config = get_moe_test_config()
        model = DeepseekV32Model(config)

        moe_layer = model.layers[1].mlp
        experts = moe_layer.experts

        # Should have 3D parameter tensors (FSDP shards these)
        self.assertIsInstance(experts.gate_up_proj, nn.Parameter)
        self.assertIsInstance(experts.down_proj, nn.Parameter)
        self.assertEqual(len(experts.gate_up_proj.shape), 3)
        self.assertEqual(len(experts.down_proj.shape), 3)

    def test_no_module_list(self):
        """Test that experts don't use nn.ModuleList (avoids FSDP AllGather issues)."""
        config = get_moe_test_config()
        model = DeepseekV32Model(config)

        moe_layer = model.layers[1].mlp
        experts = moe_layer.experts

        # Should NOT have an 'experts' ModuleList attribute
        self.assertFalse(
            hasattr(experts, 'experts') and isinstance(getattr(experts, 'experts', None), nn.ModuleList)
        )

    def test_expert_parameter_shapes(self):
        """Test that expert parameters have correct 3D shapes."""
        config = get_moe_test_config(n_routed_experts=8)
        model = DeepseekV32Model(config)

        moe_layer = model.layers[1].mlp
        experts = moe_layer.experts

        # gate_up_proj should be (num_experts, 2 * intermediate_size, hidden_size)
        self.assertEqual(
            experts.gate_up_proj.shape,
            (8, 2 * config.intermediate_size, config.hidden_size)
        )
        # down_proj should be (num_experts, hidden_size, intermediate_size)
        self.assertEqual(
            experts.down_proj.shape,
            (8, config.hidden_size, config.intermediate_size)
        )

    def test_expert_slicing_produces_correct_shape(self):
        """Test that slicing expert tensor produces correct 2D shape for F.linear."""
        config = get_moe_test_config(n_routed_experts=8)
        model = DeepseekV32Model(config)

        moe_layer = model.layers[1].mlp
        experts = moe_layer.experts

        # Slicing gate_up_proj[0] should give (2 * intermediate_size, hidden_size)
        expert_0_gate_up = experts.gate_up_proj[0]
        self.assertEqual(
            expert_0_gate_up.shape,
            (2 * config.intermediate_size, config.hidden_size)
        )

        # Slicing down_proj[0] should give (hidden_size, intermediate_size)
        expert_0_down = experts.down_proj[0]
        self.assertEqual(
            expert_0_down.shape,
            (config.hidden_size, config.intermediate_size)
        )

    def test_moe_layer_no_ep_attributes(self):
        """Test that MoE layer doesn't have old EP-specific attributes."""
        config = get_moe_test_config()
        model = DeepseekV32Model(config)

        moe_layer = model.layers[1].mlp

        # Old EP attributes should not exist
        self.assertFalse(hasattr(moe_layer, 'ep_size'))
        self.assertFalse(hasattr(moe_layer, 'ep_rank'))

    def test_all_experts_are_used(self):
        """Test that routing can potentially use all experts."""
        config = get_moe_test_config(n_routed_experts=8, num_experts_per_tok=2)
        model = DeepseekV32Model(config)
        model.eval()

        # With enough varied inputs, all experts should be touched
        torch.manual_seed(123)
        all_experts_touched = set()

        # Run multiple forward passes with different inputs
        for i in range(20):
            input_ids = torch.randint(1, 99, (1, 10))
            with torch.no_grad():
                # Hook to capture which experts are used
                moe_layer = model.layers[1].mlp

                # Get routing logits
                hidden = model.embed_tokens(input_ids)
                hidden = model.layers[0](hidden, position_embeddings=model.rotary_emb(hidden, position_ids=torch.arange(10).unsqueeze(0)))[0]
                hidden = model.layers[1].input_layernorm(hidden)

                # Get attention output first
                attn_out = model.layers[1].self_attn(
                    hidden,
                    position_embeddings=model.rotary_emb(hidden, position_ids=torch.arange(10).unsqueeze(0)),
                    attention_mask=None,
                )[0]
                hidden = hidden + attn_out
                hidden = model.layers[1].post_attention_layernorm(hidden)

                router_logits = moe_layer.gate(hidden)
                topk_indices, _ = moe_layer.route_tokens_to_experts(router_logits.view(-1, router_logits.shape[-1]))

                for idx in topk_indices.flatten().tolist():
                    all_experts_touched.add(idx)

        # Should touch most experts
        self.assertGreater(len(all_experts_touched), 4, "Should touch at least half of the experts")


class DeepseekV32MoERoutingTest(unittest.TestCase):
    """Test MoE routing functionality."""

    def test_routing_produces_valid_indices(self):
        """Test that routing produces valid expert indices."""
        config = get_moe_test_config(n_routed_experts=8, num_experts_per_tok=2)
        model = DeepseekV32Model(config)
        model.eval()

        moe_layer = model.layers[1].mlp

        # Simulate routing
        hidden = torch.randn(5, config.hidden_size)  # 5 tokens
        router_logits = moe_layer.gate(hidden)
        topk_indices, topk_weights = moe_layer.route_tokens_to_experts(router_logits)

        # Check shapes
        self.assertEqual(topk_indices.shape, (5, config.num_experts_per_tok))
        self.assertEqual(topk_weights.shape, (5, config.num_experts_per_tok))

        # Check indices are valid
        self.assertTrue((topk_indices >= 0).all())
        self.assertTrue((topk_indices < config.n_routed_experts).all())

        # Check weights are positive (after sigmoid + scaling)
        self.assertTrue((topk_weights > 0).all())

    def test_routing_normalization(self):
        """Test that routing weights are normalized when norm_topk_prob=True."""
        config = get_moe_test_config(norm_topk_prob=True)
        model = DeepseekV32Model(config)
        model.eval()

        moe_layer = model.layers[1].mlp

        hidden = torch.randn(5, config.hidden_size)
        router_logits = moe_layer.gate(hidden)
        _, topk_weights = moe_layer.route_tokens_to_experts(router_logits)

        # After normalization and scaling, weights should sum to routed_scaling_factor
        expected_sum = config.routed_scaling_factor
        actual_sums = topk_weights.sum(dim=-1)
        self.assertTrue(torch.allclose(actual_sums, torch.full_like(actual_sums, expected_sum), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
