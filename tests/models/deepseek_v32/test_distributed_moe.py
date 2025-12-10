import unittest
from unittest.mock import MagicMock, patch

import torch

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


class DeepseekV32ExpertParallelismTest(unittest.TestCase):
    """Test Expert Parallelism (EP) configuration."""

    def test_ep_size_configuration(self):
        """Test ep_size is correctly propagated."""
        config = get_moe_test_config(ep_size=4)
        model = DeepseekV32Model(config)

        moe_layer = model.layers[1].mlp
        self.assertEqual(moe_layer.ep_size, 4)
        self.assertEqual(moe_layer.experts.ep_size, 4)

    def test_ep_local_expert_range(self):
        """Test local expert range is computed correctly."""
        config = get_moe_test_config(n_routed_experts=8, ep_size=2)
        model = DeepseekV32Model(config)

        # ep_rank defaults to 0 when distributed is not initialized
        moe_layer = model.layers[1].mlp
        experts = moe_layer.experts

        # With ep_size=2, ep_rank=0, should have experts 0-3
        self.assertEqual(experts.local_expert_start, 0)
        self.assertEqual(experts.local_expert_end, 4)

    def test_ep_disabled_by_default(self):
        """Test EP is disabled (ep_size=1) by default."""
        config = get_moe_test_config()  # No ep_size specified
        model = DeepseekV32Model(config)

        moe_layer = model.layers[1].mlp
        self.assertEqual(moe_layer.ep_size, 1)
        # All experts should be local
        self.assertEqual(moe_layer.experts.local_expert_start, 0)
        self.assertEqual(moe_layer.experts.local_expert_end, 8)


class DeepseekV32DistributedMoETest(unittest.TestCase):
    @patch("transformers.models.deepseek_v32.modeling_deepseek_v32.rotate_activation", side_effect=lambda x: x)
    def test_distributed_moe_all_reduce(self, mock_rotate):
        config = DeepseekV32Config(
            hidden_size=128,
            intermediate_size=256,
            moe_intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=4,
            n_routed_experts=8,
            num_experts_per_tok=2,
            n_group=2,
            topk_group=1,  # Must be less than n_group
            first_k_dense_replace=1,  # Layer 1, 2, 3 will be MoE
            ep_size=2,  # Enable Expert Parallelism with 2 GPUs
        )
        model = DeepseekV32Model(config)

        # Layer 1 should be MoE with EP enabled
        moe_layer = model.layers[1].mlp
        self.assertEqual(moe_layer.ep_size, 2)

        input_ids = torch.LongTensor([[0, 1, 2]])
        
        # Mock distributed
        with patch("torch.distributed.is_initialized", return_value=True), \
             patch("torch.distributed.get_world_size", return_value=2), \
             patch("torch.distributed.all_reduce") as mock_all_reduce:
            
            model(input_ids)
            
            # Check if all_reduce was called
            # We have 3 MoE layers (1, 2, 3), so it should be called at least 3 times
            self.assertTrue(mock_all_reduce.called)
            self.assertGreaterEqual(mock_all_reduce.call_count, 3)

    @patch("transformers.models.deepseek_v32.modeling_deepseek_v32.rotate_activation", side_effect=lambda x: x)
    def test_distributed_moe_disabled_by_default(self, mock_rotate):
        config = DeepseekV32Config(
            hidden_size=128,
            intermediate_size=256,
            moe_intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=4,
            n_routed_experts=8,
            num_experts_per_tok=2,
            n_group=2,
            topk_group=1,  # Must be less than n_group
            first_k_dense_replace=1,
            # ep_size=1 by default (no EP)
        )
        model = DeepseekV32Model(config)
        
        input_ids = torch.LongTensor([[0, 1, 2]])
        
        # Mock distributed to ensure it's not called even if env looks distributed
        with patch("torch.distributed.is_initialized", return_value=True), \
             patch("torch.distributed.get_world_size", return_value=2), \
             patch("torch.distributed.all_reduce") as mock_all_reduce:
            
            model(input_ids)
            
            self.assertFalse(mock_all_reduce.called)

