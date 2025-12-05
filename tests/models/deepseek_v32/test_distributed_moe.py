import unittest
from unittest.mock import MagicMock, patch

import torch

from transformers import DeepseekV32Config, DeepseekV32Model


class DeepseekV32DistributedMoETest(unittest.TestCase):
    @patch("transformers.models.deepseek_v32.modeling_deepseek_v32.hadamard_transform_activation", side_effect=lambda x: x)
    def test_distributed_moe_all_reduce(self, mock_hadamard):
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
            use_distributed_moe=True,
        )
        model = DeepseekV32Model(config)
        
        # Layer 1 should be MoE
        moe_layer = model.layers[1].mlp
        self.assertTrue(moe_layer.use_distributed_moe)

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

    @patch("transformers.models.deepseek_v32.modeling_deepseek_v32.hadamard_transform_activation", side_effect=lambda x: x)
    def test_distributed_moe_disabled_by_default(self, mock_hadamard):
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
            # use_distributed_moe=False by default
        )
        model = DeepseekV32Model(config)
        
        input_ids = torch.LongTensor([[0, 1, 2]])
        
        # Mock distributed to ensure it's not called even if env looks distributed
        with patch("torch.distributed.is_initialized", return_value=True), \
             patch("torch.distributed.get_world_size", return_value=2), \
             patch("torch.distributed.all_reduce") as mock_all_reduce:
            
            model(input_ids)
            
            self.assertFalse(mock_all_reduce.called)

