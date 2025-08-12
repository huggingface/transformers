# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Testing torch.export compatibility for Mixtral models."""

import unittest

import torch
import torch.export as te

from transformers import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
from transformers.testing_utils import require_torch, torch_device


@require_torch
class MixtralTorchExportTest(unittest.TestCase):
    """Test torch.export compatibility for Mixtral MoE components."""

    def setUp(self):
        """Set up test configuration."""
        self.config = MixtralConfig(
            hidden_size=128,
            intermediate_size=256,
            num_local_experts=8,
            num_experts_per_tok=2,
            router_jitter_noise=0.0,
        )

    def test_moe_block_torch_export(self):
        """Test that MixtralSparseMoeBlock can be exported with torch.export."""
        # Create MoE block
        moe_block = MixtralSparseMoeBlock(self.config)
        moe_block.eval()
        
        # Move to meta device for export testing
        moe_block = moe_block.to("meta")
        
        # Create test input
        batch_size, seq_len = 2, 8
        hidden_states = torch.randn(
            batch_size, seq_len, self.config.hidden_size, 
            device="meta"
        )
        
        # Test torch.export - should not raise GuardOnDataDependentSymNode error
        try:
            exported_program = te.export(
                moe_block,
                args=(hidden_states,),
                kwargs={},
                strict=False
            )
            # If export succeeds, the test passes
            self.assertIsNotNone(exported_program)
        except Exception as e:
            # Check if it's the specific error we're trying to avoid
            error_msg = str(e)
            if "GuardOnDataDependentSymNode" in error_msg or "nonzero" in error_msg.lower():
                self.fail(
                    f"torch.export failed with data-dependent operation error: {error_msg}\n"
                    "This suggests the .nonzero() fix is not working properly."
                )
            else:
                # Re-raise other unexpected errors
                raise

    def test_moe_block_functionality(self):
        """Test that MoE block maintains correct functionality after the fix."""
        # Create MoE block
        moe_block = MixtralSparseMoeBlock(self.config)
        moe_block.eval()
        
        # Create test input
        batch_size, seq_len = 2, 4
        hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_size)
        
        # Forward pass
        with torch.no_grad():
            output, router_logits = moe_block(hidden_states)
        
        # Verify output shapes
        self.assertEqual(output.shape, hidden_states.shape)
        self.assertEqual(
            router_logits.shape, 
            (batch_size * seq_len, self.config.num_local_experts)
        )
        
        # Verify that outputs are not all zeros (computation happened)
        self.assertFalse(torch.allclose(output, torch.zeros_like(output)))
        
        # Test with different input to ensure different outputs
        hidden_states2 = torch.randn(batch_size, seq_len, self.config.hidden_size)
        with torch.no_grad():
            output2, _ = moe_block(hidden_states2)
        
        # Outputs should be different for different inputs
        self.assertFalse(torch.allclose(output, output2))

    def test_moe_block_export_with_different_configs(self):
        """Test torch.export with various expert configurations."""
        test_configs = [
            # (num_experts, top_k)
            (4, 2),
            (8, 2),
            (16, 2),
            (8, 4),
        ]
        
        for num_experts, top_k in test_configs:
            with self.subTest(num_experts=num_experts, top_k=top_k):
                config = MixtralConfig(
                    hidden_size=64,
                    intermediate_size=128,
                    num_local_experts=num_experts,
                    num_experts_per_tok=top_k,
                    router_jitter_noise=0.0,
                )
                
                moe_block = MixtralSparseMoeBlock(config)
                moe_block.eval()
                moe_block = moe_block.to("meta")
                
                hidden_states = torch.randn(1, 4, config.hidden_size, device="meta")
                
                # Should export without errors
                try:
                    exported_program = te.export(
                        moe_block,
                        args=(hidden_states,),
                        kwargs={},
                        strict=False
                    )
                    self.assertIsNotNone(exported_program)
                except Exception as e:
                    if "GuardOnDataDependentSymNode" in str(e):
                        self.fail(f"Export failed for config ({num_experts}, {top_k}): {e}")
                    else:
                        raise


if __name__ == "__main__":
    unittest.main()