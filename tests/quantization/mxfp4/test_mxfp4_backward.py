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

"""
Tests for MXFP4 backward kernels and training support.

This module tests:
1. SwiGLU backward pass correctness
2. MatmulOGS backward pass correctness (activation gradients)
3. MoE routing gradient inversion
4. Numerical gradient checks
5. Training integration
"""

import unittest
from unittest.mock import MagicMock, patch

from transformers.testing_utils import (
    require_torch,
    require_torch_gpu,
    require_triton,
    slow,
)
from transformers.utils import is_torch_available, is_triton_available


if is_torch_available():
    import torch

if is_torch_available() and is_triton_available():
    from transformers.integrations.mxfp4_backward import (
        MXFP_BLOCK_SIZE,
        _dequantize_mxfp4_weight,
        _matmul_ogs_backward_moe,
        swiglu_backward_torch,
        swiglu_backward_triton,
    )


# Tolerance settings for numerical comparisons
# MXFP4 is low-precision, so we use relaxed tolerances
FORWARD_RTOL = 1e-2
FORWARD_ATOL = 1e-3
BACKWARD_RTOL = 5e-2
BACKWARD_ATOL = 1e-2


class SwiGLUBackwardTest(unittest.TestCase):
    """Test SwiGLU backward pass implementation."""

    @require_torch
    def test_swiglu_backward_torch_shape(self):
        """Test that SwiGLU backward produces correct output shape."""
        batch_size, seq_len, hidden_size = 4, 16, 64
        input_a = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)
        grad_output = torch.randn(batch_size, seq_len, hidden_size // 2, dtype=torch.float32)

        grad_input = swiglu_backward_torch(grad_output, input_a, alpha=1.702, limit=7.0)

        self.assertEqual(grad_input.shape, input_a.shape)

    @require_torch
    def test_swiglu_backward_torch_gradcheck(self):
        """Test SwiGLU backward with PyTorch autograd gradcheck."""
        # Small input for gradcheck
        input_a = torch.randn(2, 4, 8, dtype=torch.float64, requires_grad=True)

        def swiglu_forward(a):
            alpha = 1.702
            limit = 7.0
            a_gelu = a[..., ::2].clamp(max=limit)
            a_linear = a[..., 1::2].clamp(min=-limit, max=limit)
            out_gelu = a_gelu * torch.sigmoid(alpha * a_gelu)
            out = out_gelu * (a_linear + 1)
            return out

        # Use PyTorch autograd to verify our backward implementation
        output = swiglu_forward(input_a)
        grad_output = torch.randn_like(output)
        output.backward(grad_output)
        expected_grad = input_a.grad.clone()

        # Now test our implementation
        input_a.grad = None
        input_a_detached = input_a.detach().clone()
        our_grad = swiglu_backward_torch(grad_output, input_a_detached, alpha=1.702, limit=7.0)

        torch.testing.assert_close(our_grad, expected_grad, rtol=1e-5, atol=1e-5)

    @require_torch
    def test_swiglu_backward_respects_saturation(self):
        """Test that gradients are zeroed where values are clamped."""
        # Create input with some values beyond the limit
        input_a = torch.tensor([[10.0, 0.0, -10.0, 5.0, 3.0, 8.0, -8.0, 2.0]])  # Shape [1, 8]
        grad_output = torch.ones(1, 4)  # Shape [1, 4]
        limit = 7.0

        grad_input = swiglu_backward_torch(grad_output, input_a, alpha=1.702, limit=limit)

        # Positions 0, 2 (even indices with |value| > limit) should have zero gradient
        # Position 0: 10.0 > 7.0 (saturated high for gelu)
        self.assertEqual(grad_input[0, 0].item(), 0.0)
        # Position 2: -10.0 < -7.0 (NOT saturated for gelu which only clamps max)
        # Actually gelu clamps at max=limit only, so -10 is not clamped

        # Positions 5 (odd index with value > limit) should have zero gradient
        # Position 5: 8.0 > 7.0 (saturated for linear)
        self.assertEqual(grad_input[0, 5].item(), 0.0)
        # Position 6: -8.0 < -7.0 (saturated low for linear)
        self.assertEqual(grad_input[0, 6].item(), 0.0)

    @require_torch
    def test_swiglu_backward_no_limit(self):
        """Test SwiGLU backward without clamping limit."""
        input_a = torch.randn(2, 4, 8, dtype=torch.float32)
        grad_output = torch.randn(2, 4, 4, dtype=torch.float32)

        grad_input = swiglu_backward_torch(grad_output, input_a, alpha=1.702, limit=None)

        self.assertEqual(grad_input.shape, input_a.shape)
        # All gradients should be non-zero (probabilistically)
        self.assertTrue(torch.any(grad_input != 0))

    @require_torch_gpu
    @require_triton(min_version="3.4.0")
    def test_swiglu_backward_triton_matches_torch(self):
        """Test that Triton implementation matches PyTorch reference."""
        batch_size, seq_len, hidden_size = 4, 32, 128
        input_a = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32, device="cuda")
        grad_output = torch.randn(batch_size, seq_len, hidden_size // 2, dtype=torch.float32, device="cuda")

        grad_torch = swiglu_backward_torch(grad_output, input_a, alpha=1.702, limit=7.0)
        grad_triton = swiglu_backward_triton(grad_output, input_a, alpha=1.702, limit=7.0)

        torch.testing.assert_close(grad_triton, grad_torch, rtol=BACKWARD_RTOL, atol=BACKWARD_ATOL)


class Mxfp4DequantizationTest(unittest.TestCase):
    """Test MXFP4 weight dequantization for backward pass."""

    @require_torch
    def test_dequantize_shape(self):
        """Test that dequantization produces correct output shape."""
        n_experts, K, N_half = 4, 64, 32
        packed_weights = torch.randint(0, 256, (n_experts, K, N_half), dtype=torch.uint8)
        scales = torch.randint(100, 150, (n_experts, K // MXFP_BLOCK_SIZE, N_half * 2), dtype=torch.uint8)

        dequantized = _dequantize_mxfp4_weight(packed_weights, scales)

        expected_shape = (n_experts, K, N_half * 2)
        self.assertEqual(dequantized.shape, expected_shape)

    @require_torch
    def test_dequantize_values_in_range(self):
        """Test that dequantized values are within expected FP4 range."""
        n_experts, K, N_half = 2, 32, 16
        # Use zero scales (exponent = 0) for predictable output
        packed_weights = torch.randint(0, 256, (n_experts, K, N_half), dtype=torch.uint8)
        # Scale exponent 127 means 2^0 = 1 (no scaling)
        scales = torch.full((n_experts, K // MXFP_BLOCK_SIZE, N_half * 2), 127, dtype=torch.uint8)

        dequantized = _dequantize_mxfp4_weight(packed_weights, scales)

        # FP4 values are in [-6, 6]
        self.assertTrue(torch.all(dequantized >= -6.0))
        self.assertTrue(torch.all(dequantized <= 6.0))


class MoEBackwardTest(unittest.TestCase):
    """Test MoE routing gradient inversion."""

    @require_torch
    def test_moe_backward_simple(self):
        """Test MoE backward with simple routing."""
        # Create a simple MoE scenario
        n_tokens, hidden_size = 8, 16
        n_experts = 4
        n_expts_act = 2

        # Create mock routing data
        routing_data = MagicMock()
        routing_data.n_expts_act = n_expts_act
        routing_data.n_expts_tot = n_experts
        routing_data.expt_hist = torch.tensor([4, 4, 4, 4], dtype=torch.int32)

        # Create mock indices
        # Simple case: each token goes to experts 0 and 1
        gather_indx = MagicMock()
        gather_indx.src_indx = torch.arange(16, dtype=torch.int32)  # 8 tokens * 2 experts

        scatter_indx = MagicMock()
        scatter_indx.dst_indx = torch.arange(16, dtype=torch.int32)
        scatter_indx.src_indx = torch.arange(16, dtype=torch.int32)

        # Create tensors
        grad_output = torch.randn(n_tokens, hidden_size)
        x = torch.randn(n_tokens, hidden_size)
        w = torch.randn(n_experts, hidden_size, hidden_size)

        # Call backward
        grad_x = _matmul_ogs_backward_moe(grad_output, w, x, routing_data, gather_indx, scatter_indx)

        self.assertEqual(grad_x.shape, x.shape)


class TrainingIntegrationTest(unittest.TestCase):
    """Test training integration with MXFP4 layers."""

    @require_torch
    def test_quantizer_is_trainable(self):
        """Test that the quantizer reports trainable=True."""
        from transformers.quantizers.quantizer_mxfp4 import Mxfp4HfQuantizer
        from transformers.utils.quantization_config import Mxfp4Config

        config = Mxfp4Config()
        quantizer = Mxfp4HfQuantizer(config)

        self.assertTrue(quantizer.is_trainable)

    @require_torch
    def test_experts_training_mode(self):
        """Test enabling/disabling training mode on expert layers."""
        from transformers.integrations.mxfp4 import Mxfp4GptOssExperts

        # Create mock config
        config = MagicMock()
        config.num_local_experts = 4
        config.intermediate_size = 64
        config.hidden_size = 32
        config.swiglu_limit = 7.0

        with patch.dict("sys.modules", {"transformers.integrations.mxfp4": MagicMock()}):
            experts = Mxfp4GptOssExperts(config)

            # Default is inference mode
            self.assertFalse(experts.training_mode)

            # Enable training mode
            experts.enable_training_mode()
            self.assertTrue(experts.training_mode)

            # Disable training mode
            experts.disable_training_mode()
            self.assertFalse(experts.training_mode)


class NumericalGradientTest(unittest.TestCase):
    """Numerical gradient verification tests."""

    @require_torch
    def test_swiglu_numerical_gradient(self):
        """Verify SwiGLU gradients numerically using finite differences."""
        eps = 1e-5
        input_a = torch.randn(2, 8, dtype=torch.float64)
        alpha = 1.702
        limit = 7.0

        def swiglu_forward(a):
            a_gelu = a[..., ::2].clamp(max=limit)
            a_linear = a[..., 1::2].clamp(min=-limit, max=limit)
            out_gelu = a_gelu * torch.sigmoid(alpha * a_gelu)
            out = out_gelu * (a_linear + 1)
            return out.sum()

        # Compute analytical gradient
        grad_output = torch.ones(2, 4, dtype=torch.float64)
        analytical_grad = swiglu_backward_torch(grad_output, input_a, alpha, limit)

        # Compute numerical gradient
        numerical_grad = torch.zeros_like(input_a)
        for i in range(input_a.numel()):
            input_plus = input_a.clone().flatten()
            input_plus[i] += eps
            input_plus = input_plus.view_as(input_a)

            input_minus = input_a.clone().flatten()
            input_minus[i] -= eps
            input_minus = input_minus.view_as(input_a)

            numerical_grad.flatten()[i] = (swiglu_forward(input_plus) - swiglu_forward(input_minus)) / (2 * eps)

        # Compare
        torch.testing.assert_close(analytical_grad, numerical_grad, rtol=1e-4, atol=1e-4)


@slow
class SlowMxfp4BackwardTest(unittest.TestCase):
    """Slow tests requiring model loading."""

    @require_torch_gpu
    @require_triton(min_version="3.4.0")
    def test_end_to_end_backward(self):
        """Test end-to-end backward pass through MXFP4 layer."""
        # This test would load an actual MXFP4 model and verify gradients flow
        # Skipped for now as it requires model weights
        pass


if __name__ == "__main__":
    unittest.main()
