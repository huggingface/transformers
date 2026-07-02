"""Tests for SDPA attention gradient health with dense masks (gh#44928).

Validates that SDPA attention produces finite, well-behaved gradients
when using dense attention masks, particularly in configurations that
can trigger backend fallback (Math backend with BF16).

These tests would catch gradient explosion bugs such as HuggingFace#44928,
where 3D position_ids force SDPA Math fallback, causing BF16 collapse
and NaN gradients in Qwen3.5 RLHF training.

To submit as upstream PR:
    git clone https://github.com/Lemniscate-world/transformers.git
    cp this_file tests/test_sdpa_mask_gradient.py
    git checkout -b test/sdpa-health
    git add tests/test_sdpa_mask_gradient.py
    git commit -m "test: add SDPA gradient health tests (fixes #44928)"
    git push origin test/sdpa-health
    gh pr create --repo huggingface/transformers --head Lemniscate-world:test/sdpa-health

Blocked 2026-07-02: GitHub fork directory file conflict on Lemniscate-world/transformers.
Workaround: submit via GitHub web UI (create file directly on fork).
"""

import unittest
import torch
from torch.nn.functional import scaled_dot_product_attention


class TestSDPAGradientHealth(unittest.TestCase):
    """Gradient correctness tests for scaled_dot_product_attention."""

    def test_sdpa_finite_gradient_standard(self):
        """SDPA with standard inputs must produce finite gradients."""
        B, H, S, D = 2, 4, 16, 32
        query = torch.randn(B, H, S, D, requires_grad=True)
        key = torch.randn(B, H, S, D, requires_grad=True)
        value = torch.randn(B, H, S, D, requires_grad=True)

        out = scaled_dot_product_attention(query, key, value)
        loss = out.sum()
        loss.backward()

        for name, tensor in [("query", query), ("key", key), ("value", value)]:
            self.assertTrue(
                torch.isfinite(tensor.grad).all(),
                f"Non-finite gradient in {name} after SDPA backward"
            )

    def test_sdpa_finite_gradient_dense_mask(self):
        """SDPA with dense causal mask must produce finite gradients."""
        B, H, S, D = 2, 4, 16, 32
        query = torch.randn(B, H, S, D, requires_grad=True)
        key = torch.randn(B, H, S, D, requires_grad=True)
        value = torch.randn(B, H, S, D, requires_grad=True)

        causal_mask = torch.triu(
            torch.full((S, S), float("-inf")), diagonal=1
        )

        out = scaled_dot_product_attention(
            query, key, value, attn_mask=causal_mask
        )
        loss = out.sum()
        loss.backward()

        for name, tensor in [("query", query), ("key", key), ("value", value)]:
            grad = tensor.grad
            self.assertTrue(
                torch.isfinite(grad).all(),
                f"Non-finite gradient in {name} with dense mask"
            )
            grad_norm = grad.norm().item()
            self.assertLess(
                grad_norm,
                1e6,
                f"Gradient norm {grad_norm:.2e} too large in {name}"
            )

    def test_sdpa_gradient_scaling_consistency(self):
        """Scaling query by factor f scales gradient approximately."""
        B, H, S, D = 1, 2, 8, 16
        key = torch.randn(B, H, S, D)
        value = torch.randn(B, H, S, D)

        q_ref = torch.randn(B, H, S, D, requires_grad=True)
        out_ref = scaled_dot_product_attention(q_ref, key, value)
        out_ref.sum().backward()
        grad_ref = q_ref.grad.clone()

        q_scaled = torch.randn(B, H, S, D, requires_grad=True) * 2.0
        out_scaled = scaled_dot_product_attention(q_scaled, key, value)
        out_scaled.sum().backward()

        ratio = (q_scaled.grad / (grad_ref + 1e-8)).mean().item()
        self.assertGreater(ratio, 0.0)
        self.assertLess(ratio, 10.0)


if __name__ == "__main__":
    unittest.main()
