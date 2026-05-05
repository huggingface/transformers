import unittest
from unittest.mock import patch

from transformers.testing_utils import require_torch, require_torch_gpu


@require_torch
class UseGqaInSdpaLogicTest(unittest.TestCase):
    """Pure-logic gating tests for `use_gqa_in_sdpa`.

    These tests mock the torch SDPA-eligibility surface so they run on any
    backend (CPU CI inclusive). They cover the failure mode that motivated
    the change: `enable_gqa=True` must only be returned when FA can actually
    be selected for these inputs.
    """

    def setUp(self):
        from types import SimpleNamespace

        import torch

        from transformers.integrations import sdpa_attention as mod

        self.mod = mod
        self.torch = torch
        # Real CPU tensors for tests that must hit the CPU short-circuit branch.
        self.cpu_q = torch.randn(1, 8, 16, 128, dtype=torch.bfloat16)
        self.cpu_k = torch.randn(1, 2, 16, 128, dtype=torch.bfloat16)
        self.cpu_v = torch.randn(1, 2, 16, 128, dtype=torch.bfloat16)
        # Fake "cuda" tensors for tests that should exercise the FA-probe path
        # without actually requiring CUDA. The probe is fully mocked, so the
        # function never indexes shape/dtype/strides — only `.device.type`.
        cuda_dev = SimpleNamespace(type="cuda")
        self.cuda_q = SimpleNamespace(device=cuda_dev)
        self.cuda_k = SimpleNamespace(device=cuda_dev)
        self.cuda_v = SimpleNamespace(device=cuda_dev)

    def _call_cpu(self, attn_mask=None, is_causal=True):
        return self.mod.use_gqa_in_sdpa(attn_mask, self.cpu_q, self.cpu_k, self.cpu_v, is_causal, 0.0)

    def _call_cuda(self, attn_mask=None, is_causal=True):
        return self.mod.use_gqa_in_sdpa(attn_mask, self.cuda_q, self.cuda_k, self.cuda_v, is_causal, 0.0)

    def test_attention_mask_rejects(self):
        # Mask -> dispatch falls back to MATH; never use enable_gqa.
        mask = self.torch.zeros(1, 1, 16, 16)
        with patch.object(self.mod, "_is_torch_greater_or_equal_than_2_5", True):
            self.assertFalse(self._call_cuda(attn_mask=mask))

    def test_old_torch_rejects(self):
        with patch.object(self.mod, "_is_torch_greater_or_equal_than_2_5", False):
            self.assertFalse(self._call_cuda())

    def test_cpu_inputs_keep_bc_without_probing(self):
        # CPU SDPA honors `enable_gqa=True` natively. Preserve pre-PR behavior
        # for non-CUDA / non-XPU devices: return True without touching the
        # CUDA-only `torch.backends.cuda.*` probe surface.
        with (
            patch.object(self.mod, "_is_torch_greater_or_equal_than_2_5", True),
            patch.object(self.mod, "_is_torch_xpu_available", False),
            patch.object(self.torch.backends.cuda, "flash_sdp_enabled") as fse,
            patch.object(self.torch.backends.cuda, "can_use_flash_attention") as cufa,
        ):
            self.assertTrue(self._call_cpu())
            fse.assert_not_called()
            cufa.assert_not_called()

    def test_flash_disabled_rejects(self):
        # User wrapped the call in sdpa_kernel([EFFICIENT]) (or globally
        # disabled FA). enable_gqa=True would silently misroute.
        with (
            patch.object(self.mod, "_is_torch_greater_or_equal_than_2_5", True),
            patch.object(self.mod, "_is_torch_xpu_available", False),
            patch.object(self.torch.backends.cuda, "flash_sdp_enabled", return_value=False),
        ):
            self.assertFalse(self._call_cuda())

    def test_flash_enabled_but_inputs_ineligible_rejects(self):
        # FA flag is on, but pytorch says FA can't actually run these inputs
        # (e.g. head_dim>256, fp32, sm<80). Must fall back to repeat_kv.
        # SDPAParams must be mocked too — its C++ binding rejects our fake
        # SimpleNamespace tensors, which would otherwise raise and route into
        # the defensive `except` branch.
        with (
            patch.object(self.mod, "_is_torch_greater_or_equal_than_2_5", True),
            patch.object(self.mod, "_is_torch_xpu_available", False),
            patch.object(self.torch.backends.cuda, "flash_sdp_enabled", return_value=True),
            patch.object(self.torch.backends.cuda, "SDPAParams", return_value=object()),
            patch.object(self.torch.backends.cuda, "can_use_flash_attention", return_value=False),
        ):
            self.assertFalse(self._call_cuda())

    def test_flash_eligible_returns_true(self):
        # Happy path: regression guard for the existing default-enabled behavior.
        with (
            patch.object(self.mod, "_is_torch_greater_or_equal_than_2_5", True),
            patch.object(self.mod, "_is_torch_xpu_available", False),
            patch.object(self.torch.backends.cuda, "flash_sdp_enabled", return_value=True),
            patch.object(self.torch.backends.cuda, "SDPAParams", return_value=object()),
            patch.object(self.torch.backends.cuda, "can_use_flash_attention", return_value=True),
        ):
            self.assertTrue(self._call_cuda())

    def test_xpu_keeps_existing_path(self):
        # XPU branch is unchanged: torch>=2.8 -> True regardless of probe.
        with (
            patch.object(self.mod, "_is_torch_xpu_available", True),
            patch.object(self.mod, "_is_torch_greater_or_equal_than_2_8", True),
        ):
            self.assertTrue(self._call_cpu())
        with (
            patch.object(self.mod, "_is_torch_xpu_available", True),
            patch.object(self.mod, "_is_torch_greater_or_equal_than_2_8", False),
        ):
            self.assertFalse(self._call_cpu())

    def test_probe_exception_is_swallowed(self):
        # SDPAParams may not exist on very old torch; defensive fallback.
        with (
            patch.object(self.mod, "_is_torch_greater_or_equal_than_2_5", True),
            patch.object(self.mod, "_is_torch_xpu_available", False),
            patch.object(self.torch.backends.cuda, "flash_sdp_enabled", return_value=True),
            patch.object(self.torch.backends.cuda, "SDPAParams", side_effect=RuntimeError("boom")),
        ):
            self.assertFalse(self._call_cuda())


@require_torch_gpu
class UseGqaInSdpaCudaTest(unittest.TestCase):
    """End-to-end probe on real CUDA inputs.

    Exercises the actual `torch.backends.cuda.can_use_flash_attention` path —
    no mocks. Each case is a shape/dtype/context that must produce a specific
    GQA decision under the hood.
    """

    def _probe(self, query, key, value, attention_mask=None, is_causal=True, dropout=0.0):
        from transformers.integrations.sdpa_attention import use_gqa_in_sdpa

        return use_gqa_in_sdpa(attention_mask, query, key, value, is_causal, dropout)

    def _gqa_tensors(self, head_dim=128, dtype=None):
        import torch

        if dtype is None:
            dtype = torch.bfloat16
        B, T, Hq, Hkv = 1, 32, 8, 2
        q = torch.randn(B, Hq, T, head_dim, dtype=dtype, device="cuda")
        k = torch.randn(B, Hkv, T, head_dim, dtype=dtype, device="cuda")
        v = torch.randn(B, Hkv, T, head_dim, dtype=dtype, device="cuda")
        return q, k, v

    def test_default_context_fa_friendly_shape_returns_true(self):
        q, k, v = self._gqa_tensors(head_dim=128)
        self.assertTrue(self._probe(q, k, v))

    def test_efficient_only_context_returns_false(self):
        # Reproduces the user-side `sdpa_kernel([EFFICIENT_ATTENTION])` case.
        # Without the fix, transformers would still pass enable_gqa=True and
        # the dispatcher would fall through.
        from torch.nn.attention import SDPBackend, sdpa_kernel

        q, k, v = self._gqa_tensors(head_dim=128)
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            self.assertFalse(self._probe(q, k, v))

    def test_math_only_context_returns_false(self):
        from torch.nn.attention import SDPBackend, sdpa_kernel

        q, k, v = self._gqa_tensors(head_dim=128)
        with sdpa_kernel([SDPBackend.MATH]):
            self.assertFalse(self._probe(q, k, v))

    def test_head_dim_too_large_returns_false(self):
        # Gemma 4: head_dim=320 > FA's library cap. EFFICIENT is the only
        # viable backend, so enable_gqa=True must not be set.
        q, k, v = self._gqa_tensors(head_dim=320)
        self.assertFalse(self._probe(q, k, v))

    def test_fp32_returns_false(self):
        import torch

        q, k, v = self._gqa_tensors(head_dim=128, dtype=torch.float32)
        self.assertFalse(self._probe(q, k, v))


if __name__ == "__main__":
    unittest.main()
