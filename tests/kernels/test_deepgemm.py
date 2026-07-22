# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Integration tests for the DeepGEMM linear / experts dispatches.

Two layers, both mocking only what needs a Hopper/Blackwell GPU + a JIT CUDA toolkit +
`kernels-community/deep-gemm`:

* `DeepGemmLoaderTest` mocks the environment probes and `lazy_load_kernel` to drive
  `load_deepgemm_kernel` with no GPU/toolkit — asserting it gates correctly (raises when a
  precondition is unmet, returns the bundle otherwise, honours the per-call `requires_sm100` gate)
  and stays torch-compile safe. Copied verbatim from `tests/test_kernel_loaders.py`.
* `DeepGemmForwardTest` mocks only the loaded-kernel ops (the `DeepGEMM` bundle callables:
  `per_token_cast_to_fp8`, `fp8_fp4_matmul`, the grouped GEMMs, the Mega MoE ops) to return
  correctly shaped tensors, and runs the REAL public forwards so their scale-factor coercion /
  TMA-aligned grouped layout / weight-selection / dtype-cast glue executes for real (device-agnostic,
  so it runs on CPU); it then asserts the tensors handed to the kernel are what the kernel expects
  (packed int32 UE8M0 SFs, int32 grouped layout, `(qtensor, sf)` operand tuples, recipes, transformed
  Mega MoE weights, ...).

Arch-gated paths are exercised by faking the device capability (`is_sm100()` reads it): the
SF-packing / TMA-alignment / psum-layout code it selects is pure tensor arithmetic, so mocking the
capability to SM100 drives the Blackwell paths on any device.
"""

import contextlib
import inspect
import os
import tempfile
import types
import unittest
from unittest import mock

import torch
from parameterized import parameterized
from test_utils import make_experts, make_fp8_experts

import transformers.integrations.deepgemm as dg
from transformers.integrations.deepgemm import (
    deepgemm_bf16_experts_forward,
    deepgemm_fp8_fp4_experts_forward,
    deepgemm_fp8_fp4_linear,
    deepgemm_fp8_fp4_megamoe_experts_forward,
)
from transformers.testing_utils import (
    require_torch,
    require_torch_greater_or_equal,
    torch_device,
)


def _add_one(x, *args, **kwargs):
    return x + 1


# ── Fake kernels (a "good" one exposing every symbol the loader resolves, plus a variant missing one) ──


class _FakeDeepGemmKernel:
    fp8_fp4_gemm_nt = staticmethod(_add_one)
    m_grouped_fp8_fp4_gemm_nt_contiguous = staticmethod(_add_one)
    m_grouped_fp8_fp4_gemm_nn_contiguous = staticmethod(_add_one)
    m_grouped_bf16_gemm_nt_contiguous = staticmethod(_add_one)
    m_grouped_bf16_gemm_nn_contiguous = staticmethod(_add_one)
    per_token_cast_to_fp8 = staticmethod(_add_one)
    transform_sf_into_required_layout = staticmethod(_add_one)
    transform_weights_for_mega_moe = staticmethod(_add_one)
    get_symm_buffer_for_mega_moe = staticmethod(_add_one)
    fp8_fp4_mega_moe = staticmethod(_add_one)

    @staticmethod
    def get_mk_alignment_for_contiguous_layout():
        return 128


class _DeepGemmKernelMissingSymbol(_FakeDeepGemmKernel):
    fp8_fp4_mega_moe = None  # a resolved symbol that comes back as None


@require_torch
class DeepGemmLoaderTest(unittest.TestCase):
    def setUp(self):
        dg._DEEPGEMM = None
        self.addCleanup(setattr, dg, "_DEEPGEMM", None)

    def _env(
        self,
        *,
        kernels_available=True,
        cuda_available=True,
        capability=(9, 0),
        toolkit_present=True,
        nvcc_present=True,
        nvcc_version=(12, 9),
        kernel=_FakeDeepGemmKernel,
    ):
        stack = contextlib.ExitStack()
        # a REAL fake toolkit on disk (bin/nvcc touched iff nvcc_present) — faking the
        # filesystem state instead of file APIs keeps global machinery untouched
        # (patching `os.path.isfile` corrupts e.g. setuptools' distutils import shim
        # under torch.compile).
        cuda_home = None
        if toolkit_present:
            cuda_home = stack.enter_context(tempfile.TemporaryDirectory())
            if nvcc_present:
                os.makedirs(os.path.join(cuda_home, "bin"), exist_ok=True)
                open(os.path.join(cuda_home, "bin", "nvcc"), "w").close()
        stack.enter_context(mock.patch.object(dg, "is_kernels_available", return_value=kernels_available))
        # Fake a "CUDA + `capability`" environment for the loader *only* (scoped to its call stack): the
        # loader's availability/arch gate passes, while torch.compile / inductor still see the real
        # device — so the compile-safety test runs under the default (inductor) backend on any host, not
        # just SM90+. The real fallbacks are lazy: only non-loader callers reach them (inductor, under
        # compile), never the CPU-only gating tests (where querying real CUDA would raise).
        real_is_available = torch.cuda.is_available
        real_capability = torch.cuda.get_device_capability

        def _in_loader():
            # The loader's only capability query happens inside `is_deepgemm_loadable` (which the gating
            # tests also call directly), so fake the device whenever that frame is on the stack. Inductor's
            # own capability query (under compile, outside that frame) still reads the real device.
            return any(f.function == "is_deepgemm_loadable" for f in inspect.stack())

        stack.enter_context(
            mock.patch.object(
                torch.cuda,
                "is_available",
                side_effect=lambda *a, **k: cuda_available if _in_loader() else real_is_available(),
            )
        )
        stack.enter_context(
            mock.patch.object(
                torch.cuda,
                "get_device_capability",
                side_effect=lambda *a, **k: capability if _in_loader() else real_capability(),
            )
        )
        stack.enter_context(mock.patch.object(dg, "_get_cuda_home", return_value=cuda_home))
        stack.enter_context(mock.patch.object(dg, "_get_nvcc_version", return_value=nvcc_version))
        stack.enter_context(mock.patch.object(dg, "lazy_load_kernel", return_value=kernel))
        return stack

    def test_loads_when_environment_is_valid(self):
        with self._env():
            bundle = dg.load_deepgemm_kernel()
        self.assertIsInstance(bundle, dg.DeepGEMM)
        self.assertEqual(bundle.m_alignment, 128)

    @parameterized.expand(
        [
            ("valid_env", {}, None),
            ("no_kernels", {"kernels_available": False}, "kernel unavailable"),
            ("no_cuda", {"cuda_available": False}, "requires CUDA"),
            ("bad_arch", {"capability": (8, 0)}, "requires Hopper"),
            ("no_toolkit", {"toolkit_present": False}, "needs a CUDA toolkit"),
            ("no_nvcc", {"nvcc_present": False}, "compiles with nvcc"),
            ("unreadable_nvcc", {"nvcc_version": None}, "could not read its CUDA version"),
            ("old_nvcc", {"nvcc_version": (12, 0)}, "too old"),
        ]
    )
    def test_is_deepgemm_loadable(self, _name, env_kwargs, pattern):
        # The single gating source: valid env -> True; each unmet precondition -> False, or the specific
        # `ImportError` when `raise_error=True` (what the loader uses).
        with self._env(**env_kwargs):
            self.assertEqual(dg.is_deepgemm_loadable(), pattern is None)
        if pattern is not None:
            with self._env(**env_kwargs), self.assertRaisesRegex(ImportError, pattern):
                dg.is_deepgemm_loadable(raise_error=True)

    @parameterized.expand(
        [
            ("unloadable_env", {"cuda_available": False}, "requires CUDA"),
            ("kernel_load_fails", {"kernel": None}, "Failed to load"),
            ("missing_symbols", {"kernel": _DeepGemmKernelMissingSymbol}, "missing required symbols"),
        ]
    )
    def test_loader_raises(self, _name, env_kwargs, pattern):
        # The loader delegates gating to `is_deepgemm_loadable(raise_error=True)`, then resolves symbols;
        # confirm each failure surfaces through `load_deepgemm_kernel`.
        with self._env(**env_kwargs), self.assertRaisesRegex(ImportError, pattern):
            dg.load_deepgemm_kernel()

    def test_loader_is_compile_safe(self):
        # Cold path: the compiled call is first to load, so the opaque loader node runs its full body
        # under compile and must return None, never the bundle (`Unsupported: torch.* op returned
        # non-Tensor`). Default (inductor) backend; `_env` fakes CUDA+SM90 only for the loader so
        # torch.compile sees the real device — no GPU required.
        with self._env():
            torch.compiler.reset()

            @torch.compile(fullgraph=True)
            def run(x):
                return dg.load_deepgemm_kernel().per_token_cast_to_fp8(x)

            out = run(torch.zeros(3, device=torch_device))
        self.assertTrue(torch.equal(out, torch.ones(3, device=torch_device)))

    def test_loader_is_compile_safe_when_warm(self):
        # Warm path (production order: eager warmup, then compile). The loader runs its arch/env checks
        # every call, then hits the cache short-circuit — which must also return None, not the bundle.
        with self._env():
            dg.load_deepgemm_kernel()
            torch.compiler.reset()

            @torch.compile(fullgraph=True)
            def run(x):
                return dg.load_deepgemm_kernel().per_token_cast_to_fp8(x)

            out = run(torch.zeros(3, device=torch_device))
        self.assertTrue(torch.equal(out, torch.ones(3, device=torch_device)))


# ── Capturing fake DeepGEMM bundle ─────────────────────────────────────────────
#
# Every op records its call and returns a rightly-shaped output; the matmul-family ops zero their
# provided output buffer in place (the real kernels write it in place) so downstream reductions stay
# finite. Scale tensors follow the real recipe dtype: UE8M0 (`float8_e8m0fnu`) when `use_ue8m0`, else
# `float32` — so `_coerce_sf_for_kernel`'s dtype-driven branches execute for real.


def _make_bundle(captured):
    def per_token_cast_to_fp8(x, *, use_ue8m0=False, gran_k=128, use_packed_ue8m0=False):
        cols = -(-x.size(-1) // gran_k)
        sf_dtype = torch.float8_e8m0fnu if use_ue8m0 else torch.float32
        sf = torch.ones(x.size(0), cols, dtype=torch.float32, device=x.device).to(sf_dtype)
        captured.setdefault("per_token_cast", []).append(
            {"x": x, "use_ue8m0": use_ue8m0, "gran_k": gran_k, "use_packed_ue8m0": use_packed_ue8m0}
        )
        return x.to(torch.float8_e4m3fn), sf

    def fp8_fp4_matmul(lhs, rhs, out, *, recipe=None):
        captured.setdefault("matmul", []).append({"lhs": lhs, "rhs": rhs, "out": out, "recipe": recipe})
        out.zero_()

    def _grouped(name):
        def op(lhs, rhs, out, grouped_layout, *, recipe=None, use_psum_layout=False):
            captured.setdefault("grouped", []).append(
                {
                    "name": name,
                    "lhs": lhs,
                    "rhs": rhs,
                    "out": out,
                    "grouped_layout": grouped_layout,
                    "recipe": recipe,
                    "use_psum_layout": use_psum_layout,
                }
            )
            out.zero_()

        return op

    def transform_sf_into_required_layout(sf, dim_a, dim_b, *, recipe, num_groups):
        captured.setdefault("transform_sf", []).append(
            {"sf": sf, "dim_a": dim_a, "dim_b": dim_b, "recipe": recipe, "num_groups": num_groups}
        )
        return sf

    def transform_weights_for_mega_moe(gate_up_pair, down_pair):
        captured["transform_weights_in"] = (gate_up_pair, down_pair)
        # Preserve the `[E_local, 2*I, *]` leading dims the forward reads back; tag the payloads so we
        # can prove the module parameters were overwritten with exactly these transformed tensors.
        gate_up_w, gate_up_sf = gate_up_pair
        down_w, down_sf = down_pair
        new_gate_up = torch.full_like(gate_up_w, 7)
        new_down = torch.full_like(down_w, 9)
        out = ((new_gate_up, gate_up_sf), (new_down, down_sf))
        captured["transform_weights_out"] = out
        return out

    def get_symm_buffer_for_mega_moe(
        process_group, *, hidden, num_topk, num_experts, num_max_tokens_per_rank, intermediate_hidden
    ):
        captured["symm_buffer_kwargs"] = {
            "process_group": process_group,
            "hidden": hidden,
            "num_topk": num_topk,
            "num_experts": num_experts,
            "num_max_tokens_per_rank": num_max_tokens_per_rank,
            "intermediate_hidden": intermediate_hidden,
        }
        buf = types.SimpleNamespace(
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            x=torch.zeros(num_max_tokens_per_rank, hidden, dtype=torch.float8_e4m3fn, device=torch_device),
            x_sf=torch.zeros(num_max_tokens_per_rank, -(-hidden // 32), dtype=torch.float32, device=torch_device),
            topk_idx=torch.zeros(num_max_tokens_per_rank, num_topk, dtype=torch.long, device=torch_device),
            topk_weights=torch.zeros(num_max_tokens_per_rank, num_topk, dtype=torch.float32, device=torch_device),
        )
        captured["symm_buffer"] = buf
        return buf

    def fp8_fp4_mega_moe(y, gate_up_pair, down_pair, symm_buffer, *, activation_clamp=None):
        captured["mega_moe"] = {
            "y": y,
            "gate_up_pair": gate_up_pair,
            "down_pair": down_pair,
            "symm_buffer": symm_buffer,
            "activation_clamp": activation_clamp,
        }
        y.zero_()

    return dg.DeepGEMM(
        fp8_fp4_matmul=fp8_fp4_matmul,
        grouped_fp8_fp4_matmul_nt=_grouped("fp8_fp4_nt"),
        grouped_fp8_fp4_matmul_nn=_grouped("fp8_fp4_nn"),
        grouped_bf16_matmul_nt=_grouped("bf16_nt"),
        grouped_bf16_matmul_nn=_grouped("bf16_nn"),
        per_token_cast_to_fp8=per_token_cast_to_fp8,
        transform_sf_into_required_layout=transform_sf_into_required_layout,
        transform_weights_for_mega_moe=transform_weights_for_mega_moe,
        get_symm_buffer_for_mega_moe=get_symm_buffer_for_mega_moe,
        fp8_fp4_mega_moe=fp8_fp4_mega_moe,
        m_alignment=128,
    )


@require_torch
class DeepGemmForwardTest(unittest.TestCase):
    """Drives the real public forwards with only the loaded-kernel ops mocked."""

    def setUp(self):
        dg._DEEPGEMM = None
        self.addCleanup(setattr, dg, "_DEEPGEMM", None)

    @contextlib.contextmanager
    def _bundle(self, *, is_sm100):
        captured = {}
        bundle = _make_bundle(captured)
        # The forwards read the arch via `is_sm100()` (which queries `get_device_capability`), so fake the
        # device to the requested arch: lets SM100 dispatch/packing run on this SM80 box and drives the
        # Hopper-rejection guards. `[0]` is all `is_sm100()` reads.
        capability = (10, 0) if is_sm100 else (9, 0)
        with (
            mock.patch.object(dg, "load_deepgemm_kernel", return_value=bundle),
            mock.patch.object(torch.cuda, "get_device_capability", return_value=capability),
        ):
            yield captured

    # ── deepgemm_fp8_fp4_linear ────────────────────────────────────────────────

    def test_linear_fp8_sm90_kernel_inputs(self):
        # FP8 weights + float32 block SF on SM90: recipe stays None, SFs are handed over row-major
        # float32 (SM90 dispatch transforms SFA itself and only checks SFB — see `_coerce_sf_for_kernel`).
        input = torch.randn(2, 3, 128, dtype=torch.bfloat16, device=torch_device)  # 3D -> flatten to (6, 128)
        weight = torch.randn(256, 128, device=torch_device).to(torch.float8_e4m3fn)
        weight_scale = torch.ones(2, 1, dtype=torch.float32, device=torch_device)  # (N/128, K/128)
        with self._bundle(is_sm100=False) as captured:
            out = deepgemm_fp8_fp4_linear(input, weight, weight_scale, block_size=(128, 128))

        cast = captured["per_token_cast"][0]
        self.assertFalse(cast["use_ue8m0"])
        self.assertEqual(cast["gran_k"], 128)
        self.assertEqual(cast["x"].shape, (6, 128))  # flattened to 2D

        (matmul,) = captured["matmul"]
        self.assertIsNone(matmul["recipe"])  # float-SF path leaves the recipe unset
        (_, act_sf), (w, w_sf) = matmul["lhs"], matmul["rhs"]
        self.assertIs(w, weight)
        # SM90 float32 SFs pass through `_coerce_sf_for_kernel` value-exact (row-major contiguous).
        self.assertEqual(act_sf.dtype, torch.float32)
        self.assertEqual(w_sf.dtype, torch.float32)
        self.assertTrue(torch.equal(w_sf, weight_scale.contiguous()))

        self.assertEqual(out.shape, (2, 3, 256))  # reshaped back to input.shape[:-1] + (N,)

    @require_torch_greater_or_equal("2.7")  # torch.float8_e8m0fnu (UE8M0) landed in 2.7
    def test_linear_fp4_sm100_packs_ue8m0_scales(self):
        # FP4 weights (int8): SM100-only. Scales arrive UE8M0 and must reach the kernel packed into
        # int32 (4 K-bytes -> 1 int32) with the `(1, 1, gran_k=32)` recipe.
        input = torch.randn(4, 128, dtype=torch.bfloat16, device=torch_device)
        weight = torch.randint(-8, 8, (256, 64), dtype=torch.int8, device=torch_device)
        weight_scale = torch.ones(256, 4, dtype=torch.float32, device=torch_device).to(
            torch.float8_e8m0fnu
        )  # (N, K/32)
        with self._bundle(is_sm100=True) as captured:
            deepgemm_fp8_fp4_linear(input, weight, weight_scale)

        cast = captured["per_token_cast"][0]
        self.assertTrue(cast["use_ue8m0"] and cast["use_packed_ue8m0"])
        self.assertEqual(cast["gran_k"], 32)

        (matmul,) = captured["matmul"]
        self.assertEqual(matmul["recipe"], (1, 1, 32))
        (_, act_sf), (_, w_sf) = matmul["lhs"], matmul["rhs"]
        # UE8M0 bytes packed 4->1 into int32 for the kernel's `(INT, 1, gran_k)` branch.
        self.assertEqual(act_sf.dtype, torch.int32)
        self.assertEqual(w_sf.dtype, torch.int32)
        self.assertEqual(w_sf.shape, (256, 1))

    @require_torch_greater_or_equal("2.7")  # torch.float8_e8m0fnu (UE8M0) landed in 2.7
    def test_assert_sm100_requirements(self):
        # The shared before-load arch guard used by every FP8/FP4 forward: FP4 (int8) weights need
        # Blackwell; Blackwell has no float32 scale-factor path. Driven directly (faked capability)
        # instead of re-tested through linear / experts / megamoe.
        fp8_w = torch.zeros(1, 1, dtype=torch.float8_e4m3fn, device=torch_device)
        int8_w = torch.zeros(1, 1, dtype=torch.int8, device=torch_device)
        f32_sf = torch.ones(1, 1, dtype=torch.float32, device=torch_device)
        ue8m0_sf = f32_sf.to(torch.float8_e8m0fnu)
        with mock.patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)):  # SM90
            with self.assertRaisesRegex(NotImplementedError, "Blackwell"):
                dg._assert_sm100_requirements(int8_w, ue8m0_sf)  # FP4 has no Hopper kernel
            dg._assert_sm100_requirements(fp8_w, f32_sf)  # float32 SF is fine on SM90 -> no raise
        with mock.patch.object(torch.cuda, "get_device_capability", return_value=(10, 0)):  # SM100
            with self.assertRaisesRegex(NotImplementedError, "float32 scale-factor path"):
                dg._assert_sm100_requirements(fp8_w, f32_sf)  # no float32 SF path on Blackwell
            dg._assert_sm100_requirements(int8_w, ue8m0_sf)  # UE8M0 on SM100 -> no raise

    def test_linear_adds_bias_and_ignores_deprecated_output_dtype(self):
        input = torch.randn(4, 128, dtype=torch.bfloat16, device=torch_device)
        weight = torch.randn(16, 128, device=torch_device).to(torch.float8_e4m3fn)
        weight_scale = torch.ones(1, 1, dtype=torch.float32, device=torch_device)
        bias = torch.randn(16, dtype=torch.bfloat16, device=torch_device)
        with self._bundle(is_sm100=False):
            with self.assertWarnsRegex(FutureWarning, "output_dtype"):
                out = deepgemm_fp8_fp4_linear(
                    input, weight, weight_scale, bias=bias, block_size=(128, 128), output_dtype=torch.float32
                )
        # output_dtype is deprecated and ignored: output follows input.dtype, not the requested float32.
        self.assertEqual(out.dtype, torch.bfloat16)
        # The fake matmul zeros the output buffer, so the result is exactly the broadcast bias.
        self.assertTrue(torch.equal(out, bias.expand(4, 16)))

    # ── deepgemm_bf16_experts_forward ──────────────────────────────────────────

    def _bf16_hidden(self, experts, *, num_tokens=3, top_k=2):
        # hidden = the projection's input dim: (E, out, H) non-transposed, (E, H, out) transposed.
        proj = experts.gate_up_proj if experts.has_gate else experts.up_proj
        hidden = proj.shape[1] if experts.is_transposed else proj.shape[-1]
        hidden_states = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=torch_device)
        top_k_index = torch.tensor([[0, 1], [1, 2], [0, 3]], device=torch_device)[:num_tokens, :top_k]
        top_k_weights = torch.rand(top_k_index.shape, dtype=torch.bfloat16, device=torch_device)
        return hidden_states, top_k_index, top_k_weights

    def test_bf16_experts_kernel_inputs_sm90(self):
        experts = make_experts(num_experts=4, hidden=8, inter=16)
        hidden_states, top_k_index, top_k_weights = self._bf16_hidden(experts)
        with self._bundle(is_sm100=False) as captured:
            deepgemm_bf16_experts_forward(experts, hidden_states, top_k_index, top_k_weights)

        up, down = captured["grouped"]
        # Non-transposed -> NT kernel for both projections.
        self.assertEqual(up["name"], "bf16_nt")
        self.assertEqual(down["name"], "bf16_nt")
        # Weights are forwarded as-is (gate_up for the up proj, down for the down proj).
        self.assertIs(up["rhs"], experts.gate_up_proj)
        self.assertIs(down["rhs"], experts.down_proj)
        # SM90 grouped layout: int32 per-row expert id, `-1` in padding rows the kernel skips.
        self.assertEqual(up["grouped_layout"].dtype, torch.int32)
        self.assertEqual(up["grouped_layout"].dim(), 1)
        self.assertFalse(up["use_psum_layout"])
        self.assertTrue((up["grouped_layout"] == -1).any())  # alignment padding present
        self.assertEqual(set(up["grouped_layout"].unique().tolist()), {-1, 0, 1, 2, 3})
        # Up-proj output is the padded (rows, 2I) buffer; hidden activation padded to the same rows.
        rows = up["out"].shape[0]
        self.assertEqual(up["out"].shape, (rows, 32))
        self.assertEqual(up["lhs"].shape, (rows, 8))

    def test_bf16_experts_sm100_uses_psum_cumsum_layout(self):
        # SM100 grouped layout is the cumsum of per-expert aligned counts (not a per-row id vector).
        experts = make_experts(num_experts=4, hidden=8, inter=16)
        hidden_states, top_k_index, top_k_weights = self._bf16_hidden(experts)
        with self._bundle(is_sm100=True) as captured:
            deepgemm_bf16_experts_forward(experts, hidden_states, top_k_index, top_k_weights)
        up = captured["grouped"][0]
        self.assertTrue(up["use_psum_layout"])
        layout = up["grouped_layout"]
        self.assertEqual(layout.dtype, torch.int32)
        self.assertEqual(layout.numel(), experts.num_experts)  # one boundary per expert
        # Monotonic non-decreasing cumsum, each entry a multiple of the M-alignment (128).
        self.assertTrue(torch.all(layout[1:] >= layout[:-1]))
        self.assertTrue(torch.all(layout % 128 == 0))

    def test_bf16_experts_marshals_bias_by_expert(self):
        # Bias is gathered per routed pair (up_bias[expert_ids]) and scattered into the padded rows.
        # With the fake matmul zeroing its output, the (non-gated) down input is `act_fn(padded bias)`,
        # so its column-sum equals `act_fn(per-pair gathered up_bias)` summed over all pairs (padding rows
        # contribute act_fn(0)=0; order-invariant).
        experts = make_experts(num_experts=4, hidden=8, inter=16, has_gate=False, has_bias=True)
        hidden_states, top_k_index, top_k_weights = self._bf16_hidden(experts)
        with self._bundle(is_sm100=False) as captured:
            deepgemm_bf16_experts_forward(experts, hidden_states, top_k_index, top_k_weights)
        down_act = captured["grouped"][1]["lhs"]
        expected = experts.act_fn(experts.up_proj_bias[top_k_index.reshape(-1)]).sum(dim=0)
        self.assertTrue(torch.allclose(down_act.sum(dim=0).float(), expected.float(), atol=1e-2))

    def test_bf16_experts_rejects_non_bf16_hidden_states(self):
        experts = make_experts()
        hidden_states = torch.randn(3, 8, dtype=torch.float16, device=torch_device)
        top_k_index = torch.zeros(3, 2, dtype=torch.long, device=torch_device)
        top_k_weights = torch.rand(3, 2, dtype=torch.bfloat16, device=torch_device)
        with self.assertRaisesRegex(ValueError, "requires bfloat16 hidden states"):
            deepgemm_bf16_experts_forward(experts, hidden_states, top_k_index, top_k_weights)

    # ── deepgemm_fp8_fp4_experts_forward ───────────────────────────────────────

    def _fp8_hidden(self, num_tokens=3, hidden=8, top_k=2):
        hidden_states = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=torch_device)
        top_k_index = torch.tensor([[0, 1], [1, 2], [0, 3]], device=torch_device)[:num_tokens, :top_k]
        top_k_weights = torch.rand(top_k_index.shape, dtype=torch.bfloat16, device=torch_device)
        return hidden_states, top_k_index, top_k_weights

    def test_fp8_experts_kernel_inputs_sm90(self):
        experts = make_fp8_experts(num_experts=4, hidden=8, inter=16)  # FP8 weights, float32 block SFs
        hidden_states, top_k_index, top_k_weights = self._fp8_hidden()
        with self._bundle(is_sm100=False) as captured:
            deepgemm_fp8_fp4_experts_forward(experts, hidden_states, top_k_index, top_k_weights)

        # FP8 + float32 SF -> non-UE8M0 gran_k=128 cast, no packed recipe.
        cast = captured["per_token_cast"][0]
        self.assertFalse(cast["use_ue8m0"])
        self.assertEqual(cast["gran_k"], 128)

        up, down = captured["grouped"]
        self.assertEqual(up["name"], "fp8_fp4_nt")  # non-transposed -> NT
        self.assertIsNone(up["recipe"])
        self.assertFalse(up["use_psum_layout"])
        # Operands are `(qtensor, sf)` tuples; the up-proj weight/SF are the module's gate_up pair.
        (_, act_sf), (w_up, w_up_sf) = up["lhs"], up["rhs"]
        self.assertEqual(act_sf.dtype, torch.float32)
        self.assertIs(w_up, experts.gate_up_proj)
        self.assertIs(w_up_sf, experts.gate_up_proj_scale_inv)
        # Down-proj weight/SF come from the down pair.
        self.assertIs(down["rhs"][0], experts.down_proj)
        self.assertIs(down["rhs"][1], experts.down_proj_scale_inv)
        # Up-proj output buffer is (padded rows, 2I).
        self.assertEqual(up["out"].shape[1], 32)

    # ── deepgemm_fp8_fp4_megamoe_experts_forward ───────────────────────────────

    def _make_megamoe(self, *, num_experts=4, hidden_dim=64, inter=32, swiglu_limit=None):
        gate_up = torch.randint(
            -8, 8, (num_experts, 2 * inter, hidden_dim // 2), dtype=torch.int8, device=torch_device
        )
        down = torch.randint(-8, 8, (num_experts, hidden_dim, inter // 2), dtype=torch.int8, device=torch_device)
        return types.SimpleNamespace(
            num_experts=num_experts,
            hidden_dim=hidden_dim,
            intermediate_dim=inter,
            gate_up_proj=torch.nn.Parameter(gate_up, requires_grad=False),
            down_proj=torch.nn.Parameter(down, requires_grad=False),
            gate_up_proj_scale_inv=torch.nn.Parameter(
                torch.ones(num_experts, 2 * inter, hidden_dim // 32, device=torch_device).to(torch.float8_e8m0fnu),
                requires_grad=False,
            ),
            down_proj_scale_inv=torch.nn.Parameter(
                torch.ones(num_experts, hidden_dim, inter // 32, device=torch_device).to(torch.float8_e8m0fnu),
                requires_grad=False,
            ),
            config=types.SimpleNamespace(swiglu_limit=swiglu_limit),
        )

    @require_torch_greater_or_equal("2.7")  # torch.float8_e8m0fnu (UE8M0) landed in 2.7
    def test_megamoe_transforms_weights_and_marshals_symm_buffer(self):
        module = self._make_megamoe(num_experts=4, hidden_dim=64, inter=32, swiglu_limit=7.0)
        num_tokens, num_top_k = 3, 2
        hidden_states = torch.randn(num_tokens, 64, dtype=torch.bfloat16, device=torch_device)
        top_k_index = torch.randint(0, 4, (num_tokens, num_top_k), dtype=torch.long, device=torch_device)
        top_k_weights = torch.rand(num_tokens, num_top_k, dtype=torch.float32, device=torch_device)
        pg = types.SimpleNamespace(size=lambda: 2)

        with self._bundle(is_sm100=True) as captured:
            deepgemm_fp8_fp4_megamoe_experts_forward(
                module, hidden_states, top_k_index, top_k_weights, process_group=pg
            )

        # setup_megamoe_weights: SFs transformed with the UTCCP recipe (1, 32) at the right dims.
        gate_up_sf_call, down_sf_call = captured["transform_sf"]
        self.assertEqual(gate_up_sf_call["dim_a"], 2 * 32)  # 2 * intermediate
        self.assertEqual(gate_up_sf_call["dim_b"], 64)  # hidden_dim
        self.assertEqual(gate_up_sf_call["recipe"], (1, 32))
        self.assertEqual(gate_up_sf_call["num_groups"], 4)  # num_local_experts
        self.assertEqual(down_sf_call["dim_a"], 64)
        self.assertEqual(down_sf_call["dim_b"], 32)

        # The module weights were overwritten in place with the transform_weights_for_mega_moe output,
        # and those exact transformed tensors reach the kernel.
        (new_gate_up, _), (new_down, _) = captured["transform_weights_out"]
        self.assertTrue(torch.equal(module.gate_up_proj.data, new_gate_up))
        self.assertTrue(torch.equal(module.down_proj.data, new_down))
        self.assertTrue(self._megamoe_flag_set(module))

        # Symm buffer sized from global expert count (num_local * process_group.size()).
        self.assertEqual(captured["symm_buffer_kwargs"]["num_experts"], 4 * 2)

        # Hidden states cast FP8/UE8M0 (gran_k=32 packed) and staged into the symm buffer.
        cast = captured["per_token_cast"][0]
        self.assertTrue(cast["use_ue8m0"] and cast["use_packed_ue8m0"])
        self.assertEqual(cast["gran_k"], 32)
        buf = captured["symm_buffer"]
        self.assertTrue(torch.equal(buf.topk_idx[:num_tokens], top_k_index))
        self.assertTrue(torch.allclose(buf.topk_weights[:num_tokens], top_k_weights))

        # Kernel receives the transformed L1/L2 weight pairs and the SwiGLU clamp from config.
        mega = captured["mega_moe"]
        self.assertIs(mega["gate_up_pair"][0], module.gate_up_proj)
        self.assertIs(mega["down_pair"][0], module.down_proj)
        self.assertIs(mega["symm_buffer"], buf)
        self.assertEqual(mega["activation_clamp"], 7.0)

    @require_torch_greater_or_equal("2.7")  # torch.float8_e8m0fnu (UE8M0) landed in 2.7
    def test_megamoe_requires_fp4_weights_and_process_group(self):
        hidden_states = torch.randn(3, 64, dtype=torch.bfloat16, device=torch_device)
        top_k_index = torch.randint(0, 4, (3, 2), dtype=torch.long, device=torch_device)
        top_k_weights = torch.rand(3, 2, dtype=torch.float32, device=torch_device)

        # Non-FP4 (non-int8) expert weights are rejected — Mega MoE is FP4-packed only.
        module = self._make_megamoe()
        module.gate_up_proj = torch.nn.Parameter(module.gate_up_proj.data.to(torch.bfloat16), requires_grad=False)
        with self._bundle(is_sm100=True):
            with self.assertRaisesRegex(NotImplementedError, "FP4-packed expert weights"):
                deepgemm_fp8_fp4_megamoe_experts_forward(
                    module,
                    hidden_states,
                    top_k_index,
                    top_k_weights,
                    process_group=types.SimpleNamespace(size=lambda: 1),
                )

        # Missing process_group (EP group) is rejected — required for the symm-buffer rendezvous.
        module2 = self._make_megamoe()
        with self._bundle(is_sm100=True):
            with self.assertRaisesRegex(ValueError, "requires a .process_group."):
                deepgemm_fp8_fp4_megamoe_experts_forward(module2, hidden_states, top_k_index, top_k_weights)

    @staticmethod
    def _megamoe_flag_set(module):
        return getattr(module, "_megamoe_transformed", False)
