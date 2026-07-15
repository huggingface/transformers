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
  correctly shaped tensors, and runs the REAL public forwards on a CUDA device so their scale-factor
  coercion / TMA-aligned grouped layout / weight-selection / dtype-cast glue executes for real; it
  then asserts the tensors handed to the kernel are what the kernel expects (packed int32 UE8M0 SFs,
  int32 grouped layout, `(qtensor, sf)` operand tuples, recipes, transformed Mega MoE weights, ...).

The box is SM80 (Ampere), so paths that gate on `_is_sm100` are exercised by patching `dg._is_sm100`:
the SF-packing / TMA-alignment / psum-layout code it selects is pure tensor arithmetic that runs on any
CUDA device — only the loader's real arch check (bypassed here by mocking `load_deepgemm_kernel`) needs
the actual silicon.
"""

import contextlib
import types
import unittest
from unittest import mock

import torch

import transformers.integrations.deepgemm as dg
from transformers.integrations.deepgemm import (
    deepgemm_bf16_experts_forward,
    deepgemm_fp8_fp4_experts_forward,
    deepgemm_fp8_fp4_linear,
    deepgemm_fp8_fp4_megamoe_experts_forward,
)
from transformers.testing_utils import require_torch, require_torch_gpu


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
        cuda_home="/fake/cuda",
        nvcc_present=True,
        nvcc_version=(12, 9),
        kernel=_FakeDeepGemmKernel,
    ):
        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.object(dg, "is_kernels_available", return_value=kernels_available))
        stack.enter_context(mock.patch.object(torch.cuda, "is_available", return_value=cuda_available))
        stack.enter_context(mock.patch.object(torch.cuda, "get_device_capability", return_value=capability))
        stack.enter_context(mock.patch.object(dg, "_get_cuda_home", return_value=cuda_home))
        stack.enter_context(mock.patch.object(dg, "_get_nvcc_version", return_value=nvcc_version))
        stack.enter_context(mock.patch("transformers.integrations.deepgemm.os.path.isfile", return_value=nvcc_present))
        stack.enter_context(mock.patch.object(dg, "lazy_load_kernel", return_value=kernel))
        return stack

    def test_loads_when_environment_is_valid(self):
        with self._env():
            bundle = dg.load_deepgemm_kernel()
        self.assertIsInstance(bundle, dg.DeepGEMM)
        self.assertEqual(bundle.m_alignment, 128)

    def test_loads_on_blackwell_when_sm100_required(self):
        with self._env(capability=(10, 0)):
            bundle = dg.load_deepgemm_kernel(requires_sm100=True)
        self.assertIsInstance(bundle, dg.DeepGEMM)

    def test_raises_without_kernels_package(self):
        with (
            self._env(kernels_available=False),
            self.assertRaisesRegex(ImportError, "requires the .kernels. package"),
        ):
            dg.load_deepgemm_kernel()

    def test_raises_without_cuda(self):
        with self._env(cuda_available=False), self.assertRaisesRegex(ImportError, "requires CUDA"):
            dg.load_deepgemm_kernel()

    def test_raises_on_unsupported_arch(self):
        with self._env(capability=(8, 0)), self.assertRaisesRegex(ImportError, "requires Hopper"):
            dg.load_deepgemm_kernel()

    def test_raises_on_hopper_when_sm100_required(self):
        with self._env(capability=(9, 0)), self.assertRaisesRegex(ImportError, "requires Blackwell"):
            dg.load_deepgemm_kernel(requires_sm100=True)

    def test_raises_without_cuda_toolkit(self):
        with self._env(cuda_home=None), self.assertRaisesRegex(ImportError, "needs a CUDA toolkit"):
            dg.load_deepgemm_kernel()

    def test_raises_without_nvcc(self):
        with self._env(nvcc_present=False), self.assertRaisesRegex(ImportError, "compiles with nvcc"):
            dg.load_deepgemm_kernel()

    def test_raises_on_unreadable_nvcc_version(self):
        with self._env(nvcc_version=None), self.assertRaisesRegex(ImportError, "could not read its CUDA version"):
            dg.load_deepgemm_kernel()

    def test_raises_on_old_nvcc(self):
        with self._env(nvcc_version=(12, 0)), self.assertRaisesRegex(ImportError, "too old"):
            dg.load_deepgemm_kernel()

    def test_raises_when_kernel_fails_to_load(self):
        with self._env(kernel=None), self.assertRaisesRegex(ImportError, "Failed to load"):
            dg.load_deepgemm_kernel()

    def test_raises_on_missing_symbols(self):
        with (
            self._env(kernel=_DeepGemmKernelMissingSymbol),
            self.assertRaisesRegex(ImportError, "missing required symbols"),
        ):
            dg.load_deepgemm_kernel()

    def test_loader_is_compile_safe(self):
        with self._env():
            torch.compiler.reset()

            @torch.compile(fullgraph=True)
            def run(x):
                return dg.load_deepgemm_kernel().per_token_cast_to_fp8(x)

        out = run(torch.zeros(3))
        self.assertTrue(torch.equal(out, torch.ones(3)))

    def test_loader_is_compile_safe_when_warm(self):
        # Warm the loader BEFORE compiling: Dynamo then executes the opaque loader node's
        # warm-cache short-circuit at trace time — the path that must return None, not the
        # bundle (a leaked bundle is `Unsupported: torch.* op returned non-Tensor`). The
        # production sequence (eager warmup, then compile) always traces warm; the cold
        # test above cannot catch a leak on this path.
        with self._env():
            dg.load_deepgemm_kernel()
            torch.compiler.reset()

            @torch.compile(fullgraph=True)
            def run(x):
                return dg.load_deepgemm_kernel().per_token_cast_to_fp8(x)

        out = run(torch.zeros(3))
        self.assertTrue(torch.equal(out, torch.ones(3)))


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
            x=torch.zeros(num_max_tokens_per_rank, hidden, dtype=torch.float8_e4m3fn, device="cuda"),
            x_sf=torch.zeros(num_max_tokens_per_rank, -(-hidden // 32), dtype=torch.float32, device="cuda"),
            topk_idx=torch.zeros(num_max_tokens_per_rank, num_topk, dtype=torch.long, device="cuda"),
            topk_weights=torch.zeros(num_max_tokens_per_rank, num_topk, dtype=torch.float32, device="cuda"),
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


def _make_experts(
    *,
    num_experts=4,
    hidden=8,
    inter=16,
    has_gate=True,
    has_bias=False,
    is_transposed=False,
    weight_dtype=torch.float8_e4m3fn,
    scale_dtype=torch.float32,
    activation_scheme="dynamic",
    block_size=(128, 128),
    fp8_experts=True,
    device="cuda",
):
    """A minimal stand-in for an FP8/BF16 experts module carrying exactly the attributes the forwards
    read. Weights are `(E, 2I, H)` (non-transposed) or `(E, H, 2I)` (transposed) for gate_up and
    `(E, H, I)` / `(E, I, H)` for down.
    """
    if is_transposed:
        gate_up = torch.randn(num_experts, hidden, 2 * inter, device=device).to(weight_dtype)
        up = torch.randn(num_experts, hidden, inter, device=device).to(weight_dtype)
        down = torch.randn(num_experts, inter, hidden, device=device).to(weight_dtype)
    else:
        gate_up = torch.randn(num_experts, 2 * inter, hidden, device=device).to(weight_dtype)
        up = torch.randn(num_experts, inter, hidden, device=device).to(weight_dtype)
        down = torch.randn(num_experts, hidden, inter, device=device).to(weight_dtype)

    gate_up_scale = torch.ones(num_experts, 1, 1, device=device).to(scale_dtype)
    up_scale = torch.ones(num_experts, 1, 1, device=device).to(scale_dtype)
    down_scale = torch.ones(num_experts, 1, 1, device=device).to(scale_dtype)

    ns = types.SimpleNamespace(
        num_experts=num_experts,
        has_gate=has_gate,
        has_bias=has_bias,
        is_transposed=is_transposed,
        gate_up_proj=gate_up,
        up_proj=up,
        down_proj=down,
        gate_up_proj_bias=torch.randn(num_experts, 2 * inter, dtype=torch.bfloat16, device=device)
        if has_bias
        else None,
        up_proj_bias=torch.randn(num_experts, inter, dtype=torch.bfloat16, device=device) if has_bias else None,
        down_proj_bias=torch.randn(num_experts, hidden, dtype=torch.bfloat16, device=device) if has_bias else None,
        # SwiGLU on the concatenated gate/up halves; identity gate keeps dims trivial for the fakes.
        _apply_gate=lambda x: x[..., : x.shape[-1] // 2],
        act_fn=lambda x: x,
    )
    if fp8_experts:
        ns.gate_up_proj_scale_inv = gate_up_scale
        ns.up_proj_scale_inv = up_scale
        ns.down_proj_scale_inv = down_scale
        ns.block_size = block_size
        ns.activation_scheme = activation_scheme
        ns._deepgemm_disabled = False
    return ns


@require_torch_gpu
class DeepGemmForwardTest(unittest.TestCase):
    """Drives the real public forwards with only the loaded-kernel ops mocked."""

    def setUp(self):
        dg._DEEPGEMM = None
        self.addCleanup(setattr, dg, "_DEEPGEMM", None)

    @contextlib.contextmanager
    def _bundle(self, *, is_sm100=None):
        captured = {}
        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.object(dg, "load_deepgemm_kernel", return_value=_make_bundle(captured)))
        if is_sm100 is not None:
            stack.enter_context(mock.patch.object(dg, "_is_sm100", return_value=is_sm100))
        with stack:
            yield captured

    # ── deepgemm_fp8_fp4_linear ────────────────────────────────────────────────

    def test_linear_fp8_sm90_kernel_inputs(self):
        # FP8 weights + float32 block SF on SM90: recipe stays None, SFs are handed over row-major
        # float32 (SM90 dispatch transforms SFA itself and only checks SFB — see `_coerce_sf_for_kernel`).
        input = torch.randn(2, 3, 128, dtype=torch.bfloat16, device="cuda")  # 3D -> flatten to (6, 128)
        weight = torch.randn(256, 128, device="cuda").to(torch.float8_e4m3fn)
        weight_scale = torch.ones(2, 1, dtype=torch.float32, device="cuda")  # (N/128, K/128)
        with self._bundle(is_sm100=False) as captured:
            out = deepgemm_fp8_fp4_linear(input, weight, weight_scale, block_size=(128, 128))

        cast = captured["per_token_cast"][0]
        self.assertFalse(cast["use_ue8m0"])
        self.assertEqual(cast["gran_k"], 128)
        self.assertEqual(cast["x"].shape, (6, 128))  # flattened to 2D

        (matmul,) = captured["matmul"]
        self.assertIsNone(matmul["recipe"])  # float-SF path leaves the recipe unset
        (qinput, act_sf), (w, w_sf) = matmul["lhs"], matmul["rhs"]
        self.assertEqual(qinput.dtype, torch.float8_e4m3fn)
        self.assertIs(w, weight)
        # SM90 float32 SFs pass through `_coerce_sf_for_kernel` value-exact (row-major contiguous).
        self.assertEqual(act_sf.dtype, torch.float32)
        self.assertEqual(w_sf.dtype, torch.float32)
        self.assertTrue(torch.equal(w_sf, weight_scale.contiguous()))

        self.assertEqual(out.shape, (2, 3, 256))  # reshaped back to input.shape[:-1] + (N,)
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_linear_fp4_sm100_packs_ue8m0_scales(self):
        # FP4 weights (int8): SM100-only. Scales arrive UE8M0 and must reach the kernel packed into
        # int32 (4 K-bytes -> 1 int32) with the `(1, 1, gran_k=32)` recipe.
        input = torch.randn(4, 128, dtype=torch.bfloat16, device="cuda")
        weight = torch.randint(-8, 8, (256, 64), dtype=torch.int8, device="cuda")
        weight_scale = torch.ones(256, 4, dtype=torch.float32, device="cuda").to(torch.float8_e8m0fnu)  # (N, K/32)
        with self._bundle(is_sm100=True) as captured:
            out = deepgemm_fp8_fp4_linear(input, weight, weight_scale)

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
        self.assertEqual(out.shape, (4, 256))

    def test_linear_adds_bias_and_honours_output_dtype(self):
        input = torch.randn(4, 128, dtype=torch.bfloat16, device="cuda")
        weight = torch.randn(16, 128, device="cuda").to(torch.float8_e4m3fn)
        weight_scale = torch.ones(1, 1, dtype=torch.float32, device="cuda")
        bias = torch.randn(16, dtype=torch.float32, device="cuda")
        with self._bundle(is_sm100=False):
            out = deepgemm_fp8_fp4_linear(
                input, weight, weight_scale, bias=bias, block_size=(128, 128), output_dtype=torch.float32
            )
        # The fake matmul zeros the output buffer, so the result is exactly the broadcast bias.
        self.assertEqual(out.dtype, torch.float32)
        self.assertTrue(torch.equal(out, bias.expand(4, 16)))

    # ── deepgemm_bf16_experts_forward ──────────────────────────────────────────

    def _bf16_hidden(self, experts, *, num_tokens=3, top_k=2):
        hidden = experts.gate_up_proj.shape[-1] if experts.is_transposed else experts.gate_up_proj.shape[-1]
        hidden_states = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device="cuda")
        top_k_index = torch.tensor([[0, 1], [1, 2], [0, 3]], device="cuda")[:num_tokens, :top_k]
        top_k_weights = torch.rand(top_k_index.shape, dtype=torch.bfloat16, device="cuda")
        return hidden_states, top_k_index, top_k_weights

    def test_bf16_experts_kernel_inputs_sm90(self):
        experts = _make_experts(num_experts=4, hidden=8, inter=16, weight_dtype=torch.bfloat16, fp8_experts=False)
        hidden_states, top_k_index, top_k_weights = self._bf16_hidden(experts)
        with self._bundle(is_sm100=False) as captured:
            out = deepgemm_bf16_experts_forward(experts, hidden_states, top_k_index, top_k_weights)

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

        self.assertEqual(out.shape, (3, 8))
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_bf16_experts_transposed_uses_nn_kernel(self):
        experts = _make_experts(
            num_experts=4, hidden=8, inter=16, is_transposed=True, weight_dtype=torch.bfloat16, fp8_experts=False
        )
        hidden_states, top_k_index, top_k_weights = self._bf16_hidden(experts)
        with self._bundle(is_sm100=False) as captured:
            deepgemm_bf16_experts_forward(experts, hidden_states, top_k_index, top_k_weights)
        up, down = captured["grouped"]
        self.assertEqual(up["name"], "bf16_nn")
        self.assertEqual(down["name"], "bf16_nn")
        # Transposed gate_up is `(E, H, 2I)` -> up-projection output dim is the last axis (2I=32).
        self.assertEqual(up["out"].shape[1], 32)

    def test_bf16_experts_sm100_uses_psum_cumsum_layout(self):
        # SM100 grouped layout is the cumsum of per-expert aligned counts (not a per-row id vector).
        experts = _make_experts(num_experts=4, hidden=8, inter=16, weight_dtype=torch.bfloat16, fp8_experts=False)
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
        # With the fake matmul zeroing its output, the second (down) matmul's input IS the padded bias,
        # so its column-sum equals the per-pair gathered up_bias summed over all pairs (order-invariant).
        experts = _make_experts(
            num_experts=4,
            hidden=8,
            inter=16,
            has_gate=False,
            has_bias=True,
            weight_dtype=torch.bfloat16,
            fp8_experts=False,
        )
        hidden_states, top_k_index, top_k_weights = self._bf16_hidden(experts)
        with self._bundle(is_sm100=False) as captured:
            deepgemm_bf16_experts_forward(experts, hidden_states, top_k_index, top_k_weights)
        down_act = captured["grouped"][1]["lhs"]  # act_fn is identity, so this is the biased up-proj out
        expected = experts.up_proj_bias[top_k_index.reshape(-1)].sum(dim=0)
        self.assertTrue(torch.allclose(down_act.sum(dim=0).float(), expected.float(), atol=1e-2))

    def test_bf16_experts_rejects_non_bf16_hidden_states(self):
        experts = _make_experts(weight_dtype=torch.bfloat16, fp8_experts=False)
        hidden_states = torch.randn(3, 8, dtype=torch.float16, device="cuda")
        top_k_index = torch.zeros(3, 2, dtype=torch.long, device="cuda")
        top_k_weights = torch.rand(3, 2, dtype=torch.bfloat16, device="cuda")
        with self.assertRaisesRegex(ValueError, "requires bfloat16 hidden states"):
            deepgemm_bf16_experts_forward(experts, hidden_states, top_k_index, top_k_weights)

    # ── deepgemm_fp8_fp4_experts_forward ───────────────────────────────────────

    def _fp8_hidden(self, num_tokens=3, hidden=8, top_k=2):
        hidden_states = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device="cuda")
        top_k_index = torch.tensor([[0, 1], [1, 2], [0, 3]], device="cuda")[:num_tokens, :top_k]
        top_k_weights = torch.rand(top_k_index.shape, dtype=torch.bfloat16, device="cuda")
        return hidden_states, top_k_index, top_k_weights

    def test_fp8_experts_kernel_inputs_sm90(self):
        experts = _make_experts(num_experts=4, hidden=8, inter=16)  # FP8 weights, float32 block SFs
        hidden_states, top_k_index, top_k_weights = self._fp8_hidden()
        with self._bundle(is_sm100=False) as captured:
            out = deepgemm_fp8_fp4_experts_forward(experts, hidden_states, top_k_index, top_k_weights)

        # FP8 + float32 SF -> non-UE8M0 gran_k=128 cast, no packed recipe.
        cast = captured["per_token_cast"][0]
        self.assertFalse(cast["use_ue8m0"])
        self.assertEqual(cast["gran_k"], 128)

        up, down = captured["grouped"]
        self.assertEqual(up["name"], "fp8_fp4_nt")  # non-transposed -> NT
        self.assertIsNone(up["recipe"])
        self.assertFalse(up["use_psum_layout"])
        # Operands are `(qtensor, sf)` tuples; the up-proj weight/SF are the module's gate_up pair.
        (act_fp8, act_sf), (w_up, w_up_sf) = up["lhs"], up["rhs"]
        self.assertEqual(act_fp8.dtype, torch.float8_e4m3fn)
        self.assertEqual(act_sf.dtype, torch.float32)
        self.assertIs(w_up, experts.gate_up_proj)
        self.assertIs(w_up_sf, experts.gate_up_proj_scale_inv)
        # Down-proj weight/SF come from the down pair.
        self.assertIs(down["rhs"][0], experts.down_proj)
        self.assertIs(down["rhs"][1], experts.down_proj_scale_inv)
        # Up-proj output buffer is (padded rows, 2I).
        self.assertEqual(up["out"].shape[1], 32)

        self.assertEqual(out.shape, (3, 8))
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_fp8_experts_transposed_uses_nn_kernel(self):
        experts = _make_experts(num_experts=4, hidden=8, inter=16, is_transposed=True)
        hidden_states, top_k_index, top_k_weights = self._fp8_hidden()
        with self._bundle(is_sm100=False) as captured:
            deepgemm_fp8_fp4_experts_forward(experts, hidden_states, top_k_index, top_k_weights)
        self.assertEqual(captured["grouped"][0]["name"], "fp8_fp4_nn")

    def test_fp8_experts_asserts_ue8m0_scales_on_sm100(self):
        # The `_assert_sm100_scales_are_ue8m0` loud-failure: on SM100 a plain float32 expert SF means a
        # non-UE8M0 checkpoint that would be silently corrupted by rounding — fail early and clearly.
        experts = _make_experts(scale_dtype=torch.float32)
        hidden_states, top_k_index, top_k_weights = self._fp8_hidden()
        with self._bundle(is_sm100=True):
            with self.assertRaisesRegex(ValueError, "power-of-two .UE8M0. scale"):
                deepgemm_fp8_fp4_experts_forward(experts, hidden_states, top_k_index, top_k_weights)

        # The converse (no false positives): the guard is a no-op for UE8M0 SFs on SM100 (kernel-ready
        # as-is) and for any SF on SM90 (float32 SFs are consumed directly). Checked on the guard itself
        # rather than a full forward — the SM100 UE8M0 path needs realistically wide SFs (last dim
        # divisible by 4 to pack into int32) that a toy expert shape can't provide.
        f32 = torch.ones(4, 1, device="cuda", dtype=torch.float32)
        ue8m0 = f32.to(torch.float8_e8m0fnu)
        with mock.patch.object(dg, "_is_sm100", return_value=True):
            dg._assert_sm100_scales_are_ue8m0(ue8m0)  # no raise
        with mock.patch.object(dg, "_is_sm100", return_value=False):
            dg._assert_sm100_scales_are_ue8m0(f32)  # no raise

    # ── deepgemm_fp8_fp4_megamoe_experts_forward ───────────────────────────────

    def _make_megamoe(self, *, num_experts=4, hidden_dim=64, inter=32, swiglu_limit=None):
        gate_up = torch.randint(-8, 8, (num_experts, 2 * inter, hidden_dim // 2), dtype=torch.int8, device="cuda")
        down = torch.randint(-8, 8, (num_experts, hidden_dim, inter // 2), dtype=torch.int8, device="cuda")
        return types.SimpleNamespace(
            num_experts=num_experts,
            hidden_dim=hidden_dim,
            intermediate_dim=inter,
            gate_up_proj=torch.nn.Parameter(gate_up, requires_grad=False),
            down_proj=torch.nn.Parameter(down, requires_grad=False),
            gate_up_proj_scale_inv=torch.nn.Parameter(
                torch.ones(num_experts, 2 * inter, hidden_dim // 32, device="cuda").to(torch.float8_e8m0fnu),
                requires_grad=False,
            ),
            down_proj_scale_inv=torch.nn.Parameter(
                torch.ones(num_experts, hidden_dim, inter // 32, device="cuda").to(torch.float8_e8m0fnu),
                requires_grad=False,
            ),
            config=types.SimpleNamespace(swiglu_limit=swiglu_limit),
        )

    def test_megamoe_transforms_weights_and_marshals_symm_buffer(self):
        module = self._make_megamoe(num_experts=4, hidden_dim=64, inter=32, swiglu_limit=7.0)
        num_tokens, num_top_k = 3, 2
        hidden_states = torch.randn(num_tokens, 64, dtype=torch.bfloat16, device="cuda")
        top_k_index = torch.randint(0, 4, (num_tokens, num_top_k), dtype=torch.long, device="cuda")
        top_k_weights = torch.rand(num_tokens, num_top_k, dtype=torch.float32, device="cuda")
        pg = types.SimpleNamespace(size=lambda: 2)

        with self._bundle(is_sm100=True) as captured:
            out = deepgemm_fp8_fp4_megamoe_experts_forward(
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
        self.assertEqual(captured["symm_buffer_kwargs"]["hidden"], 64)
        self.assertEqual(captured["symm_buffer_kwargs"]["intermediate_hidden"], 32)

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

        self.assertEqual(out.shape, (num_tokens, 64))
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_megamoe_requires_fp4_weights_and_process_group(self):
        hidden_states = torch.randn(3, 64, dtype=torch.bfloat16, device="cuda")
        top_k_index = torch.randint(0, 4, (3, 2), dtype=torch.long, device="cuda")
        top_k_weights = torch.rand(3, 2, dtype=torch.float32, device="cuda")

        # Non-FP4 (non-int8) expert weights are rejected — Mega MoE is FP4-packed only.
        module = self._make_megamoe()
        module.gate_up_proj = torch.nn.Parameter(module.gate_up_proj.data.to(torch.bfloat16), requires_grad=False)
        with self._bundle(is_sm100=True):
            with self.assertRaisesRegex(RuntimeError, "FP4-packed expert weights"):
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


if __name__ == "__main__":
    unittest.main()
