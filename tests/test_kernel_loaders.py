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
"""Environment gating and torch-compile safety of the kernel loaders.

The `deepgemm`, `finegrained_fp8`, and `sonicmoe` loaders resolve their kernel bundle inside an
`@torch._dynamo.allow_in_graph` opaque node that populates a module global; the public `load_*_kernel`
helper returns that global. These tests mock the environment probes and `lazy_load_kernel` so the
loaders can be driven with no GPU / hub build, and check that each `load_*_kernel`:

* raises `ImportError` when a precondition is not met (missing package, wrong arch, missing toolkit,
  missing kernel symbol, ...), and
* returns the real bundle when everything is satisfied.

A final per-loader case runs `load_*_kernel` inside `torch.compile(backend="eager", fullgraph=True)`
and asserts the compiled call still gets the real bundle — i.e. the opaque-node-populates-global /
traced-code-reads-global pattern does not bake in the pre-load `None`. `backend="eager"` exercises the
Dynamo frontend (where that would break) without an accelerator; `fullgraph=True` turns any graph
break into an error.
"""

import contextlib
import unittest
from unittest import mock

import torch

import transformers.integrations.deepgemm as dg
import transformers.integrations.finegrained_fp8 as fg
import transformers.integrations.sonicmoe as sm
from transformers.testing_utils import require_torch


def _add_one(x, *args, **kwargs):
    return x + 1


# ── Fake kernels (a "good" one exposing every symbol the loader resolves, plus a variant missing one) ──


class _FakeFinegrainedKernel:
    matmul_2d = staticmethod(_add_one)
    matmul_batched = staticmethod(_add_one)
    matmul_grouped = staticmethod(_add_one)


class _FinegrainedKernelMissingSymbol:
    matmul_2d = staticmethod(_add_one)  # missing matmul_batched / matmul_grouped


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


class _FakeActivationType:
    SWIGLU = "swiglu"


class _FakeSonicMoeEnums:
    ActivationType = _FakeActivationType


class _FakeSonicMoeKernel:
    enums = _FakeSonicMoeEnums

    @staticmethod
    def moe_general_routing_inputs(x, *args, **kwargs):
        return x + 1, None


class _SonicMoeKernelMissingSymbol:
    enums = _FakeSonicMoeEnums  # missing moe_general_routing_inputs


@require_torch
class FinegrainedFp8LoaderTest(unittest.TestCase):
    def setUp(self):
        fg._FINEGRAINED_FP8 = None
        self.addCleanup(setattr, fg, "_FINEGRAINED_FP8", None)

    def _env(self, *, kernels_available=True, kernel=_FakeFinegrainedKernel):
        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.object(fg, "is_kernels_available", return_value=kernels_available))
        stack.enter_context(mock.patch.object(fg, "lazy_load_kernel", return_value=kernel))
        return stack

    def test_loads_when_environment_is_valid(self):
        with self._env():
            bundle = fg.load_finegrained_fp8_kernel()
        self.assertIsInstance(bundle, fg.FineGrainedFP8)
        self.assertIs(bundle.matmul, _add_one)

    def test_raises_without_kernels_package(self):
        with self._env(kernels_available=False), self.assertRaisesRegex(ImportError, "requires the .kernels. package"):
            fg.load_finegrained_fp8_kernel()

    def test_raises_when_kernel_fails_to_load(self):
        with self._env(kernel=None), self.assertRaisesRegex(ImportError, "Failed to load the finegrained-fp8 kernel"):
            fg.load_finegrained_fp8_kernel()

    def test_raises_on_missing_symbols(self):
        with (
            self._env(kernel=_FinegrainedKernelMissingSymbol),
            self.assertRaisesRegex(ImportError, "missing required symbols"),
        ):
            fg.load_finegrained_fp8_kernel()

    def test_loader_is_compile_safe(self):
        with self._env():
            torch.compiler.reset()

            @torch.compile(backend="eager", fullgraph=True)
            def run(x):
                return fg.load_finegrained_fp8_kernel().matmul(x)

            out = run(torch.zeros(3))
        self.assertTrue(torch.equal(out, torch.ones(3)))


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

            @torch.compile(backend="eager", fullgraph=True)
            def run(x):
                return dg.load_deepgemm_kernel().per_token_cast_to_fp8(x)

            out = run(torch.zeros(3))
        self.assertTrue(torch.equal(out, torch.ones(3)))


@require_torch
class SonicMoeLoaderTest(unittest.TestCase):
    def setUp(self):
        sm._SONICMOE = None
        self.addCleanup(setattr, sm, "_SONICMOE", None)

    def _env(self, *, cuda_available=True, capability=(9, 0), versions_ok=True, kernel=_FakeSonicMoeKernel):
        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.object(torch.cuda, "is_available", return_value=cuda_available))
        stack.enter_context(mock.patch.object(torch.cuda, "get_device_capability", return_value=capability))
        if versions_ok:
            stack.enter_context(mock.patch.object(sm, "require_version", return_value=None))
        else:
            stack.enter_context(
                mock.patch.object(sm, "require_version", side_effect=ImportError("nvidia-cutlass-dsl>4.5.2"))
            )
        stack.enter_context(mock.patch.object(sm, "lazy_load_kernel", return_value=kernel))
        return stack

    def test_loads_when_environment_is_valid(self):
        with self._env():
            bundle = sm.load_sonicmoe_kernel()
        self.assertIsInstance(bundle, sm.SonicMoE)
        self.assertIs(bundle.activation_type_enum, _FakeActivationType)

    def test_raises_without_cuda(self):
        with self._env(cuda_available=False), self.assertRaisesRegex(ImportError, "requires CUDA"):
            sm.load_sonicmoe_kernel()

    def test_raises_on_unsupported_arch(self):
        with self._env(capability=(8, 0)), self.assertRaisesRegex(ImportError, "requires a Hopper"):
            sm.load_sonicmoe_kernel()

    def test_raises_on_incompatible_dependency_versions(self):
        with self._env(versions_ok=False), self.assertRaisesRegex(ImportError, "dependency requirements are not met"):
            sm.load_sonicmoe_kernel()

    def test_raises_when_kernel_fails_to_load(self):
        with self._env(kernel=None), self.assertRaisesRegex(ImportError, "Failed to load the sonic-moe kernel"):
            sm.load_sonicmoe_kernel()

    def test_raises_on_missing_symbols(self):
        with (
            self._env(kernel=_SonicMoeKernelMissingSymbol),
            self.assertRaisesRegex(ImportError, "missing required symbols"),
        ):
            sm.load_sonicmoe_kernel()

    def test_loader_is_compile_safe(self):
        with self._env():
            torch.compiler.reset()

            @torch.compile(backend="eager", fullgraph=True)
            def run(x):
                out, _ = sm.load_sonicmoe_kernel().moe_general_routing_inputs(x)
                return out

            out = run(torch.zeros(3))
        self.assertTrue(torch.equal(out, torch.ones(3)))


if __name__ == "__main__":
    unittest.main()
