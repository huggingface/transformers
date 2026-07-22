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
"""Integration tests for the sonic-moe experts implementation.

Two layers, both mocking only what needs a Hopper GPU + CuteDSL / `nvidia-cutlass-dsl`:

* `SonicMoeLoaderTest` mocks the environment probes and `lazy_load_kernel` to drive
  `load_sonicmoe_kernel` with no GPU — asserting it gates correctly (raises when a precondition is
  unmet, returns the bundle otherwise) and stays torch-compile safe.
* `SonicMoeExpertsForwardTest` mocks only the kernel dispatch (`moe_general_routing_inputs`) to return
  a correctly shaped output, and runs the real `sonicmoe_experts_forward` on a CUDA device so its
  weight-permutation / routing-flatten / dtype-cast glue executes for real; it then asserts the tensors
  handed to the kernel are what the kernel expects (int32 indices, permuted weights, ...).
"""

import contextlib
import importlib.metadata
import inspect
import unittest
from unittest import mock

import torch
from parameterized import parameterized
from test_utils import make_experts

import transformers.integrations.sonicmoe as sm
from transformers.integrations.sonicmoe import sonicmoe_experts_forward
from transformers.testing_utils import require_torch, require_torch_gpu, torch_device


class _FakeActivationType:
    SWIGLU = "swiglu"
    GEGLU = "geglu"
    REGLU = "reglu"


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
class SonicMoeLoaderTest(unittest.TestCase):
    def setUp(self):
        sm._SONICMOE = None
        self.addCleanup(setattr, sm, "_SONICMOE", None)

    def _env(
        self,
        *,
        kernels_available=True,
        cuda_available=True,
        capability=(9, 0),
        versions_ok=True,
        kernel=_FakeSonicMoeKernel,
    ):
        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.object(sm, "is_kernels_available", return_value=kernels_available))
        # Fake a "CUDA + `capability`" environment for `is_sonicmoe_loadable` *only* (scoped to its call
        # stack): its arch gate passes while torch.compile / inductor still see the real device — so the
        # compile-safety test runs under the default (inductor) backend on any host, not just SM90+. The
        # real fallbacks are lazy: only non-loader callers reach them (inductor, under compile), never the
        # CPU-only gating tests (where querying real CUDA would raise).
        real_is_available = torch.cuda.is_available
        real_capability = torch.cuda.get_device_capability

        def _in_loader():
            return any(f.function == "is_sonicmoe_loadable" for f in inspect.stack())

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
        # Report each build dependency at its validated max when ok, one major past it when not; delegate
        # unknown distributions to the real lookup so nothing else the process imports is disturbed.
        real_version = importlib.metadata.version

        def _version(distribution):
            max_version = sm.SONICMOE_DEPENDENCIES.get(distribution)
            if max_version is None:
                return real_version(distribution)
            return max_version if versions_ok else f"{int(max_version.split('.')[0]) + 1}.0.0"

        stack.enter_context(mock.patch.object(importlib.metadata, "version", side_effect=_version))
        stack.enter_context(mock.patch.object(sm, "lazy_load_kernel", return_value=kernel))
        return stack

    def test_loads_when_environment_is_valid(self):
        with self._env():
            bundle = sm.load_sonicmoe_kernel()
        self.assertIsInstance(bundle, sm.SonicMoE)
        self.assertIs(bundle.activation_type_enum, _FakeActivationType)

    @parameterized.expand(
        [
            ("valid_env", {}, None),
            ("no_kernels", {"kernels_available": False}, "`kernels` package"),
            ("no_cuda", {"cuda_available": False}, "requires CUDA"),
            ("bad_arch", {"capability": (8, 0)}, "requires a Hopper"),
            ("incompatible_versions", {"versions_ok": False}, "unvalidated"),
        ]
    )
    def test_is_sonicmoe_loadable(self, _name, env_kwargs, pattern):
        # The single gating source: valid env -> True; each unmet precondition -> False, or the specific
        # `ImportError` when `raise_error=True` (what the loader uses).
        with self._env(**env_kwargs):
            self.assertEqual(sm.is_sonicmoe_loadable(), pattern is None)
        if pattern is not None:
            with self._env(**env_kwargs), self.assertRaisesRegex(ImportError, pattern):
                sm.is_sonicmoe_loadable(raise_error=True)

    @parameterized.expand(
        [
            ("unloadable_env", {"cuda_available": False}, "requires CUDA"),
            ("kernel_load_fails", {"kernel": None}, "Failed to load the sonic-moe kernel"),
            ("missing_symbols", {"kernel": _SonicMoeKernelMissingSymbol}, "missing required symbols"),
        ]
    )
    def test_loader_raises(self, _name, env_kwargs, pattern):
        # The loader delegates gating to `is_sonicmoe_loadable(raise_error=True)`, then resolves symbols;
        # confirm each failure surfaces through `load_sonicmoe_kernel`.
        with self._env(**env_kwargs), self.assertRaisesRegex(ImportError, pattern):
            sm.load_sonicmoe_kernel()

    def test_loader_is_compile_safe(self):
        # Cold path: the compiled call is first to load, so the opaque loader node runs its full body
        # under compile and must return None, never the bundle (`Unsupported: torch.* op returned
        # non-Tensor`). Default (inductor) backend; `_env` fakes CUDA+SM90 only for the loader so
        # torch.compile sees the real device — no GPU required.
        with self._env():
            torch.compiler.reset()

            @torch.compile(fullgraph=True)
            def run(x):
                out, _ = sm.load_sonicmoe_kernel().moe_general_routing_inputs(x)
                return out

            out = run(torch.zeros(3, device=torch_device))
        self.assertTrue(torch.equal(out, torch.ones(3, device=torch_device)))

    def test_loader_is_compile_safe_when_warm(self):
        # Warm path (production order: eager warmup, then compile). The loader hits its short-circuit at
        # trace time — the branch that must also return None, not the already-loaded bundle.
        with self._env():
            sm.load_sonicmoe_kernel()
            torch.compiler.reset()

            @torch.compile(fullgraph=True)
            def run(x):
                out, _ = sm.load_sonicmoe_kernel().moe_general_routing_inputs(x)
                return out

            out = run(torch.zeros(3, device=torch_device))
        self.assertTrue(torch.equal(out, torch.ones(3, device=torch_device)))

    def test_wrapper_is_compile_safe(self):
        # `_sonicmoe_wrapper`'s `@allow_in_graph` must keep the CuteDSL dispatch opaque — sonic-moe's
        # kernel asserts `not is_compiling()`. Targets the wrapper directly (not the full forward, which
        # requires a CUDA device); the wrapper has no device dependency, so like the loader tests above it
        # needs no GPU. Args past `hidden_states` are dummies — the fake only reads it (for `zeros_like`).
        def fake_moe(hidden_states, *args, **kwargs):
            assert not torch.compiler.is_compiling()
            return torch.zeros_like(hidden_states), None

        bundle = sm.SonicMoE(activation_type_enum=_FakeActivationType, moe_general_routing_inputs=fake_moe)
        d = torch.zeros(1, device=torch_device)
        with mock.patch.object(sm, "load_sonicmoe_kernel", return_value=bundle):
            torch.compiler.reset()

            @torch.compile(fullgraph=True)
            def run(x):
                return sm._sonicmoe_wrapper(x, d, d, d, d, None, d, None, "silu", 4, True, True)

            out = run(torch.zeros(6, 8, dtype=torch.bfloat16, device=torch_device))
        self.assertEqual(out.shape, (6, 8))


@require_torch_gpu
class SonicMoeExpertsForwardTest(unittest.TestCase):
    """Drives the real `sonicmoe_experts_forward` with only `moe_general_routing_inputs` mocked."""

    def setUp(self):
        sm._SONICMOE = None
        self.addCleanup(setattr, sm, "_SONICMOE", None)

    @contextlib.contextmanager
    def _mocked_kernel(self, activation_type_enum=_FakeActivationType, assert_not_compiling=False):
        captured = {}

        def fake_moe(
            hidden_states,
            router_scores,
            token_idx,
            expert_ids,
            w1,
            b1,
            w2,
            b2,
            *,
            E,
            activation_type,
            is_inference_mode_enabled,
            concat_layout,
            stream_id,
        ):
            # sonic-moe's real kernel refuses to run while Dynamo is tracing (CuteDSL is untraceable);
            # `_sonicmoe_wrapper`'s `@allow_in_graph` must keep this dispatch opaque so it runs at runtime.
            if assert_not_compiling:
                assert not torch.compiler.is_compiling()
            captured.update(
                hidden_states=hidden_states,
                router_scores=router_scores,
                token_idx=token_idx,
                expert_ids=expert_ids,
                w1=w1,
                b1=b1,
                w2=w2,
                b2=b2,
                E=E,
                activation_type=activation_type,
                is_inference_mode_enabled=is_inference_mode_enabled,
                concat_layout=concat_layout,
            )
            return torch.zeros_like(hidden_states), None

        bundle = sm.SonicMoE(activation_type_enum=activation_type_enum, moe_general_routing_inputs=fake_moe)
        with mock.patch.object(sm, "load_sonicmoe_kernel", return_value=bundle):
            yield captured

    def _run(self, experts, *, top_k_index=None, top_k_weights=None, num_tokens=3, top_k=2):
        # gate_up_proj is (E, H, 2I) when transposed else (E, 2I, H) — hidden is axis 1 or 2 accordingly.
        hidden = experts.gate_up_proj.shape[1] if experts.is_transposed else experts.gate_up_proj.shape[2]
        hidden_states = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=torch_device)
        if top_k_index is None:
            top_k_index = torch.randint(0, experts.num_experts, (num_tokens, top_k), device=torch_device)
        if top_k_weights is None:
            top_k_weights = torch.rand(top_k_index.shape, dtype=torch.bfloat16, device=torch_device)
        with self._mocked_kernel() as captured:
            out = sonicmoe_experts_forward(experts, hidden_states, top_k_index, top_k_weights)
        return out, captured, hidden_states, top_k_index, top_k_weights

    def test_forward_marshals_routing_and_weights(self):
        experts = make_experts(num_experts=4, hidden=8, inter=16)
        _, captured, _, top_k_index, top_k_weights = self._run(experts, num_tokens=3, top_k=2)

        # Routing is flattened to (num_tokens * top_k,): token_idx repeats each token index top_k times
        # as int32 ([0,0,1,1,2,2]); expert_ids / router_scores are the real routing tensors, recast.
        self.assertEqual(captured["token_idx"].dtype, torch.int32)
        self.assertTrue(
            torch.equal(captured["token_idx"], torch.arange(3, device=torch_device).repeat_interleave(2).int())
        )
        self.assertEqual(captured["expert_ids"].dtype, torch.int32)
        self.assertTrue(torch.equal(captured["expert_ids"], top_k_index.reshape(-1).int()))
        self.assertTrue(torch.equal(captured["router_scores"], top_k_weights.reshape(-1)))

        # Weights are permuted to the (..., E) layout the kernel expects (value-exact).
        self.assertTrue(torch.equal(captured["w1"], experts.gate_up_proj.permute(1, 2, 0)))
        self.assertTrue(torch.equal(captured["w2"], experts.down_proj.permute(1, 2, 0)))

    def test_forward_passes_sentinel_expert_ids_unclamped(self):
        # EP sentinels (expert_ids >= num_experts) must reach the kernel unclamped — unlike the eager
        # path, sonic-moe drops them in its metadata stage. Regression guard against a stray clamp.
        experts = make_experts(num_experts=4, hidden=8, inter=16)
        top_k_index = torch.tensor([[0, 4], [1, 4], [2, 4]], device=torch_device)  # 4 == num_experts -> sentinel
        _, captured, _, _, _ = self._run(experts, top_k_index=top_k_index)
        self.assertTrue(torch.equal(captured["expert_ids"], top_k_index.reshape(-1).int()))
        self.assertEqual(int(captured["expert_ids"].max()), 4)

    def test_forward_sets_inference_mode_flag(self):
        experts = make_experts(num_experts=4, hidden=8, inter=16)
        with torch.no_grad():
            _, captured, _, _, _ = self._run(experts)
        self.assertTrue(captured["is_inference_mode_enabled"])
        with torch.enable_grad():
            _, captured, _, _, _ = self._run(experts)
        self.assertFalse(captured["is_inference_mode_enabled"])

    def test_forward_with_bias(self):
        experts = make_experts(num_experts=4, hidden=8, inter=16, has_bias=True)
        _, captured, _, _, _ = self._run(experts)
        self.assertTrue(torch.equal(captured["b1"], experts.gate_up_proj_bias))
        self.assertTrue(torch.equal(captured["b2"], experts.down_proj_bias))

    def test_forward_raises_on_unsupported_activation(self):
        experts = make_experts(hidden_act="tanh")
        with self.assertRaisesRegex(ValueError, "does not support the 'tanh' activation"):
            self._run(experts)
