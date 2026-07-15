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
  handed to the kernel are what the kernel expects (int32 indices, permuted weights, `E`, ...).
"""

import contextlib
import types
import unittest
from unittest import mock

import torch

import transformers.integrations.sonicmoe as sm
from transformers.integrations.sonicmoe import sonicmoe_experts_forward
from transformers.testing_utils import require_torch, require_torch_gpu


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

            @torch.compile(backend="aot_eager", fullgraph=True)
            def run(x):
                out, _ = sm.load_sonicmoe_kernel().moe_general_routing_inputs(x)
                return out

            out = run(torch.zeros(3))
        self.assertTrue(torch.equal(out, torch.ones(3)))


def _make_experts(
    *,
    num_experts=4,
    hidden=8,
    inter=16,
    has_bias=False,
    is_transposed=False,
    is_concatenated=True,
    hidden_act="silu",
    dtype=torch.bfloat16,
    device="cuda",
):
    if is_transposed:
        gate_up = torch.randn(num_experts, hidden, 2 * inter, dtype=dtype, device=device)
        down = torch.randn(num_experts, inter, hidden, dtype=dtype, device=device)
    else:
        gate_up = torch.randn(num_experts, 2 * inter, hidden, dtype=dtype, device=device)
        down = torch.randn(num_experts, hidden, inter, dtype=dtype, device=device)
    return types.SimpleNamespace(
        has_gate=True,
        has_bias=has_bias,
        gate_up_proj=gate_up,
        down_proj=down,
        gate_up_proj_bias=torch.randn(num_experts, 2 * inter, dtype=dtype, device=device) if has_bias else None,
        down_proj_bias=torch.randn(num_experts, hidden, dtype=dtype, device=device) if has_bias else None,
        config=types.SimpleNamespace(hidden_act=hidden_act),
        is_transposed=is_transposed,
        num_experts=num_experts,
        is_concatenated=is_concatenated,
    )


@require_torch_gpu
class SonicMoeExpertsForwardTest(unittest.TestCase):
    """Drives the real `sonicmoe_experts_forward` with only `moe_general_routing_inputs` mocked."""

    def setUp(self):
        sm._SONICMOE = None
        self.addCleanup(setattr, sm, "_SONICMOE", None)

    @contextlib.contextmanager
    def _mocked_kernel(self, activation_type_enum=_FakeActivationType):
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
        hidden_states = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device="cuda")
        if top_k_index is None:
            top_k_index = torch.randint(0, experts.num_experts, (num_tokens, top_k), device="cuda")
        if top_k_weights is None:
            top_k_weights = torch.rand(top_k_index.shape, dtype=torch.bfloat16, device="cuda")
        with self._mocked_kernel() as captured:
            out = sonicmoe_experts_forward(experts, hidden_states, top_k_index, top_k_weights)
        return out, captured, hidden_states, top_k_index, top_k_weights

    def test_forward_shapes_and_kernel_inputs(self):
        experts = _make_experts(num_experts=4, hidden=8, inter=16)
        out, captured, hidden_states, top_k_index, top_k_weights = self._run(experts, num_tokens=3, top_k=2)

        # Output mirrors the (num_tokens, hidden) input.
        self.assertEqual(out.shape, hidden_states.shape)
        self.assertEqual(out.dtype, hidden_states.dtype)

        # token_idx is int32 and repeats each token index top_k times, ascending: [0,0,1,1,2,2].
        expected_token_idx = torch.arange(3, device="cuda").repeat_interleave(2)
        self.assertEqual(captured["token_idx"].dtype, torch.int32)
        self.assertTrue(torch.equal(captured["token_idx"], expected_token_idx.int()))

        # expert_ids / router_scores are the real routing tensors, flattened row-major and recast.
        self.assertEqual(captured["expert_ids"].dtype, torch.int32)
        self.assertTrue(torch.equal(captured["expert_ids"], top_k_index.reshape(-1).int()))
        self.assertEqual(captured["router_scores"].dtype, hidden_states.dtype)
        self.assertTrue(torch.equal(captured["router_scores"], top_k_weights.reshape(-1)))

        # Weights are permuted to (..., E) — value-exact, not just shape; activation maps silu -> SWIGLU.
        self.assertTrue(torch.equal(captured["w1"], experts.gate_up_proj.permute(1, 2, 0)))
        self.assertTrue(torch.equal(captured["w2"], experts.down_proj.permute(1, 2, 0)))
        self.assertEqual(captured["w1"].shape, (2 * 16, 8, 4))
        self.assertEqual(captured["w2"].shape, (8, 16, 4))
        self.assertEqual(captured["E"], 4)
        self.assertEqual(captured["activation_type"], _FakeActivationType.SWIGLU)
        self.assertEqual(captured["concat_layout"], experts.is_concatenated)
        self.assertIsNone(captured["b1"])
        self.assertIsNone(captured["b2"])

    def test_forward_passes_sentinel_expert_ids_unclamped(self):
        # EP sentinels (expert_ids >= num_experts) must reach the kernel unclamped — unlike the eager
        # path, sonic-moe drops them in its metadata stage. Regression guard against a stray clamp.
        experts = _make_experts(num_experts=4, hidden=8, inter=16)
        top_k_index = torch.tensor([[0, 4], [1, 4], [2, 4]], device="cuda")  # 4 == num_experts -> sentinel
        _, captured, _, _, _ = self._run(experts, top_k_index=top_k_index)
        self.assertTrue(torch.equal(captured["expert_ids"], top_k_index.reshape(-1).int()))
        self.assertEqual(int(captured["expert_ids"].max()), 4)

    def test_forward_sets_inference_mode_flag(self):
        experts = _make_experts(num_experts=4, hidden=8, inter=16)
        with torch.no_grad():
            _, captured, _, _, _ = self._run(experts)
        self.assertTrue(captured["is_inference_mode_enabled"])
        with torch.enable_grad():
            _, captured, _, _, _ = self._run(experts)
        self.assertFalse(captured["is_inference_mode_enabled"])

    def test_forward_with_bias(self):
        experts = _make_experts(num_experts=4, hidden=8, inter=16, has_bias=True)
        _, captured, _, _, _ = self._run(experts)
        self.assertTrue(torch.equal(captured["b1"], experts.gate_up_proj_bias))
        self.assertTrue(torch.equal(captured["b2"], experts.down_proj_bias))

    def test_forward_transposed_weight_layout(self):
        # Transposed uses permute(2, 1, 0): gate_up (E, H, 2I) -> (2I, H, E); down (E, I, H) -> (H, I, E),
        # i.e. transposed and non-transposed weights normalize to the SAME (..., E) kernel layout.
        experts = _make_experts(num_experts=4, hidden=8, inter=16, is_transposed=True)
        _, captured, _, _, _ = self._run(experts)
        self.assertTrue(torch.equal(captured["w1"], experts.gate_up_proj.permute(2, 1, 0)))
        self.assertTrue(torch.equal(captured["w2"], experts.down_proj.permute(2, 1, 0)))
        self.assertEqual(captured["w1"].shape, (2 * 16, 8, 4))
        self.assertEqual(captured["w2"].shape, (8, 16, 4))

    def test_forward_activation_mapping(self):
        for act, expected in [
            ("silu", _FakeActivationType.SWIGLU),
            ("gelu", _FakeActivationType.GEGLU),
            ("relu", _FakeActivationType.REGLU),
        ]:
            with self.subTest(act=act):
                experts = _make_experts(hidden_act=act)
                _, captured, _, _, _ = self._run(experts)
                self.assertEqual(captured["activation_type"], expected)

    def test_forward_raises_on_unsupported_activation(self):
        experts = _make_experts(hidden_act="tanh")
        with self.assertRaisesRegex(ValueError, "does not support the 'tanh' activation"):
            self._run(experts)


if __name__ == "__main__":
    unittest.main()
