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
"""Tests for the built-in experts implementations in `integrations/moe.py`.

Expert parallelism leaves each rank only its own experts, and the router marks the slots it does not own
with a sentinel expert id and a zero routing weight. `batched_mm_experts_forward` clamps those slots into
a real expert to keep the weight gather in bounds, so they have to be kept out of the output and of the
gradient explicitly.
"""

import copy
import types
import unittest
from unittest.mock import patch

import torch

from transformers.activations import ACT2FN
from transformers.integrations.moe import (
    ExpertsHookContext,
    batched_mm_experts_forward,
    grouped_mm_experts_forward,
    use_experts_implementation,
)
from transformers.testing_utils import require_torch, torch_device


def make_experts(num_experts, hidden, inter, is_expert_parallel):
    """The attributes `batched_mm_experts_forward` reads off an experts module, with real weights."""
    act_fn = ACT2FN["silu"]

    def apply_gate(gate_up):
        gate, up = gate_up.chunk(2, dim=-1)
        return act_fn(gate) * up

    return types.SimpleNamespace(
        num_experts=num_experts,
        has_gate=True,
        has_bias=False,
        is_transposed=False,
        act_fn=act_fn,
        _apply_gate=apply_gate,
        gate_up_proj=torch.randn(num_experts, 2 * inter, hidden, device=torch_device),
        gate_up_proj_bias=None,
        down_proj=torch.randn(num_experts, hidden, inter, device=torch_device),
        down_proj_bias=None,
        _is_expert_parallel=is_expert_parallel,
        has_post_expert_norm=False,
    )


NUM_EXPERTS = 4
# Slots (0, 1) and (2, 0) are sentinels: expert id `NUM_EXPERTS`, routing weight 0.
TOP_K_INDEX = [[0, NUM_EXPERTS], [1, 2], [NUM_EXPERTS, 3]]
TOP_K_WEIGHTS = [[0.7, 0.0], [0.4, 0.6], [0.0, 0.9]]


@require_torch
class BatchedMmExpertsForwardTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        # The sentinel slots only exist under expert parallelism, which is also what gates the handling.
        self.experts = make_experts(num_experts=NUM_EXPERTS, hidden=8, inter=16, is_expert_parallel=True)
        self.hidden_states = torch.randn(3, 8, device=torch_device)
        self.top_k_index = torch.tensor(TOP_K_INDEX, device=torch_device)

    def _weights(self, requires_grad=False):
        return torch.tensor(TOP_K_WEIGHTS, device=torch_device, dtype=torch.float32, requires_grad=requires_grad)

    def test_sentinel_slots_get_no_routing_weight_gradient(self):
        top_k_weights = self._weights(requires_grad=True)
        out = batched_mm_experts_forward(self.experts, self.hidden_states, self.top_k_index, top_k_weights)
        out.sum().backward()

        # The clamp leaves a sentinel slot a real expert output, so only the zero weight keeps it out of the
        # forward, and multiplying by zero does not stop the gradient reaching the weight on the other side.
        sentinel = self.top_k_index >= NUM_EXPERTS
        self.assertTrue(torch.all(top_k_weights.grad[sentinel] == 0))
        # The routed slots must still get one, or the assert above would pass on an all-zero gradient.
        self.assertTrue(torch.all(top_k_weights.grad[~sentinel] != 0))

    def test_sentinel_slots_do_not_reach_the_output(self):
        out = batched_mm_experts_forward(self.experts, self.hidden_states, self.top_k_index, self._weights())

        # Which expert a sentinel slot is clamped into cannot matter, so moving it leaves the output alone.
        moved = self.top_k_index.masked_fill(self.top_k_index >= NUM_EXPERTS, NUM_EXPERTS + 1)
        out_moved = batched_mm_experts_forward(self.experts, self.hidden_states, moved, self._weights())
        torch.testing.assert_close(out, out_moved)


@require_torch
class GroupedMmFallbackCompileTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

        self.experts = make_experts(num_experts=NUM_EXPERTS, hidden=8, inter=16, is_expert_parallel=False)
        self.experts.gate_up_proj.requires_grad_(True)
        self.experts.down_proj.requires_grad_(True)

        self.hidden_states = torch.randn(3, 8, device=torch_device, requires_grad=True)
        # Leave experts 1 and 3 empty
        self.top_k_index = torch.tensor([[0, 2], [2, 0], [0, 2]], device=torch_device)
        self.top_k_weights = torch.tensor(
            [[0.7, 0.3], [0.4, 0.6], [0.2, 0.8]],
            device=torch_device,
        )

    def test_forward_backward_fullgraph(self):
        def forward():
            return grouped_mm_experts_forward(
                self.experts,
                self.hidden_states,
                self.top_k_index,
                self.top_k_weights,
            )

        def run(fn):
            self.hidden_states.grad = None
            self.experts.gate_up_proj.grad = None
            self.experts.down_proj.grad = None

            output = fn()
            output.square().sum().backward()  # arbitrary we only want the bwd to be triggered

            return (
                output.detach().clone(),
                self.hidden_states.grad.detach().clone(),
                self.experts.gate_up_proj.grad.detach().clone(),
                self.experts.down_proj.grad.detach().clone(),
            )

        # Force the fallback to be selected
        with patch(
            "transformers.integrations.moe._can_use_grouped_mm",
            new=lambda *args: False,
        ):
            eager_results = run(forward)
            compiled_results = run(torch.compile(forward, fullgraph=True))

        for eager, compiled in zip(eager_results, compiled_results):
            torch.testing.assert_close(compiled, eager, rtol=1e-4, atol=1e-4)

        # Unused experts should have zero gradients
        for grad in compiled_results[2:]:
            self.assertTrue(torch.all(grad[1::2] == 0))


@require_torch
class ExpertsHooksTest(unittest.TestCase):
    def _make_module(self, impl="grouped_mm", has_gate=True, is_expert_parallel=False, dtype=torch.float32):
        @use_experts_implementation(has_gate=has_gate)
        class DummyExperts(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.num_experts = config.num_experts
                self.act_fn = ACT2FN["silu"]
                if has_gate:
                    self.gate_up_proj = torch.nn.Parameter(
                        torch.randn(
                            config.num_experts, 2 * config.inter, config.hidden, device=torch_device, dtype=dtype
                        )
                    )
                else:
                    self.up_proj = torch.nn.Parameter(
                        torch.randn(config.num_experts, config.inter, config.hidden, device=torch_device, dtype=dtype)
                    )
                self.down_proj = torch.nn.Parameter(
                    torch.randn(config.num_experts, config.hidden, config.inter, device=torch_device, dtype=dtype)
                )

            def forward(self, hidden_states, top_k_index, top_k_weights):
                return torch.zeros_like(hidden_states)

        cfg = types.SimpleNamespace(
            num_experts=NUM_EXPERTS,
            hidden=8,
            inter=16,
            _experts_implementation=impl,
        )
        mod = DummyExperts(cfg)
        mod._is_expert_parallel = is_expert_parallel
        return mod

    def setUp(self):
        torch.manual_seed(42)
        self.hidden_states = torch.randn(4, 8, device=torch_device)
        self.top_k_index = torch.tensor([[2, 0], [1, 3], [0, 2], [3, 1]], device=torch_device)
        self.top_k_weights = torch.tensor(
            [[0.7, 0.3], [0.6, 0.4], [0.2, 0.8], [0.5, 0.5]],
            device=torch_device,
        )

    def test_read_hooks_match_reference_and_are_bit_identical(self):
        for impl in ("grouped_mm", "batched_mm"):
            for has_gate in (True, False):
                torch.manual_seed(7)
                mod = self._make_module(impl=impl, has_gate=has_gate)
                baseline = mod(self.hidden_states, self.top_k_index, self.top_k_weights)

                captured_act = []
                captured_out = []

                def act_hook(module, activations, ctx):
                    self.assertIs(module, mod)
                    self.assertIsInstance(ctx, ExpertsHookContext)
                    self.assertIsNone(ctx.valid_mask)
                    self.assertEqual(ctx.num_tokens, 4)
                    self.assertEqual(ctx.num_top_k, 2)
                    captured_act.append((ctx.to_token_order(activations).detach().clone(), ctx))
                    return None

                def out_hook(module, outputs, ctx):
                    self.assertIs(module, mod)
                    self.assertIsInstance(ctx, ExpertsHookContext)
                    captured_out.append(ctx.to_token_order(outputs).detach().clone())
                    return None

                with mod.register_activation_hook(act_hook), mod.register_output_hook(out_hook):
                    with_hooks = mod(self.hidden_states, self.top_k_index, self.top_k_weights)

                # Read-only hooks must be bit-identical to unhooked forward.
                self.assertTrue(torch.equal(baseline, with_hooks))
                self.assertEqual(len(captured_act), 1)
                self.assertEqual(len(captured_out), 1)

                ordered_act, ctx = captured_act[0]
                ordered_out = captured_out[0]
                for t in range(4):
                    for s in range(2):
                        e = int(self.top_k_index[t, s])
                        x = self.hidden_states[t : t + 1]
                        if has_gate:
                            g, u = torch.nn.functional.linear(x, mod.gate_up_proj[e]).chunk(2, dim=-1)
                            expected_act = mod.act_fn(g) * u
                        else:
                            expected_act = mod.act_fn(torch.nn.functional.linear(x, mod.up_proj[e]))
                        expected_out = torch.nn.functional.linear(expected_act, mod.down_proj[e])
                        torch.testing.assert_close(ordered_act[t, s : s + 1], expected_act, rtol=2e-3, atol=2e-3)
                        torch.testing.assert_close(ordered_out[t, s : s + 1], expected_out, rtol=2e-3, atol=2e-3)

                for row in range(ctx.expert_ids.numel()):
                    t = int(ctx.token_ids[row])
                    s = int(ctx.top_k_slot[row])
                    self.assertEqual(int(ctx.expert_ids[row]), int(self.top_k_index[t, s]))
                    self.assertAlmostEqual(float(ctx.top_k_weights[row]), float(self.top_k_weights[t, s]), places=6)

                # After context managers exit, hooks must be removed.
                _ = mod(self.hidden_states, self.top_k_index, self.top_k_weights)
                self.assertEqual(len(captured_act), 1)
                self.assertEqual(len(captured_out), 1)

    def test_write_hooks_multi_hook_composition_and_autograd(self):
        for impl in ("grouped_mm", "batched_mm"):
            torch.manual_seed(11)
            mod = self._make_module(impl=impl, has_gate=True)
            x = self.hidden_states.detach().clone().requires_grad_(True)

            # Activation hooks: zero expert 2, then scale all activations by 2.
            # Output hook: scale expert 1 outputs by 0.5.
            h1 = mod.register_activation_hook(
                lambda m, act, ctx: torch.where((ctx.expert_ids == 2).unsqueeze(-1), torch.zeros_like(act), act)
            )
            h2 = mod.register_activation_hook(lambda m, act, ctx: act * 2.0)
            h3 = mod.register_output_hook(
                lambda m, out, ctx: torch.where((ctx.expert_ids == 1).unsqueeze(-1), out * 0.5, out)
            )
            try:
                out = mod(x, self.top_k_index, self.top_k_weights)
                out.square().sum().backward()
            finally:
                h1.remove()
                h2.remove()
                h3.remove()

            # Compute exact manual reference with the same interventions.
            x_ref = self.hidden_states.detach().clone().requires_grad_(True)
            gate_up_ref = mod.gate_up_proj.detach().clone().requires_grad_(True)
            down_ref = mod.down_proj.detach().clone().requires_grad_(True)
            out_ref = torch.zeros_like(x_ref)
            for t in range(4):
                for s in range(2):
                    e = int(self.top_k_index[t, s])
                    g, u = torch.nn.functional.linear(x_ref[t : t + 1], gate_up_ref[e]).chunk(2, dim=-1)
                    h = mod.act_fn(g) * u
                    if e == 2:
                        h = torch.zeros_like(h)
                    h = h * 2.0
                    y = torch.nn.functional.linear(h, down_ref[e])
                    if e == 1:
                        y = y * 0.5
                    out_ref[t : t + 1] = out_ref[t : t + 1] + y * self.top_k_weights[t, s]
            out_ref.square().sum().backward()

            torch.testing.assert_close(out, out_ref, rtol=2e-3, atol=2e-3)
            torch.testing.assert_close(x.grad, x_ref.grad, rtol=2e-3, atol=2e-3)
            torch.testing.assert_close(mod.gate_up_proj.grad, gate_up_ref.grad, rtol=2e-3, atol=2e-3)
            torch.testing.assert_close(mod.down_proj.grad, down_ref.grad, rtol=2e-3, atol=2e-3)
            # Zeroed expert 2 must receive zero gradient on gate_up_proj and down_proj.
            self.assertTrue(torch.all(mod.gate_up_proj.grad[2] == 0))
            self.assertTrue(torch.all(mod.down_proj.grad[2] == 0))

    def test_ep_sentinel_isolation_under_functional_and_inplace_hooks(self):
        top_k_index = torch.tensor(TOP_K_INDEX, device=torch_device)
        for impl in ("grouped_mm", "batched_mm"):
            for use_inplace in (False, True):
                torch.manual_seed(13)
                mod = self._make_module(impl=impl, has_gate=True, is_expert_parallel=True)
                x = torch.randn(3, 8, device=torch_device, requires_grad=not use_inplace)
                w = torch.tensor(TOP_K_WEIGHTS, device=torch_device, dtype=torch.float32, requires_grad=True)

                seen_valid = []

                def steer_act(module, act, ctx):
                    self.assertIsNotNone(ctx.valid_mask)
                    self.assertTrue(torch.all(act[~ctx.valid_mask] == 0))
                    seen_valid.append(ctx.valid_mask.detach().clone())
                    if use_inplace:
                        act.add_(1.5)
                        return None
                    return act + 1.5

                def steer_out(module, out_t, ctx):
                    self.assertIsNotNone(ctx.valid_mask)
                    self.assertTrue(torch.all(out_t[~ctx.valid_mask] == 0))
                    if use_inplace:
                        out_t.add_(2.0)
                        return None
                    return out_t + 2.0

                with mod.register_activation_hook(steer_act), mod.register_output_hook(steer_out):
                    out = mod(x, top_k_index, w)
                    out.sum().backward()

                self.assertEqual(len(seen_valid), 1)
                self.assertEqual(int(seen_valid[0].sum()), 4)
                sentinel = top_k_index >= NUM_EXPERTS
                # Sentinel slots must receive zero routing weight gradient even under additive steering.
                self.assertTrue(torch.all(w.grad[sentinel] == 0))
                self.assertTrue(torch.all(w.grad[~sentinel] != 0))

    def test_fp32_hook_return_autocasts_to_bf16_with_autograd(self):
        for impl in ("grouped_mm", "batched_mm"):
            torch.manual_seed(19)
            mod = self._make_module(impl=impl, has_gate=True, dtype=torch.bfloat16)
            x = torch.randn(4, 8, device=torch_device, dtype=torch.bfloat16, requires_grad=True)
            w = self.top_k_weights.to(torch.bfloat16)

            with (
                mod.register_activation_hook(lambda m, act, ctx: act.float() * 0.5),
                mod.register_output_hook(lambda m, out_t, ctx: out_t.float() * 2.0),
            ):
                out = mod(x, self.top_k_index, w)
                out.float().sum().backward()

            self.assertEqual(out.dtype, torch.bfloat16)
            self.assertIsNotNone(x.grad)
            self.assertTrue(torch.isfinite(x.grad).all())

    def test_torch_compile_fullgraph_unhooked_and_with_active_hooks(self):
        torch.manual_seed(17)
        mod = self._make_module(impl="grouped_mm", has_gate=True)
        x = self.hidden_states.detach().clone().requires_grad_(True)

        def run(fn):
            x.grad = None
            mod.gate_up_proj.grad = None
            mod.down_proj.grad = None
            out = fn()
            out.square().sum().backward()
            return out.detach().clone(), x.grad.detach().clone(), mod.down_proj.grad.detach().clone()

        def forward():
            return mod(x, self.top_k_index, self.top_k_weights)

        with patch("transformers.integrations.moe._can_use_grouped_mm", new=lambda *args: False):
            # 1. Unhooked through use_experts_implementation.forward (zero graph breaks).
            eager_base = run(forward)
            compiled_base = run(torch.compile(forward, fullgraph=True))
            for e, c in zip(eager_base, compiled_base):
                torch.testing.assert_close(c, e, rtol=1e-4, atol=1e-4)

            # 2. Active activation + output edit hooks (zero graph breaks).
            def act_scale(module, act, ctx):
                return act * torch.where((ctx.expert_ids == 0).unsqueeze(-1), 0.5, 1.0)

            def out_scale(module, out_t, ctx):
                return out_t * torch.where((ctx.expert_ids == 2).unsqueeze(-1), 1.5, 1.0)

            with mod.register_activation_hook(act_scale), mod.register_output_hook(out_scale):
                eager = run(forward)
                compiled = run(torch.compile(forward, fullgraph=True))

        for e, c in zip(eager, compiled):
            torch.testing.assert_close(c, e, rtol=1e-4, atol=1e-4)

    def test_shape_and_device_mismatch_raise_value_error(self):
        mod = self._make_module(impl="grouped_mm", has_gate=True)
        with mod.register_activation_hook(lambda m, act, ctx: act[:, :4]):
            with self.assertRaisesRegex(ValueError, "shape"):
                mod(self.hidden_states, self.top_k_index, self.top_k_weights)
        with mod.register_output_hook(lambda m, out_t, ctx: out_t[:, :4]):
            with self.assertRaisesRegex(ValueError, "shape"):
                mod(self.hidden_states, self.top_k_index, self.top_k_weights)
        if torch_device != "cpu":
            with mod.register_activation_hook(lambda m, act, ctx: act.cpu()):
                with self.assertRaisesRegex(ValueError, "device"):
                    mod(self.hidden_states, self.top_k_index, self.top_k_weights)

    def test_unsupported_backend_raises_not_implemented_error(self):
        mod = self._make_module(impl="eager", has_gate=True)
        with mod.register_activation_hook(lambda m, act, ctx: None):
            with self.assertRaisesRegex(NotImplementedError, "not supported"):
                mod(self.hidden_states, self.top_k_index, self.top_k_weights)
        with mod.register_output_hook(lambda m, out_t, ctx: None):
            with self.assertRaisesRegex(NotImplementedError, "not supported"):
                mod(self.hidden_states, self.top_k_index, self.top_k_weights)

    def test_deepgemm_hooks_functional_and_inplace_and_output_site(self):
        from transformers.integrations.deepgemm import (
            _combine_routed_output,
            _dispatch_routed_input,
            _maybe_apply_deepgemm_activation_hooks,
            _pad_for_deepgemm,
            _unpad_from_deepgemm_contiguous_layout,
        )

        mod = self._make_module(impl="grouped_mm", has_gate=True, is_expert_parallel=True)
        top_k_index = torch.tensor(TOP_K_INDEX, device=torch_device)
        top_k_weights = torch.tensor(TOP_K_WEIGHTS, device=torch_device, dtype=torch.float32)
        x = torch.randn(3, 8, device=torch_device)

        _, sorted_weights, expert_ids_g, sentinel_mask, perm, sorted_to_padded, _, total_padded = (
            _dispatch_routed_input(
                x,
                top_k_index,
                top_k_weights,
                NUM_EXPERTS,
                m_alignment=4,
                use_psum_layout=False,
                is_expert_parallel=True,
            )
        )
        raw_unpadded = torch.randn(6, 16, device=torch_device)
        padded = _pad_for_deepgemm(raw_unpadded, sorted_to_padded, total_padded)

        # Unhooked returns exact same tensor object.
        self.assertIs(
            _maybe_apply_deepgemm_activation_hooks(
                mod,
                padded,
                sorted_to_padded=sorted_to_padded,
                expert_ids=expert_ids_g,
                perm=perm,
                top_k_weights=sorted_weights,
                num_tokens=3,
                num_top_k=2,
                sentinel_mask=sentinel_mask,
            ),
            padded,
        )

        # Functional modification writes back valid rows and keeps sentinels zeroed.
        with mod.register_activation_hook(lambda m, act, ctx: act * 3.0 + 1.0):
            updated = _maybe_apply_deepgemm_activation_hooks(
                mod,
                padded.clone(),
                sorted_to_padded=sorted_to_padded,
                expert_ids=expert_ids_g,
                perm=perm,
                top_k_weights=sorted_weights,
                num_tokens=3,
                num_top_k=2,
                sentinel_mask=sentinel_mask,
            )
        expected = torch.where(sentinel_mask, torch.zeros_like(raw_unpadded), raw_unpadded * 3.0 + 1.0)
        torch.testing.assert_close(_unpad_from_deepgemm_contiguous_layout(updated, sorted_to_padded), expected)

        # In-place mutation returning None must also propagate back to padded tensor!
        def inplace_hook(m, act, ctx):
            act.mul_(2.0).add_(0.5)
            return None

        with mod.register_activation_hook(inplace_hook):
            updated_inplace = _maybe_apply_deepgemm_activation_hooks(
                mod,
                padded.clone(),
                sorted_to_padded=sorted_to_padded,
                expert_ids=expert_ids_g,
                perm=perm,
                top_k_weights=sorted_weights,
                num_tokens=3,
                num_top_k=2,
                sentinel_mask=sentinel_mask,
            )
        expected_ip = torch.where(sentinel_mask, torch.zeros_like(raw_unpadded), raw_unpadded * 2.0 + 0.5)
        torch.testing.assert_close(
            _unpad_from_deepgemm_contiguous_layout(updated_inplace, sorted_to_padded), expected_ip
        )

        # Output hook inside _combine_routed_output.
        raw_out = torch.randn(6, 8, device=torch_device)
        out_padded = _pad_for_deepgemm(raw_out, sorted_to_padded, total_padded)
        base_combined = _combine_routed_output(
            out_padded,
            sorted_weights,
            sentinel_mask,
            perm,
            sorted_to_padded,
            3,
            2,
            8,
            torch.float32,
            module=mod,
            expert_ids=expert_ids_g,
        )
        with mod.register_output_hook(lambda m, o, ctx: o * 2.0):
            scaled_combined = _combine_routed_output(
                out_padded,
                sorted_weights,
                sentinel_mask,
                perm,
                sorted_to_padded,
                3,
                2,
                8,
                torch.float32,
                module=mod,
                expert_ids=expert_ids_g,
            )
        torch.testing.assert_close(scaled_combined, base_combined * 2.0)

    def test_fp8_backends_end_to_end_with_mocked_kernels(self):
        from transformers.integrations.finegrained_fp8 import (
            fp8_batched_mm_experts_forward,
            fp8_grouped_mm_experts_forward,
        )

        torch.manual_seed(23)
        mod = self._make_module(impl="grouped_mm", has_gate=True, is_expert_parallel=False)
        mod.activation_scheme = "dynamic"
        mod.block_size = (128, 128)
        mod.gate_up_proj_scale_inv = torch.ones(NUM_EXPERTS, 1, 1, device=torch_device)
        mod.down_proj_scale_inv = torch.ones(NUM_EXPERTS, 1, 1, device=torch_device)

        def fake_batched_mm(x, w, w_s, block_size, expert_ids):
            return torch.bmm(w[expert_ids], x.unsqueeze(-1)).squeeze(-1)

        def fake_grouped_mm(x, w, w_s, offs, tokens_per_expert, block_size):
            out = torch.zeros(x.size(0), w.size(1), device=x.device, dtype=x.dtype)
            start = 0
            for i, end in enumerate(offs.tolist()):
                if start < end:
                    out[start:end] = torch.nn.functional.linear(x[start:end], w[i])
                start = end
            return out

        fake_kernel = types.SimpleNamespace(batched_matmul=fake_batched_mm)

        with (
            patch("transformers.integrations.finegrained_fp8.load_finegrained_fp8_kernel", return_value=fake_kernel),
            patch("transformers.integrations.finegrained_fp8.fp8_grouped_matmul", side_effect=fake_grouped_mm),
        ):
            for fn in (fp8_batched_mm_experts_forward, fp8_grouped_mm_experts_forward):
                base = fn(mod, self.hidden_states, self.top_k_index, self.top_k_weights)
                seen_act, seen_out = [], []
                with (
                    mod.register_activation_hook(
                        lambda m, a, ctx: seen_act.append(ctx.to_token_order(a)) or (a * 2.0)
                    ),
                    mod.register_output_hook(lambda m, o, ctx: seen_out.append(ctx.to_token_order(o)) or (o * 0.5)),
                ):
                    # *2 on activations then *0.5 on outputs cancels exactly (down_proj is linear, no bias).
                    hooked = fn(mod, self.hidden_states, self.top_k_index, self.top_k_weights)
                self.assertEqual(len(seen_act), 1)
                self.assertEqual(len(seen_out), 1)
                self.assertEqual(seen_act[0].shape, (4, 2, 16))
                self.assertEqual(seen_out[0].shape, (4, 2, 8))
                torch.testing.assert_close(hooked, base, rtol=2e-3, atol=2e-3)

    def test_no_state_dict_submodule_pollution_and_deepcopy_safe(self):
        mod = self._make_module(impl="grouped_mm", has_gate=True)
        before_modules = [name for name, _ in mod.named_modules()]
        before_keys = list(mod.state_dict().keys())

        with mod.register_activation_hook(lambda m, act, ctx: act * 0.5):
            self.assertEqual([name for name, _ in mod.named_modules()], before_modules)
            self.assertEqual(list(mod.state_dict().keys()), before_keys)
            _ = mod(self.hidden_states, self.top_k_index, self.top_k_weights)

        cloned = copy.deepcopy(mod)
        out_orig = mod(self.hidden_states, self.top_k_index, self.top_k_weights)
        out_cloned = cloned(self.hidden_states, self.top_k_index, self.top_k_weights)
        torch.testing.assert_close(out_orig, out_cloned)
        self.assertEqual([name for name, _ in mod.named_modules()], before_modules)
        self.assertEqual(list(mod.state_dict().keys()), before_keys)
