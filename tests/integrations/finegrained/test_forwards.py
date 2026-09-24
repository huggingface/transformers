# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""The same chains as the marshalling tests, against the build the integration actually loads:
what those assert is passed to the kernels, these assert comes back out of them. Needs an
accelerator, since nothing here is mocked."""

import unittest
from unittest import mock

import torch

import transformers.integrations.finegrained.core as fg
from transformers.integrations.finegrained import (
    FineGrainedExperts,
    finegrained_linear,
    load_finegrained_kernel,
)
from transformers.testing_utils import (
    backend_synchronize,
    require_kernels,
    require_torch_accelerator,
    torch_device,
)

from .test_core import _Cfg


class _InputScaledRMSNorm(torch.nn.Module):
    """A post-expert norm that scales by ``1 + weight`` BEFORE normalizing, so the row's
    mean square is taken on the scaled values (not the same function as normalizing first)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.zeros(dim, dtype=torch.bfloat16))

    def forward(self, x):
        out = x.float() * (1.0 + self.weight.float())
        return (out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)


class _UnfusableNorm(torch.nn.Module):
    """A norm whose math is none of the fused forms (a bias the kernels have no slot for): the
    binder has to leave it a module call rather than pick the nearest form."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim, dtype=torch.bfloat16))
        self.bias = torch.nn.Parameter(torch.full((dim,), 0.5, dtype=torch.bfloat16))

    def forward(self, x):
        xf = x.float()
        normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed * self.weight.float() + self.bias.float()).type_as(x)


def _nvfp4_two_level(kernel, weight):
    """Canonical two-level NVFP4 quant of one matrix: the global is ``amax / (6 * 448)`` — the
    smallest one that keeps every e4m3 block scale in range — and the block quant runs on the
    normalized values. Returns ``(packed_e2m1, e4m3 block scales, fp32 global)``."""
    global_scale = (weight.abs().amax() / (6.0 * 448.0)).clamp(min=torch.finfo(torch.float32).tiny)
    packed, block = kernel.nvfp4_act_quant((weight.float() / global_scale).contiguous())
    return packed.view(torch.int8), block, global_scale


def _unpack_e2m1(packed):
    """Packed-E2M1 bytes (two codes per byte along the last axis) back to fp32 values."""
    lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device=packed.device,
    )
    b = packed.view(torch.uint8)
    pairs = torch.stack([lut[(b & 0x0F).long()], lut[(b >> 4).long()]], dim=-1)
    return pairs.reshape(*packed.shape[:-1], packed.shape[-1] * 2)


@require_kernels
@require_torch_accelerator
class FineGrainedForwardTest(unittest.TestCase):
    """`test_core.py` pins the calling contract; this pins that the contract produces correct
    numbers, which no amount of argument-matching can show.

    The load is NOT wrapped in a skip: `kernels` missing is the one environment reason to sit these
    out and the decorator states it, so anything else that stops the bundle loading — a broken
    loader, no build for this torch, a bad hub entry — has to fail, not quietly pass."""

    @classmethod
    def setUpClass(cls):
        load_finegrained_kernel()

    def _block_fp8_weight(self, N, K, *, ue8m0):
        """(N, K) E4M3 + its (N/128, K/128) inv-scale grid, and the exact dequantized values the
        kernel will see — so the reference is the quantization floor, not the pre-quant weight."""
        w = torch.randn(N, K, device=torch_device, dtype=torch.float32)
        blocks = w.reshape(N // 128, 128, K // 128, 128)
        amax = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12)
        inv = amax / 448.0
        if ue8m0:  # power-of-two scales: the tcgen05 dot_scaled format
            inv = torch.pow(2.0, torch.ceil(torch.log2(inv)))
        q = (blocks / inv).to(torch.float8_e4m3fn)
        deq = (q.float() * inv).reshape(N, K)
        return q.reshape(N, K), inv.reshape(N // 128, K // 128).contiguous(), deq

    def test_linear_forward_matches_the_quantization_floor(self):
        """A real quantized linear against its own dequantized weight. Loose vs bf16 would pass
        even if the kernel silently dropped a K-block; comparing to the floor does not."""
        M, N, K = 256, 512, 384
        for ue8m0 in (False, True):
            with self.subTest(ue8m0=ue8m0):
                w, s, deq = self._block_fp8_weight(N, K, ue8m0=ue8m0)
                x = torch.randn(M, K, device=torch_device, dtype=torch.bfloat16)
                out = finegrained_linear(x, w, s, block_size=[128, 128])
                ref = x.float() @ deq.t()
                rel = ((out.float() - ref).norm() / ref.norm()).item()
                self.assertLess(rel, 5e-2, f"ue8m0={ue8m0}: {rel:.2e} vs the dequantized weight")

    def test_on_the_fly_group_formats_round_trip_within_their_floor(self):
        """MXFP8 / MXFP4 / NVFP4 on-the-fly quantization through the kernels' quantizers, emitted in
        the module's layout (swizzled where held so); dequantizing them back lands within each
        format's floor."""
        from transformers.integrations.finegrained.conversions import FineGrainedDequantize, FineGrainedQuantize

        torch.manual_seed(0)
        cfg = _Cfg()
        cfg.hidden_size, cfg.intermediate_size, cfg.num_local_experts = 512, 256, 4
        cfg._experts_implementation = "grouped_mm"
        for fmt, floor in (("mxfp8", 4e-2), ("mxfp4", 2e-1), ("nvfp4", 2e-1)):
            with self.subTest(fmt=fmt):
                experts = FineGrainedExperts(cfg, weight_format=fmt).to(torch_device)
                model = torch.nn.Module()
                model.experts = experts
                original = (torch.randn(4, 512, 512, device=torch_device) * 0.02).to(torch.bfloat16)
                out = FineGrainedQuantize(hf_quantizer=None).convert({"experts.gate_up_proj": original}, model=model)
                weight, scale = out["experts.gate_up_proj"], out["experts.gate_up_proj_scale_inv"]
                self.assertEqual(
                    (weight.dtype, weight.shape), (experts.gate_up_proj.dtype, experts.gate_up_proj.shape)
                )
                self.assertEqual(
                    (scale.dtype, scale.shape),
                    (experts.gate_up_proj_scale_inv.dtype, experts.gate_up_proj_scale_inv.shape),
                )
                if scale.ndim == 5:
                    kernel = load_finegrained_kernel()
                    scale = kernel.unswizzle_mx_scales(scale, 512, scale.shape[2] * 4, num_experts=4)
                deq = FineGrainedDequantize(None)._dequantize_one(weight, scale.float(), output_dtype=torch.float32)
                if fmt == "nvfp4":
                    global_scale = out["experts.gate_up_proj_weight_global_scale"]
                    deq = deq * global_scale.reshape(-1, 1, 1)
                else:
                    self.assertNotIn("experts.gate_up_proj_weight_global_scale", out)
                rel = ((deq - original.float()).norm() / original.float().norm()).item()
                self.assertLess(rel, floor, f"{fmt}: {rel:.3f}")
                self.assertGreater(rel, 0.0)

    def _nvfp4_experts_modelopt_layout(self, cfg, activation_format=None, magnitudes=(8.0, 1.0)):
        """NVFP4 experts in the layout a modelopt checkpoint delivers: every (expert, half) of the
        gate|up stack quantized against its OWN global (``weight_scale_2`` is per projection, and
        the two can differ per expert), the down per expert, plus calibrated activation
        globals. Returns the module, the exact dequantized weights (so the reference is the
        quantization floor rather than the pre-quant weight), and the dequantization a reader that
        applied ONE of the two globals to both halves would get."""
        from transformers.integrations.finegrained.conversions import FineGrainedWeightGlobals

        kernel = load_finegrained_kernel()
        experts = FineGrainedExperts(cfg, weight_format="nvfp4", activation_format=activation_format).to(torch_device)
        dequantized, single_global, checkpoint_globals = {}, {}, {}
        for proj, rows, cols in (
            ("gate_up_proj", 2 * cfg.intermediate_size, cfg.hidden_size),
            ("down_proj", cfg.hidden_size, cfg.intermediate_size),
        ):
            reference = torch.randn(cfg.num_local_experts, rows, cols, device=torch_device)
            if proj == "gate_up_proj":  # the halves interleave by row; draw them far apart
                reference[:, 0::2] *= magnitudes[0]
                reference[:, 1::2] *= magnitudes[1]
            packed = torch.empty(cfg.num_local_experts, rows, cols // 2, dtype=torch.int8, device=torch_device)
            block = torch.empty(
                cfg.num_local_experts, rows, cols // 16, dtype=torch.float8_e4m3fn, device=torch_device
            )
            halves = 2 if proj == "gate_up_proj" else 1
            globals_ = torch.empty(cfg.num_local_experts, halves, device=torch_device)
            deq, deq_one = torch.empty_like(reference), torch.empty_like(reference)
            for e in range(cfg.num_local_experts):
                for h in range(halves):
                    rows_h = slice(h, None, halves)
                    q, b, g = _nvfp4_two_level(kernel, reference[e, rows_h])
                    packed[e, rows_h], block[e, rows_h], globals_[e, h] = q, b, g
                    values = _unpack_e2m1(q) * b.float().repeat_interleave(16, dim=-1)
                    deq[e, rows_h] = values * g
                if halves == 2:
                    # the control: BOTH halves read the up half's global, which is what a kernel
                    # that ignored the layout does. It has to be the up one — putting the gate's
                    # global on both scales the expert's whole output uniformly, and the
                    # post-expert RMSNorm divides exactly that out; a wrong GATE scale goes
                    # through the SiLU and survives.
                    for h in (0, 1):
                        rows_h = slice(h, None, 2)
                        deq_one[e, rows_h] = deq[e, rows_h] * (globals_[e, 1] / globals_[e, h])
            scale = block if getattr(experts, f"{proj}_scale_inv").dim() != 5 else kernel.swizzle_mx_scales(block)
            setattr(experts, proj, torch.nn.Parameter(packed, requires_grad=False))
            setattr(experts, f"{proj}_scale_inv", torch.nn.Parameter(scale, requires_grad=False))
            checkpoint_globals[f"experts.{proj}_weight_scale_2"] = globals_
            dequantized[proj], single_global[proj] = deq, deq_one
        # calibrated input_scale: one value for the gate_up (quantized before routing), one per
        # expert for the down (its rows are per expert). A weight-only module holds neither —
        # nothing there quantizes activations — so the checkpoint's keys stay unused.
        if activation_format != "bf16":
            experts.gate_up_proj_input_global_scale = torch.nn.Parameter(
                torch.full((1,), 0.05, device=torch_device), requires_grad=False
            )
            checkpoint_globals["experts.down_proj_input_scale"] = torch.linspace(
                0.02, 0.08, cfg.num_local_experts, device=torch_device
            )
        single_global["down_proj"] = dequantized["down_proj"]  # the down has one global either way
        # The globals go in through the loader's own converter, so the module holds exactly what a
        # real checkpoint puts there: one per expert, with the gate|up pair merged onto the down.
        targets = [
            "experts.gate_up_proj_weight_global_scale",
            "experts.down_proj_weight_global_scale",
            "experts.down_proj_input_global_scale",
        ]
        for target, value in FineGrainedWeightGlobals().convert(checkpoint_globals, target_patterns=targets).items():
            setattr(experts, target.split(".")[-1], torch.nn.Parameter(value, requires_grad=False))
        return experts, dequantized, single_global

    def _moe_reference(self, dequantized, x, idx, wts, post_norm=None):
        """The experts chain in torch over the dequantized weights: gate|up (interleaved columns),
        SwiGLU, down, the optional per-expert output norm, then the routing-weighted sum."""
        out = torch.zeros_like(x, dtype=torch.float32)
        for token in range(x.shape[0]):
            for slot in range(idx.shape[1]):
                e = int(idx[token, slot])
                pre = x[token].float() @ dequantized["gate_up_proj"][e].t()
                inter = torch.nn.functional.silu(pre[0::2]) * pre[1::2]
                row = inter @ dequantized["down_proj"][e].t()
                if post_norm is not None:
                    row = post_norm(row.to(x.dtype)).float()
                out[token] += row * float(wts[token, slot])
        return out

    def test_modelopt_nvfp4_experts_merge_their_globals_and_fuse_the_post_norm(self):
        """The stacked checkpoint shape end to end through the integration: a gate|up stack
        calibrated per half (merged at load), calibrated activation globals, and a per-expert
        output norm the binder fuses, on all three forwards. Weight-only pins the numbers against the dequantized weights (nothing else
        rounds); the W4A4 chain rides the 4-bit activation floor, so it is pinned by being far
        closer to the right globals than to one global folded over both halves — what a kernel
        that ignored the layout computes."""
        torch.manual_seed(0)
        cfg = _Cfg()
        cfg.hidden_size, cfg.intermediate_size, cfg.num_local_experts = 512, 256, 4
        cfg._experts_implementation = "grouped_mm"
        x = torch.randn(8, cfg.hidden_size, device=torch_device, dtype=torch.bfloat16) * 0.1
        idx = torch.randint(0, cfg.num_local_experts, (8, 2), device=torch_device, dtype=torch.long)
        wts = torch.rand(8, 2, device=torch_device, dtype=torch.bfloat16)
        forwards = (
            ("grouped", fg.finegrained_grouped_mm_experts_forward),
            ("batched", fg.finegrained_batched_mm_experts_forward),
            ("eager", lambda module, *args: module(*args)),
        )

        def relative(out, reference):
            return ((out.float() - reference).norm() / reference.norm()).item()

        for activation_format, floor in (("bf16", 0.02), (None, 0.35)):
            experts, dequantized, one_global = self._nvfp4_experts_modelopt_layout(cfg, activation_format)
            reference = self._moe_reference(dequantized, x, idx, wts)
            wrong = self._moe_reference(one_global, x, idx, wts)
            for name, forward in forwards:
                with self.subTest(forward=name, activation_format=activation_format):
                    out = forward(experts, x, idx, wts)
                    rel, rel_wrong = relative(out, reference), relative(out, wrong)
                    self.assertLess(rel, floor, f"{name} diverged from the dequantized reference: {rel:.3f}")
                    self.assertLess(
                        3 * rel,
                        rel_wrong,
                        f"{name} is as close to ONE global over both gate|up halves ({rel_wrong:.3f}) "
                        f"as to the per-half ones ({rel:.3f}) — the fold is not reading the halves",
                    )

            # the same chain with the model's per-expert output norm, which runs on the routed
            # rows: every forward has to apply it where the reference forwards do. A name the
            # kernels implement is folded into the chain's reduce and anything else is the module
            # itself, so both have to land on the same numbers as the reference.
            for norm, norm_name in (
                (torch.nn.RMSNorm(cfg.hidden_size, device=torch_device, dtype=torch.bfloat16), "rms_norm"),
                (_InputScaledRMSNorm(cfg.hidden_size).to(torch_device), "input_scaled_rms_norm"),
                # a form the kernels do not implement: the module is called on the rows
                (_UnfusableNorm(cfg.hidden_size).to(torch_device), "a_models_own_norm"),
            ):
                norm.weight.data.uniform_(0.5, 1.5)
                experts.post_expert_norm, experts.post_expert_norm_name = norm, norm_name
                experts.has_post_expert_norm = True
                normed = self._moe_reference(dequantized, x, idx, wts, norm)
                self.assertGreater(relative(normed, reference), 0.05, "the norm has to change the output")
                for name, forward in forwards:
                    with self.subTest(forward=name, activation_format=activation_format, norm=norm_name):
                        rel = relative(forward(experts, x, idx, wts), normed)
                        self.assertLess(rel, floor, f"{name} diverged with the post-expert norm: {rel:.3f}")

    def test_experts_scale_layout_op_keeps_the_forward_correct(self):
        """End-to-end over the integration: real MXFP8 experts hold swizzled scales, the loader's op
        fills them from the affine grid, the fused forward matches the affine module's, and the
        reverse op hands the affine grid back bitwise."""
        from transformers.integrations.finegrained.conversions import FineGrainedSwizzleScales

        cfg = _Cfg()
        cfg.hidden_size, cfg.intermediate_size, cfg.num_local_experts = (
            512,
            256,
            4,
        )  # K >= 256: the swizzled arm has tiles
        cfg._experts_implementation = "grouped_mm"
        experts = FineGrainedExperts(cfg, weight_format="mxfp8").to(torch_device)
        with mock.patch.object(fg, "_holds_swizzled_scales", return_value=False):
            affine_experts = FineGrainedExperts(cfg, weight_format="mxfp8").to(torch_device)
        model = torch.nn.Module()
        model.experts = experts
        op = FineGrainedSwizzleScales(hf_quantizer=None)

        affine = {}
        for proj, rows in (("gate_up_proj", 2 * cfg.intermediate_size), ("down_proj", cfg.hidden_size)):
            cols = cfg.hidden_size if proj == "gate_up_proj" else cfg.intermediate_size
            weight = torch.randn(cfg.num_local_experts, rows, cols, device=torch_device).to(torch.float8_e4m3fn)
            grid = torch.randint(
                120, 134, (cfg.num_local_experts, rows, cols // 32), dtype=torch.uint8, device=torch_device
            )
            affine[proj] = grid.view(torch.float8_e8m0fnu)
            for module in (experts, affine_experts):
                setattr(module, proj, torch.nn.Parameter(weight.clone(), requires_grad=False))
            setattr(affine_experts, f"{proj}_scale_inv", torch.nn.Parameter(affine[proj].clone(), requires_grad=False))
            self.assertEqual(
                getattr(experts, f"{proj}_scale_inv").dim(), 5, "the module should hold this projection swizzled"
            )
            out = op.convert(
                {f"mlp.experts.{proj}": affine[proj]}, model=model, full_layer_name=f"experts.{proj}_scale_inv"
            )
            swizzled = out[f"mlp.experts.{proj}"]
            self.assertEqual(swizzled.shape, getattr(experts, f"{proj}_scale_inv").shape)
            setattr(experts, f"{proj}_scale_inv", torch.nn.Parameter(swizzled, requires_grad=False))

        x = torch.randn(8, cfg.hidden_size, device=torch_device, dtype=torch.bfloat16)
        idx = torch.randint(0, cfg.num_local_experts, (8, 2), device=torch_device, dtype=torch.long)
        wts = torch.rand(8, 2, device=torch_device, dtype=torch.bfloat16)
        reference = fg.finegrained_grouped_mm_experts_forward(affine_experts, x, idx, wts)
        out = fg.finegrained_grouped_mm_experts_forward(experts, x, idx, wts)
        backend_synchronize(torch_device)
        rel = ((out.float() - reference.float()).norm() / reference.float().norm().clamp(min=1e-9)).item()
        # different tuned arms reorder the fp32 reduction (~1e-5); a layout bug is O(1)
        self.assertLess(rel, 1e-3, f"swizzled forward diverged from the affine one: {rel:.2e}")
        # the eager per-expert loop reads one expert's slice of the swizzled stack directly
        eager_reference = affine_experts(x, idx, wts)
        eager_out = experts(x, idx, wts)
        rel = (
            (eager_out.float() - eager_reference.float()).norm() / eager_reference.float().norm().clamp(min=1e-9)
        ).item()
        self.assertLess(rel, 1e-3, f"eager forward on swizzled scales diverged from affine: {rel:.2e}")

        reverse = op.reverse_op
        for proj, grid in affine.items():
            restored = reverse.convert({proj: getattr(experts, f"{proj}_scale_inv").data})[proj]
            self.assertTrue(torch.equal(restored.view(torch.uint8), grid.view(torch.uint8)), proj)
