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
"""Shared experts-module fixtures for the MoE-kernel integration tests.

`make_experts` builds a BF16 stand-in (no scales) — used by the sonic-moe tests and the DeepGEMM
BF16-experts tests. `make_fp8_experts` builds an FP8/FP4 stand-in (per-projection `_scale_inv`) — used by
the DeepGEMM FP8 and finegrained-fp8 tests. Both are `SimpleNamespace`s carrying exactly the attributes
the `*_experts_forward` glue reads; the kernels are mocked in the tests, so the weights/scales are
arbitrary but are the same tensor objects handed to the kernel (so `torch.equal` checks marshalling).

Layout: gate_up is `(E, 2I, H)` (non-transposed) / `(E, H, 2I)` (transposed); `has_gate=False` swaps it
for a plain `up_proj` of half the width. `down_proj` is `(E, H, I)` / `(E, I, H)`.
"""

import types

import torch

from transformers.activations import ACT2FN
from transformers.testing_utils import torch_device


def _build_experts(
    *, num_experts, hidden, inter, has_gate, has_bias, is_transposed, weight_dtype, scale_dtype, hidden_act, **extra
):
    def weight(out_dim, in_dim):
        # Non-transposed weights are (E, out, in); transposed are (E, in, out).
        shape = (num_experts, in_dim, out_dim) if is_transposed else (num_experts, out_dim, in_dim)
        return torch.randn(*shape, device=torch_device).to(weight_dtype)

    def bias(dim):
        return torch.randn(num_experts, dim, dtype=torch.bfloat16, device=torch_device) if has_bias else None

    act_fn = ACT2FN[hidden_act]

    def apply_gate(gate_up):
        # SwiGLU over the concatenated gate/up halves: act_fn(gate) * up (the 2*inter -> inter collapse).
        gate, up = gate_up.chunk(2, dim=-1)
        return act_fn(gate) * up

    # Gated experts pack gate+up into one `2*inter` projection; non-gated carry a plain `up` of `inter`.
    proj, proj_out = ("gate_up_proj", 2 * inter) if has_gate else ("up_proj", inter)
    ns = types.SimpleNamespace(
        num_experts=num_experts,
        has_gate=has_gate,
        has_bias=has_bias,
        is_transposed=is_transposed,
        act_fn=act_fn,
        _apply_gate=apply_gate,
        down_proj=weight(hidden, inter),
        down_proj_bias=bias(hidden),
        **{proj: weight(proj_out, hidden), f"{proj}_bias": bias(proj_out)},
    )
    if scale_dtype is not None:
        ns.down_proj_scale_inv = torch.ones(num_experts, 1, 1, device=torch_device).to(scale_dtype)
        setattr(ns, f"{proj}_scale_inv", torch.ones(num_experts, 1, 1, device=torch_device).to(scale_dtype))
    for name, value in extra.items():
        setattr(ns, name, value)
    return ns


def make_experts(
    *,
    num_experts=4,
    hidden=8,
    inter=16,
    has_gate=True,
    has_bias=False,
    is_transposed=False,
    hidden_act="silu",
    is_concatenated=True,
    weight_dtype=torch.bfloat16,
):
    """BF16 experts stand-in (no scales) for the sonic-moe and DeepGEMM BF16 forwards. Carries
    `config.hidden_act` / `is_concatenated` (read by sonic-moe)."""
    return _build_experts(
        num_experts=num_experts,
        hidden=hidden,
        inter=inter,
        has_gate=has_gate,
        has_bias=has_bias,
        is_transposed=is_transposed,
        weight_dtype=weight_dtype,
        scale_dtype=None,
        hidden_act=hidden_act,
        config=types.SimpleNamespace(hidden_act=hidden_act),
        is_concatenated=is_concatenated,
    )


def make_fp8_experts(
    *,
    num_experts=4,
    hidden=8,
    inter=16,
    has_gate=True,
    is_transposed=False,
    hidden_act="silu",
    weight_dtype=torch.float8_e4m3fn,
    scale_dtype=torch.float32,
    activation_scheme="dynamic",
    block_size=(128, 128),
):
    """FP8/FP4 experts stand-in (per-projection `_scale_inv`) for the DeepGEMM FP8 and finegrained-fp8
    forwards, plus the `_deepgemm_disabled` multi-device flag."""
    return _build_experts(
        num_experts=num_experts,
        hidden=hidden,
        inter=inter,
        has_gate=has_gate,
        has_bias=False,
        is_transposed=is_transposed,
        weight_dtype=weight_dtype,
        scale_dtype=scale_dtype,
        hidden_act=hidden_act,
        activation_scheme=activation_scheme,
        block_size=block_size,
        _deepgemm_disabled=False,
    )
