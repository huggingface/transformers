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

"""
Tests for `torch_chunk_gated_delta_rule` and `torch_recurrent_gated_delta_rule`, the gated delta rule implementations
shared by Qwen3-Next, Qwen3.5, Qwen3.5-MoE and OLMo-Hybrid.

Adapted from flash-linear-attention's `tests/ops/test_gdn.py` (https://github.com/fla-org/flash-linear-attention),
MIT licensed, Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li. The rest of this docstring is the full diff
against that file, so there is no need to open it to know what was changed and why.

Test mapping
    test_fused_recurrent       -> test_recurrent, renamed since nothing is fused here, it is a plain python loop
    test_chunk                 -> test_chunk, same structure: forward, backward, then one assertion block
    test_chunk_with_chunk_size -> test_chunk_with_chunk_size, same `run_ref()` / `run_tri(chunk_size)` closures
    test_chunk_varlen          -> test_chunk_varlen, but sequence by sequence, see "no cu_seqlens" below
    (nothing upstream)         -> test_prefill_then_decode, added because it is the path generation actually takes

Differences forced by the transformers API
    - No `scale`: scale default invsqrt(k_head_dim)
    - No GQA inside the ops: repeat-interleaves is called before the kernel
    - No `cu_seqlens`: the ops take no packing argument, so `test_chunk_varlen` runs each sequence on its own
    - Separate K and V head dims
"""

import pytest
import torch
import torch.nn.functional as F

from transformers.models.qwen3_next.modeling_qwen3_next import (
    l2norm,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)
from transformers.testing_utils import require_torch, torch_device


def naive_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
):
    """
    Reference PyTorch implementation of recurrent gated delta rule. Taken straight from the upstream file.

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        v: [B, T, H, V]
        beta: [B, T, H]
        g: [B, T, H]
        scale: float, optional
        initial_state: [B, H, K, V], optional
        output_final_state: bool

    Returns:
        o: [B, T, H, V]
        final_state: [B, H, K, V] if output_final_state else None
    """
    q, k, v, beta, g = (x.transpose(1, 2).contiguous().to(torch.float32) for x in [q, k, v, beta, g])
    B, H, T, K, V = *k.shape, v.shape[-1]
    o = torch.zeros(B, H, T, V).to(v)
    h = torch.zeros(B, H, K, V).to(v)
    if initial_state is not None:
        h = initial_state.to(torch.float32)
    scale = 1 / (q.shape[-1] ** 0.5)
    q = q * scale

    for i in range(T):
        b_q = q[:, :, i]
        b_k = k[:, :, i]
        b_v = v[:, :, i].clone()
        h = h.clone() * g[:, :, i].exp()[..., None, None]
        b_beta = beta[:, :, i]
        b_v = b_v - (h.clone() * b_k[..., None]).sum(-2)
        b_v = b_v * b_beta[..., None]
        h = h.clone() + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)
        o[:, :, i] = torch.einsum("bhd,bhdm->bhm", b_q, h)

    if not output_final_state:
        h = None
    o = o.transpose(1, 2).contiguous()
    return o, h


def assert_close(prefix, ref, tri, ratio):
    """Mean relative error check, in the same spirit as flash-linear-attention's `assert_close`."""
    ref, tri = ref.float(), tri.float()
    error = ((ref - tri).abs().mean() / ref.abs().mean().clamp(min=1e-8)).item()
    assert error < ratio, f"{prefix}: mean relative error {error:.3e} exceeds {ratio}"


@require_torch
@pytest.mark.parametrize(
    ("B", "T", "H", "HV", "K", "V", "gate_logit_normalizer", "dtype"),
    [
        pytest.param(*test, id="B{}-T{}-H{}-HV{}-K{}-V{}-gate_logit_normalizer{}-{}".format(*test))
        for test in [
            (1, 63, 1, 1, 64, 64, 1, torch.float32),
            (2, 500, 4, 4, 60, 60, 1, torch.float32),
            (2, 1000, 2, 8, 128, 128, 0.1, torch.float32),
            (3, 1024, 2, 2, 128, 128, 1, torch.float32),
            (4, 1024, 3, 3, 128, 128, 10, torch.float32),
            (4, 2048, 4, 4, 64, 64, 1, torch.float32),
            (2, 1024, 4, 4, 128, 128, 0.1, torch.float16),
            (2, 1024, 4, 8, 128, 128, 10, torch.float16),
            (2, 512, 2, 2, 64, 128, 1, torch.float32),
        ]
    ],
)
def test_recurrent(B: int, T: int, H: int, HV: int, K: int, V: int, gate_logit_normalizer: float, dtype: torch.dtype):
    torch.manual_seed(42)
    assert HV % H == 0
    G = HV // H

    q = torch.randn(B, T, H, K, dtype=dtype, device=torch_device)
    k = torch.randn(B, T, H, K, dtype=dtype, device=torch_device)
    v = torch.randn(B, T, HV, V, dtype=dtype, device=torch_device)
    beta = torch.rand(B, T, HV, dtype=torch.float32, device=torch_device).sigmoid()
    g = F.logsigmoid(torch.rand(B, T, HV, dtype=torch.float32, device=torch_device)) / gate_logit_normalizer
    h0 = torch.randn(B, HV, K, V, dtype=torch.float32, device=torch_device)

    # The op does not expand grouped query/key heads itself, the `GatedDeltaNet` module does it beforehand
    q, k = q.repeat_interleave(G, dim=2), k.repeat_interleave(G, dim=2)
    tolerance = 0.005 if dtype == torch.float32 else 0.01

    ref, ref_ht = naive_recurrent_gated_delta_rule(
        q=l2norm(q.clone(), dim=-1, eps=1e-6),
        k=l2norm(k.clone(), dim=-1, eps=1e-6),
        v=v.clone(),
        beta=beta.clone(),
        g=g.clone(),
        initial_state=h0.clone(),
        output_final_state=True,
    )
    tri, tri_ht = torch_recurrent_gated_delta_rule(
        q.clone(),
        k.clone(),
        v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        initial_state=h0.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )

    assert tri.shape == (B, T, HV, V)
    assert_close("o", ref, tri, tolerance)
    assert_close("ht", ref_ht, tri_ht, tolerance)


@require_torch
@pytest.mark.parametrize(
    ("B", "T", "H", "HV", "K", "V", "gate_logit_normalizer", "mask_p", "use_qk_l2norm_in_kernel", "dtype"),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-HV{}-K{}-V{}-gate_logit_normalizer{}-mask_p{}-use_qk_l2norm_in_kernel{}-{}".format(*test),
        )
        for test in [
            (4, 1024, 4, 4, 128, 128, 1, 1.0, False, torch.float16),
            (2, 75, 4, 4, 64, 64, 0.01, 0, False, torch.float16),
            (2, 500, 3, 3, 60, 60, 1, 0, False, torch.float16),
            (2, 1000, 3, 3, 64, 64, 1, 0.5, False, torch.float16),
            (3, 1024, 4, 4, 100, 100, 0.1, 0, False, torch.float16),
            (4, 1024, 4, 4, 128, 128, 1, 0, True, torch.float16),
            (2, 1500, 4, 4, 128, 128, 10, 0, False, torch.float16),
            (4, 2048, 8, 8, 64, 64, 1, 0, False, torch.float16),
            (2, 256, 2, 4, 64, 64, 1, 0, False, torch.float16),
            (2, 512, 2, 8, 64, 64, 0.1, 0, True, torch.float16),
            (2, 1024, 4, 8, 128, 128, 1, 0, False, torch.float16),
            (2, 512, 2, 2, 64, 128, 1, 0, False, torch.float32),
        ]
    ],
)
def test_chunk(
    B: int,
    T: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    gate_logit_normalizer: float,
    mask_p: float,
    use_qk_l2norm_in_kernel: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    assert HV % H == 0
    G = HV // H

    q = torch.randn(B, T, H, K, dtype=dtype, device=torch_device)
    k = torch.randn(B, T, H, K, dtype=dtype, device=torch_device)
    v = torch.randn(B, T, HV, V, dtype=dtype, device=torch_device)
    beta = torch.rand(B, T, HV, dtype=torch.float32, device=torch_device).sigmoid()
    g = F.logsigmoid(torch.rand(B, T, HV, dtype=torch.float32, device=torch_device)) / gate_logit_normalizer
    g = g * (torch.rand_like(g) > mask_p)
    h0 = torch.randn(B, HV, K, V, dtype=torch.float32, device=torch_device)
    q, k, v, beta, g, h0 = (x.requires_grad_(True) for x in (q, k, v, beta, g, h0))

    # The op does not expand grouped query/key heads itself, the `GatedDeltaNet` module does it beforehand
    q_expanded, k_expanded = q.repeat_interleave(G, dim=2), k.repeat_interleave(G, dim=2)
    # `l2norm` and not `F.normalize`: it is what the op uses internally, so both sides normalize identically
    tolerance = 0.005 if dtype == torch.float32 else 0.01

    tri, tri_ht = torch_chunk_gated_delta_rule(
        q_expanded if use_qk_l2norm_in_kernel else l2norm(q_expanded, dim=-1, eps=1e-6),
        k_expanded if use_qk_l2norm_in_kernel else l2norm(k_expanded, dim=-1, eps=1e-6),
        v,
        g=g,
        beta=beta,
        initial_state=h0,
        output_final_state=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dg, tri_dh0 = q.grad, k.grad, v.grad, beta.grad, g.grad, h0.grad
    q.grad = k.grad = v.grad = beta.grad = g.grad = h0.grad = None

    ref, ref_ht = naive_recurrent_gated_delta_rule(
        q=l2norm(q_expanded, dim=-1, eps=1e-6),
        k=l2norm(k_expanded, dim=-1, eps=1e-6),
        v=v,
        beta=beta,
        g=g,
        initial_state=h0,
        output_final_state=True,
    )
    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dbeta, ref_dg, ref_dh0 = q.grad, k.grad, v.grad, beta.grad, g.grad, h0.grad

    # Sequence lengths that are not a multiple of the chunk size are padded internally, so this also checks that the
    # output is cut back to `T` and that the padded tail leaves the final state untouched
    assert tri.shape == (B, T, HV, V)
    assert_close("o", ref, tri, tolerance)
    assert_close("ht", ref_ht, tri_ht, tolerance)
    assert_close("dq", ref_dq, tri_dq, 0.01)
    assert_close("dk", ref_dk, tri_dk, 0.01)
    assert_close("dv", ref_dv, tri_dv, 0.01)
    assert_close("db", ref_dbeta, tri_dbeta, 0.02)
    assert_close("dg", ref_dg, tri_dg, 0.02)
    assert_close("dh0", ref_dh0, tri_dh0, 0.01)


@require_torch
@pytest.mark.parametrize(
    ("B", "T", "H", "HV", "K", "V", "gate_logit_normalizer", "dtype", "chunk_size"),
    [
        pytest.param(*test, id="B{}-T{}-H{}-HV{}-K{}-V{}-gate{}-{}-chunk{}".format(*test))
        for chunk_size in [16, 32, 64]
        for test in [
            (1, 64, 2, 4, 32, 32, 1.0, torch.float32, chunk_size),
            (2, 130, 2, 2, 64, 128, 1.0, torch.float32, chunk_size),
        ]
    ],
)
def test_chunk_with_chunk_size(
    B: int,
    T: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    gate_logit_normalizer: float,
    dtype: torch.dtype,
    chunk_size: int,
):
    torch.manual_seed(42)
    assert HV % H == 0
    G = HV // H

    q = torch.randn(B, T, H, K, dtype=dtype, device=torch_device)
    k = torch.randn(B, T, H, K, dtype=dtype, device=torch_device)
    v = torch.randn(B, T, HV, V, dtype=dtype, device=torch_device)
    beta = torch.rand(B, T, HV, dtype=torch.float32, device=torch_device).sigmoid()
    g = F.logsigmoid(torch.rand(B, T, HV, dtype=torch.float32, device=torch_device)) / gate_logit_normalizer
    h0 = torch.randn(B, HV, K, V, dtype=torch.float32, device=torch_device)
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)

    def normalized_qk(q_, k_):
        # The op does not expand grouped query/key heads itself, the `GatedDeltaNet` module does it beforehand
        return (
            l2norm(q_.repeat_interleave(G, dim=2), dim=-1, eps=1e-6),
            l2norm(k_.repeat_interleave(G, dim=2), dim=-1, eps=1e-6),
        )

    def run_ref():
        q_, k_, v_, beta_, g_, h0_ = (x.detach().clone().requires_grad_(True) for x in (q, k, v, beta, g, h0))
        q_expanded, k_expanded = normalized_qk(q_, k_)
        o, ht = naive_recurrent_gated_delta_rule(
            q=q_expanded,
            k=k_expanded,
            v=v_,
            beta=beta_,
            g=g_,
            initial_state=h0_,
            output_final_state=True,
        )
        ((o * do).sum() + (ht * dht).sum()).backward()
        return o, ht, q_.grad, k_.grad, v_.grad, beta_.grad, g_.grad, h0_.grad

    def run_tri(chunk_size: int):
        q_, k_, v_, beta_, g_, h0_ = (x.detach().clone().requires_grad_(True) for x in (q, k, v, beta, g, h0))
        q_expanded, k_expanded = normalized_qk(q_, k_)
        o, ht = torch_chunk_gated_delta_rule(
            q_expanded,
            k_expanded,
            v_,
            g=g_,
            beta=beta_,
            chunk_size=chunk_size,
            initial_state=h0_,
            output_final_state=True,
        )
        ((o * do).sum() + (ht * dht).sum()).backward()
        return o, ht, q_.grad, k_.grad, v_.grad, beta_.grad, g_.grad, h0_.grad

    ref_o, ref_ht, ref_dq, ref_dk, ref_dv, ref_dbeta, ref_dg, ref_dh0 = run_ref()
    tri_o, tri_ht, tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dg, tri_dh0 = run_tri(chunk_size)

    assert_close(f"o@{chunk_size}", ref_o, tri_o, 0.005)
    assert_close(f"ht@{chunk_size}", ref_ht, tri_ht, 0.005)
    assert_close(f"dq@{chunk_size}", ref_dq, tri_dq, 0.01)
    assert_close(f"dk@{chunk_size}", ref_dk, tri_dk, 0.01)
    assert_close(f"dv@{chunk_size}", ref_dv, tri_dv, 0.01)
    assert_close(f"db@{chunk_size}", ref_dbeta, tri_dbeta, 0.02)
    assert_close(f"dg@{chunk_size}", ref_dg, tri_dg, 0.02)
    assert_close(f"dh0@{chunk_size}", ref_dh0, tri_dh0, 0.01)


@require_torch
@pytest.mark.parametrize(
    ("H", "HV", "K", "V", "mask_p", "cu_seqlens", "dtype"),
    [
        pytest.param(*test, id="H{}-HV{}-K{}-V{}-mask_p{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (4, 4, 60, 60, 0, [0, 15], torch.float16),
            (4, 4, 64, 64, 0, [0, 256, 500, 1000], torch.float16),
            (4, 4, 64, 64, 0.5, [0, 256, 500, 1000], torch.float16),
            (4, 4, 100, 100, 0, [0, 15, 100, 300, 1200, 2000], torch.float16),
            (2, 4, 64, 64, 0, [0, 256, 500, 1000], torch.float16),
            (2, 8, 64, 64, 0, [0, 256, 500, 1000], torch.float16),
            (2, 2, 64, 128, 0, [0, 256, 500, 1000], torch.float32),
        ]
    ],
)
def test_chunk_varlen(H: int, HV: int, K: int, V: int, mask_p: float, cu_seqlens: list[int], dtype: torch.dtype):
    """The torch ops take no `cu_seqlens`, so each sequence is run on its own with its own initial state."""
    torch.manual_seed(42)
    assert HV % H == 0
    G = HV // H
    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1

    # seq-first, as would be required for inputs with variable lengths
    q = torch.randn(1, T, H, K, dtype=dtype, device=torch_device)
    k = torch.randn(1, T, H, K, dtype=dtype, device=torch_device)
    v = torch.randn(1, T, HV, V, dtype=dtype, device=torch_device)
    beta = torch.rand(1, T, HV, dtype=torch.float32, device=torch_device).sigmoid()
    g = F.logsigmoid(torch.rand(1, T, HV, dtype=torch.float32, device=torch_device))
    g = g * (torch.rand_like(g) > mask_p)
    h0 = torch.randn(N, HV, K, V, dtype=torch.float32, device=torch_device)
    q, k, v, beta, g, h0 = (x.requires_grad_(True) for x in (q, k, v, beta, g, h0))

    # The op does not expand grouped query/key heads itself, the `GatedDeltaNet` module does it beforehand
    q_expanded = l2norm(q.repeat_interleave(G, dim=2), dim=-1, eps=1e-6)
    k_expanded = l2norm(k.repeat_interleave(G, dim=2), dim=-1, eps=1e-6)
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)

    def run(op):
        outputs, states = [], []
        for i in range(N):
            sequence = slice(cu_seqlens[i], cu_seqlens[i + 1])
            o_i, ht_i = op(
                q_expanded[:, sequence],
                k_expanded[:, sequence],
                v[:, sequence],
                g[:, sequence],
                beta[:, sequence],
                h0[i : i + 1],
            )
            outputs.append(o_i)
            states.append(ht_i)
        return torch.cat(outputs, 1), torch.cat(states, 0)

    tri, tri_ht = run(
        lambda q_, k_, v_, g_, beta_, h0_: torch_chunk_gated_delta_rule(
            q_, k_, v_, g=g_, beta=beta_, initial_state=h0_, output_final_state=True
        )
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dg, tri_dh0 = q.grad, k.grad, v.grad, beta.grad, g.grad, h0.grad
    q.grad = k.grad = v.grad = beta.grad = g.grad = h0.grad = None

    ref, ref_ht = run(
        lambda q_, k_, v_, g_, beta_, h0_: naive_recurrent_gated_delta_rule(
            q=q_, k=k_, v=v_, beta=beta_, g=g_, initial_state=h0_, output_final_state=True
        )
    )
    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dbeta, ref_dg, ref_dh0 = q.grad, k.grad, v.grad, beta.grad, g.grad, h0.grad

    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)
    assert_close("dq", ref_dq, tri_dq, 0.01)
    assert_close("dk", ref_dk, tri_dk, 0.01)
    assert_close("dv", ref_dv, tri_dv, 0.01)
    assert_close("db", ref_dbeta, tri_dbeta, 0.02)
    assert_close("dg", ref_dg, tri_dg, 0.02)
    assert_close("dh0", ref_dh0, tri_dh0, 0.01)


@require_torch
@pytest.mark.parametrize(
    ("B", "T_prefill", "T_decode", "H", "K", "V", "dtype"),
    [
        pytest.param(*test, id="B{}-Tprefill{}-Tdecode{}-H{}-K{}-V{}-{}".format(*test))
        for test in [
            (2, 1024, 8, 4, 128, 128, torch.float32),
            (2, 500, 8, 2, 64, 128, torch.float32),
            (1, 63, 4, 3, 32, 32, torch.float32),
        ]
    ],
)
def test_prefill_then_decode(B: int, T_prefill: int, T_decode: int, H: int, K: int, V: int, dtype: torch.dtype):
    """The path generation takes: a chunked prefill, then one recurrent step per new token, carrying the state."""
    torch.manual_seed(42)
    T = T_prefill + T_decode

    q = l2norm(torch.randn(B, T, H, K, dtype=dtype, device=torch_device), dim=-1, eps=1e-6)
    k = l2norm(torch.randn(B, T, H, K, dtype=dtype, device=torch_device), dim=-1, eps=1e-6)
    v = torch.randn(B, T, H, V, dtype=dtype, device=torch_device)
    beta = torch.rand(B, T, H, dtype=torch.float32, device=torch_device).sigmoid()
    g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.float32, device=torch_device))
    h0 = torch.randn(B, H, K, V, dtype=torch.float32, device=torch_device)

    ref, ref_ht = naive_recurrent_gated_delta_rule(
        q=q, k=k, v=v, beta=beta, g=g, initial_state=h0, output_final_state=True
    )

    prefill = slice(None, T_prefill)
    prefill_out, state = torch_chunk_gated_delta_rule(
        q[:, prefill],
        k[:, prefill],
        v[:, prefill],
        g=g[:, prefill],
        beta=beta[:, prefill],
        initial_state=h0,
        output_final_state=True,
    )
    outputs = [prefill_out]
    for t in range(T_prefill, T):
        step = slice(t, t + 1)
        step_out, state = torch_recurrent_gated_delta_rule(
            q[:, step],
            k[:, step],
            v[:, step],
            g=g[:, step],
            beta=beta[:, step],
            initial_state=state,
            output_final_state=True,
        )
        outputs.append(step_out)

    assert_close("o", ref, torch.cat(outputs, dim=1), 0.005)
    assert_close("ht", ref_ht, state, 0.005)
