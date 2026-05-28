# Copyright 2025 The HuggingFace Team. All rights reserved.
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
Ulysses-style Context Parallelism (CP) primitives.

Each rank in a CP process group holds a contiguous *sequence slice* of every
per-token tensor: ``hidden_states.shape == (B, N_local, H)`` with
``N_local = N_total // cp_world``. All point-wise ops (norms, projections,
RoPE, MLP, residual add) run locally on the shard. Attention is the only
global op, and it is handled by ``ulysses_attention``: a single all-to-all
on the head axis swaps ``head_axis`` for ``seq_axis`` so the SDPA call sees
the full causal sequence, then an inverse all-to-all returns to the
sequence-sharded layout.

This module has no dependency on ``transformers`` internals — it is pure
``torch`` + ``torch.distributed`` so it can be reused as a stand-alone
attention kernel. The user-facing wiring (model walking, registering the
attention impl, threading ``cp_group``) lives in
``transformers.integrations.context_parallel``.

References:
- Jacobs et al., "DeepSpeed-Ulysses: System Optimizations for Training
  Extreme Long Sequence Transformer Models" (https://arxiv.org/abs/2309.14509)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F


@dataclass
class CPGroup:
    """Lightweight wrapper around a CP-axis ``torch.distributed.ProcessGroup``.

    Holding the process group in a small object (rather than passing
    ``ProcessGroup`` around directly) lets call sites query
    ``world`` / ``rank`` without re-importing ``torch.distributed`` and
    keeps the ``cp_world == 1`` degenerate case as ``CPGroup(group=None)``.

    Args:
        group (`torch.distributed.ProcessGroup`, *optional*):
            The CP-axis process group, or ``None`` to disable CP (the
            attention kernel falls back to local SDPA).
    """

    group: dist.ProcessGroup | None

    @property
    def world(self) -> int:
        if self.group is None:
            return 1
        return dist.get_world_size(self.group)

    @property
    def rank(self) -> int:
        if self.group is None:
            return 0
        return dist.get_rank(self.group)

    def is_active(self) -> bool:
        return self.group is not None and self.world > 1


def build_cp_group_from_default() -> CPGroup:
    """Build a :class:`CPGroup` from the default world process group.

    Convenience for single-axis CP runs (every rank is in the CP group).
    For 2-D meshes (e.g. EP × CP), construct sub-groups explicitly and pass
    them to :func:`ulysses_attention`.
    """
    if not dist.is_initialized():
        return CPGroup(group=None)
    return CPGroup(group=dist.group.WORLD)


def _alltoall_heads_seq_to_head(x: torch.Tensor, cp: CPGroup) -> torch.Tensor:
    """All-to-all that swaps the head axis (split) for the seq axis (gather).

    Input shape:  ``(B, H, N_local, D)`` — seq-sharded.
    Output shape: ``(B, H // cp.world, N_total, D)`` — head-sharded.

    Used for both Q (``H = num_q_heads``) and KV
    (``H = num_kv_heads``). The only constraint is
    ``H % cp.world == 0``.

    The implementation permutes so that the split axis (``P = cp.world``)
    is the *leading* dim before slicing. This avoids a subtle NCCL bug
    where non-contiguous views of a ``(B, P, H/P, N, D)`` tensor with
    ``B > 1`` are silently corrupted by ``all_to_all`` (the per-rank
    slices look contiguous when ``B == 1`` because ``stride[0]`` happens
    to match, but break when ``B > 1``).
    """
    B, H, n_local, D = x.shape
    P = cp.world
    if H % P != 0:
        raise ValueError(f"head count {H} must be divisible by cp_world {P}")

    y = x.reshape(B, P, H // P, n_local, D).permute(1, 0, 2, 3, 4).contiguous()
    in_list = [y[r] for r in range(P)]
    out_buf = torch.empty_like(y)
    out_list = [out_buf[r] for r in range(P)]
    dist.all_to_all(out_list, in_list, group=cp.group)
    out = out_buf.permute(1, 2, 0, 3, 4).contiguous()
    out = out.reshape(B, H // P, P * n_local, D)
    return out


def _alltoall_heads_head_to_seq(x: torch.Tensor, cp: CPGroup) -> torch.Tensor:
    """Inverse of :func:`_alltoall_heads_seq_to_head`.

    Input shape:  ``(B, H_local, N_total, D)`` — head-sharded.
    Output shape: ``(B, H_local * cp.world, N_local, D)`` — seq-sharded.
    """
    B, H_local, N_total, D = x.shape
    P = cp.world
    if N_total % P != 0:
        raise ValueError(f"N_total {N_total} must be divisible by cp_world {P}")
    n_local = N_total // P

    y = x.reshape(B, H_local, P, n_local, D).permute(2, 0, 1, 3, 4).contiguous()
    in_list = [y[r] for r in range(P)]
    out_buf = torch.empty_like(y)
    out_list = [out_buf[r] for r in range(P)]
    dist.all_to_all(out_list, in_list, group=cp.group)
    out = out_buf.permute(1, 0, 2, 3, 4).contiguous()
    out = out.reshape(B, P * H_local, n_local, D)
    return out


def _local_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool,
    scale: float,
    sinks: torch.Tensor | None = None,
    sliding_window: int | None = None,
) -> torch.Tensor:
    """Local attention on head-sharded, full-sequence tensors.

    Uses :func:`torch.nn.functional.scaled_dot_product_attention` on the
    hot path (no sinks, no sliding-window) and falls back to an explicit
    softmax otherwise. Handles GQA by ``repeat_interleave`` over the head
    axis.

    Args:
        q (`torch.Tensor`): Query, shape ``(B, H_q, N, D)``.
        k (`torch.Tensor`): Key, shape ``(B, H_kv, N, D)``.
        v (`torch.Tensor`): Value, shape ``(B, H_kv, N, D)``.
        is_causal (`bool`): Whether to apply a causal mask.
        scale (`float`): Softmax scaling factor (typically ``1/sqrt(D)``).
        sinks (`torch.Tensor`, *optional*):
            Per-head attention sink logit, shape ``(H_q,)``. Adds an
            implicit "always-zero" key with the per-head bias.
        sliding_window (`int`, *optional*):
            If set, also applies a band mask of the given width on top of
            the causal mask.

    Returns:
        `torch.Tensor` of shape ``(B, H_q, N, D)``.
    """
    Hq = q.size(1)
    Hkv = k.size(1)
    g = Hq // Hkv

    if g > 1:
        k = k.repeat_interleave(g, dim=1)
        v = v.repeat_interleave(g, dim=1)

    if sinks is None and sliding_window is None:
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=scale,
        )

    B, _, N, _ = q.shape
    logits = torch.matmul(q, k.transpose(-1, -2)) * scale

    if is_causal:
        causal_mask = torch.ones(N, N, dtype=torch.bool, device=q.device).tril()
        if sliding_window is not None:
            band = torch.ones(N, N, dtype=torch.bool, device=q.device).triu(diagonal=-sliding_window + 1)
            causal_mask = causal_mask & band
        logits = logits.masked_fill(~causal_mask, float("-inf"))

    if sinks is not None:
        sink_logits = sinks.reshape(1, -1, 1, 1).expand(B, -1, N, 1)
        logits = torch.cat([logits, sink_logits.to(logits.dtype)], dim=-1)
        attn = torch.softmax(logits, dim=-1)
        attn = attn[..., :-1]
    else:
        attn = torch.softmax(logits, dim=-1)

    return torch.matmul(attn, v)


class _UlyssesFn(torch.autograd.Function):
    """Ulysses autograd function.

    Forward:
        1. All-to-all on Q and KV head axis (seq → head).
        2. Local attention on the head-sharded, full-sequence tensors.
        3. Inverse all-to-all on output (head → seq).

    Backward:
        1. All-to-all on ``grad_out`` head axis.
        2. Recompute local attention with ``torch.enable_grad`` and use
           :func:`torch.autograd.grad` to get ``dQ``, ``dK``, ``dV``,
           and optionally ``d_sinks`` on the local head shard.
        3. Inverse all-to-all on ``dQ``, ``dK``, ``dV`` to seq-sharded.
        4. ``all_gather`` on ``d_sinks`` along the cp axis to reconstruct
           the full sink gradient (shape ``(H_q,)``).
    """

    @staticmethod
    def forward(ctx, q, k, v, sinks, is_causal, scale, sliding_window, cp_group):
        cp = CPGroup(group=cp_group)
        Hq_total = q.size(1)

        q_h = _alltoall_heads_seq_to_head(q, cp)
        k_h = _alltoall_heads_seq_to_head(k, cp)
        v_h = _alltoall_heads_seq_to_head(v, cp)

        sinks_local = None
        if sinks is not None:
            head_slice = slice(
                cp.rank * (Hq_total // cp.world),
                (cp.rank + 1) * (Hq_total // cp.world),
            )
            sinks_local = sinks[head_slice]

        out_h = _local_attention(
            q_h,
            k_h,
            v_h,
            is_causal=is_causal,
            scale=scale,
            sinks=sinks_local,
            sliding_window=sliding_window,
        )
        out = _alltoall_heads_head_to_seq(out_h, cp)

        ctx.save_for_backward(q, k, v, sinks if sinks is not None else q.new_empty(0))
        ctx.is_causal = is_causal
        ctx.scale = scale
        ctx.sliding_window = sliding_window
        ctx.cp_group = cp_group
        ctx.has_sinks = sinks is not None
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, sinks_buf = ctx.saved_tensors
        cp = CPGroup(group=ctx.cp_group)
        sinks = sinks_buf if ctx.has_sinks else None
        Hq_total = q.size(1)

        grad_out_h = _alltoall_heads_seq_to_head(grad_out, cp)

        with torch.enable_grad():
            q_h = _alltoall_heads_seq_to_head(q.detach(), cp).requires_grad_(True)
            k_h = _alltoall_heads_seq_to_head(k.detach(), cp).requires_grad_(True)
            v_h = _alltoall_heads_seq_to_head(v.detach(), cp).requires_grad_(True)
            sinks_local = None
            if ctx.has_sinks:
                head_slice = slice(
                    cp.rank * (Hq_total // cp.world),
                    (cp.rank + 1) * (Hq_total // cp.world),
                )
                sinks_local = sinks[head_slice].detach().requires_grad_(True)
            out_h_local = _local_attention(
                q_h,
                k_h,
                v_h,
                is_causal=ctx.is_causal,
                scale=ctx.scale,
                sinks=sinks_local,
                sliding_window=ctx.sliding_window,
            )
            grad_inputs = [q_h, k_h, v_h] + ([sinks_local] if ctx.has_sinks else [])
            grads = torch.autograd.grad(
                out_h_local,
                grad_inputs,
                grad_outputs=grad_out_h,
                retain_graph=False,
                create_graph=False,
            )
        dq_h, dk_h, dv_h = grads[0], grads[1], grads[2]
        d_sinks_local = grads[3] if ctx.has_sinks else None

        dq = _alltoall_heads_head_to_seq(dq_h, cp)
        dk = _alltoall_heads_head_to_seq(dk_h, cp)
        dv = _alltoall_heads_head_to_seq(dv_h, cp)

        d_sinks = None
        if ctx.has_sinks:
            if cp.is_active():
                buf = [torch.empty_like(d_sinks_local) for _ in range(cp.world)]
                dist.all_gather(buf, d_sinks_local, group=cp.group)
            else:
                buf = [d_sinks_local]
            d_sinks = torch.cat(buf, dim=0)

        return dq, dk, dv, d_sinks, None, None, None, None


def ulysses_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = True,
    scale: float | None = None,
    sinks: torch.Tensor | None = None,
    sliding_window: int | None = None,
    cp_group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Ulysses-style attention with GQA + optional per-head sink logit.

    Args:
        q (`torch.Tensor`): Query, seq-sharded, shape ``(B, H_q, N_local, D)``.
        k (`torch.Tensor`): Key, seq-sharded, shape ``(B, H_kv, N_local, D)``.
        v (`torch.Tensor`): Value, seq-sharded, shape ``(B, H_kv, N_local, D)``.
        is_causal (`bool`, *optional*, defaults to ``True``):
            Whether to apply a causal mask inside the local attention.
        scale (`float`, *optional*):
            Softmax scaling factor. Defaults to ``D ** -0.5``.
        sinks (`torch.Tensor`, *optional*):
            Per-head sink logit, shape ``(H_q,)``. Required for GPT-OSS-style
            attention; ``None`` for plain LLaMA-style.
        sliding_window (`int`, *optional*):
            If set, applies a band mask of the given width on top of the
            causal mask.
        cp_group (`torch.distributed.ProcessGroup`, *optional*):
            CP-axis process group. ``None`` or world-size-1 group falls
            back to local attention.

    Returns:
        `torch.Tensor` of shape ``(B, H_q, N_local, D)`` (seq-sharded).
    """
    if scale is None:
        scale = q.size(-1) ** -0.5
    if cp_group is None or dist.get_world_size(cp_group) == 1:
        return _local_attention(
            q,
            k,
            v,
            is_causal=is_causal,
            scale=float(scale),
            sinks=sinks,
            sliding_window=sliding_window,
        )
    return _UlyssesFn.apply(
        q,
        k,
        v,
        sinks,
        is_causal,
        float(scale),
        sliding_window,
        cp_group,
    )


__all__ = ["CPGroup", "build_cp_group_from_default", "ulysses_attention"]
