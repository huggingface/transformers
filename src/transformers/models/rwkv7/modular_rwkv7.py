# Copyright 2026 The RWKV team and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch RWKV-7 ("Goose") model.

Attention-free and fully recurrent: state is O(1) in sequence length, so there is
no KV cache and no attention mask over past tokens.

NAMING. Submodule and parameter names follow the upstream RWKV reference
(`BlinkDL/RWKV-LM`): `blocks.N.att.{receptance,key,value,output}`, the LoRA
factors as raw `w1/w2`-style parameters rather than `nn.Linear` pairs, `emb`,
`head`, `ln0/ln1/ln2`, `att.ln_x`, so that converting a native `.pth` is a copy
rather than a rename table. That is a deliberate departure from the `rwkv` (v4)
port in this repo, whose renamed parameters make every conversion script carry
its own mapping.

Per-layer time-mix (att):
    shifted = prev_token(x);  x_* = x + x_*·(shifted - x)     for * in r,w,k,v,a,g
    r = receptance(xr); k = key(xk); v = value(xv)
    w_log = -e^-0.5 · sigmoid( tanh(xw @ w1) @ w2 + w0 )              # log decay
    a     = sigmoid( xa @ a1 @ a2 + a0 )                              # in-context LR
    g     = sigmoid(xg @ g1) @ g2                                     # output gate
    v    += (v_first - v) · sigmoid( xv @ v1 @ v2 + v0 )              # layer > 0
    kk = l2norm_per_head(k · k_k);  k = k + k·(a - 1)·k_a
    y  = WKV(r, w_log, k, v, kk, a)
    y  = ln_x(y) + (r·k·r_k).sum(-1, keepdim) · v ;  out = output(y · g)

Channel-mix (ffn):
    shifted = prev_token(x);  xk = x + x_k·(shifted - x)
    out = value(relu(key(xk))**2)
"""

import math
from dataclasses import dataclass

import torch
import torch.utils.checkpoint
from torch import nn

from ...cache_utils import Cache, LinearAttentionLayer
from ...generation import GenerationMixin
from ...initialization import zeros_
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring, can_return_tuple, logging
from .configuration_rwkv7 import Rwkv7Config


logger = logging.get_logger(__name__)

# e^-0.5. The decay LoRA emits w_log = -INV_SQRT_E * sigmoid(...), so w_log lies in
# (-e^-0.5, 0) and the per-step decay exp(w_log) lies in (exp(-e^-0.5), 1), i.e.
# (0.5452, 1). See RWKV-7 reference. Note the floor is exp(-e^-0.5), not e^-0.5: this
# comment said the latter, and `rwkv7_chunked` built its chunk_size bound on it.
_INV_SQRT_E = 0.6065306597126334


def rwkv7_recurrent(
    r: torch.Tensor,
    w_log: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    state: torch.Tensor,
    compute_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference RWKV-7 WKV recurrence (generalised delta rule).

    All inputs are `[batch, seq_len, num_heads, head_dim]`; `state` is
    `[batch, num_heads, head_dim, head_dim]` and is carried across calls.

    Per token, with `S` the per-head state and every product outer over the two
    head_dim axes::

        decay = exp(w_log)
        sa    = (-kk) @ S                 # uses the PRE-update state
        S     = decay * S + (kk * a) ⊗ sa + k ⊗ v
        out   = r @ S                     # uses the POST-update state

    The state axes are (key, value); `S[i, j]` accumulates key channel `i` against
    value channel `j`. Accumulation is fp32 regardless of the activation dtype:
    the recurrence is unrolled over the whole sequence, so a lower-precision state
    drifts. This is the portable path; a fused kernel may replace it as long as it
    reproduces these values.
    """
    batch, seq_len, num_heads, head_dim = r.shape
    dtype = r.dtype
    r, w_log, k, v, kk, a = (t.to(compute_dtype) for t in (r, w_log, k, v, kk, a))
    state = state.to(compute_dtype)
    out = torch.empty(batch, seq_len, num_heads, head_dim, device=r.device, dtype=compute_dtype)

    for t in range(seq_len):
        decay = torch.exp(w_log[:, t])[..., None]  # [B, H, K, 1]
        kk_t = kk[:, t]
        b = (kk_t * a[:, t])[..., None]  # [B, H, K, 1]
        # sa[b, h, j] = sum_i (-kk[b, h, i]) * S[b, h, i, j]
        sa = torch.einsum("bhi,bhij->bhj", -kk_t, state)[:, :, None, :]  # [B, H, 1, V]
        state = decay * state + b * sa + k[:, t][..., None] * v[:, t][:, :, None, :]
        out[:, t] = torch.einsum("bhi,bhij->bhj", r[:, t], state)

    return out.to(dtype), state


def _unit_lower_triangular_inverse(strict: torch.Tensor, block: int = 8) -> torch.Tensor:
    """Inverse of a batch of unit lower triangular matrices `I + strict`.

    `strict` is strictly lower triangular, so it is nilpotent and the Neumann series
    `I - strict + strict^2 - ...` terminates -- which makes an inverse by repeated
    squaring look like the obvious choice, and it is wrong here. The intermediate
    powers are not bounded by the answer: on a real chunk from a 3-layer model at
    T=1024, `strict` has entries at most 0.977 and the true inverse has entries at
    most 1.0, but `strict^32` reaches 1.3e11. Summing a series whose terms exceed the
    result by eleven orders of magnitude cancels away every digit float32 has -- both
    Newton doubling and a plain Neumann sum came back with entries of 1e4 where the
    answer is 1, and the NaNs that followed reached the states two layers on. It shows
    up in a fifth of randomly initialised 3-layer models at T=1024, one in forty at
    T=256 and none at all at T=128 or below: it needs several chunks to compound, so a
    short-prompt test cannot see it however many initialisations it tries.

    Block forward substitution never forms a high power. `block` bounds the largest
    power taken, and the same chunk that breaks the series gives 6.6e-7 at block 8 and
    1.3e-7 at block 4, against float64 `linalg.solve_triangular`; 16 already costs a
    digit and a half (1.9e-5) because the within-block series runs longer.

    Random triangular matrices do not show any of this, which is worth stating because
    it is what a test would reach for first: they agree with the series to 1e-6, since
    their own inverses are as large as the intermediate powers. The cancellation needs
    an inverse that stays near 1 while the powers do not, which is what the delta rule
    produces and what a random matrix does not.

    `linalg.solve_triangular` is the obvious call and is the accurate one, but it
    decomposes for export into a graph carrying a scalar tensor constant, which
    `lift_fresh` turns into `aten.alias`, functionalization turns into the in-place
    `aten.detach_`, and aot_autograd rejects -- one of the two ops that made every ONNX
    export subtest fail. Inverting once for every chunk at a time, rather than solving
    inside the serial chunk loop, is what keeps that affordable; it is not free even
    so. Measured against the per-chunk solve, loop included, on an RTX 5090 at 1.5B
    shapes: 2.5x at T=1024 where 16 chunks cannot fill the card, falling to 1.2x at
    T=4096, and flat in batch and head count. End to end on CPU, a T=1024 forward
    runs about a third slower than it did on the solve.
    """
    span = strict.shape[-1]
    block = min(block, span)
    # Padding with zeros extends the matrix by an identity block, whose inverse is
    # itself, so the leading span x span block of the result is unchanged.
    pad = -span % block
    if pad:
        strict = torch.nn.functional.pad(strict, (0, pad, 0, pad))
    width = span + pad
    blocks = width // block

    eye = torch.eye(block, device=strict.device, dtype=strict.dtype)
    # Every diagonal block at once: [..., blocks, block, block].
    diag = strict.reshape(*strict.shape[:-2], blocks, block, blocks, block)
    diag = diag.diagonal(dim1=-4, dim2=-2).movedim(-1, -3)
    # Exact after `block` terms, and `block` is small enough that the powers stay near
    # the answer -- which is the whole point of the blocking.
    inverse, term = eye - diag, -diag
    for _ in range(block - 2):
        term = -(diag @ term)
        inverse = inverse + term

    # `out` grows one block row at a time and is the inverse of the leading principal
    # submatrix throughout, so the substitution reads exactly the columns it has
    # already filled and never multiplies the zeros above the diagonal.
    out = inverse[..., 0, :, :]
    for i in range(1, blocks):
        start, stop = i * block, (i + 1) * block
        d = inverse[..., i, :, :]
        row = torch.cat([-d @ (strict[..., start:stop, :start] @ out), d], dim=-1)
        out = torch.cat([torch.nn.functional.pad(out, (0, block)), row], dim=-2)
    return out[..., :span, :span] if pad else out


def rwkv7_chunked(
    r: torch.Tensor,
    w_log: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    state: torch.Tensor,
    chunk_size: int = 64,
    compute_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunk-parallel form of [`rwkv7_recurrent`], for sequences.

    The step is `S_t = A_t S_{t-1} + k_t v_t^T` with `A_t = diag(w_t) - b_t kk_t^T`
    and `b_t = kk_t * a_t`, i.e. diagonal-plus-rank-one. Substituting
    `S_t = diag(c_t) P_t` with the running decay `c_t = prod_{s<=t} w_s` removes the
    diagonal part and leaves a plain delta rule::

        P_t = (I - b~_t q~_t^T) P_{t-1} + k~_t v_t^T
        b~ = b / c,  k~ = k / c,  q~ = kk * c_{t-1},  r~ = r * c

    Writing `u_t = q~_t^T P_{t-1}` turns a whole chunk into one unit-lower-triangular
    system, so the chunk is a handful of matmuls instead of `chunk_size` sequential
    steps; only the chunk-to-chunk carry stays serial::

        (I + tril(Q~ B~^T, -1)) U = Q~ P_0 + tril(Q~ K~^T, -1) V
        O   = R~ P_0 + tril(R~ K~^T, 0) V - tril(R~ B~^T, 0) U
        P_C = P_0 + K~^T V - B~^T U

    `chunk_size` is bounded by that division by `c`, but by OVERFLOW rather than by
    precision, and the difference is worth a factor of four. The per-step decay is at
    least `exp(-e^-0.5)` = 0.5452, so at the worst case, every channel pinned at that
    floor, `1/c` grows like `e^(e^-0.5 * chunk_size)`. That reaches 7.2e16 at 64 and
    fp32 tops out near 3.4e38, so there are twenty-odd decades of headroom left at the
    default, and the ceiling is `ln(finfo.max) / e^-0.5` = 146 in fp32.

    That derivation is worth stating carefully because the first version of it was
    wrong in a way the numbers hid. It said the decay floor was `e^-0.5` rather than
    `exp(-e^-0.5)`, hence growth like `e^(0.5 * chunk_size)`, which puts the ceiling at
    177, and measured at the decay floor this function returns all-NaN from 147 up. The
    quoted 7.2e16 came from the correct law all along (`e^(0.5*64)` is 7.9e13), so the
    arithmetic had been done right and written up wrong, and following the prose rather
    than the number led into the overflow band. `chunk_size` is checked below rather
    than left to that reasoning.

    Precision does not degrade with it, because the substitution is a similarity
    transform: whatever `1/c` inflates, `c` deflates again on the way out, and a
    common scale factor does not move a floating-point relative error. Measured at
    the decay floor, `T=256`, against the sequential form: chunk 16 gives 2.0e-07 and
    chunk 64 gives 2.7e-07, both fp32 noise, while the recurrence runs 3.9x faster
    (3.771 ms -> 0.913 ms). 16 was the conservative reading of the same bound.
    """
    batch, seq_len, num_heads, head_dim = r.shape
    dtype = r.dtype
    r, w_log, k, v, kk, a = (t.to(compute_dtype) for t in (r, w_log, k, v, kk, a))
    state = state.to(compute_dtype)
    # Everything that does not touch the carried state is computed for ALL chunks at
    # once. Only the state recurrence is serial, and it was pulling the rest of the
    # arithmetic into the Python loop with it: at T=256 and chunk 64 that was four
    # iterations of ten-odd launches each, plus three constant matrices rebuilt every
    # iteration. The loop below now does the state-dependent terms and nothing else.
    # Never pad past the sequence: a 16-token prefill grouped into 64-wide chunks does
    # four times the arithmetic for the same answer, which cost 1x16 nine points before
    # this line existed. The old loop avoided it by shortening its last chunk.
    chunk_size = min(chunk_size, seq_len)
    # Refuse rather than return NaN. `1/c` reaches `e^(e^-0.5 * chunk_size)` at the
    # decay floor, so the widest chunk this dtype can carry is derived from the dtype
    # rather than written down: 146 in fp32, 12 in fp16. A caller who raises
    # `chunk_size` for a longer prefill gets an error naming the limit instead of
    # all-NaN logits several layers later, which is what the previous version did.
    widest = int(math.log(torch.finfo(compute_dtype).max) / _INV_SQRT_E)
    if chunk_size > widest:
        raise ValueError(
            f"chunk_size={chunk_size} overflows {compute_dtype}: the running decay's "
            f"reciprocal grows like e^(e^-0.5 * chunk_size), so the widest chunk that "
            f"stays finite is {widest}. Lower chunk_size, or raise compute_dtype."
        )
    chunks = (seq_len + chunk_size - 1) // chunk_size
    padded = chunks * chunk_size
    if padded != seq_len:
        pad = (0, 0, 0, 0, 0, padded - seq_len)
        r, k, v, kk, a = (torch.nn.functional.pad(t, pad) for t in (r, k, v, kk, a))
        # A pad step must be the identity for the recurrence: decay 1, nothing added.
        w_log = torch.nn.functional.pad(w_log, pad)
    grouped = lambda t: t.reshape(batch, chunks, chunk_size, num_heads, head_dim)  # noqa: E731
    rg, kg, vg, kkg, ag, wg = (grouped(t) for t in (r, k, v, kk, a, w_log))

    # `cumprod(exp(x))` and `exp(cumsum(x))` are the same function, and the running
    # decay is only ever needed in the exponentiated form, so the sum is taken first.
    # Two reasons to prefer it. Numerically, a cumulative product of up-to-1 factors
    # underflows the further it runs while the equivalent sum of logs does not, and
    # `c_prev` becomes a subtraction instead of a division by a possibly-tiny `w_c`.
    # For export, `aten.cumprod` decomposes into a graph carrying a scalar tensor
    # constant; `aten.lift_fresh` on that constant decomposes to `aten.alias`, which
    # functionalization rewrites to the in-place `aten.detach_`, and aot_autograd
    # rejects the result -- which is what made every ONNX export subtest fail.
    cum = torch.cumsum(wg, dim=2)
    c = torch.exp(cum)
    c_prev = torch.exp(cum - wg)
    bg = kkg * ag
    b_t, k_t = bg / c, kg / c
    q_t, r_t = kkg * c_prev, rg * c

    span = chunk_size
    tri = torch.ones(span, span, device=r.device, dtype=r.dtype).tril(-1)
    causal = torch.ones(span, span, device=r.device, dtype=r.dtype).tril(0)

    # [batch, chunks, heads, span, span] -- one launch each instead of one per chunk.
    qb = torch.einsum("bcthn,bcshn->bchts", q_t, b_t) * tri
    qk = torch.einsum("bcthn,bcshn->bchts", q_t, k_t) * tri
    rk = torch.einsum("bcthn,bcshn->bchts", r_t, k_t) * causal
    rb = torch.einsum("bcthn,bcshn->bchts", r_t, b_t) * causal
    lhs_inv = _unit_lower_triangular_inverse(qb)
    qkv = torch.einsum("bchts,bcshv->bchtv", qk, vg)
    rkv = torch.einsum("bchts,bcshv->bchtv", rk, vg)
    c_last = c[:, :, -1].unsqueeze(-1)

    # `k~^T v` carries no state either, so it joins the batched half above; only the
    # two products that read the carried state and the solve that depends on them are
    # genuinely serial.
    kv = torch.einsum("bcthn,bcthv->bchnv", k_t, vg)
    # Both state products read the SAME state, so they are one matmul over a stacked
    # token axis rather than two -- the loop is short and launch-bound, and this is one
    # launch per chunk instead of two.
    qr = torch.cat([q_t, r_t], dim=2)

    outputs = []
    for i in range(chunks):
        qr_s = torch.einsum("bthn,bhnv->bhtv", qr[:, i], state)
        rhs = qr_s[:, :, :chunk_size] + qkv[:, i]
        u = lhs_inv[:, i] @ rhs
        out_c = qr_s[:, :, chunk_size:] + rkv[:, i] - torch.einsum("bhts,bhsv->bhtv", rb[:, i], u)
        outputs.append(out_c.permute(0, 2, 1, 3))
        state = c_last[:, i] * (state + kv[:, i] - torch.einsum("bthn,bhtv->bhnv", b_t[:, i], u))

    # The chunked loop pads the sequence up to a multiple of the chunk size and
    # computes in a wider dtype, so both the trailing slice and the cast back are
    # no-ops whenever the input needed neither. A no-op slice or cast still returns
    # a view, ONNX export turns that view into `aten.alias`, functionalization
    # rewrites the alias to the in-place `aten.detach_`, and aot_autograd rejects
    # the graph. Doing each step only when it changes something keeps the common
    # path free of views without changing any value.
    out = torch.cat(outputs, dim=1)
    if out.shape[1] != seq_len:
        out = out[:, :seq_len]
    if out.dtype != dtype:
        out = out.to(dtype)
    return out, state


def rwkv7_eager(
    r: torch.Tensor,
    w_log: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    state: torch.Tensor,
    cu_seq_lens_q: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Portable WKV: the sequential step for a single token, chunk-parallel otherwise.

    With `cu_seq_lens_q` the row is a *packed* batch: several independent sequences
    laid end to end, the layout a varlen kernel consumes. Each one has to start
    from a fresh state, so they are run in turn and their outputs concatenated;
    the state returned is the last segment's, which is the one a continuation
    would resume from. This is the reference behaviour, not the fast path: a
    fused varlen kernel does the same segments in one launch, which is the shape
    of optimised implementation the hub-kernels mechanism can substitute in.
    """
    if cu_seq_lens_q is not None:
        bounds = cu_seq_lens_q.tolist()
        outputs = []
        for start, stop in zip(bounds[:-1], bounds[1:]):
            if stop <= start:
                continue
            segment_state = torch.zeros_like(state)
            out, state = rwkv7_eager(
                r[:, start:stop],
                w_log[:, start:stop],
                k[:, start:stop],
                v[:, start:stop],
                kk[:, start:stop],
                a[:, start:stop],
                segment_state,
            )
            outputs.append(out)
        return torch.cat(outputs, dim=1), state

    kernel = rwkv7_recurrent if r.shape[1] == 1 else rwkv7_chunked
    return kernel(r, w_log, k, v, kk, a, state)


class Rwkv7TokenShift(nn.Module):
    """`prev_token(x)`: the previous token's hidden state, zero at sequence start.

    Carries one vector per layer per stream across forward calls, which is what
    makes incremental decoding exact rather than approximate.
    """

    def forward(
        self,
        x: torch.Tensor,
        shift_state: torch.Tensor | None,
        cu_seq_lens_q: torch.Tensor | None = None,
        keep: torch.Tensor | None = None,
    ):
        # x: [batch, seq_len, hidden]; shift_state: [batch, hidden] or None
        if shift_state is None:
            prev = torch.zeros_like(x[:, :1])
        else:
            prev = shift_state[:, None]
        shifted = torch.cat([prev, x[:, :-1]], dim=1)
        if cu_seq_lens_q is not None:
            # In a packed row the token before a segment's first one belongs to the
            # PREVIOUS sequence. Resetting the recurrent state per segment does not
            # cover this (the shift reaches back through it), so the first token of
            # each segment gets the zero shift a sequence start is supposed to see.
            positions = torch.arange(x.shape[1], device=x.device)
            starts = (positions[:, None] == cu_seq_lens_q[None, :-1]).any(dim=1)
            shifted = torch.where(starts[None, :, None], torch.zeros_like(shifted), shifted)
        if keep is None:
            return shifted, x[:, -1]
        # The state handed back must be the last REAL token, not the last position.
        # They coincide under left padding, which is why this went unnoticed; under
        # right padding `x[:, -1]` is a blanked pad, so a continuation resumes from
        # zero instead of from where the sequence actually got to.
        # Integer arithmetic, not the activation dtype. `keep` arrives cast to the
        # model's dtype, and bf16 carries eight mantissa bits: past position 256 the
        # products are no longer distinct, `argmax` returns the first of a tie, and
        # the state comes back from a token several places short of the last real one
        # -- 1022 for 1023 at length 1024, 4088 for 4095 at 4096. An all-ones mask is
        # enough to trigger it, which is what `generate` sends, so this was not
        # confined to padded batches.
        positions = torch.arange(x.shape[1], device=x.device)
        last_real = ((keep.squeeze(-1) > 0) * positions).argmax(dim=-1)
        return shifted, x[torch.arange(x.shape[0], device=x.device), last_real]


class Rwkv7Attention(nn.Module):
    """RWKV-7 time-mixing block (the recurrent replacement for self-attention)."""

    def __init__(self, config: Rwkv7Config, layer_id: int):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        C = config.hidden_size
        self.hidden_size = C
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.state_dtype = getattr(torch, config.wkv_state_dtype)

        self.time_shift = Rwkv7TokenShift()

        # per-channel token-shift mixes (kept at Bo's (1, 1, C) shape)
        self.x_r = nn.Parameter(torch.zeros(1, 1, C))
        self.x_w = nn.Parameter(torch.zeros(1, 1, C))
        self.x_k = nn.Parameter(torch.zeros(1, 1, C))
        self.x_v = nn.Parameter(torch.zeros(1, 1, C))
        self.x_a = nn.Parameter(torch.zeros(1, 1, C))
        self.x_g = nn.Parameter(torch.zeros(1, 1, C))

        # LoRA factors as raw tensors, exactly as the reference stores them
        self.w1 = nn.Parameter(torch.zeros(C, config.decay_low_rank_dim))
        self.w2 = nn.Parameter(torch.zeros(config.decay_low_rank_dim, C))
        self.w0 = nn.Parameter(torch.zeros(1, 1, C))
        self.a1 = nn.Parameter(torch.zeros(C, config.a_low_rank_dim))
        self.a2 = nn.Parameter(torch.zeros(config.a_low_rank_dim, C))
        self.a0 = nn.Parameter(torch.zeros(1, 1, C))
        self.g1 = nn.Parameter(torch.zeros(C, config.gate_low_rank_dim))
        self.g2 = nn.Parameter(torch.zeros(config.gate_low_rank_dim, C))
        # The value-residual LoRA exists on every layer in a reference checkpoint,
        # but layer 0 PRODUCES v_first instead of mixing towards it, so its copy is
        # never read. It is registered anyway so that loading is lossless.
        self.v1 = nn.Parameter(torch.zeros(C, config.v_low_rank_dim))
        self.v2 = nn.Parameter(torch.zeros(config.v_low_rank_dim, C))
        self.v0 = nn.Parameter(torch.zeros(1, 1, C))

        self.k_k = nn.Parameter(torch.zeros(1, 1, C))
        self.k_a = nn.Parameter(torch.zeros(1, 1, C))
        self.r_k = nn.Parameter(torch.zeros(config.num_heads, config.head_dim))

        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)
        # GroupNorm over heads, matching the reference's per-head normalisation.
        #
        # The reference hardcodes `eps=64e-5`, which is `head_dim * 1e-5` at its
        # head_dim of 64 -- NOT `num_heads * 1e-5`. The two coincide only when a
        # model happens to have as many heads as channels per head, which for
        # head_dim 64 means hidden_size 4096 exactly. Scaling by `num_heads` is
        # therefore right on the 7.2B and wrong everywhere else: 2x low on the 1.5B,
        # 5.33x low on the 0.1B. It is written against `head_dim` because that is
        # the axis GroupNorm actually reduces over, so the constant tracks the
        # reference at any width.
        self.ln_x = nn.GroupNorm(config.num_heads, C, eps=config.norm_eps * config.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        v_first: torch.Tensor | None,
        shift_state: torch.Tensor | None,
        wkv_state: torch.Tensor | None,
        keep: torch.Tensor | None = None,
        cu_seq_lens_q: torch.Tensor | None = None,
    ):
        batch, seq_len, C = hidden_states.shape
        H, N = self.num_heads, self.head_dim

        shifted, new_shift_state = self.time_shift(hidden_states, shift_state, cu_seq_lens_q, keep)
        delta = shifted - hidden_states
        xr = hidden_states + self.x_r * delta
        xw = hidden_states + self.x_w * delta
        xk = hidden_states + self.x_k * delta
        xv = hidden_states + self.x_v * delta
        xa = hidden_states + self.x_a * delta
        xg = hidden_states + self.x_g * delta

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)

        w_log, a, g, v_gate = self.lora_gates(xw, xa, xg, None if self.layer_id == 0 else xv)

        if keep is not None:
            # A padding position has to leave the recurrent state exactly as it
            # found it, which takes three things and not one.
            #
            # The decay is held at w = exp(0) = 1, so the transition is the identity.
            #
            # `k` and `v` are zeroed EXPLICITLY rather than relying on the blanked
            # hidden state to make them zero: the projections are bias-free, so that
            # only holds for a pad with nothing before it.
            # A pad that FOLLOWS a real token still receives that token's hidden
            # state through the token shift, so `delta = shifted - 0` is non-zero,
            # and so are `k` and `v`. The update term `k v^T` then entered the state
            # on every right-padded batch and on any left-padded batch continued
            # from a carried shift state.
            w_log = w_log * keep
            k = k * keep
            v = v * keep

        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * v_gate

        kk = k * self.k_k
        kk = torch.nn.functional.normalize(kk.view(batch, seq_len, H, N), dim=-1, p=2.0).view(batch, seq_len, C)
        if keep is not None:
            # A blanked padding position makes `k` exactly zero, so this normalises a
            # zero vector. `F.normalize` divides by `max(norm, 1e-12)`, and 1e-12 is
            # below the smallest fp16 subnormal -- so in fp16 the divisor really is
            # zero and every padded row comes out NaN. (In fp32 it is representable
            # and the whole thing looks fine, which is why a test on a small fp32
            # model missed it entirely.) `where` is used rather than a multiply
            # because NaN * 0 is still NaN.
            kk = torch.where(keep.bool().expand_as(kk), kk, torch.zeros_like(kk))
        k = k * (1 + (a - 1) * self.k_a)

        if wkv_state is None:
            wkv_state = torch.zeros(batch, H, N, N, device=r.device, dtype=self.state_dtype)

        def _heads(t):
            return t.view(batch, seq_len, H, N)

        y, wkv_state = rwkv7_eager(
            _heads(r),
            _heads(w_log),
            _heads(k),
            _heads(v),
            _heads(kk),
            _heads(a),
            wkv_state,
            cu_seq_lens_q=cu_seq_lens_q,
        )

        # `reshape`, not `view`: what comes back from the WKV is whatever layout that
        # implementation produced, and a registered kernel -- or inductor, which is
        # free to pick its own -- can hand back a `[batch, seq_len, heads, head_dim]`
        # that is strided as if it were `[batch, heads, seq_len, head_dim]`. `view`
        # then raises, and it raises only when batch and seq_len are BOTH greater
        # than one, which is why a compiled 16x16 forward failed while 1xT and Bx1
        # both passed.
        y = self.ln_x(y.reshape(batch * seq_len, C)).view(batch, seq_len, C)
        # r·k·r_k summed per head, broadcast back over the head's value channels
        bonus = ((_heads(r) * _heads(k) * self.r_k).sum(dim=-1, keepdim=True) * _heads(v)).reshape(batch, seq_len, C)
        y = self.output((y + bonus) * g)
        return y, v_first, new_shift_state, wkv_state

    def lora_gates(self, xw, xa, xg, xv):
        """The four low-rank chains, as one unit.

        Each is a rank-r down projection, an activation, and an up projection, and
        they differ only in where the activation sits: `w` has `tanh` between the
        two, `g` has `sigmoid` between them and nothing after, `a` and `v` have
        nothing between and `sigmoid` after. Together they are a dozen tiny
        matrix-vector products per layer whose weights are a rounding error of the
        model's bytes -- so at batch 1 they cost latency, not bandwidth, which is
        the shape a fused kernel improves and a portable implementation cannot.

        Kept as one method for exactly that reason: it is the unit worth replacing,
        and replacing it should not need a fork. `xv` is None on layer 0, which
        produces `v_first` rather than mixing towards it and never reads that chain.
        """
        w_log = -_INV_SQRT_E * torch.sigmoid(torch.tanh(xw @ self.w1) @ self.w2 + self.w0)
        a = torch.sigmoid(xa @ self.a1 @ self.a2 + self.a0)
        g = torch.sigmoid(xg @ self.g1) @ self.g2
        v_gate = None if xv is None else torch.sigmoid(xv @ self.v1 @ self.v2 + self.v0)
        return w_log, a, g, v_gate


class Rwkv7FeedForward(nn.Module):
    """RWKV-7 channel-mixing block: squared-ReLU over a single token shift."""

    def __init__(self, config: Rwkv7Config, layer_id: int):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        C = config.hidden_size
        self.time_shift = Rwkv7TokenShift()
        self.x_k = nn.Parameter(torch.zeros(1, 1, C))
        self.key = nn.Linear(C, config.intermediate_size, bias=False)
        self.value = nn.Linear(config.intermediate_size, C, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        shift_state: torch.Tensor | None,
        deep_embed: torch.Tensor | None = None,
        cu_seq_lens_q: torch.Tensor | None = None,
        keep: torch.Tensor | None = None,
    ):
        shifted, new_shift_state = self.time_shift(hidden_states, shift_state, cu_seq_lens_q, keep)
        xk = hidden_states + self.x_k * (shifted - hidden_states)
        inner = torch.relu(self.key(xk)) ** 2

        # RWKV-8 DeepEmbed: a per-layer, per-token vector that channelwise modulates
        # the channel-mix, supplied by the caller rather than stored as a weight
        # because the design keeps the table in RAM/SSD and prefetches per token.
        # Its width says which side it attaches to -- `intermediate_size` scales the
        # projection's INPUT (the reference "4x" variant), `hidden_size` its OUTPUT
        # ("1x"). Resolved once here rather than per branch, so a later branch cannot
        # drop the 4x variant silently.
        scale_output = None
        if deep_embed is not None:
            if deep_embed.shape[-1] == inner.shape[-1]:
                inner = inner * deep_embed  # only rescales, so exact zeros survive
            else:
                scale_output = deep_embed

        out = self._project(inner)
        if scale_output is not None:
            out = out * scale_output
        return out, new_shift_state

    def _project(self, inner: torch.Tensor) -> torch.Tensor:
        return self.value(inner)


class Rwkv7CacheLayer(LinearAttentionLayer):
    """One block's slice of the recurrent state: the WKV matrix and two token shifts.

    Everything a beam search or a batched generate needs to do to a cache is a
    permutation of its batch axis, and for RWKV-7 that is the whole job -- the state
    is O(1) in sequence length, so there is no time axis to gather along and no
    length bookkeeping to keep consistent.

    The slot layout is [`Rwkv7Cache`]'s; this class only moves whatever is in them.
    """

    def lazy_initialization(self, conv_states=None, recurrent_states=None, state_idx: int = 0, **kwargs) -> None:
        super().lazy_initialization(conv_states, recurrent_states, state_idx, **kwargs)
        # Upstream records device/dtype only on the conv branch, since a linear
        # attention layer normally has a convolution in front of it. This model has
        # none, so without this every recurrent-only layer would keep `device=None`
        # and `reorder_cache`'s `beam_idx.to(self.device)` would fail.
        if recurrent_states is not None and self.device is None:
            self.dtype, self.device = recurrent_states.dtype, recurrent_states.device

    def allocate(
        self, batch: int, shapes: dict[int, tuple], device, dtypes: dict[int, torch.dtype], state_dtype
    ) -> None:
        """Create every slot up front, zeroed, at pinned addresses.

        The lazy path allocates a slot the first time it is written, which is inside
        the compiled region -- and a buffer that first appears there cannot be given
        a static address, so inductor declines CUDA graphs for a recurrent decode,
        the one workload that most needs them.
        """
        for slot, shape in shapes.items():
            buffer = torch.zeros((batch, *shape), device=device, dtype=dtypes[slot])
            if not torch.compiler.is_compiling():
                torch._dynamo.mark_static_address(buffer)
            self.recurrent_states[slot] = buffer
            self.is_recurrent_states_initialized[slot] = True
            self.has_previous_state[slot] = True
        self.dtype, self.device = state_dtype, device

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Permute the batch onto the beams that survived, without moving house.

        Upstream rebinds each slot to the `index_select` result. That is correct but
        it hands back a freshly allocated tensor, so the address pinned at allocation
        is gone for the rest of the generation and the compiled decode quietly loses
        its CUDA graphs. Copying back into the same buffer costs one temporary and
        keeps the pinning.
        """
        for slot in range(self.number_of_states):
            if self.is_recurrent_states_initialized[slot]:
                buffer = self.recurrent_states[slot]
                buffer.copy_(buffer.index_select(0, beam_idx.to(buffer.device)))

    def batch_repeat_interleave(self, repeats: int) -> None:
        """Fan each sequence's state out to `repeats` copies, for one-prompt-many-samples.

        This changes the batch size, so unlike `reorder_cache` it cannot preserve the
        pinned addresses -- the buffers are necessarily new ones. Callers that then
        compile should re-pin, which `Rwkv7Model.allocate_state` does.
        """
        for slot in range(self.number_of_states):
            if self.is_recurrent_states_initialized[slot]:
                self.recurrent_states[slot] = self.recurrent_states[slot].repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        """Keep only the given batch rows. Same address caveat as `batch_repeat_interleave`."""
        for slot in range(self.number_of_states):
            if self.is_recurrent_states_initialized[slot]:
                self.recurrent_states[slot] = self.recurrent_states[slot][indices, ...]


class Rwkv7Cache(Cache):
    """The recurrent state of every block, as a `Cache`.

    It replaces a KV cache and is a constant size: one `[num_heads, head_dim,
    head_dim]` matrix and two `[hidden]` token shifts per layer per sequence,
    whatever the context length. That is the property the architecture is for, so
    `get_max_length()` is -1 (no limit) and nothing here grows as tokens arrive.
    """

    # Slot numbering inside one block's layer. All three are recurrent states:
    # RWKV-7's token shift is a one-token history, not a convolution window, so it
    # lives in a recurrent slot, where the update is a plain copy rather than the
    # rolling concatenate a conv slot would do.
    WKV, ATT_SHIFT, FFN_SHIFT = 0, 1, 2

    def __init__(
        self,
        config: Rwkv7Config,
        batch_size: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(layers=[Rwkv7CacheLayer(number_of_states=3) for _ in range(config.num_hidden_layers)])
        self.config = config
        if batch_size is not None:
            self.allocate(batch_size, device, dtype)

    def allocate(self, batch_size: int, device=None, dtype=None) -> "Rwkv7Cache":
        config = self.config
        shapes = {
            self.WKV: (config.num_heads, config.head_dim, config.head_dim),
            self.ATT_SHIFT: (config.hidden_size,),
            self.FFN_SHIFT: (config.hidden_size,),
        }
        # The WKV state carries the whole history and is the one place where a
        # narrower dtype actually costs accuracy, so it is configured separately
        # from the activation dtype the shifts follow.
        dtypes = {
            self.WKV: getattr(torch, config.wkv_state_dtype),
            self.ATT_SHIFT: dtype,
            self.FFN_SHIFT: dtype,
        }
        for layer in self.layers:
            layer.allocate(batch_size, shapes, device, dtypes, dtypes[self.WKV])
        return self

    def crop(self, tokens_to_remove: int) -> None:
        # The WKV matrix is a lossy running summary, not a sequence of entries:
        # there is nothing to remove N tokens from. Refuse loudly -- the inherited
        # implementation would index conv slots this cache does not have, and a
        # rolled-back state that silently kept its future would be worse than the
        # error. (Same stance as MiniMaxCache.)
        raise RuntimeError("Rwkv7Cache cannot be cropped: the recurrent state is not invertible.")

    def read(self, layer_idx: int):
        """This block's `(att_shift, ffn_shift, wkv)`, each None before allocation."""
        states = self.layers[layer_idx].recurrent_states
        return states[self.ATT_SHIFT], states[self.FFN_SHIFT], states[self.WKV]

    def write(self, layer_idx: int, att_shift: torch.Tensor, ffn_shift: torch.Tensor, wkv: torch.Tensor) -> None:
        states = (
            (self.ATT_SHIFT, att_shift),
            (self.FFN_SHIFT, ffn_shift),
            (self.WKV, wkv),
        )
        # Copying into the pre-allocated slot is what keeps its address fixed, which
        # is what lets a captured CUDA graph replay the decode loop. It is also an
        # in-place write on a tensor autograd may be holding, and backward through a
        # cached forward then dies on "a variable needed for gradient computation has
        # been modified by an inplace operation". Rebinding instead costs nothing
        # here, because a training step is not the workload that wants graphs.
        if torch.is_grad_enabled() and any(state.requires_grad for _, state in states):
            layer = self.layers[layer_idx]
            for slot, state in states:
                layer.recurrent_states[slot] = state
                layer.is_recurrent_states_initialized[slot] = True
                layer.has_previous_state[slot] = True
            return
        for slot, state in states:
            self.update_recurrent_state(state, layer_idx, slot)


class Rwkv7Block(GradientCheckpointingLayer):
    def __init__(self, config: Rwkv7Config, layer_id: int):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        eps, bias = config.norm_eps, config.norm_bias
        # Layer 0 carries the extra input norm of the reference implementation.
        if layer_id == 0:
            self.ln0 = nn.LayerNorm(config.hidden_size, eps=eps, bias=bias)
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=eps, bias=bias)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=eps, bias=bias)
        self.att = Rwkv7Attention(config, layer_id)
        self.ffn = Rwkv7FeedForward(config, layer_id)

    def forward(
        self,
        hidden_states: torch.Tensor,
        v_first: torch.Tensor | None,
        state: Cache | None,
        deep_embed: torch.Tensor | None = None,
        keep: torch.Tensor | None = None,
        cu_seq_lens_q: torch.Tensor | None = None,
    ):
        if self.layer_id == 0:
            hidden_states = self.ln0(hidden_states)

        att_shift, ffn_shift, wkv = state.read(self.layer_id) if state is not None else (None, None, None)

        # Padding is blanked AFTER each norm, not before: a LayerNorm maps the zero
        # vector to its own bias, so masking the residual stream instead would let
        # every pad position come back to life on the way into the next mixer -- and
        # a live pad both moves the state and leaks into the next token's shift.
        attn_in = self.ln1(hidden_states)
        if keep is not None:
            attn_in = attn_in * keep
        attn_out, v_first, att_shift, wkv = self.att(attn_in, v_first, att_shift, wkv, keep, cu_seq_lens_q)
        hidden_states = hidden_states + attn_out

        ffn_in = self.ln2(hidden_states)
        if keep is not None:
            ffn_in = ffn_in * keep
        ffn_out, ffn_shift = self.ffn(ffn_in, ffn_shift, deep_embed, cu_seq_lens_q, keep)
        hidden_states = hidden_states + ffn_out

        if state is not None:
            # `update_recurrent_state` copies into the pre-allocated slot rather than
            # rebinding it, so the buffers keep fixed addresses across steps -- which
            # is what lets a captured CUDA graph replay the decode loop.
            state.write(self.layer_id, att_shift, ffn_shift, wkv)
        return hidden_states, v_first, state


@dataclass
class Rwkv7Output(ModelOutput):
    r"""
    state (`Rwkv7Cache`, *optional*):
        The recurrent state of every block. Feed it back to continue a sequence; it
        replaces the KV cache and is a constant size, whatever the context length.
    """

    last_hidden_state: torch.FloatTensor | None = None
    state: Rwkv7Cache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class Rwkv7CausalLMOutput(ModelOutput):
    r"""
    state (`Rwkv7Cache`, *optional*):
        The recurrent state, as in [`Rwkv7Output`].
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    state: Rwkv7Cache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None


@auto_docstring
class Rwkv7PreTrainedModel(PreTrainedModel):
    config: Rwkv7Config
    base_model_prefix = "rwkv7"
    _no_split_modules = ["Rwkv7Block"]
    supports_gradient_checkpointing = True
    _is_stateful = True
    # Beam search reorders through `Rwkv7Cache.reorder_cache`. Defining
    # `_reorder_cache` here instead would take precedence over it in `generate` and
    # bypass the cache's own bookkeeping, so it is deliberately absent.

    def _init_weights(self, module):
        """Also initialise the parameters that are not inside a standard submodule.

        The time-mix and channel-mix hold twenty-one raw `nn.Parameter`s -- the six
        token-shift mixes, the four LoRA factor pairs and their biases, `k_k`, `k_a`,
        `r_k`. The inherited `_init_weights` knows about `nn.Linear`, `nn.Embedding`
        and `nn.LayerNorm` and nothing else, so none of those were reachable: a model
        built on a meta device and then materialised came back with them uninitialised,
        which `test_can_init_all_missing_weights` says plainly and which nothing here
        was running until this suite was first executed on a GPU.

        Zeros, because that is exactly what `__init__` does -- this makes materialising
        from meta agree with ordinary construction rather than inventing a second
        answer. It is deliberately NOT the reference training initialisation, which
        gives the shift mixes a per-layer ramp and the LoRA factors particular scales.
        Reproducing that belongs with training support, which this port does not claim;
        every real use loads a converted checkpoint over these values.

        Through `initialization.zeros_` rather than `parameter.data.zero_()`, and the
        difference is not cosmetic. The framework skips re-initialising anything already
        loaded by checking an `_is_hf_initialized` flag that those helpers set and a raw
        in-place write does not. The first version of this method wrote in place, so the
        flag was never set, `_init_weights` ran anyway, and 249 of a converted 0.1B
        checkpoint's parameters were zeroed after being loaded -- the model produced
        "civil civil civil" where the reference runtime produced "Paris, France".
        """
        super()._init_weights(module)
        if isinstance(module, (Rwkv7Attention, Rwkv7FeedForward)):
            for parameter in module._parameters.values():
                if parameter is not None:
                    zeros_(parameter)


@auto_docstring
class Rwkv7Model(Rwkv7PreTrainedModel):
    def __init__(self, config: Rwkv7Config):
        super().__init__(config)
        self.emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([Rwkv7Block(config, i) for i in range(config.num_hidden_layers)])
        self.ln_out = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.emb

    def set_input_embeddings(self, new_embeddings):
        self.emb = new_embeddings

    def allocate_state(self, batch: int, device=None, dtype=None) -> Rwkv7Cache:
        """A zeroed cache, plus everything else the decode path would build lazily.

        Call this before compiling, and pass the result in as `state=`.
        `mark_static_address` cannot run during tracing, so anything first allocated
        *inside* the compiled region stays unpinned, and inductor declines CUDA graphs
        for a region that mutates its inputs. Starting from `state=None` loses the
        state buffers that way, and the decode then runs several times slower while
        saying so only in a line of warning.
        """
        state = self._empty_state(
            batch,
            device if device is not None else self.emb.weight.device,
            dtype if dtype is not None else self.emb.weight.dtype,
        )
        return state

    def _empty_state(self, batch: int, device, dtype) -> Rwkv7Cache:
        return Rwkv7Cache(self.config, batch_size=batch, device=device, dtype=dtype)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        state: Rwkv7Cache | None = None,
        deep_embeds: torch.FloatTensor | None = None,
        cu_seq_lens_q: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> Rwkv7Output:
        r"""
        attention_mask (`torch.LongTensor`, *optional*):
            1 on real tokens, 0 on padding. Read as the **tail** of whatever is
            given, so a decode step may hand over the whole conversation's mask and
            only the last position is used. A prefix chunk must therefore slice its
            own mask to match the `input_ids` it passes, or it silently masks the
            wrong positions. There is nothing here for a mask to hide behind: an
            all-recurrent model feeds pads through the recurrence like any other
            token unless they are neutralised.
        state (`Rwkv7Cache`, *optional*):
            Recurrent state returned by a previous call; pass it back to continue
            the sequence. Allocated on the first forward if omitted. Ignored as a
            *history* when `cu_seq_lens_q` is given -- see there.
        deep_embeds (`torch.FloatTensor`, *optional*):
            RWKV-8 DeepEmbed vectors for this batch, shaped
            `[num_layers, batch, seq_len, hidden_size or intermediate_size]` (or broadcastable).
            Only meaningful when `config.use_deep_embed` is set; the table itself is
            external to the checkpoint by design.
        cu_seq_lens_q (`torch.LongTensor`, *optional*):
            Cumulative sequence lengths for a *packed* batch: several sequences
            concatenated into one row instead of padded to a rectangle, starting at
            0 and ending at `seq_len`, and non-decreasing. Each segment then decodes
            from a fresh recurrent state, as if it had been run on its own. This is
            the varlen layout; use it instead of padding when the lengths vary a
            lot, since a recurrent model pays for pad tokens in time as well as
            memory.

            A packed batch is a set of *new* sequences, not a continuation, so a
            `state` passed alongside contributes its shape and dtype and nothing
            else -- its contents are not read. What comes back is the last segment's
            state, which is the one a continuation of this row would resume from.
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        # Gradient checkpointing replays the forward during backward. If the cache is
        # live, the replay re-reads a state the first pass already advanced, and the
        # gradients that come back are wrong rather than absent. `GradientCheckpointingLayer`
        # neutralises this itself, but only for a cache arriving as a keyword named
        # `use_cache` / `past_key_values` / `layer_past`; this model hands its state to
        # the block positionally, so none of those guards see it and the whole thing
        # has to be caught here.
        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.emb(input_ids)

        if use_cache and state is None:
            state = self._empty_state(inputs_embeds.shape[0], inputs_embeds.device, inputs_embeds.dtype)

        # An attention-free recurrence has to be told where the padding is: pads are
        # fed through the recurrence like any other token, so a left-padded batch
        # (what `generate` produces) would otherwise start every short row from a
        # state the pads had already moved. A single decoded token is never padding,
        # so the decode step takes none of this.
        keep = None
        if attention_mask is not None:
            # The mask is taken as the TAIL of whatever is passed, so a decode step
            # can hand over the whole conversation's mask. A prefix chunk must slice
            # its own mask to match, or it silently reads the wrong positions.
            #
            # Applied at seq_len == 1 as well. Skipping it there assumes a single
            # decoded token is never padding -- true of `generate`, but an
            # assumption about the caller rather than a property
            # of the model, and a fully-masked 1-token row was moving the state.
            keep = attention_mask[:, -inputs_embeds.shape[1] :, None].to(inputs_embeds.dtype)

        if cu_seq_lens_q is not None:
            # Checked rather than trusted: a malformed boundary list does not fail,
            # it silently splits the recurrence in the wrong places and returns
            # fluent output computed from the wrong states.
            if inputs_embeds.shape[0] != 1:
                raise ValueError(
                    f"cu_seq_lens_q describes one packed row, but got batch size {inputs_embeds.shape[0]}. "
                    "Pack the sequences into a single row, or use attention_mask with a padded batch."
                )
            if cu_seq_lens_q.ndim != 1 or cu_seq_lens_q[0] != 0 or cu_seq_lens_q[-1] != inputs_embeds.shape[1]:
                raise ValueError(
                    f"cu_seq_lens_q must be 1-D, start at 0 and end at seq_len ({inputs_embeds.shape[1]}); "
                    f"got {cu_seq_lens_q.tolist()}"
                )
            # Endpoints alone do not pin the list down. A pair that goes backwards
            # is skipped rather than rejected further in, so the segments emit fewer
            # tokens than came in and the row silently changes length -- and one that
            # merely repeats a boundary contributes an empty segment, which is
            # harmless but is never what the caller meant.
            if bool((cu_seq_lens_q[1:] <= cu_seq_lens_q[:-1]).any()):
                raise ValueError(
                    "cu_seq_lens_q must be strictly increasing (each segment needs at least one token); "
                    f"got {cu_seq_lens_q.tolist()}"
                )

        hidden_states = inputs_embeds
        v_first = None
        all_hidden_states = () if output_hidden_states else None

        for block in self.blocks:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            # Gated on the config flag as well as on the argument: passing
            # `deep_embeds` to a model configured without them would otherwise modulate
            # the channel-mix anyway, which is a silently different model.
            layer_deep_embed = (
                deep_embeds[block.layer_id] if deep_embeds is not None and self.config.use_deep_embed else None
            )
            hidden_states, v_first, state = block(hidden_states, v_first, state, layer_deep_embed, keep, cu_seq_lens_q)

        hidden_states = self.ln_out(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return Rwkv7Output(last_hidden_state=hidden_states, state=state, hidden_states=all_hidden_states)


@auto_docstring
class Rwkv7ForCausalLM(Rwkv7PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"head.weight": "rwkv7.emb.weight"}

    def __init__(self, config: Rwkv7Config):
        super().__init__(config)
        self.rwkv7 = Rwkv7Model(config)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_output_embeddings(self):
        return self.head

    def set_output_embeddings(self, new_embeddings):
        self.head = new_embeddings

    def prepare_inputs_for_generation(
        self, input_ids, state=None, inputs_embeds=None, is_first_iteration: bool = False, **kwargs
    ):
        # Recurrent: once the prompt has been consumed, only the newest token is
        # needed. The gate is the generation step (`is_first_iteration`, which
        # `generate` passes every call), NOT `state is not None`: the compile
        # contract tells callers to hand the FIRST call a pre-allocated state
        # (`allocate_state`), and a warm state is how a chat turn resumes.
        # Truncating on existence silently dropped every prompt token but the
        # last one in both of those cases -- same gate Mamba uses. Outside
        # `generate` the flag defaults to False, so a bare caller with a state
        # keeps the old truncate-on-existence behaviour: NOT truncating there
        # would replay the whole running sequence into a state that already
        # contains it, measured as a slow score drift, not a crash.
        if state is not None and not is_first_iteration:
            input_ids = input_ids[:, -1:]
        model_inputs = (
            {"input_ids": input_ids}
            if inputs_embeds is None or state is not None
            else {"inputs_embeds": inputs_embeds}
        )
        model_inputs["state"] = state
        # Everything else the caller passed goes through, minus what this model does
        # not take. An allowlist of specific names instead would make
        # `generate(output_hidden_states=True)` return a tuple of `None`: the flag
        # would be dropped here, the forward would never see it, and generate collects the
        # nothing it got back. Any user kwarg met the same fate, silently. The two
        # excluded here are `labels`, which would make generate compute a loss it
        # never reads, and the KV-cache bookkeeping that belongs to models with a KV
        # cache; this one carries its history in `state`. (`is_first_iteration` is
        # absorbed by the signature above: it gates the truncation and is not a
        # forward argument.)
        skip = ("labels", "next_sequence_length", "past_key_values", "cache_position")
        model_inputs.update({k: v for k, v in kwargs.items() if k not in skip and k not in model_inputs})
        return model_inputs

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        state: Rwkv7Cache | None = None,
        deep_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> Rwkv7CausalLMOutput:
        r"""
        state (`Rwkv7Cache`, *optional*):
            Recurrent state returned by a previous call.
        deep_embeds (`torch.FloatTensor`, *optional*):
            RWKV-8 DeepEmbed vectors; see [`Rwkv7Model.forward`].
        logits_to_keep (`int` or `torch.Tensor`, *optional*, defaults to 0):
            Compute the head over only the last `logits_to_keep` positions, or over the
            positions this tensor indexes; 0 means all of them. Worth having on a
            recurrent model for the same reason as on any other: a prefill needs one
            row of logits and the vocabulary is the widest matrix in the model, so
            running the head over the whole prompt is the single largest avoidable cost
            in a prefill. Until this argument existed it was swallowed by `**kwargs`,
            so `generate` declined to pass it and a caller who passed it was quietly
            ignored.
        """
        outputs = self.rwkv7(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            state=state,
            deep_embeds=deep_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )
        keep = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.head(outputs.last_hidden_state[:, keep, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.config.vocab_size, **kwargs)

        return Rwkv7CausalLMOutput(loss=loss, logits=logits, state=outputs.state, hidden_states=outputs.hidden_states)


__all__ = ["Rwkv7Cache", "Rwkv7PreTrainedModel", "Rwkv7Model", "Rwkv7ForCausalLM"]
