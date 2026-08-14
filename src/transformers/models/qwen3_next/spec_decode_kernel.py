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
Triton kernel for GatedDeltaNet speculative-decoding state capture.

During speculative decoding the target model verifies N candidate tokens but
may accept only k < N.  The standard `chunk_gated_delta_rule` returns only the
FINAL recurrent state (after all N tokens), so correcting for k < N requires an
extra forward pass.  This kernel instead processes all N tokens in ONE call and
writes state[0..N] to a pre-allocated buffer so the caller can select state[k]
without any rerun.

Compared to the vLLM `fused_sigmoid_gating_delta_rule_update_kernel`:
  - Removes `ssm_state_indices` / paged-cache indirection (no paged KV cache
    in transformers standard generation).
  - Removes IS_VARLEN, IS_CONTINUOUS_BATCHING (not needed here).
  - State buffer is [T+1, B, HV, K, V]; index 0 = pre-verify, index t+1 = after
    token t.

Falls back to a pure-PyTorch loop when Triton is not installed.
"""

import torch


try:
    import triton
    import triton.language as tl

    @triton.jit(do_not_specialize=["T"])
    def _spec_recurrent_gdn_fwd_kernel(
        q_ptr,  # [B, T, HV, K]  contiguous; K is fast axis
        k_ptr,  # [B, T, HV, K]
        v_ptr,  # [B, T, HV, V]  contiguous; V is fast axis
        g_ptr,  # [B, T, HV]     contiguous; HV is fast axis (log-decay, not yet exp'd)
        beta_ptr,  # [B, T, HV]     contiguous; HV is fast axis
        h0_ptr,  # [B, HV, K, V]  contiguous; V is fast axis (may be null ptr)
        ht_ptr,  # [T+1, B, HV, K, V]  contiguous; V is fast axis (output)
        o_ptr,  # [B, T, HV, V]  contiguous; V is fast axis (output)
        # runtime scalars
        T,  # number of candidate tokens
        scale,  # query scale = head_k_dim ** -0.5
        # shapes (constexpr for tile-size computation and masking)
        B: tl.constexpr,
        HV: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BK: tl.constexpr,  # = next_power_of_2(K); covers all of K in one tile (NK==1)
        BV: tl.constexpr,  # tile size along V
        # strides for q/k  [B, T, HV, K]
        sq_b: tl.constexpr,
        sq_t: tl.constexpr,
        sq_h: tl.constexpr,
        # strides for v    [B, T, HV, V]
        sv_b: tl.constexpr,
        sv_t: tl.constexpr,
        sv_h: tl.constexpr,
        # strides for g/beta [B, T, HV]
        sg_b: tl.constexpr,
        sg_t: tl.constexpr,
        sg_h: tl.constexpr,
        # strides for h0   [B, HV, K, V]  (also used for inner dims of ht)
        sh_b: tl.constexpr,
        sh_h: tl.constexpr,
        sh_k: tl.constexpr,
        # stride for ht outer time dimension  [T+1, ...]
        sht_t: tl.constexpr,
        # strides for o    [B, T, HV, V]
        so_b: tl.constexpr,
        so_t: tl.constexpr,
        so_h: tl.constexpr,
        # flags
        USE_INITIAL_STATE: tl.constexpr,
        USE_QK_L2NORM: tl.constexpr,
    ):
        """
        Grid: (NV, B * HV)
          i_v  = program_id(0)  → which BV-slice of the V dimension
          i_bh = program_id(1)  → flattened (batch, head) index
        """
        i_v = tl.program_id(0)
        i_bh = tl.program_id(1)
        i_b = i_bh // HV
        i_h = i_bh % HV

        # Coordinate vectors for K and V tiles.
        # BK covers all of K (NK==1 asserted in wrapper); BV is a V-dimension tile.
        o_k = tl.arange(0, BK)  # [BK]
        o_v = i_v * BV + tl.arange(0, BV)  # [BV]  absolute indices

        mask_k = o_k < K
        mask_v = o_v < V
        mask_h = mask_v[:, None] & mask_k[None, :]  # [BV, BK]

        # ---------------------------------------------------------------
        # Base pointers to the first token of this (batch, head) slice.
        # All tensors are contiguous (enforced in Python wrapper).
        # q/k: [B, T, HV, K] → element [b, t, h, k] at b*sq_b + t*sq_t + h*sq_h + k
        # ---------------------------------------------------------------
        p_q = q_ptr + i_b * sq_b + i_h * sq_h + o_k  # advances sq_t per token
        p_k = k_ptr + i_b * sq_b + i_h * sq_h + o_k
        p_v = v_ptr + i_b * sv_b + i_h * sv_h + o_v  # advances sv_t per token
        p_g = g_ptr + i_b * sg_b + i_h * sg_h  # scalar; advances sg_t per token
        p_beta = beta_ptr + i_b * sg_b + i_h * sg_h
        p_o = o_ptr + i_b * so_b + i_h * so_h + o_v  # advances so_t per token

        # ---------------------------------------------------------------
        # Initialise recurrent state b_h[BV, BK] = state[b, h, BK_range, BV_range]
        # Note: b_h[bv, bk] ↔ state element at (K=bk, V=bv).
        # Memory layout of state tensor [*, B, HV, K, V]:
        #   element [b, h, k, v] at b*sh_b + h*sh_h + k*sh_k + v*1
        # So b_h[bv, bk] lives at: b*sh_b + h*sh_h + bk*sh_k + bv
        # ---------------------------------------------------------------
        b_h = tl.zeros([BV, BK], dtype=tl.float32)

        if USE_INITIAL_STATE:
            # h0: [B, HV, K, V]  element [b, h, k, v] at b*sh_b + h*sh_h + k*sh_k + v
            p_h0 = (
                h0_ptr
                + i_b * sh_b
                + i_h * sh_h
                + o_k[None, :] * sh_k  # [1, BK] → selects K dimension
                + o_v[:, None]
            )  # [BV, 1] → selects V dimension (stride=1)
            b_h = tl.load(p_h0, mask=mask_h, other=0.0).to(tl.float32)

        # Write initial state to ht[0]: state BEFORE any speculative token.
        p_ht = ht_ptr + 0 * sht_t + i_b * sh_b + i_h * sh_h + o_k[None, :] * sh_k + o_v[:, None]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

        # ---------------------------------------------------------------
        # Main recurrent loop over T candidate tokens.
        # ---------------------------------------------------------------
        for i_t in range(T):
            b_q = tl.load(p_q, mask=mask_k, other=0.0).to(tl.float32)  # [BK]
            b_k = tl.load(p_k, mask=mask_k, other=0.0).to(tl.float32)  # [BK]
            b_v = tl.load(p_v, mask=mask_v, other=0.0).to(tl.float32)  # [BV]
            b_g = tl.load(p_g).to(tl.float32)  # scalar
            b_beta = tl.load(p_beta).to(tl.float32)  # scalar

            # Optional L2-normalise query and key (use_qk_l2norm_in_kernel=True path).
            if USE_QK_L2NORM:
                b_q = b_q * tl.rsqrt(tl.sum(b_q * b_q) + 1e-6)
                b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)

            b_q = b_q * scale

            # GDN recurrent update (matches torch_recurrent_gated_delta_rule):
            #   1. Decay:      h     ← h * exp(g)               [BV, BK]
            #   2. Retrieval:  kv    ← sum_K h[V,K] * k[K]     → [BV]
            #   3. Residual:   delta ← (v - kv) * beta          → [BV]
            #   4. Write:      h     ← h + outer(delta, k)      [BV, BK]
            #   5. Readout:    o     ← sum_K h[V,K] * q[K]     → [BV]
            b_h = b_h * tl.exp(b_g)
            b_kv = tl.sum(b_h * b_k[None, :], axis=1)  # [BV]
            b_dv = (b_v - b_kv) * b_beta  # [BV]
            b_h = b_h + b_dv[:, None] * b_k[None, :]  # [BV, BK]
            b_o = tl.sum(b_h * b_q[None, :], axis=1)  # [BV]

            tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

            # Write state AFTER token i_t to ht[i_t + 1].
            p_ht_cur = ht_ptr + (i_t + 1) * sht_t + i_b * sh_b + i_h * sh_h + o_k[None, :] * sh_k + o_v[:, None]
            tl.store(p_ht_cur, b_h.to(p_ht_cur.dtype.element_ty), mask=mask_h)

            # Advance per-token pointers.
            p_q += sq_t
            p_k += sq_t
            p_v += sv_t
            p_g += sg_t
            p_beta += sg_t
            p_o += so_t

    def speculative_recurrent_gdn(
        q,
        k,
        v,
        g,
        beta,
        initial_state=None,
        scale=None,
        use_qk_l2norm_in_kernel=True,
    ):
        """
        Single-pass speculative-decoding recurrence for a GDN layer.

        Processes all T candidate tokens in one Triton kernel call and emits:
          o          [B, T, HV, V]          per-token output activations
          all_states [T+1, B, HV, K, V]     recurrent state at every boundary:
                     all_states[0]   = initial_state (before any token)
                     all_states[t+1] = state after token t

        After verification the caller selects ``all_states[k]`` as the corrected
        recurrent state for ``k`` accepted tokens, with zero extra model forward
        passes.

        Args:
            q, k  : [B, T, HV, K]  query / key (already repeat_interleave'd to HV heads)
            v     : [B, T, HV, V]  value
            g     : [B, T, HV]     log-decay = -A_log.exp() * softplus(a + dt_bias)
            beta  : [B, T, HV]     = sigmoid(b)
            initial_state : [B, HV, K, V] or None
            scale : float  (default: K ** -0.5)
            use_qk_l2norm_in_kernel : bool
        """
        B, T, HV, K = q.shape
        V = v.shape[-1]

        if scale is None:
            scale = K**-0.5

        BK = triton.next_power_of_2(K)
        BV = min(triton.next_power_of_2(V), 32)
        NV = triton.cdiv(V, BV)
        assert triton.cdiv(K, BK) == 1, f"K={K} must fit in one BK tile (BK={BK}). NK>1 is not supported."

        # Make inputs contiguous in float32 for g and beta; keep q/k/v dtype.
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        g = g.contiguous().to(torch.float32)
        beta = beta.contiguous().to(torch.float32)

        # Allocate outputs.
        o = torch.empty(B, T, HV, V, dtype=q.dtype, device=q.device)
        all_states = torch.empty(T + 1, B, HV, K, V, dtype=torch.float32, device=q.device)

        h0 = None
        if initial_state is not None:
            h0 = initial_state.contiguous().to(torch.float32)

        # Strides for q/k [B, T, HV, K]  (K fast axis)
        sq_b, sq_t, sq_h, _ = q.stride()
        # Strides for v   [B, T, HV, V]  (V fast axis)
        sv_b, sv_t, sv_h, _ = v.stride()
        # Strides for g   [B, T, HV]     (HV fast axis)
        sg_b, sg_t, sg_h = g.stride()
        # Strides for h0 / inner dims of ht  [*, B, HV, K, V]  (V fast axis)
        sh_b, sh_h, sh_k, _ = all_states.stride()[1:]  # skip the T+1 outer dim
        sht_t = all_states.stride(0)  # stride for the T+1 outer dim
        # Strides for o   [B, T, HV, V]  (V fast axis)
        so_b, so_t, so_h, _ = o.stride()

        grid = (NV, B * HV)
        _spec_recurrent_gdn_fwd_kernel[grid](
            q_ptr=q,
            k_ptr=k,
            v_ptr=v,
            g_ptr=g,
            beta_ptr=beta,
            h0_ptr=h0 if h0 is not None else q,  # unused when USE_INITIAL_STATE=False
            ht_ptr=all_states,
            o_ptr=o,
            T=T,
            scale=scale,
            B=B,
            HV=HV,
            K=K,
            V=V,
            BK=BK,
            BV=BV,
            sq_b=sq_b,
            sq_t=sq_t,
            sq_h=sq_h,
            sv_b=sv_b,
            sv_t=sv_t,
            sv_h=sv_h,
            sg_b=sg_b,
            sg_t=sg_t,
            sg_h=sg_h,
            sh_b=sh_b,
            sh_h=sh_h,
            sh_k=sh_k,
            sht_t=sht_t,
            so_b=so_b,
            so_t=so_t,
            so_h=so_h,
            USE_INITIAL_STATE=(h0 is not None),
            USE_QK_L2NORM=use_qk_l2norm_in_kernel,
            num_warps=4,
            num_stages=3,
        )
        return o, all_states

    _spec_decode_kernel_available = True

except ImportError:
    _spec_decode_kernel_available = False

    def speculative_recurrent_gdn(q, k, v, g, beta, initial_state=None, scale=None, use_qk_l2norm_in_kernel=True):
        """
        Pure-PyTorch fallback: N sequential recurrent calls, each capturing one state.
        Used when Triton is not available.  Produces the same output as the Triton
        kernel but with N kernel launches instead of one.
        """
        B, T, HV, K = q.shape
        V = v.shape[-1]
        dtype = q.dtype

        if scale is None:
            scale = K**-0.5

        def l2norm(x):
            return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)

        state = (
            initial_state.to(torch.float32).clone()
            if initial_state is not None
            else torch.zeros(B, HV, K, V, dtype=torch.float32, device=q.device)
        )
        all_states = torch.empty(T + 1, B, HV, K, V, dtype=torch.float32, device=q.device)
        all_states[0] = state

        outputs = []
        for t in range(T):
            q_t = q[:, t].to(torch.float32)  # [B, HV, K]
            k_t = k[:, t].to(torch.float32)
            v_t = v[:, t].to(torch.float32)
            g_t = g[:, t].to(torch.float32)  # [B, HV]
            beta_t = beta[:, t].to(torch.float32)  # [B, HV]

            if use_qk_l2norm_in_kernel:
                q_t = l2norm(q_t)
                k_t = l2norm(k_t)

            q_t = q_t * scale

            # GDN recurrent update
            state = state * g_t.exp()[:, :, None, None]
            kv_mem = (state * k_t[:, :, :, None]).sum(dim=2)  # [B, HV, V]
            delta = (v_t - kv_mem) * beta_t[:, :, None]  # [B, HV, V]
            state = state + k_t[:, :, :, None] * delta[:, :, None, :]  # [B, HV, K, V]
            o_t = (state * q_t[:, :, :, None]).sum(dim=2)  # [B, HV, V]

            all_states[t + 1] = state
            outputs.append(o_t.to(dtype).unsqueeze(1))

        o = torch.cat(outputs, dim=1)  # [B, T, HV, V]
        return o, all_states
