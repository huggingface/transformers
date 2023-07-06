"""
Copied from
https://github.com/HazyResearch/flash-attention/blob/eff9fe6b8076df59d64d7a3f464696738a3c7c24/flash_attn/flash_attn_triton.py
update imports to use 'triton_pre_mlir' *Experimental* implementation of FlashAttention in Triton. Tested with
triton==2.0.0.dev20221202. Triton 2.0 has a new backend (MLIR) but seems like it doesn't yet work for head dimensions
other than 64:
https://github.com/openai/triton/blob/d376020f90002757eea3ea9475d4f7cfc2ec5ead/python/triton/ops/flash_attention.py#L207
We'll update this implementation with the new Triton backend once this is fixed. We use the FlashAttention
implementation from Phil Tillet a starting point.
https://github.com/openai/triton/blob/master/python/tutorials/06-fused-attention.py Changes:
- Implement both causal and non-causal attention.
- Implement both self-attention and cross-attention.
- Support arbitrary seqlens (not just multiples of 128), for both forward and backward.
- Support all head dimensions up to 128 (not just 16, 32, 64, 128), for both forward and backward.
- Support attention bias.
- Speed up the forward pass a bit, and only store the LSE instead of m and l.
- Make the backward for d=128 much faster by reducing register spilling.
- Optionally parallelize the backward pass across seqlen_k, to deal with the case of
small batch size * nheads. Caution:
- This is an *experimental* implementation. The forward pass should be quite robust but
I'm not 100% sure that the backward pass doesn't have race conditions (due to the Triton compiler).
- This implementation has only been tested on A100.
- If you plan to use headdim other than 64 and 128, you should test for race conditions
(due to the Triton compiler), as done in tests/test_flash_attn.py "test_flash_attn_triton_race_condition". I've tested
and fixed many race conditions for different head dimensions (40, 48, 64, 128, 80, 88, 96), but I'm still not 100%
confident that there are none left for other head dimensions. Differences between this Triton version and the CUDA
version:
- Triton version doesn't support dropout.
- Triton forward is generally faster than CUDA forward, while Triton backward is
generally slower than CUDA backward. Overall Triton forward + backward is slightly slower than CUDA forward + backward.
- Triton version doesn't support different sequence lengths in a batch (i.e., RaggedTensor/NestedTensor).
- Triton version supports attention bias, while CUDA version doesn't.
"""
import math
import torch

# TODO: can we change it with triton?
import triton_pre_mlir as triton
import triton_pre_mlir.language as tl

@triton.heuristics({'EVEN_M': lambda args: args['seqlen_q'] % args['BLOCK_M'] == 0, 'EVEN_N': lambda args: args['seqlen_k'] % args['BLOCK_N'] == 0, 'EVEN_HEADDIM': lambda args: args['headdim'] == args['BLOCK_HEADDIM']})
@triton.jit
def _fwd_kernel(Q, K, V, Bias, Out, Lse, TMP, softmax_scale, stride_qb, stride_qh, stride_qm, stride_kb, stride_kh, stride_kn, stride_vb, stride_vh, stride_vn, stride_bb, stride_bh, stride_bm, stride_ob, stride_oh, stride_om, nheads, seqlen_q, seqlen_k, seqlen_q_rounded, headdim, CACHE_KEY_SEQLEN_Q, CACHE_KEY_SEQLEN_K, BIAS_TYPE: tl.constexpr, IS_CAUSAL: tl.constexpr, BLOCK_HEADDIM: tl.constexpr, EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_HEADDIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])
    if BIAS_TYPE == 'vector':
        b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + offs_n
    elif BIAS_TYPE == 'matrix':
        b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + (offs_m[:, None] * stride_bm + offs_n[None, :])
    t_ptrs = TMP + off_hb * seqlen_q_rounded + offs_m
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    if EVEN_M & EVEN_N:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    elif EVEN_HEADDIM:
        q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
    else:
        q = tl.load(q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        if EVEN_N & EVEN_M:
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_d[None, :] < headdim, other=0.0)
        elif EVEN_HEADDIM:
            k = tl.load(k_ptrs + start_n * stride_kn, mask=(start_n + offs_n)[:, None] < seqlen_k, other=0.0)
        else:
            k = tl.load(k_ptrs + start_n * stride_kn, mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k, trans_b=True)
        if not EVEN_N:
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float('-inf'))
        if IS_CAUSAL:
            qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float('-inf'))
        if BIAS_TYPE != 'none':
            if BIAS_TYPE == 'vector':
                if EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(b_ptrs + start_n, mask=start_n + offs_n < seqlen_k, other=0.0).to(tl.float32)
                bias = bias[None, :]
            elif BIAS_TYPE == 'matrix':
                if EVEN_M & EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(b_ptrs + start_n, mask=(offs_m[:, None] < seqlen_q) & ((start_n + offs_n)[None, :] < seqlen_k), other=0.0).to(tl.float32)
            qk = qk * softmax_scale + bias
            m_ij = tl.maximum(tl.max(qk, 1), lse_i)
            p = tl.exp(qk - m_ij[:, None])
        else:
            m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
            p = tl.exp(qk * softmax_scale - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        acc_o_scale = tl.exp(m_i - m_ij)
        tl.store(t_ptrs, acc_o_scale)
        acc_o_scale = tl.load(t_ptrs)
        acc_o = acc_o * acc_o_scale[:, None]
        if EVEN_N & EVEN_M:
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_vn)
            else:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_d[None, :] < headdim, other=0.0)
        elif EVEN_HEADDIM:
            v = tl.load(v_ptrs + start_n * stride_vn, mask=(start_n + offs_n)[:, None] < seqlen_k, other=0.0)
        else:
            v = tl.load(v_ptrs + start_n * stride_vn, mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)
        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)
    o_scale = tl.exp(m_i - lse_i)
    tl.store(t_ptrs, o_scale)
    o_scale = tl.load(t_ptrs)
    acc_o = acc_o * o_scale[:, None]
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m
    tl.store(lse_ptrs, lse_i)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = Out + off_b * stride_ob + off_h * stride_oh + (offs_m[:, None] * stride_om + offs_d[None, :])
    if EVEN_M:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o)
        else:
            tl.store(out_ptrs, acc_o, mask=offs_d[None, :] < headdim)
    elif EVEN_HEADDIM:
        tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < seqlen_q)
    else:
        tl.store(out_ptrs, acc_o, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim))

@triton.jit
def _bwd_preprocess_do_o_dot(Out, DO, Delta, stride_ob, stride_oh, stride_om, stride_dob, stride_doh, stride_dom, nheads, seqlen_q, seqlen_q_rounded, headdim, BLOCK_M: tl.constexpr, BLOCK_HEADDIM: tl.constexpr):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    o = tl.load(Out + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :], mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0).to(tl.float32)
    do = tl.load(DO + off_b * stride_dob + off_h * stride_doh + offs_m[:, None] * stride_dom + offs_d[None, :], mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + off_hb * seqlen_q_rounded + offs_m, delta)

@triton.jit
def _bwd_store_dk_dv(dk_ptrs, dv_ptrs, dk, dv, offs_n, offs_d, seqlen_k, headdim, EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_HEADDIM: tl.constexpr):
    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            tl.store(dv_ptrs, dv)
            tl.store(dk_ptrs, dk)
        else:
            tl.store(dv_ptrs, dv, mask=offs_d[None, :] < headdim)
            tl.store(dk_ptrs, dk, mask=offs_d[None, :] < headdim)
    elif EVEN_HEADDIM:
        tl.store(dv_ptrs, dv, mask=offs_n[:, None] < seqlen_k)
        tl.store(dk_ptrs, dk, mask=offs_n[:, None] < seqlen_k)
    else:
        tl.store(dv_ptrs, dv, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))
        tl.store(dk_ptrs, dk, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))

@triton.jit
def _bwd_kernel_one_col_block(start_n, Q, K, V, Bias, DO, DQ, DK, DV, LSE, D, softmax_scale, stride_qm, stride_kn, stride_vn, stride_bm, stride_dom, stride_dqm, stride_dkn, stride_dvn, seqlen_q, seqlen_k, headdim, ATOMIC_ADD: tl.constexpr, BIAS_TYPE: tl.constexpr, IS_CAUSAL: tl.constexpr, BLOCK_HEADDIM: tl.constexpr, EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_HEADDIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    begin_m = 0 if not IS_CAUSAL else start_n * BLOCK_N // BLOCK_M * BLOCK_M
    offs_qm = begin_m + tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    q_ptrs = Q + (offs_qm[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_d[None, :])
    do_ptrs = DO + (offs_qm[:, None] * stride_dom + offs_d[None, :])
    dq_ptrs = DQ + (offs_qm[:, None] * stride_dqm + offs_d[None, :])
    if BIAS_TYPE == 'vector':
        b_ptrs = Bias + offs_n
    elif BIAS_TYPE == 'matrix':
        b_ptrs = Bias + (offs_qm[:, None] * stride_bm + offs_n[None, :])
    dv = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    if begin_m >= seqlen_q:
        dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d[None, :])
        dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])
        _bwd_store_dk_dv(dk_ptrs, dv_ptrs, dk, dv, offs_n, offs_d, seqlen_k, headdim, EVEN_M=EVEN_M, EVEN_N=EVEN_N, EVEN_HEADDIM=EVEN_HEADDIM)
        return
    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
        else:
            k = tl.load(k_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
            v = tl.load(v_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    elif EVEN_HEADDIM:
        k = tl.load(k_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
        v = tl.load(v_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
    else:
        k = tl.load(k_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
        v = tl.load(v_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
    num_block_m = tl.cdiv(seqlen_q, BLOCK_M)
    for start_m in range(begin_m, num_block_m * BLOCK_M, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m_curr = start_m + offs_m
        if EVEN_M & EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        elif EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0)
        else:
            q = tl.load(q_ptrs, mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)
        qk = tl.dot(q, k, trans_b=True)
        if not EVEN_N:
            qk = tl.where(offs_n[None, :] < seqlen_k, qk, float('-inf'))
        if IS_CAUSAL:
            qk = tl.where(offs_m_curr[:, None] >= offs_n[None, :], qk, float('-inf'))
        if BIAS_TYPE != 'none':
            tl.debug_barrier()
            if BIAS_TYPE == 'vector':
                if EVEN_N:
                    bias = tl.load(b_ptrs).to(tl.float32)
                else:
                    bias = tl.load(b_ptrs, mask=offs_n < seqlen_k, other=0.0).to(tl.float32)
                bias = bias[None, :]
            elif BIAS_TYPE == 'matrix':
                if EVEN_M & EVEN_N:
                    bias = tl.load(b_ptrs).to(tl.float32)
                else:
                    bias = tl.load(b_ptrs, mask=(offs_m_curr[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k), other=0.0).to(tl.float32)
            qk = qk * softmax_scale + bias
        if not EVEN_M & EVEN_HEADDIM:
            tl.debug_barrier()
        lse_i = tl.load(LSE + offs_m_curr)
        if BIAS_TYPE == 'none':
            p = tl.exp(qk * softmax_scale - lse_i[:, None])
        else:
            p = tl.exp(qk - lse_i[:, None])
        if EVEN_M & EVEN_HEADDIM:
            do = tl.load(do_ptrs)
        else:
            do = tl.load(do_ptrs, mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)
        dv += tl.dot(p.to(do.dtype), do, trans_a=True)
        if not EVEN_M & EVEN_HEADDIM:
            tl.debug_barrier()
        dp = tl.dot(do, v, trans_b=True)
        if not EVEN_HEADDIM:
            tl.debug_barrier()
        Di = tl.load(D + offs_m_curr)
        ds = (p * (dp - Di[:, None]) * softmax_scale).to(q.dtype)
        dk += tl.dot(ds, q, trans_a=True)
        if not EVEN_M & EVEN_HEADDIM:
            tl.debug_barrier()
        if not ATOMIC_ADD:
            if EVEN_M & EVEN_HEADDIM:
                dq = tl.load(dq_ptrs, eviction_policy='evict_last')
                dq += tl.dot(ds, k)
                tl.store(dq_ptrs, dq, eviction_policy='evict_last')
            elif EVEN_HEADDIM:
                dq = tl.load(dq_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0, eviction_policy='evict_last')
                dq += tl.dot(ds, k)
                tl.store(dq_ptrs, dq, mask=offs_m_curr[:, None] < seqlen_q, eviction_policy='evict_last')
            else:
                dq = tl.load(dq_ptrs, mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0, eviction_policy='evict_last')
                dq += tl.dot(ds, k)
                tl.store(dq_ptrs, dq, mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim), eviction_policy='evict_last')
        else:
            dq = tl.dot(ds, k)
            if EVEN_M & EVEN_HEADDIM:
                tl.atomic_add(dq_ptrs, dq)
            elif EVEN_HEADDIM:
                tl.atomic_add(dq_ptrs, dq, mask=offs_m_curr[:, None] < seqlen_q)
            else:
                tl.atomic_add(dq_ptrs, dq, mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim))
        dq_ptrs += BLOCK_M * stride_dqm
        q_ptrs += BLOCK_M * stride_qm
        do_ptrs += BLOCK_M * stride_dom
        if BIAS_TYPE == 'matrix':
            b_ptrs += BLOCK_M * stride_bm
    dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d[None, :])
    dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])
    _bwd_store_dk_dv(dk_ptrs, dv_ptrs, dk, dv, offs_n, offs_d, seqlen_k, headdim, EVEN_M=EVEN_M, EVEN_N=EVEN_N, EVEN_HEADDIM=EVEN_HEADDIM)

def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()

@triton.autotune(configs=[triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'SEQUENCE_PARALLEL': False}, num_warps=8, num_stages=1, pre_hook=init_to_zero('DQ')), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'SEQUENCE_PARALLEL': True}, num_warps=8, num_stages=1, pre_hook=init_to_zero('DQ'))], key=['CACHE_KEY_SEQLEN_Q', 'CACHE_KEY_SEQLEN_K', 'BIAS_TYPE', 'IS_CAUSAL', 'BLOCK_HEADDIM'])
@triton.heuristics({'EVEN_M': lambda args: args['seqlen_q'] % args['BLOCK_M'] == 0, 'EVEN_N': lambda args: args['seqlen_k'] % args['BLOCK_N'] == 0, 'EVEN_HEADDIM': lambda args: args['headdim'] == args['BLOCK_HEADDIM']})
@triton.jit
def _bwd_kernel(Q, K, V, Bias, DO, DQ, DK, DV, LSE, D, softmax_scale, stride_qb, stride_qh, stride_qm, stride_kb, stride_kh, stride_kn, stride_vb, stride_vh, stride_vn, stride_bb, stride_bh, stride_bm, stride_dob, stride_doh, stride_dom, stride_dqb, stride_dqh, stride_dqm, stride_dkb, stride_dkh, stride_dkn, stride_dvb, stride_dvh, stride_dvn, nheads, seqlen_q, seqlen_k, seqlen_q_rounded, headdim, CACHE_KEY_SEQLEN_Q, CACHE_KEY_SEQLEN_K, BIAS_TYPE: tl.constexpr, IS_CAUSAL: tl.constexpr, BLOCK_HEADDIM: tl.constexpr, SEQUENCE_PARALLEL: tl.constexpr, EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_HEADDIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    Q += off_b * stride_qb + off_h * stride_qh
    K += off_b * stride_kb + off_h * stride_kh
    V += off_b * stride_vb + off_h * stride_vh
    DO += off_b * stride_dob + off_h * stride_doh
    DQ += off_b * stride_dqb + off_h * stride_dqh
    DK += off_b * stride_dkb + off_h * stride_dkh
    DV += off_b * stride_dvb + off_h * stride_dvh
    if BIAS_TYPE != 'none':
        Bias += off_b * stride_bb + off_h * stride_bh
    D += off_hb * seqlen_q_rounded
    LSE += off_hb * seqlen_q_rounded
    if not SEQUENCE_PARALLEL:
        num_block_n = tl.cdiv(seqlen_k, BLOCK_N)
        for start_n in range(0, num_block_n):
            _bwd_kernel_one_col_block(start_n, Q, K, V, Bias, DO, DQ, DK, DV, LSE, D, softmax_scale, stride_qm, stride_kn, stride_vn, stride_bm, stride_dom, stride_dqm, stride_dkn, stride_dvn, seqlen_q, seqlen_k, headdim, ATOMIC_ADD=False, BIAS_TYPE=BIAS_TYPE, IS_CAUSAL=IS_CAUSAL, BLOCK_HEADDIM=BLOCK_HEADDIM, EVEN_M=EVEN_M, EVEN_N=EVEN_N, EVEN_HEADDIM=EVEN_HEADDIM, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    else:
        start_n = tl.program_id(0)
        _bwd_kernel_one_col_block(start_n, Q, K, V, Bias, DO, DQ, DK, DV, LSE, D, softmax_scale, stride_qm, stride_kn, stride_vn, stride_bm, stride_dom, stride_dqm, stride_dkn, stride_dvn, seqlen_q, seqlen_k, headdim, ATOMIC_ADD=True, BIAS_TYPE=BIAS_TYPE, IS_CAUSAL=IS_CAUSAL, BLOCK_HEADDIM=BLOCK_HEADDIM, EVEN_M=EVEN_M, EVEN_N=EVEN_N, EVEN_HEADDIM=EVEN_HEADDIM, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)

def _flash_attn_forward(q, k, v, bias=None, causal=False, softmax_scale=None):
    (batch, seqlen_q, nheads, d) = q.shape
    (_, seqlen_k, _, _) = k.shape
    assert k.shape == (batch, seqlen_k, nheads, d)
    assert v.shape == (batch, seqlen_k, nheads, d)
    assert d <= 128, 'FlashAttention only support head dimensions up to 128'
    assert q.dtype == k.dtype == v.dtype, 'All tensors must have the same type'
    assert q.dtype in [torch.float16, torch.bfloat16], 'Only support fp16 and bf16'
    assert q.is_cuda and k.is_cuda and v.is_cuda
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)
    has_bias = bias is not None
    bias_type = 'none'
    if has_bias:
        assert bias.dtype in [q.dtype, torch.float]
        assert bias.is_cuda
        assert bias.dim() == 4
        if bias.stride(-1) != 1:
            bias = bias.contiguous()
        if bias.shape[2:] == (1, seqlen_k):
            bias_type = 'vector'
        elif bias.shape[2:] == (seqlen_q, seqlen_k):
            bias_type = 'matrix'
        else:
            raise RuntimeError('Last 2 dimensions of bias must be (1, seqlen_k) or (seqlen_q, seqlen_k)')
        bias = bias.expand(batch, nheads, seqlen_q, seqlen_k)
    bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2)) if has_bias else (0, 0, 0)
    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    lse = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    tmp = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    o = torch.empty_like(q)
    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    BLOCK = 128
    num_warps = 4 if d <= 64 else 8
    grid = lambda META: (triton.cdiv(seqlen_q, META['BLOCK_M']), batch * nheads)
    _fwd_kernel[grid](q, k, v, bias, o, lse, tmp, softmax_scale, q.stride(0), q.stride(2), q.stride(1), k.stride(0), k.stride(2), k.stride(1), v.stride(0), v.stride(2), v.stride(1), *bias_strides, o.stride(0), o.stride(2), o.stride(1), nheads, seqlen_q, seqlen_k, seqlen_q_rounded, d, seqlen_q // 32, seqlen_k // 32, bias_type, causal, BLOCK_HEADDIM, BLOCK_M=BLOCK, BLOCK_N=BLOCK, num_warps=num_warps, num_stages=1)
    return (o, lse, softmax_scale)

def _flash_attn_backward(do, q, k, v, o, lse, dq, dk, dv, bias=None, causal=False, softmax_scale=None):
    if do.stride(-1) != 1:
        do = do.contiguous()
    (batch, seqlen_q, nheads, d) = q.shape
    (_, seqlen_k, _, _) = k.shape
    assert d <= 128
    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    assert lse.shape == (batch, nheads, seqlen_q_rounded)
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == o.stride(-1) == 1
    assert dq.stride(-1) == dk.stride(-1) == dv.stride(-1) == 1
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)
    dq_accum = torch.empty_like(q, dtype=torch.float32)
    delta = torch.empty_like(lse)
    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    grid = lambda META: (triton.cdiv(seqlen_q, META['BLOCK_M']), batch * nheads)
    _bwd_preprocess_do_o_dot[grid](o, do, delta, o.stride(0), o.stride(2), o.stride(1), do.stride(0), do.stride(2), do.stride(1), nheads, seqlen_q, seqlen_q_rounded, d, BLOCK_M=128, BLOCK_HEADDIM=BLOCK_HEADDIM)
    has_bias = bias is not None
    bias_type = 'none'
    if has_bias:
        assert bias.dtype in [q.dtype, torch.float]
        assert bias.is_cuda
        assert bias.dim() == 4
        assert bias.stride(-1) == 1
        if bias.shape[2:] == (1, seqlen_k):
            bias_type = 'vector'
        elif bias.shape[2:] == (seqlen_q, seqlen_k):
            bias_type = 'matrix'
        else:
            raise RuntimeError('Last 2 dimensions of bias must be (1, seqlen_k) or (seqlen_q, seqlen_k)')
        bias = bias.expand(batch, nheads, seqlen_q, seqlen_k)
    bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2)) if has_bias else (0, 0, 0)
    grid = lambda META: (triton.cdiv(seqlen_k, META['BLOCK_N']) if META['SEQUENCE_PARALLEL'] else 1, batch * nheads)
    _bwd_kernel[grid](q, k, v, bias, do, dq_accum, dk, dv, lse, delta, softmax_scale, q.stride(0), q.stride(2), q.stride(1), k.stride(0), k.stride(2), k.stride(1), v.stride(0), v.stride(2), v.stride(1), *bias_strides, do.stride(0), do.stride(2), do.stride(1), dq_accum.stride(0), dq_accum.stride(2), dq_accum.stride(1), dk.stride(0), dk.stride(2), dk.stride(1), dv.stride(0), dv.stride(2), dv.stride(1), nheads, seqlen_q, seqlen_k, seqlen_q_rounded, d, seqlen_q // 32, seqlen_k // 32, bias_type, causal, BLOCK_HEADDIM)
    dq.copy_(dq_accum)

class FlashAttnQKVPackedFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, qkv, bias=None, causal=False, softmax_scale=None):
        """
            qkv: (batch, seqlen, 3, nheads, headdim) bias: optional, shape broadcastible to (batch, nheads, seqlen,
            seqlen).
                For example, ALiBi mask for causal would have shape (1, nheads, 1, seqlen). ALiBi mask for non-causal
                would have shape (1, nheads, seqlen, seqlen)
        """
        if qkv.stride(-1) != 1:
            qkv = qkv.contiguous()
        (o, lse, ctx.softmax_scale) = _flash_attn_forward(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2], bias=bias, causal=causal, softmax_scale=softmax_scale)
        ctx.save_for_backward(qkv, o, lse, bias)
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        (qkv, o, lse, bias) = ctx.saved_tensors
        assert not ctx.needs_input_grad[1], 'FlashAttention does not support bias gradient yet'
        with torch.inference_mode():
            dqkv = torch.empty_like(qkv)
            _flash_attn_backward(do, qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2], o, lse, dqkv[:, :, 0], dqkv[:, :, 1], dqkv[:, :, 2], bias=bias, causal=ctx.causal, softmax_scale=ctx.softmax_scale)
        return (dqkv, None, None, None)
flash_attn_qkvpacked_func = FlashAttnQKVPackedFunc.apply

class FlashAttnKVPackedFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, kv, bias=None, causal=False, softmax_scale=None):
        """
            q: (batch, seqlen_q, nheads, headdim) kv: (batch, seqlen_k, 2, nheads, headdim) bias: optional, shape
            broadcastible to (batch, nheads, seqlen_q, seqlen_k).
                For example, ALiBi mask for causal would have shape (1, nheads, 1, seqlen_k). ALiBi mask for non-causal
                would have shape (1, nheads, seqlen_q, seqlen_k)
        """
        (q, kv) = [x if x.stride(-1) == 1 else x.contiguous() for x in [q, kv]]
        (o, lse, ctx.softmax_scale) = _flash_attn_forward(q, kv[:, :, 0], kv[:, :, 1], bias=bias, causal=causal, softmax_scale=softmax_scale)
        ctx.save_for_backward(q, kv, o, lse, bias)
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        (q, kv, o, lse, bias) = ctx.saved_tensors
        if len(ctx.needs_input_grad) >= 3:
            assert not ctx.needs_input_grad[2], 'FlashAttention does not support bias gradient yet'
        with torch.inference_mode():
            dq = torch.empty_like(q)
            dkv = torch.empty_like(kv)
            _flash_attn_backward(do, q, kv[:, :, 0], kv[:, :, 1], o, lse, dq, dkv[:, :, 0], dkv[:, :, 1], bias=bias, causal=ctx.causal, softmax_scale=ctx.softmax_scale)
        return (dq, dkv, None, None, None)
flash_attn_kvpacked_func = FlashAttnKVPackedFunc.apply

class FlashAttnFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, bias=None, causal=False, softmax_scale=None):
        """
            q: (batch_size, seqlen_q, nheads, headdim) k, v: (batch_size, seqlen_k, nheads, headdim) bias: optional,
            shape broadcastible to (batch, nheads, seqlen_q, seqlen_k).
                For example, ALiBi mask for causal would have shape (1, nheads, 1, seqlen_k). ALiBi mask for non-causal
                would have shape (1, nheads, seqlen_q, seqlen_k)
        """
        (q, k, v) = [x if x.stride(-1) == 1 else x.contiguous() for x in [q, k, v]]
        (o, lse, ctx.softmax_scale) = _flash_attn_forward(q, k, v, bias=bias, causal=causal, softmax_scale=softmax_scale)
        ctx.save_for_backward(q, k, v, o, lse, bias)
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        (q, k, v, o, lse, bias) = ctx.saved_tensors
        assert not ctx.needs_input_grad[3], 'FlashAttention does not support bias gradient yet'
        with torch.inference_mode():
            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            _flash_attn_backward(do, q, k, v, o, lse, dq, dk, dv, bias=bias, causal=ctx.causal, softmax_scale=ctx.softmax_scale)
        return (dq, dk, dv, None, None, None)
flash_attn_func = FlashAttnFunc.apply