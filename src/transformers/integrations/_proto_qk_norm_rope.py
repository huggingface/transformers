# [scratch] Qwen3-style attention input in one kernel: per-head RMSNorm of q and k then rotary (rotate_half layout),
# in place on the output of the fused q/k/v projection.
import torch
import triton
import triton.language as tl


@triton.jit
def _qk_norm_rope_kernel(
    QKV, Cos, Sin, Wq, Wk, eps,
    NUM_Q_HEADS: tl.constexpr, NUM_K_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
):  # fmt: skip
    t = tl.program_id(0)
    stride_t = (NUM_Q_HEADS + 2 * NUM_K_HEADS) * HEAD_DIM
    stride_cs = HEAD_DIM
    half = tl.arange(0, HEAD_DIM // 2)
    cos1 = tl.load(Cos + t * stride_cs + half).to(tl.float32)
    cos2 = tl.load(Cos + t * stride_cs + HEAD_DIM // 2 + half).to(tl.float32)
    sin1 = tl.load(Sin + t * stride_cs + half).to(tl.float32)
    sin2 = tl.load(Sin + t * stride_cs + HEAD_DIM // 2 + half).to(tl.float32)
    for which in tl.static_range(2):
        if which == 0:
            heads = tl.arange(0, NUM_Q_HEADS)
            base = QKV + t * stride_t + heads[:, None] * HEAD_DIM
            w1 = tl.load(Wq + half).to(tl.float32)
            w2 = tl.load(Wq + HEAD_DIM // 2 + half).to(tl.float32)
        else:
            heads = tl.arange(0, NUM_K_HEADS)
            base = QKV + t * stride_t + (NUM_Q_HEADS + heads[:, None]) * HEAD_DIM
            w1 = tl.load(Wk + half).to(tl.float32)
            w2 = tl.load(Wk + HEAD_DIM // 2 + half).to(tl.float32)
        x1 = tl.load(base + half[None, :]).to(tl.float32)
        x2 = tl.load(base + HEAD_DIM // 2 + half[None, :]).to(tl.float32)
        rstd = tl.rsqrt((tl.sum(x1 * x1, 1) + tl.sum(x2 * x2, 1)) / HEAD_DIM + eps)
        y1 = x1 * rstd[:, None] * w1[None, :]
        y2 = x2 * rstd[:, None] * w2[None, :]
        tl.store(base + half[None, :], (y1 * cos1[None, :] - y2 * sin1[None, :]).to(QKV.dtype.element_ty))
        tl.store(
            base + HEAD_DIM // 2 + half[None, :], (y2 * cos2[None, :] + y1 * sin2[None, :]).to(QKV.dtype.element_ty)
        )


def qk_norm_rope_(qkv, cos, sin, q_weight, k_weight, eps, num_q_heads, num_k_heads, head_dim):
    """In place on `qkv` (contiguous [..., tokens, (num_q_heads + 2 * num_k_heads) * head_dim]): q and k heads
    RMS-normalized with their weights, then rotated by `cos`/`sin` ([..., tokens, head_dim])."""
    num_tokens = qkv.numel() // qkv.shape[-1]
    _qk_norm_rope_kernel[(num_tokens,)](
        qkv,
        cos.contiguous(),
        sin.contiguous(),
        q_weight,
        k_weight,
        eps,
        num_q_heads,
        num_k_heads,
        head_dim,
        num_warps=4,
    )
    return qkv


def fuse_qkv_weights(model: torch.nn.Module) -> int:
    """Makes the q/k/v projection weights of the attention layers that support it views of one [q; k; v] weight, so
    their projections can run as one GEMM. Returns the number of layers fused."""
    fused = 0
    for module in model.modules():
        if "_proto_qkv_weight" not in module.__dict__:
            continue
        q, k, v = module.q_proj.weight, module.k_proj.weight, module.v_proj.weight
        weight = torch.cat([q, k, v]).detach()
        nq, nk = q.shape[0], k.shape[0]
        q.data, k.data, v.data = weight[:nq], weight[nq : nq + nk], weight[nq + nk :]
        module._proto_qkv_weight = weight
        fused += 1
    return fused
