# [scratch] One kernel for the softmax top-k router (softmax over the experts, top-k, optional renormalization), in place
# of softmax + topk (radix select and sort) + sum + div + cast.
import os

import torch
import triton
import triton.language as tl


PROTO_ROUTER = os.environ.get("PROTO_ROUTER", "1") == "1"


@triton.jit
def _router_topk_kernel(
    Logits, Weights, Indices, num_tokens,
    NUM_EXPERTS: tl.constexpr, TOP_K: tl.constexpr, NORM: tl.constexpr, BLOCK_T: tl.constexpr,
):  # fmt: skip
    rows = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    cols = tl.arange(0, NUM_EXPERTS)
    ks = tl.arange(0, TOP_K)
    row_mask = rows < num_tokens
    x = tl.load(Logits + rows[:, None] * NUM_EXPERTS + cols[None, :], mask=row_mask[:, None], other=0.0)
    x = x.to(tl.float32)
    p = tl.exp(x - tl.max(x, 1)[:, None])
    p = p / tl.sum(p, 1)[:, None]
    values = tl.zeros([BLOCK_T, TOP_K], dtype=tl.float32)
    indices = tl.zeros([BLOCK_T, TOP_K], dtype=tl.int32)
    for k in tl.static_range(TOP_K):
        best = tl.argmax(p, 1)
        values = tl.where(ks[None, :] == k, tl.max(p, 1)[:, None], values)
        indices = tl.where(ks[None, :] == k, best[:, None], indices)
        p = tl.where(cols[None, :] == best[:, None], -1.0, p)
    if NORM:
        values = values / tl.sum(values, 1)[:, None]
    out = rows[:, None] * TOP_K + ks[None, :]
    tl.store(Weights + out, values.to(Weights.dtype.element_ty), mask=row_mask[:, None])
    tl.store(Indices + out, indices.to(tl.int64), mask=row_mask[:, None])


def router_topk(router_logits: torch.Tensor, top_k: int, norm_topk_prob: bool):
    """(weights in the logits' dtype, int64 indices) of the softmax top-k router, like
    softmax(fp32) -> topk -> optional renormalization -> cast."""
    num_tokens, num_experts = router_logits.shape
    weights = torch.empty(num_tokens, top_k, device=router_logits.device, dtype=router_logits.dtype)
    indices = torch.empty(num_tokens, top_k, device=router_logits.device, dtype=torch.int64)
    block_t = 4
    _router_topk_kernel[(triton.cdiv(num_tokens, block_t),)](
        router_logits.contiguous(),
        weights,
        indices,
        num_tokens,
        num_experts,
        top_k,
        norm_topk_prob,
        block_t,
        num_warps=1,
    )
    return weights, indices
