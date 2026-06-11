import torch
import torch.nn.functional as F

from ..utils import logging
from .sdpa_attention import sdpa_attention_forward


logger = logging.get_logger(__name__)

# `sparse_atten_func` only compiles for these per-query block size.
MSA_SUPPORTED_TOPK = (4, 8, 16, 32)

# Single-token decode reuses the bf16 varlen CSR kernel, but its fixed launch/schedule
# cost (~0.14 ms) only beats dense SDPA once enough KV blocks can be skipped. Below this
# many key blocks the selection isn't sparse enough to pay for the kernel, so we keep the
# SDPA fallback. Empirical SM100 crossover (Hq=64/Hkv=4, block=128, topk=16) is ~2.25x topk;
# 3x leaves a safe margin where the kernel is already >1.3x faster and the win grows with S.
MSA_DECODE_MIN_BLOCKS_PER_TOPK = 3

_MSA_KERNELS: dict[str, object] = {}


def _round_topk(num_selected: int) -> int | None:
    return next((t for t in MSA_SUPPORTED_TOPK if t >= num_selected), None)


def _get_msa_kernel(attn_implementation: str):
    """Lazily load (and cache) the MSA kernel from an ``attn_implementation`` string.

    The string may carry a ``paged|`` prefix and/or an ``@<revision>`` pin (e.g.
    ``kernels-staging/msa@v0``); the build currently lives on the repo's ``v0`` branch.
    """
    repo_id = attn_implementation.split("|")[-1]
    repo_id, _, rev = repo_id.partition("@")
    cache_key = f"{repo_id}@{rev}" if rev else repo_id
    if cache_key not in _MSA_KERNELS:
        from .hub_kernels import get_kernel

        _MSA_KERNELS[cache_key] = get_kernel(
            repo_id, revision=rev or None, version=None if rev else 0, allow_all_kernels=True
        )
    return _MSA_KERNELS[cache_key]


def _can_run_kernel(
    query: torch.Tensor, key: torch.Tensor, block_indices: torch.Tensor | None, block_size: int
) -> bool:
    """Gate the bf16 varlen CSR kernel (SM100, head_dim 128, supported topK).

    The same kernel serves both prefill (q_len > 1) and single-token decode (q_len == 1):
    decode is just a varlen call with one query slot. For decode we additionally require the
    cache to be long enough that skipping unselected blocks beats a dense SDPA pass.
    """
    if block_indices is None or query.device.type != "cuda":
        return False
    if torch.cuda.get_device_capability(query.device)[0] != 10:  # SM100 / Blackwell only
        return False
    if query.shape[-1] != 128:  # head_dim 128 only
        return False
    topk = _round_topk(block_indices.shape[-1])
    if topk is None:
        return False
    if query.shape[2] == 1:  # decode: only worth it when the selection is genuinely sparse
        num_kv_blocks = -(-key.shape[2] // block_size)
        return num_kv_blocks >= MSA_DECODE_MIN_BLOCKS_PER_TOPK * topk
    return True  # prefill


def _sparse_attention(module, query, key, value, scaling, block_indices, block_size):
    msa = _get_msa_kernel(module.config._attn_implementation)
    bsz, num_q_heads, q_len, head_dim = query.shape
    num_kv_heads, k_len = key.shape[1], key.shape[2]
    qheads_per_kv = num_q_heads // num_kv_heads
    num_selected = block_indices.shape[-1]
    topk = _round_topk(num_selected)

    # FA-style flatten: drop the batch dim into varlen [total, H, head_dim] + cu_seqlens.
    q = query.transpose(1, 2).reshape(bsz * q_len, num_q_heads, head_dim).contiguous()
    k = key.transpose(1, 2).reshape(bsz * k_len, num_kv_heads, head_dim).contiguous()
    v = value.transpose(1, 2).reshape(bsz * k_len, num_kv_heads, head_dim).contiguous()
    cu_seqlens_q = torch.arange(0, (bsz + 1) * q_len, q_len, device=q.device, dtype=torch.int32)
    cu_seqlens_k = torch.arange(0, (bsz + 1) * k_len, k_len, device=q.device, dtype=torch.int32)

    q2k = block_indices.to(torch.int32)
    if topk > num_selected:
        q2k = F.pad(q2k, (0, topk - num_selected), value=-1)
    q2k = q2k.reshape(bsz * q_len, topk).unsqueeze(0).expand(num_kv_heads, -1, -1).contiguous()

    # The CuTe-DSL kernel launches on the ambient ``current_device`` with no internal device
    # guard, so under ``device_map`` (where a prior layer may have left the current device on a
    # different GPU) it would reference this layer's tensors from the wrong context and raise
    # ``cudaErrorInvalidValue``. Pin the context to the tensors' device for the launch.
    with torch.cuda.device(query.device):
        k2q_row_ptr, k2q_q_indices = msa.build_k2q_csr(
            q2k,
            cu_seqlens_q,
            cu_seqlens_k,
            block_size,
            total_k=bsz * k_len,
            max_seqlen_k=k_len,
            max_seqlen_q=q_len,
            qhead_per_kv=qheads_per_kv,
        )
        attn_output = msa.sparse_atten_func(
            q,
            k,
            v,
            k2q_row_ptr,
            k2q_q_indices,
            topk,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=q_len,
            max_seqlen_k=k_len,
            blk_kv=block_size,
            causal=True,
            softmax_scale=scaling,
        )
    return attn_output.reshape(bsz, q_len, num_q_heads, head_dim)


def msa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    dropout: float = 0.0,
    scaling: float | None = None,
    block_indices: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """MiniMax Sparse Attention backend (``kernels-staging/msa``).

    On the supported path (SM100, head_dim 128) this runs true block-sparse attention via
    ``sparse_atten_func`` over the indexer's per-query selected key blocks (``block_indices``,
    batch-local ids, ``-1`` padded), for both prefill (q_len > 1) and single-token decode
    (q_len == 1, served as a one-slot varlen call over the contiguous bf16 cache). For everything
    else (short context, non-SM100, ...) it transparently falls back to SDPA over the dense block
    mask built from the same selection — the impl switch is hidden from the model and the user.
    Returns ``[B, S_q, H, head_dim]``.
    """
    if scaling is None:
        scaling = query.shape[-1] ** -0.5

    # No block selection (the dense vision tower, or non-sparse layers without an indexer) -> plain SDPA.
    if block_indices is None:
        return sdpa_attention_forward(
            module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs
        )

    block_size = module.indexer.block_size
    if _can_run_kernel(query, key, block_indices, block_size):
        attn_output = _sparse_attention(module, query, key, value, scaling, block_indices, block_size)
        return attn_output, None

    # Hidden fallback: reconstruct the dense block mask from the selection and run SDPA.
    attention_mask = module.indexer.build_block_mask(
        block_indices, attention_mask, key.shape[2], query.dtype, query.device
    )
    return sdpa_attention_forward(
        module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs
    )
