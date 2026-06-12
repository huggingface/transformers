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
import torch
import torch.nn.functional as F

from ..utils import logging
from .sdpa_attention import sdpa_attention_forward


logger = logging.get_logger(__name__)

# `sparse_atten_func` only compiles for these per-query block size.
MSA_SUPPORTED_TOPK = (4, 8, 16, 32)

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


@torch.library.custom_op("transformers_msa::sparse_atten", mutates_args=())
def _msa_sparse_atten_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    topk: int,
    block_size: int,
    total_k: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    qheads_per_kv: int,
    scaling: float,
    impl: str,
) -> torch.Tensor:
    """Opaque wrapper around the CuTe-DSL CSR build + block-sparse kernel.

    Registered as a ``torch.library`` custom op so ``torch.compile(fullgraph=True)`` treats the
    whole CSR-build + attention as a single opaque node (no graph break) and ``reduce-overhead``
    CUDA graphs can capture it. The internal ``build_k2q_csr`` output is data-dependent in shape,
    but it never escapes this op (only the fixed-shape ``[total_q, Hq, D]`` attention output does),
    so the fake/meta impl below is exact. The op is functional (no input mutation).
    """
    msa = _get_msa_kernel(impl)
    # CuTe-DSL kernel launches on the ambient ``current_device`` with no internal guard; pin context
    # to the tensors' device so device_map (mixed-GPU) layouts don't reference the wrong context.
    with torch.cuda.device(q.device):
        k2q_row_ptr, k2q_q_indices = msa.build_k2q_csr(
            q2k,
            cu_seqlens_q,
            cu_seqlens_k,
            block_size,
            total_k=total_k,
            max_seqlen_k=max_seqlen_k,
            max_seqlen_q=max_seqlen_q,
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
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            blk_kv=block_size,
            causal=True,
            softmax_scale=scaling,
        )
    return attn_output.contiguous()


@_msa_sparse_atten_op.register_fake
def _msa_sparse_atten_fake(
    q, k, v, q2k, cu_seqlens_q, cu_seqlens_k, topk, block_size, total_k,
    max_seqlen_q, max_seqlen_k, qheads_per_kv, scaling, impl,
):
    # Output matches the query varlen layout [total_q, Hq, head_dim].
    return torch.empty_like(q)


def _can_run_kernel(query: torch.Tensor, block_indices: torch.Tensor | None, block_size: int) -> bool:
    """Gate the bf16 varlen CSR kernel (SM100, head_dim 128, blk_kv 128, supported topK).

    Serves both prefill (q_len > 1) and single-token decode (q_len == 1) — decode is just a
    varlen call with one query slot. The attention op is a sub-millisecond slice of each step,
    so there's no context-length threshold: whenever the kernel is supported, it runs.
    """
    if block_indices is None or query.device.type != "cuda":
        return False
    if torch.cuda.get_device_capability(query.device)[0] != 10:  # SM100 / Blackwell only
        return False
    if query.shape[-1] != 128:  # head_dim 128 only
        return False
    if block_size != 128:  # SparseK2qCsrBuilderSm100 only supports blk_kv == 128
        return False
    return _round_topk(block_indices.shape[-1]) is not None


def _is_padded_cache(cache_position: torch.Tensor | None, k_len: int) -> bool:
    """Whether the KV buffer is longer than the valid region (a padded StaticCache).

    Forces a device->host sync, so it is only ever called on the bsz>1 branch — the bsz==1 fast
    path (the cudagraph target) short-circuits before this and stays sync-free.
    """
    if cache_position is None:
        return False
    return int(cache_position[-1]) + 1 < k_len


def _sparse_attention(module, query, key, value, scaling, block_indices, block_size, cache_position):
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
    # Under a StaticCache `k_len` is the pre-allocated buffer (`max_cache_len`), not the valid length,
    # so a packed `[0, k_len, 2*k_len, ...]` boundary would let the kernel's causal attend zero-padded
    # future slots. For bsz==1 the real boundary is `[0, valid_k]` (valid_k = cache_position[-1]+1) built
    # as a device tensor (no host sync) -- `build_k2q_csr` reads only the host shape hints `total_k`/
    # `max_seqlen_k` (kept fixed at `bsz*k_len`/`k_len` below), so this stays compile/cudagraph stable.
    # bsz>1 padded decode needs the kernel's paged layout and is routed to SDPA upstream, so the dense
    # packed boundary is exact for every batched case that reaches here.
    if bsz == 1 and cache_position is not None:
        valid_k = (cache_position[-1] + 1).to(torch.int32).reshape(1)
        cu_seqlens_k = torch.cat([torch.zeros(1, device=q.device, dtype=torch.int32), valid_k])
    else:
        cu_seqlens_k = torch.arange(0, (bsz + 1) * k_len, k_len, device=q.device, dtype=torch.int32)

    q2k = block_indices.to(torch.int32)
    if topk > num_selected:
        q2k = F.pad(q2k, (0, topk - num_selected), value=-1)
    q2k = q2k.reshape(bsz * q_len, topk).unsqueeze(0).expand(num_kv_heads, -1, -1).contiguous()

    # Opaque custom op: keeps the CuTe-DSL CSR build + block-sparse kernel as a single graph node
    # so ``torch.compile(fullgraph=True)`` doesn't break and ``reduce-overhead`` CUDA graphs capture it.
    attn_output = _msa_sparse_atten_op(
        q,
        k,
        v,
        q2k,
        cu_seqlens_q,
        cu_seqlens_k,
        topk,
        block_size,
        bsz * k_len,
        q_len,
        k_len,
        qheads_per_kv,
        scaling,
        module.config._attn_implementation,
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
    """
    TODO: this opens a door to per-layer attn implementation which is something we might want lalter on.
    """
    if scaling is None:
        scaling = query.shape[-1] ** -0.5

    # No block selection (the dense vision tower, or full-attention layers without an indexer) -> plain SDPA.
    if block_indices is None:
        return sdpa_attention_forward(
            module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs
        )

    block_size = module.indexer.block_size
    cache_position = kwargs.get("cache_position")
    # bsz>1 over a padded StaticCache needs the kernel's paged layout (the dense packed varlen
    # boundary would index the wrong rows for seq>0); route those to the SDPA fallback. bsz==1 and
    # non-padded (dynamic) batches build an exact boundary, so they keep the kernel fast path.
    paged_only = query.shape[0] > 1 and _is_padded_cache(cache_position, key.shape[2])
    if _can_run_kernel(query, block_indices, block_size) and not paged_only:
        attn_output = _sparse_attention(
            module, query, key, value, scaling, block_indices, block_size, cache_position
        )
        return attn_output, None

    # Hidden fallback: reconstruct the dense block mask from the selection and run SDPA.
    attention_mask = module.indexer.build_block_mask(
        block_indices, attention_mask, key.shape[2], query.dtype, query.device, cache_position
    )
    return sdpa_attention_forward(
        module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs
    )

