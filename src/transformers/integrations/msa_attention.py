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

from ..utils import logging
from .sdpa_attention import sdpa_attention_forward


logger = logging.get_logger(__name__)

# `sparse_atten_func` only compiles for these per-query block counts.
MSA_SUPPORTED_TOPK = (4, 8, 16, 32)
# `SparseK2qCsrBuilderSm100` only supports a 128-key block.
MSA_SUPPORTED_BLOCK_SIZE = 128
# SM100 / Blackwell head_dim 128 kernel.
MSA_SUPPORTED_HEAD_DIM = 128

_MSA_KERNEL = None


def load_and_register_msa_kernel(attn_implementation: str):
    """Load the MSA hub kernel once and verify the expected callables are present.

    The ``attn_implementation`` string may carry a ``paged|`` prefix and/or an ``@<revision>`` pin
    (e.g. ``kernels-staging/msa@v0``); the build currently lives on the repo's ``v0`` branch. The
    loaded module is cached in a module-level global so registration happens once, not per call.
    """
    global _MSA_KERNEL
    if _MSA_KERNEL is not None:
        return _MSA_KERNEL

    from .hub_kernels import get_kernel

    repo_id = attn_implementation.split("|")[-1]
    repo_id, _, rev = repo_id.partition("@")
    kernel = get_kernel(repo_id, revision=rev or None, version=None if rev else 0, allow_all_kernels=True)

    for fn_name in ("sparse_atten_func", "build_k2q_csr"):
        if not callable(getattr(kernel, fn_name, None)):
            raise ImportError(
                f"The MSA kernel loaded from `{repo_id}` does not expose a callable `{fn_name}`. "
                "Make sure you request a compatible build, e.g. `kernels-staging/msa@v0`."
            )

    _MSA_KERNEL = kernel
    return _MSA_KERNEL


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
    msa = load_and_register_msa_kernel(impl)
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
    q,
    k,
    v,
    q2k,
    cu_seqlens_q,
    cu_seqlens_k,
    topk,
    block_size,
    total_k,
    max_seqlen_q,
    max_seqlen_k,
    qheads_per_kv,
    scaling,
    impl,
):
    # Output matches the query varlen layout [total_q, Hq, head_dim].
    return torch.empty_like(q)


def _validate_msa_init(module, query: torch.Tensor, dropout: float) -> None:
    """Validate kernel capability, dropout and configured topk once per attention module.

    Mirrors the flash-attention integration, which checks capability/dropout at model init rather
    than on every forward. The check is cached on the module so the hot path never re-runs it.

    There is no SDPA fallback: a sparse layer either runs the MSA kernel or this raises. Serves both
    prefill (q_len > 1) and single-token decode (q_len == 1) -- decode is just a varlen call with one
    query slot, so there is no context-length threshold.
    """
    if query.device.type != "cuda" or torch.cuda.get_device_capability(query.device)[0] != 10:
        raise RuntimeError(
            "MSA block-sparse attention requires an SM100 / Blackwell CUDA device. "
            "Select a different `attn_implementation` on unsupported hardware."
        )
    if query.shape[-1] != MSA_SUPPORTED_HEAD_DIM:
        raise ValueError(f"MSA block-sparse attention only supports head_dim {MSA_SUPPORTED_HEAD_DIM}.")
    if module.indexer.block_size != MSA_SUPPORTED_BLOCK_SIZE:
        raise ValueError(f"MSA block-sparse attention only supports block_size {MSA_SUPPORTED_BLOCK_SIZE}.")
    if dropout != 0.0:
        raise ValueError("MSA block-sparse attention does not support attention dropout; set `attention_dropout=0`.")
    topk = module.indexer.topk_blocks
    if topk not in MSA_SUPPORTED_TOPK:
        raise ValueError(
            f"MSA block-sparse attention only supports topk in {MSA_SUPPORTED_TOPK}, got `{topk}`. "
            "Set `index_topk_blocks` to a supported value."
        )


def _sparse_attention(module, query, key, value, scaling, block_indices, block_size, cache_position):
    bsz, num_q_heads, q_len, head_dim = query.shape
    num_kv_heads, k_len = key.shape[1], key.shape[2]
    qheads_per_kv = num_q_heads // num_kv_heads
    topk = block_indices.shape[-1]

    # The indexer emits `min(index_topk_blocks, num_key_blocks)` selected blocks, so on sequences with
    # fewer key blocks than the configured budget the width lands on an arbitrary value (e.g. 12) that the
    # `SparseK2qCsrBuilderSm100` CSR builder rejects -- it only accepts a CSR width in `MSA_SUPPORTED_TOPK`.
    # Right-pad the selection up to the next supported width with `-1`, the same empty-slot sentinel the
    # kernel already skips, so behaviour is unchanged and the width is always one the builder accepts. The
    # width is a Python int (static under `torch.compile`), so this stays fullgraph / cudagraph stable.
    padded_topk = next(t for t in MSA_SUPPORTED_TOPK if t >= topk)
    if padded_topk != topk:
        pad = block_indices.new_full((*block_indices.shape[:-1], padded_topk - topk), -1)
        block_indices = torch.cat([block_indices, pad], dim=-1)
        topk = padded_topk

    # Flatten the batch dim into a packed varlen layout [total, H, head_dim] + cu_seqlens. The
    # query boundary is a fixed stride (every row is `q_len` long), built device-side with no host
    # sync so it stays compile/cudagraph stable.
    q = query.transpose(1, 2).reshape(bsz * q_len, num_q_heads, head_dim).contiguous()
    k = key.transpose(1, 2).reshape(bsz * k_len, num_kv_heads, head_dim).contiguous()
    v = value.transpose(1, 2).reshape(bsz * k_len, num_kv_heads, head_dim).contiguous()
    cu_seqlens_q = torch.arange(0, (bsz + 1) * q_len, q_len, device=q.device, dtype=torch.int32)
    # Under a StaticCache `k_len` is the pre-allocated buffer (`max_cache_len`), not the valid length,
    # so a packed `[0, k_len, 2*k_len, ...]` boundary would let the kernel's causal attend zero-padded
    # future slots. For bsz==1 the real boundary is `[0, valid_k]` (valid_k = cache_position[-1]+1) built
    # as a device tensor (no host sync) -- `build_k2q_csr` reads only the host shape hints `total_k`/
    # `max_seqlen_k` (kept fixed at `bsz*k_len`/`k_len` below), so this stays compile/cudagraph stable.
    if bsz == 1 and cache_position is not None:
        valid_k = (cache_position[-1] + 1).to(torch.int32).reshape(1)
        cu_seqlens_k = torch.cat([torch.zeros(1, device=q.device, dtype=torch.int32), valid_k])
    else:
        cu_seqlens_k = torch.arange(0, (bsz + 1) * k_len, k_len, device=q.device, dtype=torch.int32)

    q2k = block_indices.to(torch.int32)
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

    # A sparse layer always runs the MSA kernel -- there is no SDPA fallback. Capability/config is
    # validated once per module (raises on unsupported hardware or config) and cached on the module.
    if not getattr(module, "_msa_validated", False):
        _validate_msa_init(module, query, dropout)
        module._msa_validated = True

    block_size = module.indexer.block_size
    cache_position = kwargs.get("cache_position")
    attn_output = _sparse_attention(module, query, key, value, scaling, block_indices, block_size, cache_position)
    return attn_output, None
