# Copyright 2026 The HuggingFace Inc. team.
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
from torch import nn
from torch.distributed.tensor import DTensor

from .hub_kernels import lazy_load_kernel


def paged_latent_attention(
    module: nn.Module,
    query: torch.Tensor,  # shape [1, num_heads, num_sequences, qk_nope_head_dim + qk_rope_head_dim]
    latent_nope: torch.Tensor,  # shape [1, 1, num_sequences, kv_lora_rank]
    latent_rot: torch.Tensor,  # shape [1, 1, num_sequences, qk_rope_head_dim]
    **kwargs,
) -> tuple[torch.Tensor, None]:  # shape [1, num_sequences, num_heads, v_head_dim]
    """Multi-head latent attention on the decode fast path of continuous batching, where every sequence brings one new
    token: the latents are attended to where they sit in the cache pages, never expanded into keys and values.

    The new token's latent is written to its sequence's page. Each query head absorbs the key half of `kv_b_proj`, so
    that it scores the latents directly: all heads attend to the latents as one shared key head, of width
    `kv_lora_rank + qk_rope_head_dim`, which is the FlashMLA kernel's layout (576 on DeepSeek V3). The attention output
    is then a latent per head, which absorbs the value half of `kv_b_proj`.
    """
    cache = kwargs["cache"]
    latent = torch.cat((latent_nope, latent_rot), dim=-1)[0, 0]
    pages, block_table, seqlens_k = cache.update_paged_rows(module.layer_idx, {"latent": latent}, kwargs)

    # The key half and the value half of `kv_b_proj`, for the heads of this rank under tensor parallelism
    weight = module.kv_b_proj.weight
    weight = weight.to_local() if isinstance(weight, DTensor) else weight
    weight = weight.view(-1, module.qk_nope_head_dim + module.v_head_dim, module.kv_lora_rank)
    key_weight, value_weight = weight.split([module.qk_nope_head_dim, module.v_head_dim], dim=1)

    query_nope, query_rot = query[0].split([module.qk_nope_head_dim, module.qk_rope_head_dim], dim=-1)
    query = torch.cat((torch.einsum("hsn,hnr->shr", query_nope, key_weight), query_rot.transpose(0, 1)), dim=-1)

    flash_mla = lazy_load_kernel("flash-mla")
    # The scheduling metadata is computed by the kernel's first call: a fresh one per call keeps it right as the
    # sequence lengths change, including on graph replays
    latent_output, _ = flash_mla.flash_mla_with_kvcache(
        query[:, None],
        pages["latent"][:, :, None],
        block_table,
        seqlens_k,
        module.kv_lora_rank,
        flash_mla.get_mla_metadata()[0],
        softmax_scale=module.scaling,
        causal=True,
    )
    attn_output = torch.einsum("shr,hvr->shv", latent_output[:, 0], value_weight)
    return attn_output[None], None
