from typing import Tuple

import torch
import torch.nn.functional as F


try:
    from einops import rearrange
    from flash_attn.bert_padding import IndexFirstAxis
except:
    rearrange = None
    IndexFirstAxis = None


def unpad_tensor(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))

    if hidden_states is not None:
        hidden_states = IndexFirstAxis.apply(rearrange(hidden_states, "b s ... -> (b s) ..."), indices)

    return hidden_states, indices, cu_seqlens, max_seqlen


def interleave_query_key_value_tensor_for_mha(
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    interleaved = []
    for i in range(num_heads):
        start_index = i * head_dim
        end_index = start_index + head_dim

        interleaved.append(query_weight[start_index:end_index])
        interleaved.append(key_weight[start_index:end_index])
        interleaved.append(value_weight[start_index:end_index])

    return torch.cat(interleaved)


def split_query_key_value_tensor_for_mha(
    query_key_value_weight: torch.Tensor, num_heads: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    original_shape = query_key_value_weight.shape

    query_key_value_weight = query_key_value_weight.view(num_heads, -1)

    query_weight, key_weight, value_weight = query_key_value_weight.chunk(3, -1)

    query_weight = query_weight.reshape(-1, *original_shape[1:])
    key_weight = key_weight.reshape(-1, *original_shape[1:])
    value_weight = value_weight.reshape(-1, *original_shape[1:])

    return query_weight, key_weight, value_weight


def interleave_query_key_value_tensor_for_gqa(
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
    num_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> torch.Tensor:
    query_heads_per_group = num_heads // num_key_value_heads

    interleaved = []
    for i in range(num_key_value_heads):
        start_index = i * query_heads_per_group * head_dim
        end_index = start_index + query_heads_per_group * head_dim
        interleaved.append(query_weight[start_index:end_index])

        start_index = i * head_dim
        end_index = start_index + head_dim
        interleaved.append(key_weight[start_index:end_index])
        interleaved.append(value_weight[start_index:end_index])

    return torch.cat(interleaved)


def split_query_key_value_tensor_for_gqa(
    query_key_value_weight: torch.Tensor, num_heads: int, num_key_value_heads: int, head_dim: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query_heads_per_group = num_heads // num_key_value_heads
    original_shape = query_key_value_weight.shape

    query_key_value_weight = query_key_value_weight.view(num_key_value_heads, (query_heads_per_group + 2), -1)

    query_weight, key_weight, value_weight = query_key_value_weight.split((query_heads_per_group, 1, 1), 1)

    query_weight = query_weight.reshape(-1, *original_shape[1:])
    key_weight = key_weight.reshape(-1, *original_shape[1:])
    value_weight = value_weight.reshape(-1, *original_shape[1:])

    return query_weight, key_weight, value_weight


def interleave_query_key_value_tensor_for_mqa(
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
) -> torch.Tensor:
    # [:] for converting slice to tensor
    return torch.cat([query_weight[:], key_weight[:], value_weight[:]])


def split_query_key_value_tensor_for_mqa(
    query_key_value_weight: torch.Tensor, num_heads: int, head_dim: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return query_key_value_weight.split((num_heads * head_dim, head_dim, head_dim))
