# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# Modifications Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
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
from collections.abc import Callable
from contextlib import contextmanager
from typing import TYPE_CHECKING

from ..cache_utils import Cache, DynamicCache, DynamicLayer, DynamicSlidingWindowLayer, EncoderDecoderCache
from ..masking_utils import (
    ALL_MASK_ATTENTION_FUNCTIONS,
    _ignore_causal_mask_sdpa,
    and_masks,
    causal_mask_function,
    eager_mask,
    padding_mask_function,
    prepare_padding_mask,
    sdpa_mask,
)
from ..utils.import_utils import is_torch_available
from ..utils.logging import get_logger


logger = get_logger(__name__)


if is_torch_available():
    import torch

if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel


def _get_dynamic_cache_dict(cache: DynamicCache):
    """Converts DynamicCache to dictionary format for pytree operations."""
    if any(not isinstance(layer, DynamicLayer | DynamicSlidingWindowLayer) for layer in cache.layers):
        raise RuntimeError("This pytree flattening function should only be applied to DynamicCache")

    return {
        "key_cache": [layer.keys for layer in cache.layers if layer.keys is not None],
        "value_cache": [layer.values for layer in cache.layers if layer.values is not None],
    }


def get_encoder_decoder_cache_dict(cache: EncoderDecoderCache):
    """Converts EncoderDecoderCache to dictionary format for pytree operations."""
    return {
        "self_attention_cache": _get_dynamic_cache_dict(cache.self_attention_cache),
        "cross_attention_cache": _get_dynamic_cache_dict(cache.cross_attention_cache),
    }


def _unflatten_dynamic_cache(values, context: torch.utils._pytree.Context):
    dictionary = torch.utils._pytree._dict_unflatten(values, context)
    cache = DynamicCache()
    # Reconstruct layers from keys and values lists
    key_list = dictionary.get("key_cache", [])
    value_list = dictionary.get("value_cache", [])
    for idx in range(max(len(key_list), len(value_list))):
        key = key_list[idx] if idx < len(key_list) else None
        value = value_list[idx] if idx < len(value_list) else None
        cache_layer = DynamicLayer()
        cache_layer.keys = key
        cache_layer.values = value
        cache_layer.is_initialized = True
        cache.layers.append(cache_layer)
    return cache


def _unflatten_encoder_decoder_cache(values, context: torch.utils._pytree.Context):
    dictionary = torch.utils._pytree._dict_unflatten(values, context)
    self_attention_cache = _unflatten_dynamic_cache(
        [
            dictionary.get("self_attention_cache", {}).get("key_cache", []),
            dictionary.get("self_attention_cache", {}).get("value_cache", []),
        ],
        context,
    )
    cross_attention_cache = _unflatten_dynamic_cache(
        [
            dictionary.get("cross_attention_cache", {}).get("key_cache", []),
            dictionary.get("cross_attention_cache", {}).get("value_cache", []),
        ],
        context,
    )
    return EncoderDecoderCache(self_attention_cache, cross_attention_cache)


def register_dynamic_cache_for_export():
    try:
        torch.utils._pytree.register_pytree_node(
            DynamicCache,
            lambda dynamic_cache: torch.utils._pytree._dict_flatten(_get_dynamic_cache_dict(dynamic_cache)),
            _unflatten_dynamic_cache,
            serialized_type_name=f"{DynamicCache.__module__}.{DynamicCache.__name__}",
            flatten_with_keys_fn=lambda dynamic_cache: torch.utils._pytree._dict_flatten_with_keys(
                _get_dynamic_cache_dict(dynamic_cache)
            ),
        )
        # TODO (tmanlaibaatar) This won't be needed in torch 2.7.
        torch.fx._pytree.register_pytree_flatten_spec(
            DynamicCache,
            lambda cache, spec: torch.fx._pytree._dict_flatten_spec(_get_dynamic_cache_dict(cache), spec),
        )
    # Catching this in case there are multiple runs for some test runs
    except ValueError as e:
        if "already registered as pytree node" not in str(e):
            raise


def register_encoder_decoder_cache_for_export():
    try:
        torch.utils._pytree.register_pytree_node(
            EncoderDecoderCache,
            lambda cache: torch.utils._pytree._dict_flatten(get_encoder_decoder_cache_dict(cache)),
            _unflatten_encoder_decoder_cache,
            serialized_type_name=f"{EncoderDecoderCache.__module__}.{EncoderDecoderCache.__name__}",
            flatten_with_keys_fn=lambda cache: torch.utils._pytree._dict_flatten_with_keys(
                get_encoder_decoder_cache_dict(cache)
            ),
        )
        # TODO (tmanlaibaatar) This won't be needed in torch 2.7.
        torch.fx._pytree.register_pytree_flatten_spec(
            EncoderDecoderCache,
            lambda cache, spec: torch.fx._pytree._dict_flatten_spec(get_encoder_decoder_cache_dict(cache), spec),
        )
    # Catching this in case there are multiple runs for some test runs
    except ValueError as e:
        if "already registered as pytree node" not in str(e):
            raise


# TODO: won't be needed when it becomes the default in transformers
# Custom vectorized implementation of sdpa_mask without using vmap
def sdpa_mask_without_vmap(
    batch_size: int,
    cache_position: torch.Tensor,
    kv_length: int,
    kv_offset: int = 0,
    mask_function: Callable | None = None,
    attention_mask: torch.Tensor | None = None,
    local_size: int | None = None,
    allow_is_causal_skip: bool = True,
    **kwargs,
) -> torch.Tensor | None:
    if mask_function is None:
        mask_function = causal_mask_function

    q_length = cache_position.shape[0]
    # Potentially pad the 2D mask, and slice it correctly
    padding_mask = prepare_padding_mask(attention_mask, kv_length, kv_offset, _slice=False)

    # Under specific conditions, we can avoid materializing the mask, instead relying on the `is_causal` argument
    if allow_is_causal_skip and _ignore_causal_mask_sdpa(padding_mask, q_length, kv_length, kv_offset, local_size):
        return None

    # Potentially add the padding 2D mask
    if padding_mask is not None:
        mask_function = and_masks(mask_function, padding_mask_function(padding_mask))

    # Create broadcatable indices
    device = cache_position.device
    q_indices = cache_position[None, None, :, None]
    head_indices = torch.arange(1, dtype=torch.long, device=device)[None, :, None, None]
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)[:, None, None, None]
    kv_indices = torch.arange(kv_length, dtype=torch.long, device=device)[None, None, None, :] + kv_offset
    # Apply mask function element-wise through broadcasting
    causal_mask = mask_function(batch_indices, head_indices, q_indices, kv_indices)
    # Expand the mask to match batch size and query length if they weren't used in the mask function
    causal_mask = causal_mask.expand(batch_size, -1, q_length, kv_length)

    return causal_mask


# Adapted from https://github.com/huggingface/transformers/blob/v4.53.0/src/transformers/masking_utils.py#L433
def eager_mask_without_vmap(*args, **kwargs) -> torch.Tensor:
    kwargs.pop("allow_is_causal_skip", None)
    dtype = kwargs.get("dtype", torch.float32)
    mask = sdpa_mask_without_vmap(*args, allow_is_causal_skip=False, **kwargs)
    mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), torch.finfo(dtype).min)
    return mask


def get_auto_dynamic_shapes(inputs: dict[str, torch.Tensor | Cache]) -> dict[str, dict[int, torch.export.Dim]]:
    """
    Utility function to automatically generate dynamic shapes for a dictionary of model inputs.

    Args:
        inputs (`dict[str, torch.Tensor | Cache]`):
            The inputs with which the model will be exported.
    Returns:
        `dict[str, dict[int, torch.export.Dim]]`: A dictionary mapping input names to their dynamic shapes.
    """
    from torch.export import Dim

    dynamic_shapes = {}
    for name, input in inputs.items():
        if isinstance(input, DynamicCache):
            dynamic_shapes[name] = [
                [dict.fromkeys(range(len(layer.keys.shape)), Dim.AUTO) for layer in input.layers],
                [dict.fromkeys(range(len(layer.values.shape)), Dim.AUTO) for layer in input.layers],
            ]
        elif isinstance(input, torch.Tensor):
            dynamic_shapes[name] = dict.fromkeys(range(len(input.shape)), Dim.AUTO)
        else:
            raise ValueError(
                f"Input '{name}' is of unsupported type '{type(input)}'. Only 'torch.Tensor' and 'DynamicCache' are supported."
            )

    return dynamic_shapes


def generate_masks_with_special_tokens_and_transfer_map(
    input_ids: torch.LongTensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a pair of tensors used for text handling in GroundingDINO:

    Returns:
    - attention_mask (`torch.BoolTensor`, shape `(B, N, N)`):
        For each batch, a boolean matrix where True indicates tokens that should attend to
        each other. We always keep the diagonal (a token attends to itself). Additionally,
        non-special tokens that belong to the same segment (i.e. share the same closing
        special token) will attend to each other.
    - position_ids (`torch.LongTensor`, shape `(B, N)`):
        For tokens that belong to a valid segment (bounded by special tokens), this contains
        the position of the token inside its segment (0-based). Tokens not belonging to any
        valid segment get 0.

    Algorithm overview (per batch):
    1. Find positions that are special tokens.
    2. For each position p, compute next_delim_idx[p]: index of the first special token
       at or after p (or N if none).
    3. Tokens that share the same next_delim_idx (and where that next_delim is not a sentinel)
       belong to the same block/segment and are allowed to cross-attend.
    4. Exclude blocks whose closing delimiter is at position 0 or N-1 (these should only keep diagonal).
    5. Compute position_ids as distance from the previous delimiter (+1), only for valid blocks.
    """
    from transformers.models.grounding_dino.modeling_grounding_dino import SPECIAL_TOKENS

    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    # 1) Identify special token positions
    # boolean mask of special-token positions (B, N)
    special_mask = torch.isin(input_ids, torch.tensor(SPECIAL_TOKENS, device=device))

    # 2) For each position, find index of next special token (or seq_len if none)
    # indexes [0,1,2,...,N-1] broadcasted to (B, N)
    indexes = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    # at special token positions keep their index, else sentinel seq_len
    candidates = torch.where(special_mask, indexes, torch.tensor(seq_len, device=device))  # (B, N)
    # compute prefix minimum from the right to get first special >= p
    next_delim_idx = torch.flip(torch.cummin(torch.flip(candidates, dims=[1]), dim=1)[0], dims=[1])  # (B, N)

    # 3) Build group mask based on next_delim_idx
    # tokens sharing same next_delim_idx (and where that delim is real) belong to same block
    nd_i = next_delim_idx.unsqueeze(2)  # (B, N, 1)
    nd_j = next_delim_idx.unsqueeze(1)  # (B, 1, N)
    has_real_delim = next_delim_idx != seq_len
    group_mask = (nd_i == nd_j) & has_real_delim.unsqueeze(1)  # (B, N, N)

    # 4) Exclude blocks whose closing delimiter is at position 0 or N-1 (these should only keep diagonal)
    valid_block = (next_delim_idx != 0) & (next_delim_idx != (seq_len - 1)) & (next_delim_idx != seq_len)
    group_mask &= valid_block.unsqueeze(1)

    # Always allow self-attention (diagonal)
    identity = torch.eye(seq_len, device=device, dtype=torch.bool).unsqueeze(0).expand(batch_size, -1, -1)
    attention_mask = identity | group_mask  # (B, N, N)

    # 5) Compute position_ids as distance from previous delimiter (+1), only for valid blocks
    # previous delimiter index at each column: shift prefix-max of special indices right by 1
    neg_one = torch.full((batch_size, 1), -1, device=device, dtype=torch.long)
    prev_candidates = torch.where(special_mask, indexes, torch.tensor(-1, device=device))
    prefix_max = torch.cummax(prev_candidates, dim=1)[0]  # (B, N)
    prev_delim_per_col = torch.cat((neg_one, prefix_max[:, :-1]), dim=1)  # (B, N)
    prev_delim_per_col = torch.clamp(prev_delim_per_col, min=0)  # (B, N)
    # gather previous delimiter corresponding to each token's closing delimiter
    gather_idx = torch.clamp(next_delim_idx, max=seq_len - 1)  # (B, N)
    prev_delim_for_token = torch.gather(prev_delim_per_col, 1, gather_idx)  # (B, N)
    position_ids = indexes - prev_delim_for_token - 1  # distance from previous delimiter (0-based)
    # only keep position ids for tokens in valid blocks; others set to 0
    position_ids = torch.where(valid_block, position_ids, torch.zeros_like(position_ids))
    position_ids = torch.clamp(position_ids, min=0).to(torch.long)

    return attention_mask, position_ids


def range_index_map(batch_shape, num_segments, name="range_index_map"):
    """
    Constructs an index map equal to range(num_segments).

    Args:
        batch_shape (`torch.Size`):
            Batch shape
        num_segments (`int`):
            Number of segments
        name (`str`, *optional*, defaults to 'range_index_map'):
            Name for the operation. Currently not used

    Returns:
        (`IndexMap`): IndexMap of shape batch_shape with elements equal to range(num_segments).
    """
    from ..models.tapas.modeling_tapas import IndexMap

    device = num_segments.device if torch.is_tensor(num_segments) else "cpu"
    batch_shape = torch.as_tensor(batch_shape, dtype=torch.long, device=device)
    num_segments = torch.as_tensor(num_segments, device=device)

    # handle the no-batch case
    if batch_shape.numel() == 0:
        indices = torch.arange(start=0, end=num_segments, device=device)
        return IndexMap(indices=indices, num_segments=num_segments, batch_dims=0)

    # Create 1D ranges for each batch dimension (these accept tensor scalars)
    ranges = [torch.arange(0, b, device=device) for b in batch_shape]

    # Build a representative tensor of shape `batch_shape` using meshgrid,
    # then make a ones tensor of that shape to broadcast the last-dimension arange.
    grids = torch.meshgrid(*ranges, indexing="ij") if len(ranges) > 1 else (ranges[0],)
    ones = torch.ones_like(grids[0], dtype=torch.long, device=device)  # shape = batch_shape

    # create the last-dimension arange (num_segments) and broadcast-multiply
    last = torch.arange(0, num_segments, device=device, dtype=torch.long).unsqueeze(0)  # shape (1, num_segments)
    indices = ones.unsqueeze(-1) * last  # broadcasts to (*batch_shape, num_segments)

    return IndexMap(indices=indices, num_segments=num_segments, batch_dims=int(batch_shape.size(0)))


def batched_experts_forward_with_split_expert_weights(
    self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
) -> torch.Tensor:
    final_hidden_states = torch.zeros_like(hidden_states)

    # Vectorized single-pass expert dispatch (no data-dependent loop)
    num_tokens, hidden_dim = hidden_states.shape
    top_k = top_k_index.size(1)
    device = hidden_states.device
    dtype = hidden_states.dtype

    # Flatten token-expert pairs
    token_idx = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, top_k).reshape(-1)  # (P,)
    expert_ids = top_k_index.reshape(-1)  # (P,)
    pair_weights = top_k_weights.reshape(-1, 1).to(dtype)  # (P, 1)

    # Gather inputs for all (token, expert) pairs
    x = hidden_states.index_select(0, token_idx)  # (P, H)

    # TODO: instead of stacking here, patch the entire class to have its expert weights pre-stacked
    # Stack per-expert weights once, then gather per pair
    # Linear weight shapes: gate/up: (I, H), down: (H, I)
    if hasattr(self[0], "w1"):
        Wg = torch.stack([m.w1.weight for m in self], dim=0)  # (E, I, H)
        Wup = torch.stack([m.w3.weight for m in self], dim=0)  # (E, I, H)
        Wd = torch.stack([m.w2.weight for m in self], dim=0)  # (E, H, I)
    elif hasattr(self[0], "gate_proj"):
        Wg = torch.stack(
            [m.gate_proj.weight for m in self if not isinstance(m, torch.nn.Identity)], dim=0
        )  # (E, I, H)
        Wup = torch.stack([m.up_proj.weight for m in self if not isinstance(m, torch.nn.Identity)], dim=0)  # (E, I, H)
        Wd = torch.stack(
            [m.down_proj.weight for m in self if not isinstance(m, torch.nn.Identity)], dim=0
        )  # (E, H, I)
    else:
        raise RuntimeError("Unexpected expert MLP structure")

    # Select weights for each pair and reshape for bmm
    Wg_sel = Wg.index_select(0, expert_ids).transpose(1, 2)  # (P, H, I)
    Wup_sel = Wup.index_select(0, expert_ids).transpose(1, 2)  # (P, H, I)
    Wd_sel = Wd.index_select(0, expert_ids).transpose(1, 2)  # (P, I, H)

    x_ = x.unsqueeze(1)  # (P, 1, H)

    # gate/up projections
    s_gate = torch.bmm(x_, Wg_sel).squeeze(1)  # (P, I)
    s_up = torch.bmm(x_, Wup_sel).squeeze(1)  # (P, I)

    # activation and elementwise product
    act = self[0].act_fn(s_gate)  # (P, I)  # same act for all experts
    inter = act * s_up  # (P, I)

    # down projection
    y = torch.bmm(inter.unsqueeze(1), Wd_sel).squeeze(1)  # (P, H)

    # apply routing weights and scatter-add back to tokens
    y = (y * pair_weights).to(dtype)
    final_hidden_states.index_add_(0, token_idx, y)

    return final_hidden_states


def batched_experts_forward_with_grouped_expert_weights(
    self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
) -> torch.Tensor:
    batch_size = hidden_states.size(0)
    hidden_states = hidden_states.reshape(-1, self.ffn_hidden_size)
    next_states = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)

    # Vectorized single-pass expert dispatch (compute only hit experts)
    T = hidden_states.shape[0]
    top_k = top_k_index.shape[1]
    device = hidden_states.device
    dtype = hidden_states.dtype

    # Flatten token-expert pairs
    token_idx = torch.arange(T, device=device).unsqueeze(1).expand(-1, top_k).reshape(-1)  # (P,)
    expert_ids = top_k_index.reshape(-1)  # (P,)
    pair_weights = top_k_weights.reshape(-1, 1).to(dtype)  # (P, 1)

    # Gather input states for all pairs
    x = hidden_states.index_select(0, token_idx)  # (P, I)

    # Prepare per-expert weights and select for each pair
    I = self.ffn_hidden_size
    H = self.hidden_size
    W1 = self.mlp.w1.view(self.num_experts, I, H)  # (E, I, H)
    V1 = self.mlp.v1.view(self.num_experts, I, H)  # (E, I, H)
    W2 = self.mlp.w2.view(self.num_experts, I, H)  # (E, I, H)

    W1_sel = W1.index_select(0, expert_ids)  # (P, I, H)
    V1_sel = V1.index_select(0, expert_ids)  # (P, I, H)
    W2_sel = W2.index_select(0, expert_ids)  # (P, I, H)

    x_ = x.unsqueeze(1)  # (P, 1, I)
    s1 = torch.bmm(x_, W1_sel).squeeze(1)  # (P, H)
    s2 = torch.bmm(x_, V1_sel).squeeze(1)  # (P, H)

    inter = self.mlp.activation_fn(s1) * s2  # (P, H)
    y = torch.bmm(inter.unsqueeze(1), W2_sel.transpose(1, 2)).squeeze(1)  # (P, I)

    y = (y * pair_weights).to(dtype)  # (P, I)
    next_states.index_add_(0, token_idx, y)  # scatter-add back to tokens
    next_states = next_states.view(batch_size, -1, self.ffn_hidden_size)
    return next_states


def batched_switch_transformers_experts_forward(
    self, hidden_states: torch.Tensor, selected_experts: torch.Tensor, routing_weights: torch.Tensor
) -> torch.Tensor:
    """
    Vectorized, torch.export-friendly version of SwitchTransformersExperts.forward.

    This implementation avoids data-dependent control flow (no early returns or size-varying
    indexing). All pair computations are kept at a fixed shape (P * top_k) and invalid pairs
    (where selected_experts is all-zero) are suppressed by zeroing their routing weights so
    they contribute nothing to the final scatter-add.
    """
    P, H = hidden_states.shape
    device = hidden_states.device
    dtype = hidden_states.dtype

    # shapes
    top_k = selected_experts.shape[1]
    E = selected_experts.shape[-1]

    # Flatten token-topk pairs: length = P * top_k
    token_idx = torch.arange(P, device=device).unsqueeze(1).expand(-1, top_k).reshape(-1)  # (P*top_k,)

    # Flatten selection and compute a pair mask (no branching on it)
    sel_flat = selected_experts.reshape(-1, E)  # (P*top_k, E)
    pair_mask = sel_flat.any(dim=-1)  # (P*top_k,) boolean

    # Use argmax for expert ids (safe even when sel_flat is all-zero: will pick 0)
    expert_ids = torch.argmax(sel_flat, dim=-1)  # (P*top_k,)

    # Flatten routing weights and zero-out invalid pairs so they have no effect
    pair_weights = routing_weights.reshape(-1, 1).to(dtype) * pair_mask.to(dtype).unsqueeze(1)  # (P*top_k, 1)

    # Gather inputs for all pairs
    x = hidden_states.index_select(0, token_idx)  # (P*top_k, H)

    # Stack per-expert weights once, then select per pair
    W_i = torch.stack([self[f"expert_{i}"].wi.weight for i in range(self.num_experts)], dim=0)  # (E, d_ff, H)
    W_o = torch.stack([self[f"expert_{i}"].wo.weight for i in range(self.num_experts)], dim=0)  # (E, H, d_ff)

    # Select weights per (flattened) pair
    W_i_sel = W_i.index_select(0, expert_ids)  # (P*top_k, d_ff, H)
    W_o_sel = W_o.index_select(0, expert_ids)  # (P*top_k, H, d_ff)

    # Up projection: x (1, H) @ (H, d_ff) -> (1, d_ff)
    x_ = x.unsqueeze(1)  # (P*top_k, 1, H)
    s_up = torch.bmm(x_, W_i_sel.transpose(1, 2)).squeeze(1)  # (P*top_k, d_ff)

    # Activation: use any expert's activation (acts are the same across experts in this impl)
    act_fn = next(iter(self.values())).act
    inter = act_fn(s_up)  # (P*top_k, d_ff)

    # Align dtype to output weight dtype if needed
    if inter.dtype != W_o_sel.dtype and W_o_sel.dtype != torch.int8:
        inter = inter.to(W_o_sel.dtype)

    # Down projection: (1, d_ff) @ (d_ff, H) -> (1, H)
    y = torch.bmm(inter.unsqueeze(1), W_o_sel.transpose(1, 2)).squeeze(1)  # (P*top_k, H)

    # Apply routing weights (zeros out invalid pairs) and cast back to original dtype
    y = (y * pair_weights).to(dtype)  # (P*top_k, H)

    # Scatter-add back to final hidden states (fixed-size token_idx)
    final_hidden_states = torch.zeros_like(hidden_states)
    final_hidden_states.index_add_(0, token_idx, y)

    return final_hidden_states


def axial_position_embedding(self, position_ids):
    # broadcast weights to correct shape
    batch_size = position_ids.shape[0]
    broadcasted_weights = [
        weight.expand((batch_size,) + self.axial_pos_shape + weight.shape[-1:]) for weight in self.weights
    ]

    position_encodings = torch.cat(broadcasted_weights, dim=-1)
    position_encodings = torch.reshape(position_encodings, (batch_size, -1, position_encodings.shape[-1]))
    position_encodings = torch.gather(
        position_encodings,
        1,
        position_ids.unsqueeze(-1).expand(-1, -1, position_encodings.shape[-1]),
    )

    return position_encodings


def aria_grouped_experts_gemm(self, input: torch.Tensor, tokens_per_expert: torch.Tensor) -> torch.Tensor:
    # Create expert assignment indices [num_tokens]
    expert_indices = torch.repeat_interleave(
        torch.arange(self.groups, device=input.device, dtype=torch.long), tokens_per_expert.to(torch.long)
    )
    torch._check(expert_indices.shape[0] == input.shape[0])

    # Gather expert weights for each token: [num_tokens, in_features, out_features]
    expert_weights = self.weight.index_select(0, expert_indices)

    # Batched matrix multiplication: [num_tokens, 1, in_features] @ [num_tokens, in_features, out_features]
    output = torch.bmm(input.unsqueeze(1), expert_weights).squeeze(1)

    return output


def idefics_embedding(self, x, seq_len=None):
    torch._check(seq_len.item() > 0)
    torch._check(seq_len.item() <= max(self.cos_cached.shape[0], self.sin_cached.shape[0]))
    return (
        self.cos_cached[: seq_len.item()].to(dtype=x.dtype),
        self.sin_cached[: seq_len.item()].to(dtype=x.dtype),
    )


TRANSFORMERS_MODULE_TO_EXPORTABLE_FORWARD: dict[str, Callable] = {
    # Expert MLPs with different weight storage schemes
    "AriaGroupedExpertsGemm": aria_grouped_experts_gemm,
    "AxialPositionEmbeddings": axial_position_embedding,
    "DbrxExperts": batched_experts_forward_with_grouped_expert_weights,
    "DeepseekV2Experts": batched_experts_forward_with_split_expert_weights,
    "DeepseekV3NaiveMoe": batched_experts_forward_with_split_expert_weights,
    "Dots1NaiveMoe": batched_experts_forward_with_split_expert_weights,
    "Ernie4_5_MoeExperts": batched_experts_forward_with_split_expert_weights,
    "FlexOlmoExperts": batched_experts_forward_with_split_expert_weights,
    "Glm4MoeNaiveMoe": batched_experts_forward_with_split_expert_weights,
    "Glm4vMoeTextNaiveMoe": batched_experts_forward_with_split_expert_weights,
    "HunYuanMoEV1Experts": batched_experts_forward_with_split_expert_weights,
    "IdeficsEmbedding": idefics_embedding,
    "JambaExperts": batched_experts_forward_with_split_expert_weights,
    "Lfm2MoeExperts": batched_experts_forward_with_split_expert_weights,
    "LongcatFlashExperts": batched_experts_forward_with_split_expert_weights,
    "MiniMaxExperts": batched_experts_forward_with_split_expert_weights,
    "MixtralExperts": batched_experts_forward_with_split_expert_weights,
    "OlmoeExperts": batched_experts_forward_with_split_expert_weights,
    "PhimoeExperts": batched_experts_forward_with_split_expert_weights,
    "Qwen2MoeExperts": batched_experts_forward_with_split_expert_weights,
    "Qwen3MoeExperts": batched_experts_forward_with_split_expert_weights,
    "Qwen3NextExperts": batched_experts_forward_with_split_expert_weights,
    "Qwen3OmniMoeThinkerTextExperts": batched_experts_forward_with_split_expert_weights,
    "SwitchTransformersExperts": batched_switch_transformers_experts_forward,
}


@contextmanager
def patch_model_for_export(model: "PreTrainedModel"):
    # patch masking functions to use the non-vmap versions
    ALL_MASK_ATTENTION_FUNCTIONS["sdpa"] = sdpa_mask_without_vmap
    ALL_MASK_ATTENTION_FUNCTIONS["eager"] = eager_mask_without_vmap

    original_forwards = {}
    for module in model.modules():
        module_class_name = module.__class__.__name__
        if module_class_name in TRANSFORMERS_MODULE_TO_EXPORTABLE_FORWARD:
            original_forwards[module_class_name] = module.forward
            # patch forward method with an exportable version (non data-dependent)
            module.forward = TRANSFORMERS_MODULE_TO_EXPORTABLE_FORWARD[module_class_name].__get__(module)

    # TODO: automate the helper methods patching process as well
    original_functions = {}
    if model.config.model_type == "grounding-dino":
        import transformers.models.grounding_dino.modeling_grounding_dino as grounding_dino_module

        original_functions["generate_masks_with_special_tokens_and_transfer_map"] = (
            grounding_dino_module.generate_masks_with_special_tokens_and_transfer_map
        )
        grounding_dino_module.generate_masks_with_special_tokens_and_transfer_map = (
            generate_masks_with_special_tokens_and_transfer_map
        )
    elif model.config.model_type == "mm-grounding-dino":
        import transformers.models.mm_grounding_dino.modeling_mm_grounding_dino as mm_grounding_dino_module

        original_functions["generate_masks_with_special_tokens_and_transfer_map"] = (
            mm_grounding_dino_module.generate_masks_with_special_tokens_and_transfer_map
        )
        mm_grounding_dino_module.generate_masks_with_special_tokens_and_transfer_map = (
            generate_masks_with_special_tokens_and_transfer_map
        )
    elif model.config.model_type == "tapas":
        import transformers.models.tapas.modeling_tapas as tapas_module

        original_functions["range_index_map"] = tapas_module.range_index_map
        tapas_module.range_index_map = range_index_map

    try:
        yield
    finally:
        # restore original masking functions and module forwards
        ALL_MASK_ATTENTION_FUNCTIONS["sdpa"] = sdpa_mask
        ALL_MASK_ATTENTION_FUNCTIONS["eager"] = eager_mask

        for module in model.modules():
            module_class_name = module.__class__.__name__
            if module_class_name in TRANSFORMERS_MODULE_TO_EXPORTABLE_FORWARD:
                # restore original forward method
                module.forward = original_forwards[module_class_name]

        if model.config.model_type == "grounding-dino":
            import transformers.models.grounding_dino.modeling_grounding_dino as grounding_dino_module

            grounding_dino_module.generate_masks_with_special_tokens_and_transfer_map = original_functions[
                "generate_masks_with_special_tokens_and_transfer_map"
            ]
        elif model.config.model_type == "mm-grounding-dino":
            import transformers.models.mm_grounding_dino.modeling_mm_grounding_dino as mm_grounding_dino_module

            mm_grounding_dino_module.generate_masks_with_special_tokens_and_transfer_map = original_functions[
                "generate_masks_with_special_tokens_and_transfer_map"
            ]
        elif model.config.model_type == "tapas":
            import transformers.models.tapas.modeling_tapas as tapas_module

            tapas_module.range_index_map = original_functions["range_index_map"]


UNSUPPORTED_MODEL_TYPES: set[str] = {
    "clvp",  # many data-dependent branches that add bos/eos tokens
    "colqwen2",  # Uses Qwen2VLModel which uses get_rope_index that is data-dependent
    "emu3",  # Emu3VQVAE.encode is data-dependent
    "encodec",  # torch.export struggles with torch.nn.functional.pad with "reflect" mode
    "esm",  # uses compute_tm function which data-dependent
    "fastspeech2_conformer",  # Even after making parts of it exportable, dynamo still struggle with the convolutions in FastSpeech2ConformerMultiLayeredConv1d
    "fastspeech2_conformer_with_hifigan",  # Even after making parts of it exportable, dynamo still struggle with the convolutions in FastSpeech2ConformerMultiLayeredConv1d
    "funnel",  # torch.export struggles with torch.einsum in FunnelRelMultiheadAttention
    "glm4v",  # Glm4vVisionAttention implementation is highly data-dependent
    "glm4v_moe",  # Glm4vMoeVisionAttention implementation is highly data-dependent
    "hiera",  # torch.export struggles with a reshape operation in HieraEncoder.reroll
    "ibert",  # Uses numpy arrays and decimal.Decimal in batch_frexp
    "lfm2_vl",  # Uses siglip2 which is not exportable
    "lightglue",  # torch.export struggles with sigmoid_log_double_softmax
    "llava_next",  # All three have the same unexplicable error during export
    "llava_next_video",  # All three have the same unexplicable error during export
    "llava_onevision",  # All three have the same unexplicable error during export
    "longformer",  # torch.export is struggling with the global attention implementation
    "mistral3",  # PixtralVisionModel uses some data-dependent truncation
    "modernbert",  # Uses torch.compile directly on some module forward methods
    "nllb-moe",  # TODO: Moe implementation needs to be patched for export
    "omdet-turbo",  #
    "oneformer",  # torch.export is failing on multiple torch methods like torch.linspace and torch.meshgrid
    "pixtral",  # PixtralModel.forward does some data-dependent truncation
    "phi4_multimodal",  # I guess the model is just broken in its current state
    "qwen2_5_omni_thinker",  # already made many parts exportable but still has some non-exportable ops
    "qwen2_5_vl",  # Qwen2_5_VisionTransformerPretrainedModel.get_window_index is data-dependent
    "qwen2_vl",  # Qwen2VLModel.get_rope_index is data-dependent
    "qwen3_omni_moe_thinker",  # Qwen3OmniMoeAudioEncoder.forward does data-dependent chunking
    "qwen3_vl",  # fast_pos_embed_interpolate is data-dependent
    "qwen3_vl_moe",  # fast_pos_embed_interpolate is highly data-dependent
    "siglip2",  # torch.export is failing on torch.nn.functional.interpolate
    "siglip2_vision_model",  # torch.export is failing on torch.nn.functional.interpolate
    "superpoint",  # torch.export is failing on torch.nn.functional.grid_sample
    "video_llama_3",  # VideoLlama3VisionAttention implementation is highly data-dependent
    "video_llama_3_vision",  # VideoLlama3VisionAttention implementation is highly data-dependent
    "vilt",  # torch.export is failing on torch.nn.functional.interpolate
    "xmod",  # XmodOutput.lang_adapter is data-dependent
}


def raise_on_unsupported_model(model: "PreTrainedModel"):
    if model.config.model_type in UNSUPPORTED_MODEL_TYPES:
        raise NotImplementedError(
            f"Dynamo export is not supported for model class '{model.__class__.__name__}' with model_type '{model.config.model_type}'."
        )


UNSUPPORTED_CACHE_CLASS_MODEL_TYPES: set[str] = {
    "falcon_mamba",
    "jamba",
    "lfm2",
    "lfm2_moe",
    "mamba",
    "mamba2",
    "minimax",
    "qwen3_next",
    "reformer",
    "xlstm",
    "zamba2",
}


def warn_on_unsupported_cache_class(model: "PreTrainedModel"):
    if model.config.model_type in UNSUPPORTED_CACHE_CLASS_MODEL_TYPES:
        logger.warning(
            f"Model class '{model.__class__.__name__}' with model_type '{model.config.model_type}' uses a cache class that is not yet fully supported for export. "
            "We will set 'use_cache=False' during export, but some functionalities may be limited."
        )
        model.config.use_cache = False
