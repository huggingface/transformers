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


def batched_experts_gemm(self, input: torch.Tensor, tokens_per_expert: torch.Tensor) -> torch.Tensor:
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


def embedding_without_dynamic_slicing_with_tensor(self, x: torch.Tensor, seq_len: torch.Tensor):
    seq_len = seq_len.item()
    torch._check(seq_len > 0)
    torch._check(seq_len <= max(self.cos_cached.shape[0], self.sin_cached.shape[0]))

    return (
        self.cos_cached[:seq_len].to(device=x.device, dtype=x.dtype),
        self.sin_cached[:seq_len].to(device=x.device, dtype=x.dtype),
    )


TRANSFORMERS_MODULE_TO_EXPORTABLE_FORWARD: dict[str, Callable] = {
    # Expert MLPs with different weight storage schemes
    "AriaGroupedExpertsGemm": batched_experts_gemm,
    "DbrxExperts": batched_experts_forward_with_grouped_expert_weights,
    "DeepseekV2Experts": batched_experts_forward_with_split_expert_weights,
    "DeepseekV3NaiveMoe": batched_experts_forward_with_split_expert_weights,
    "Dots1NaiveMoe": batched_experts_forward_with_split_expert_weights,
    "Ernie4_5_MoeExperts": batched_experts_forward_with_split_expert_weights,
    "FlexOlmoExperts": batched_experts_forward_with_split_expert_weights,
    "Glm4MoeNaiveMoe": batched_experts_forward_with_split_expert_weights,
    "Glm4vMoeTextNaiveMoe": batched_experts_forward_with_split_expert_weights,
    "HunYuanMoEV1Experts": batched_experts_forward_with_split_expert_weights,
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
    # Embedding modules with dynamic slicing
    "IdeficsEmbedding": embedding_without_dynamic_slicing_with_tensor,
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


UNSUPPORTED_MODEL_TYPES: set[str] = {
    "clvp",  # if isin_mps_friendly(each_input_id, pad_token_id).sum():
    "colqwen2",  # input_tokens = input_ids.tolist()
    "emu3",
    "encodec",
    "esm",
    "falcon_mamba",  # Uses FalconMambaCache which is a custom cache type that's not yet registered as a pytree node
    "fastspeech2_conformer",
    "fastspeech2_conformer_with_hifigan",
    "flava",
    "funnel",
    "glm4v",
    "glm4v_moe",
    "grounding-dino",
    "hiera",
    "ibert",
    "jamba",
    "led",
    "lfm2",
    "lfm2_moe",
    "lfm2_vl",
    "lightglue",
    "llava_next",
    "llava_next_video",
    "llava_onevision",
    "longformer",
    "mamba",  # Uses MambaCache which is a custom cache type that's not yet registered as a pytree node
    "mamba2",
    "mimi",
    "minimax",  # Uses MiniMaxCache which is a custom cache type that's not yet registered as a pytree node
    "mistral3",
    "mm-grounding-dino",
    "modernbert",
    "nllb-moe",
    "omdet-turbo",
    "oneformer",
    "perception_lm",
    "phi4_multimodal",
    "pixtral",
    "qwen2_5_omni",
    "qwen2_5_omni_thinker",
    "qwen2_5_vl",
    "qwen2_vl",
    "qwen3_next",  # Uses Qwen3NextDynamicCache which is a custom cache type that's not yet registered as a pytree node
    "qwen3_omni_moe",
    "qwen3_omni_moe_thinker",
    "qwen3_vl",
    "qwen3_vl_moe",
    "reformer",
    "siglip2",
    "siglip2_vision_model",
    "splinter",
    "superglue",
    "superpoint",
    "switch_transformers",
    "tapas",
    "video_llama_3",
    "video_llama_3_vision",
    "video_llava",
    "videomae",
    "vilt",
    "xlstm",  # Uses xLSTMCache which is a custom cache type that's not yet registered as a pytree node
    "xmod",
    "zamba2",
}


def raise_on_unsupported_model(model: "PreTrainedModel"):
    if model.config.model_type in UNSUPPORTED_MODEL_TYPES:
        raise NotImplementedError(
            f"Dynamo export is not supported for model class '{model.__class__.__name__}' with model_type '{model.config.model_type}'."
        )
