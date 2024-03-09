import inspect
from typing import Tuple

import torch

from ...config import GPTMegatronConfig
from ...enums import AttentionHeadType, AttentionImplementation, PositionEmbeddingType
from .base import Attention
from .flash import FlashAttention
from .math import MathAttention
from .padding_free import PackedFlashAttention
from .sdpa import SDPA
from .utils import (
    interleave_query_key_value_tensor_for_gqa,
    interleave_query_key_value_tensor_for_mha,
    interleave_query_key_value_tensor_for_mqa,
    split_query_key_value_tensor_for_gqa,
    split_query_key_value_tensor_for_mha,
    split_query_key_value_tensor_for_mqa,
)


_ATTENTION_MODULES = {
    AttentionImplementation.eager.value: MathAttention,
    AttentionImplementation.math.value: MathAttention,
    AttentionImplementation.sdpa.value: SDPA,
    AttentionImplementation.flash.value: FlashAttention,
    AttentionImplementation.padding_free.value: PackedFlashAttention,
}


_INTERLEAVE_FUNCTIONS = {
    AttentionHeadType.mha.value: interleave_query_key_value_tensor_for_mha,
    AttentionHeadType.mqa.value: interleave_query_key_value_tensor_for_mqa,
    AttentionHeadType.gqa.value: interleave_query_key_value_tensor_for_gqa,
}


_SPLIT_FUNCTIONS = {
    AttentionHeadType.mha.value: split_query_key_value_tensor_for_mha,
    AttentionHeadType.mqa.value: split_query_key_value_tensor_for_mqa,
    AttentionHeadType.gqa.value: split_query_key_value_tensor_for_gqa,
}


def get_attention_module(
    config: GPTMegatronConfig,
    causal: bool,
    layer_idx: int,
    attention_implementation: AttentionImplementation = AttentionImplementation.sdpa,
) -> Attention:
    attention_implementaton = AttentionImplementation(attention_implementation)

    if attention_implementaton.value in _ATTENTION_MODULES:
        return _ATTENTION_MODULES[attention_implementaton.value](
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            attention_head_type=AttentionHeadType(config.attention_head_type),
            position_embedding_type=PositionEmbeddingType(config.position_embedding_type),
            causal=causal,
            add_bias=config.add_bias,
            scale_attention_weights=config.scale_attn_weights,
            attention_softmax_in_fp32=config.attention_softmax_in_fp32,
            scale_attention_softmax_in_fp32=config.scale_attention_softmax_in_fp32,
            attn_pdrop=config.attn_pdrop,
            resid_pdrop=config.resid_pdrop,
            layer_idx=layer_idx,
        )

    raise ValueError(f"unexpected `attention_implementation` {attention_implementaton}")


def interleave_query_key_value_tensor_for_attention(
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
    num_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    attention_head_type: AttentionHeadType,
) -> torch.Tensor:
    if attention_head_type.value in _INTERLEAVE_FUNCTIONS:
        interleave_function = _INTERLEAVE_FUNCTIONS[attention_head_type.value]
        interleave_function_parameters = inspect.signature(interleave_function).parameters.keys()

        parameters_to_pass = {}
        this_function_parameters = locals()
        for parameter in interleave_function_parameters:
            parameters_to_pass[parameter] = this_function_parameters[parameter]

        query_key_value_weight = interleave_function(**parameters_to_pass)

        return query_key_value_weight

    raise ValueError(f"unexpected `attention_head_type` {attention_head_type}")


def split_query_key_value_tensor_for_attention(
    query_key_value_weight: torch.Tensor,
    num_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    attention_head_type: AttentionHeadType,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if attention_head_type.value in _SPLIT_FUNCTIONS:
        split_function = _SPLIT_FUNCTIONS[attention_head_type.value]
        split_function_parameters = inspect.signature(split_function).parameters.keys()

        parameters_to_pass = {}
        this_function_parameters = locals()
        for parameter in split_function_parameters:
            parameters_to_pass[parameter] = this_function_parameters[parameter]

        return split_function(**parameters_to_pass)

    raise ValueError(f"unexpected `attention_head_type` {attention_head_type}")
