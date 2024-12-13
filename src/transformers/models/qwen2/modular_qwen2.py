from typing import Callable, List, Optional, Tuple, Union, Unpack

import torch
from torch import nn
import torch.utils.checkpoint

from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...cache_utils import Cache, SlidingWindowCache, StaticCache
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...utils import logging
from ..llama.modeling_llama import (
    LlamaForQuestionAnswering,
    LlamaForSequenceClassification,
    LlamaForTokenClassification,
    LlamaModel,
    LlamaForCausalLM,
    LlamaAttention,
    LlamaDecoderLayer,
    eager_attention_forward,
    apply_rotary_pos_emb,
)
from .configuration_qwen2 import Qwen2Config


logger = logging.get_logger(__name__)

class Qwen2Attention(LlamaAttention):

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        sliding_window = None
        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            sliding_window=sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    
class Qwen2DecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        if config.sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )

class Qwen2Model(LlamaModel):
    pass

class Qwen2ForCausalLM(LlamaForCausalLM):
    pass

class Qwen2ForSequenceClassification(LlamaForSequenceClassification):
    pass

class Qwen2ForTokenClassification(LlamaForTokenClassification):
    pass

class Qwen2ForQuestionAnswering(LlamaForQuestionAnswering):
    pass