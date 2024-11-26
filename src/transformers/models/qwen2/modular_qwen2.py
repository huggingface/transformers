from typing import Optional, Tuple

import torch
import torch.nn as nn

from ...cache_utils import Cache

from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaFlashAttention2,
    LlamaForCausalLM,
    LlamaModel,
    LlamaSdpaAttention,
)
from .configuration_qwen2 import Qwen2Config


class Qwen2Attention(LlamaAttention):
    pass


class Qwen2SdpaAttention(LlamaSdpaAttention):
    pass


class Qwen2FlashAttention2(LlamaFlashAttention2):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ):
        # Decide whether to use SWA or not by layer index.
        if self.layer_idx >= self.config.max_window_layers:
            self.sliding_window = None
        return LlamaFlashAttention2.forward(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            cache_position,
            position_embeddings,
            **kwargs,
        )


QWEN2_ATTENTION_CLASSES = {
    "eager": Qwen2Attention,
    "flash_attention_2": Qwen2FlashAttention2,
    "sdpa": Qwen2SdpaAttention,
}


class Qwen2DecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.self_attn = QWEN2_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)


class Qwen2Model(LlamaModel):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )


class Qwen2ForCausalLM(LlamaForCausalLM):
    pass
