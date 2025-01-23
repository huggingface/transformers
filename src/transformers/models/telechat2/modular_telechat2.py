from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import logging
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaForCausalLM,
    LlamaForQuestionAnswering,
    LlamaForSequenceClassification,
    LlamaForTokenClassification,
    LlamaModel,
    LlamaPreTrainedModel,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from .configuration_telechat2 import TeleChat2Config


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "TeleAI/TeleChat2-3B"
_CONFIG_FOR_DOC = "TeleChat2Config"


class TeleChat2MLP(nn.Module):
    def __init__(self, config: TeleChat2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.hidden_dropout = config.hidden_dropout

    def forward(self, hidden_states):
        intermediate_output = self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
        output = F.dropout(intermediate_output, p=self.hidden_dropout, training=self.training)
        return output


class TeleChat2Attention(LlamaAttention):
    def __init__(self, config: TeleChat2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=True)


class TeleChat2PreTrainedModel(LlamaPreTrainedModel):
    pass


class TeleChat2Model(LlamaModel):
    pass


class TeleChat2ForCausalLM(LlamaForCausalLM):
    pass


class TeleChat2ForSequenceClassification(LlamaForSequenceClassification):
    pass


class TeleChat2ForQuestionAnswering(LlamaForQuestionAnswering):
    pass


class TeleChat2ForTokenClassification(LlamaForTokenClassification):
    pass


__all__ = [
    "TeleChat2ForCausalLM",
    "TeleChat2ForQuestionAnswering",
    "TeleChat2ForSequenceClassification",
    "TeleChat2ForTokenClassification",
    "TeleChat2Model",
    "TeleChat2PreTrainedModel",
]
