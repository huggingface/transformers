from torch import nn

from ...utils import logging
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaForCausalLM,
    LlamaForQuestionAnswering,
    LlamaForSequenceClassification,
    LlamaForTokenClassification,
    LlamaMLP,
    LlamaModel,
    LlamaPreTrainedModel,
)
from .configuration_telechat2 import TeleChat2Config


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "TeleAI/TeleChat2-3B"
_CONFIG_FOR_DOC = "TeleChat2Config"


class TeleChat2MLP(LlamaMLP):
    def __init__(self, config):
        super().__init__(config)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)


class TeleChat2Attention(LlamaAttention):
    def __init__(self, config: TeleChat2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
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
