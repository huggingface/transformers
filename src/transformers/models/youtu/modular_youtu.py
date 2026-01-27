import torch
from torch import nn

from ... import initialization as init
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from ..deepseek_v3.modeling_deepseek_v3 import DeepseekV3Attention
from ..llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from ..qwen3.modeling_qwen3 import Qwen3MLP


logger = logging.get_logger(__name__)


class YoutuRMSNorm(LlamaRMSNorm):
    pass


class YoutuRotaryEmbedding(LlamaRotaryEmbedding):
    pass


class YoutuMLP(Qwen3MLP):
    pass


class YoutuAttention(DeepseekV3Attention):
    pass


class YoutuDecoderLayer(LlamaDecoderLayer):
    pass


class YoutuPreTrainedModel(LlamaPreTrainedModel, PreTrainedModel):
    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        std = getattr(self.config, "initializer_range", 0.02)
        embed_std = getattr(self.config, "embedding_initializer_range", 2 * std)
        if isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=embed_std)
            if module.padding_idx is not None:
                init.zeros_(module.weight.data[module.padding_idx])


class YoutuModel(LlamaModel):
    pass


class YoutuForCausalLM(LlamaForCausalLM):
    pass


__all__ = [
    "YoutuPreTrainedModel",
    "YoutuModel",
    "YoutuForCausalLM",
]
