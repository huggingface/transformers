import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import logging
from ..llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaMLP,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
    rotate_half,
)
from ..qwen3.modeling_qwen3 import Qwen3Attention, Qwen3RotaryEmbedding
from ..deepseek_v3.modeling_deepseek_v3 import DeepseekV3MLP, DeepseekV3TopkRouter
from ..deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE

from .configuration_dots1 import Dots1Config


logger = logging.get_logger(__name__)


class Dots1RMSNorm(LlamaRMSNorm):
    pass


class Dots1RotaryEmbedding(Qwen3RotaryEmbedding):
    pass


class Dots1Attention(Qwen3Attention):
    pass


class Dots1MLP(DeepseekV3MLP):
    pass


class Dots1MoE(DeepseekV3MoE):
    pass


class Dots1TopkRouter(DeepseekV3TopkRouter):
    pass


class Dots1DecoderLayer(LlamaDecoderLayer, nn.Module):
    def __init__(self, config: Dots1Config, layer_idx: int):
        nn.Module().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Dots1Attention(config=config, layer_idx=layer_idx)

        if layer_idx >= config.first_k_dense_replace:
            self.mlp = Dots1MoE(config)
        else:
            self.mlp = Dots1MLP(config)

        self.input_layernorm = Dots1RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Dots1RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Dots1PreTrainedModel(LlamaPreTrainedModel):
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, Dots1RMSNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, Dots1TopkRouter):
            module.weight.data.normal_(mean=0.0, std=std)


class Dots1Model(LlamaModel):
    pass

class Dots1ForCausalLM(LlamaForCausalLM):
    pass


__all__ = [
    "Dots1PreTrainedModel",
    "Dots1Model",
    "Dots1ForCausalLM",
]
