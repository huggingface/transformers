# Copyright (c) 2025 Baidu, Inc. and HuggingFace Inc. team. All Rights Reserved.
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
"""PyTorch Ernie 4.5 model"""

from torch import nn

from ...utils import auto_docstring, can_return_tuple
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from .configuration_ernie4_5 import Ernie4_5Config


@auto_docstring
class Ernie4_5PreTrainedModel(LlamaPreTrainedModel):
    config_class = Ernie4_5Config
    _no_split_modules = ["Ernie4_5DecoderLayer"]


class Ernie4_5RMSNorm(LlamaRMSNorm):
    pass


class Ernie4_5MLP(LlamaMLP):
    def __init__(self, config: Ernie4_5Config):
        super().__init__()
        del self.gate_proj
        del self.up_proj
        del self.down_proj

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.use_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.use_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)


class Ernie4_5RotaryEmbedding(LlamaRotaryEmbedding):
    pass


class Ernie4_5Attention(LlamaAttention):
    def __init__(self, config: Ernie4_5Config, layer_idx: int):
        super().__init__(config, layer_idx)
        del self.q_proj
        del self.k_proj
        del self.v_proj
        del self.o_proj
        del self.attention_dropout

        self.attention_dropout = 0.0

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.use_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.use_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.use_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.use_bias)


class Ernie4_5DecoderLayer(LlamaDecoderLayer, nn.Module):
    def __init__(self, config: Ernie4_5Config, layer_idx: int):
        nn.Module().__init__()

        self.hidden_size = config.hidden_size

        self.self_attn = Ernie4_5Attention(config=config, layer_idx=layer_idx)

        self.mlp = Ernie4_5MLP(config)
        self.input_layernorm = Ernie4_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Ernie4_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


@auto_docstring
class Ernie4_5Model(LlamaModel, Ernie4_5PreTrainedModel, nn.Module):
    def __init__(self, config: Ernie4_5Config):
        nn.Module().__init__()

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Ernie4_5DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Ernie4_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Ernie4_5RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()


class Ernie4_5ForCausalLM(LlamaForCausalLM, Ernie4_5PreTrainedModel, nn.Module):
    def __init__(self, config: Ernie4_5Config):
        nn.Module().__init__()

        self.model = Ernie4_5Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(self, **super_kwargs):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        super().forward(**super_kwargs)


__all__ = ["Ernie4_5ForCausalLM", "Ernie4_5Model", "Ernie4_5PreTrainedModel"]
