# Copyright 2026 Shanghai AI Laboratory and the HuggingFace Inc. team. All rights reserved.
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

from huggingface_hub.dataclasses import strict
from torch import nn

from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring
from ..llama.configuration_llama import LlamaConfig
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaModel,
    LlamaPreTrainedModel,
)


@auto_docstring(checkpoint="internlm/internlm3-8b-instruct")
@strict
class InternLM3Config(LlamaConfig):
    r"""
    qkv_bias (`bool`, *optional*, defaults to `False`):
        Whether to use a bias in the query, key and value projection layers.
    bias (`bool`, *optional*, defaults to `False`):
        Whether to use a bias in the attention output projection and MLP layers.

    ```python
    >>> from transformers import InternLM3Model, InternLM3Config

    >>> # Initializing an InternLM3-8B style configuration
    >>> configuration = InternLM3Config()

    >>> # Initializing a model from the InternLM3-8B style configuration
    >>> model = InternLM3Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "internlm3"

    vocab_size: int = 128512
    hidden_size: int = 4096
    intermediate_size: int = 10240
    num_hidden_layers: int = 48
    num_attention_heads: int = 32
    num_key_value_heads: int | None = 2
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int | None = 2
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    attention_dropout: float | int = 0.0
    head_dim: int | None = 128
    qkv_bias: bool = False
    bias: bool = False

    attention_bias = AttributeError()
    mlp_bias = AttributeError()
    pretraining_tp = AttributeError()


class InternLM3MLP(LlamaMLP):
    def __init__(self, config: InternLM3Config):
        super().__init__(config)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.bias)


class InternLM3Attention(LlamaAttention):
    def __init__(self, config: InternLM3Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.qkv_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.qkv_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.qkv_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.bias)


class InternLM3DecoderLayer(LlamaDecoderLayer):
    pass


class InternLM3PreTrainedModel(LlamaPreTrainedModel):
    pass


class InternLM3Model(LlamaModel):
    pass


@auto_docstring(checkpoint="internlm/internlm3-8b-instruct")
class InternLM3ForCausalLM(LlamaForCausalLM):
    pass


__all__ = [
    "InternLM3Config",
    "InternLM3ForCausalLM",
    "InternLM3Model",
    "InternLM3PreTrainedModel",
]
