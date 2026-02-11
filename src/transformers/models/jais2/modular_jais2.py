# coding=utf-8
# Copyright 2025 the HuggingFace Team. All rights reserved.
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

from typing import Optional

import torch.nn as nn

from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring, can_return_tuple
from ..llama.configuration_llama import LlamaConfig
from ..llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    LlamaPreTrainedModel,
)
from ..nemotron.modeling_nemotron import NemotronMLP


class Jais2Config(LlamaConfig):
    r"""
    This is the configuration class to store the configuration of a [`Jais2Model`]. It is used to instantiate a Jais2
    model according to the specified arguments, defining the model architecture.
    [inceptionai/Jais-2-8B-Chat](https://huggingface.co/inceptionai/Jais-2-8B-Chat).

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 150272):
            Vocabulary size of the Jais2 model.
        hidden_size (`int`, *optional*, defaults to 3328):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 26624):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 26):
            Number of attention heads for each attention layer.
        num_key_value_heads (`int`, *optional*):
            Number of key_value heads for Grouped Query Attention.
        hidden_act (`str`, *optional*, defaults to `"relu2"`):
            The non-linear activation function in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 8192):
            The maximum sequence length.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to return last key/values attentions.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 0):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 150024):
            End of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings.
        attention_bias (`bool`, *optional*, defaults to `True`):
            Whether to use a bias in the query, key, value and output projection layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        mlp_bias (`bool`, *optional*, defaults to `True`):
            Whether to use a bias in up_proj, down_proj and gate_proj layers.
        head_dim (`int`, *optional*):
            The attention head dimension.
        rope_parameters (`dict`, *optional*):
            The RoPE parameters.
    """

    model_type = "jais2"

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    def __init__(
        self,
        vocab_size: Optional[int] = 150272,
        hidden_size: Optional[int] = 3328,
        intermediate_size: Optional[int] = 26624,
        num_hidden_layers: Optional[int] = 32,
        num_attention_heads: Optional[int] = 26,
        num_key_value_heads: Optional[int] = None,
        hidden_act: Optional[str] = "relu2",
        max_position_embeddings: Optional[int] = 8192,
        initializer_range: Optional[float] = 0.02,
        layer_norm_eps: Optional[float] = 1e-5,
        use_cache: Optional[bool] = True,
        pad_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = 0,
        eos_token_id: Optional[int] = 150024,
        tie_word_embeddings: Optional[bool] = False,
        attention_bias: Optional[bool] = True,
        attention_dropout: Optional[float] = 0.0,
        mlp_bias: Optional[bool] = True,
        head_dim: Optional[int] = None,
        rope_parameters: Optional[RopeParameters | dict[str, RopeParameters]] = None,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            mlp_bias=mlp_bias,
            head_dim=head_dim,
            rope_parameters=rope_parameters,
            **kwargs,
        )
        self.layer_norm_eps = layer_norm_eps
        del self.rms_norm_eps
        del self.pretraining_tp


class Jais2MLP(NemotronMLP):
    pass


class Jais2DecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: Jais2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)


class Jais2PreTrainedModel(LlamaPreTrainedModel):
    pass


class Jais2Model(LlamaModel):
    def __init__(self, config: Jais2Config):
        super().__init__(config)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)


class Jais2ForCausalLM(LlamaForCausalLM):
    @can_return_tuple
    @auto_docstring
    def forward(self, **super_kwargs):
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, Jais2ForCausalLM

        >>> model = Jais2ForCausalLM.from_pretrained("inceptionai/Jais-2-8B-Chat")
        >>> tokenizer = AutoTokenizer.from_pretrained("inceptionai/Jais-2-8B-Chat")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        return super().forward(**super_kwargs)


__all__ = [
    "Jais2Config",
    "Jais2Model",
    "Jais2ForCausalLM",
    "Jais2PreTrainedModel",
]
