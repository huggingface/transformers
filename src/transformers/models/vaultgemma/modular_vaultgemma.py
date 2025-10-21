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

import torch

from ...cache_utils import Cache
from ...modeling_rope_utils import RopeParameters
from ..gemma2.configuration_gemma2 import Gemma2Config
from ..gemma2.modeling_gemma2 import Gemma2Attention, Gemma2DecoderLayer, Gemma2ForCausalLM, Gemma2MLP, Gemma2RMSNorm


class VaultGemmaConfig(Gemma2Config):
    r"""
    This is the configuration class to store the configuration of a [`VaultGemmaModel`]. It is used to instantiate an VaultGemma
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the VaultGemma-7B.
    e.g. [google/vaultgemma-7b](https://huggingface.co/google/vaultgemma-7b)
    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 256000):
            Vocabulary size of the VaultGemma model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`VaultGemmaModel`]
        hidden_size (`int`, *optional*, defaults to 2304):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 9216):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 26):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 4):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to
            `num_attention_heads`.
        head_dim (`int`, *optional*, defaults to 256):
            The attention head dimension.
        hidden_activation (`str` or `function`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            The non-linear activation function (function or string) in the decoder. Will default to `"gelu_pytorch_tanh"`
            if not specified. `"gelu_pytorch_tanh"` uses an approximation of the `"gelu"` activation function.
        max_position_embeddings (`int`, *optional*, defaults to 8192):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        eos_token_id (`int`, *optional*, defaults to 1):
            End of stream token id.
        bos_token_id (`int`, *optional*, defaults to 2):
            Beginning of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to tie weight embeddings
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionaty should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        query_pre_attn_scalar (`float`, *optional*, defaults to 256):
            scaling factor used on the attention scores
        sliding_window (`int`, *optional*, defaults to 4096):
            in VaultGemma, every other layer uses sliding window attention. This is the size of the sliding window.
        layer_types (`list`, *optional*):
            Attention pattern for each layer.
        final_logit_softcapping (`float`, *optional*, defaults to 30.0):
            scaling factor when applying tanh softcapping on the logits.
        attn_logit_softcapping (`float`, *optional*, defaults to 50.0):
            scaling factor when applying tanh softcapping on the attention scores.

    ```python
    >>> from transformers import VaultGemmaModel, VaultGemmaConfig
    >>> # Initializing a VaultGemma vaultgemma-7b style configuration
    >>> configuration = VaultGemmaConfig()
    >>> # Initializing a model from the vaultgemma-7b style configuration
    >>> model = VaultGemmaModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    def __init__(
        self,
        vocab_size: Optional[int] = 256000,
        hidden_size: Optional[int] = 2304,
        intermediate_size: Optional[int] = 9216,
        num_hidden_layers: Optional[int] = 26,
        num_attention_heads: Optional[int] = 8,
        num_key_value_heads: Optional[int] = 4,
        head_dim: Optional[int] = 256,
        hidden_activation: Optional[str] = "gelu_pytorch_tanh",
        max_position_embeddings: Optional[int] = 8192,
        initializer_range: Optional[float] = 0.02,
        rms_norm_eps: Optional[int] = 1e-6,
        use_cache: Optional[bool] = True,
        pad_token_id: Optional[int] = 0,
        eos_token_id: Optional[int] = 1,
        bos_token_id: Optional[int] = 2,
        tie_word_embeddings: Optional[bool] = True,
        rope_parameters: Optional[RopeParameters | dict[RopeParameters]] = None,
        attention_bias: Optional[bool] = False,
        attention_dropout: Optional[float] = 0.0,
        query_pre_attn_scalar: Optional[int] = 256,
        sliding_window: Optional[int] = 4096,
        layer_types: Optional[list[str]] = None,
        final_logit_softcapping: Optional[float] = 30.0,
        attn_logit_softcapping: Optional[float] = 50.0,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            hidden_activation=hidden_activation,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            rope_parameters=rope_parameters,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            query_pre_attn_scalar=query_pre_attn_scalar,
            sliding_window=sliding_window,
            layer_types=layer_types,
            final_logit_softcapping=final_logit_softcapping,
            attn_logit_softcapping=attn_logit_softcapping,
            **kwargs,
        )

        del self.use_bidirectional_attention


class VaultGemmaRMSNorm(Gemma2RMSNorm):
    pass


class VaultGemmaMLP(Gemma2MLP):
    pass


class VaultGemmaAttention(Gemma2Attention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: VaultGemmaConfig, layer_idx: int):
        super().__init__()
        self.is_causal = True


class VaultGemmaDecoderLayer(Gemma2DecoderLayer):
    def __init__(self, **super_kwargs):
        super().__init__(**super_kwargs)
        del self.post_attention_layernorm
        del self.post_feedforward_layernorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class VaultGemmaForCausalLM(Gemma2ForCausalLM):
    pass


__all__ = [
    "VaultGemmaConfig",
    "VaultGemmaForCausalLM",
    "VaultGemmaModel",  # noqa: F822
    "VaultGemmaPreTrainedModel",  # noqa: F822
]
