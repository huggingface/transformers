# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from typing import Callable, Optional

import torch

from ...cache_utils import Cache
from ...configuration_utils import PretrainedConfig, layer_type_validation
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_rope_utils import rope_config_validation
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import logging
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaForQuestionAnswering,
    LlamaForSequenceClassification,
    LlamaForTokenClassification,
    LlamaPreTrainedModel,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from ..qwen2.modeling_qwen2 import Qwen2Model


logger = logging.get_logger(__name__)


class SmolLM3Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`SmolLM3Model`]. It is used to instantiate a
    SmolLM3 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the SmolLM3 3B.
    e.g. [HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 128256):
            Vocabulary size of the SmolLM3 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`SmolLM3Model`]
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 36):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 4):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `16`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*, defaults to 128004):
            The id of the padding token.
        bos_token_id (`int`, *optional*, defaults to 128000):
            The id of the beginning of sentence token.
        eos_token_id (`int`, *optional*, defaults to 128001):
            The id of the end of sentence token.
        rope_theta (`float`, *optional*, defaults to 2000000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention.
        sliding_window (`int`, *optional*):
            Sliding window attention (SWA) window size. If not specified, will default to `None`.
        no_rope_layers (`List[int]`, *optional*):
            List with at least the same length as the number of layers in the model.
            A `1` at an index position indicates that the corresponding layer will use RoPE,
            while a `0` indicates that it's a NoPE layer.
        no_rope_layer_interval (`int`, *optional*, defaults to 4):
            If `no_rope_layers` is `None`, it will be created using a NoPE layer every
            `no_rope_layer_interval` layers.
        layer_types (`list`, *optional*):
            Attention pattern for each layer. Automatically computed based on sliding window and NoPE settings.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    ```python
    >>> from transformers import SmolLM3Model, SmolLM3Config

    >>> # Initializing a SmolLM3 style configuration
    >>> configuration = SmolLM3Config()

    >>> # Initializing a model from the SmolLM3 style configuration
    >>> model = SmolLM3Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "smollm3"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=128256,
        hidden_size=2048,
        intermediate_size=11008,
        num_hidden_layers=36,
        num_attention_heads=16,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=128004,
        bos_token_id=128000,
        eos_token_id=128001,
        rope_theta=2000000.0,
        rope_scaling=None,
        use_sliding_window=False,
        sliding_window=None,
        no_rope_layers=None,
        no_rope_layer_interval=4,
        layer_types=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.mlp_bias = mlp_bias
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        if no_rope_layers is None:
            self.no_rope_layers = [
                int((layer_idx + 1) % no_rope_layer_interval != 0) for layer_idx in range(num_hidden_layers)
            ]
        else:
            self.no_rope_layers = no_rope_layers

        self.no_rope_layer_interval = no_rope_layer_interval

        # Update layer_types based on sliding window and NoPE pattern
        if layer_types is None:
            layer_types = []
            for layer_idx in range(num_hidden_layers):
                has_rope = self.no_rope_layers[layer_idx]
                if use_sliding_window and sliding_window is not None and not has_rope:
                    layer_types.append("sliding_attention")
                else:
                    layer_types.append("full_attention")

        self.layer_types = layer_types
        layer_type_validation(self.layer_types)

        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)


class SmolLM3Attention(LlamaAttention):
    def __init__(self, config: SmolLM3Config, layer_idx: int):
        super().__init__(config, layer_idx)

        self.use_rope = config.no_rope_layers[layer_idx]
        self.sliding_window = (
            config.sliding_window
            if config.use_sliding_window and config.layer_types[layer_idx] == "sliding_attention"
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if self.use_rope:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class SmolLM3DecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: SmolLM3Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.attention_type = config.layer_types[layer_idx]


class SmolLM3PreTrainedModel(LlamaPreTrainedModel):
    pass


class SmolLM3Model(Qwen2Model):
    pass


class SmolLM3ForCausalLM(LlamaForCausalLM):
    pass


class SmolLM3ForSequenceClassification(LlamaForSequenceClassification):
    pass


class SmolLM3ForTokenClassification(LlamaForTokenClassification):
    pass


class SmolLM3ForQuestionAnswering(LlamaForQuestionAnswering):
    pass


__all__ = [
    "SmolLM3Config",
    "SmolLM3PreTrainedModel",
    "SmolLM3Model",
    "SmolLM3ForCausalLM",
    "SmolLM3ForSequenceClassification",
    "SmolLM3ForTokenClassification",
    "SmolLM3ForQuestionAnswering",
]
