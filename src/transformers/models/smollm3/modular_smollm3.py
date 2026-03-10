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

from collections.abc import Callable

import torch

from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig, layer_type_validation
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import auto_docstring, logging
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
from ..qwen2.modeling_qwen2 import Qwen2Model, Qwen2RotaryEmbedding


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="HuggingFaceTB/SmolLM3-3B")
class SmolLM3Config(PreTrainedConfig):
    r"""
    no_rope_layers (`List[int]`, *optional*):
        List with at least the same length as the number of layers in the model.
        A `1` at an index position indicates that the corresponding layer will use RoPE,
        while a `0` indicates that it's a NoPE layer.
    no_rope_layer_interval (`int`, *optional*, defaults to 4):
        If `no_rope_layers` is `None`, it will be created using a NoPE layer every
        `no_rope_layer_interval` layers.


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
    default_theta = 2000000.0

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
        vocab_size: int | None = 128256,
        hidden_size: int | None = 2048,
        intermediate_size: int | None = 11008,
        num_hidden_layers: int | None = 36,
        num_attention_heads: int | None = 16,
        num_key_value_heads: int | None = 4,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 32768,
        initializer_range: float | None = 0.02,
        rms_norm_eps: int | None = 1e-6,
        use_cache: bool | None = True,
        pad_token_id: int | None = 128004,
        bos_token_id: int | None = 128000,
        eos_token_id: int | None = 128001,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        use_sliding_window: bool | None = False,
        sliding_window: int | None = None,
        no_rope_layers: int | None = None,
        no_rope_layer_interval: int | None = 4,
        layer_types: int | None = None,
        attention_bias: bool | None = False,
        attention_dropout: float | None = 0.0,
        mlp_bias: bool | None = False,
        tie_word_embeddings: bool | None = True,
        **kwargs,
    ):
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
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
        layer_type_validation(self.layer_types, self.num_hidden_layers)

        self.rope_parameters = rope_parameters
        super().__init__(**kwargs)


class SmolLM3RotaryEmbedding(Qwen2RotaryEmbedding):
    pass


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
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if self.use_rope:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

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
