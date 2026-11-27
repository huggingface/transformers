# Copyright 2026 The Jina-AI and HuggingFace Inc. teams. All rights reserved.
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
from huggingface_hub.dataclasses import strict

from ...cache_utils import Cache, DynamicCache
from ...masking_utils import (
    create_bidirectional_mask,
    create_bidirectional_sliding_window_mask,
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..qwen3.configuration_qwen3 import Qwen3Config
from ..qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3MLP,
    Qwen3Model,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)


@auto_docstring(checkpoint="jinaai/jina-embeddings-v5-text-small")
@strict
class JinaEmbeddingsV5Config(Qwen3Config):
    r"""
    is_decoder (`bool`, *optional*, defaults to `True`):
        Whether the model uses the causal decoder backbone with QK-norm (`jina-embeddings-v5-text-small`, based on
        Qwen3) or the bidirectional encoder backbone without QK-norm (`jina-embeddings-v5-text-nano`, based on
        EuroBERT).
    is_causal (`bool`, *optional*):
        Whether attention is causal. Defaults to `is_decoder` when unset.
    mask_token_id (`int`, *optional*):
        Mask token id used by encoder checkpoints such as `jina-embeddings-v5-text-nano`.
    task_names (`list[str]`, *optional*):
        Names of the task-specific LoRA adapters shipped alongside the checkpoint (for example `["retrieval",
        "text-matching", "clustering", "classification"]`).

    Example:

    ```python
    >>> from transformers import JinaEmbeddingsV5Config, JinaEmbeddingsV5Model

    >>> # Initializing a Jina Embeddings V5 small (decoder) style configuration
    >>> configuration = JinaEmbeddingsV5Config()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = JinaEmbeddingsV5Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "jina_embeddings_v5"
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int | None = 8
    head_dim: int | None = None
    tie_word_embeddings: bool = True
    sliding_window: int | None = None
    bos_token_id: int | None = 151643
    eos_token_id: int | list[int] | None = 151645
    is_decoder: bool = True
    is_causal: bool | None = None
    mask_token_id: int | None = None
    task_names: list[str] | None = None

    def __post_init__(self, **kwargs):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.is_causal is None:
            self.is_causal = self.is_decoder
        super().__post_init__(**kwargs)


class JinaEmbeddingsV5RMSNorm(Qwen3RMSNorm):
    pass


class JinaEmbeddingsV5MLP(Qwen3MLP):
    pass


class JinaEmbeddingsV5RotaryEmbedding(Qwen3RotaryEmbedding):
    pass


class JinaEmbeddingsV5Attention(Qwen3Attention):
    def __init__(self, config: JinaEmbeddingsV5Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.is_causal = bool(config.is_causal)
        del self.q_norm
        del self.k_norm
        if config.is_decoder:
            self.q_norm = JinaEmbeddingsV5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = JinaEmbeddingsV5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        cache: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        if self.q_norm is not None:
            query_states = self.q_norm(query_states)
        if self.k_norm is not None:
            key_states = self.k_norm(key_states)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if cache is not None:
            key_states, value_states = cache.update(key_states, value_states, self.layer_idx, kwargs=kwargs)

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


class JinaEmbeddingsV5DecoderLayer(Qwen3DecoderLayer):
    pass


class JinaEmbeddingsV5PreTrainedModel(Qwen3PreTrainedModel):
    pass


@auto_docstring
class JinaEmbeddingsV5Model(Qwen3Model):
    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        is_causal = bool(self.config.is_causal)
        use_cache = bool(use_cache and is_causal)
        if not is_causal:
            cache = None
        elif use_cache and cache is None:
            cache = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = cache.get_seq_length() if cache is not None else 0
            position_ids = (
                torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            ).unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
            }
            if is_causal:
                mask_kwargs["cache"] = cache
                mask_kwargs["position_ids"] = position_ids
                causal_mask_mapping = {
                    "full_attention": create_causal_mask(**mask_kwargs),
                }
                if self.has_sliding_layers:
                    causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
            else:
                causal_mask_mapping = {
                    "full_attention": create_bidirectional_mask(**mask_kwargs),
                }
                if self.has_sliding_layers:
                    causal_mask_mapping["sliding_attention"] = create_bidirectional_sliding_window_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                cache=cache,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            cache=cache if use_cache else None,
        )


__all__ = [
    "JinaEmbeddingsV5Config",
    "JinaEmbeddingsV5PreTrainedModel",
    "JinaEmbeddingsV5Model",
]
