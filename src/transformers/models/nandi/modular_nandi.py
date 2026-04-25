# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import torch
import torch.nn as nn
from huggingface_hub.dataclasses import strict

from ...cache_utils import Cache, DynamicCache, DynamicLayer
from ...configuration_utils import PretrainedConfig
from ...masking_utils import create_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_rope_utils import RopeParameters
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import can_return_tuple, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
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


@strict(accept_kwargs=True)
class NandiConfig(PretrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import NandiConfig, NandiForCausalLM

    >>> # Initializing a Nandi style configuration
    >>> configuration = NandiConfig()

    >>> # Initializing a model from the Nandi style configuration
    >>> model = NandiForCausalLM(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "nandi"
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

    # Defaults from the provided Nanotron training config.
    vocab_size: int = 131072
    hidden_size: int = 832
    intermediate_size: int = 2496
    num_hidden_layers: int = 16
    num_attention_heads: int = 16
    num_key_value_heads: int | None = 4
    head_dim: int | None = None
    hidden_act: str = "silu"
    max_position_embeddings: int = 2048
    initializer_range: float = 0.008
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 0
    pretraining_tp: int | None = 1
    tie_word_embeddings: bool = True
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    mlp_bias: bool = False

    # Nandi-specific options.
    factorized_embedding: bool = True
    embedding_rank: int = 196
    layer_sharing: bool = True
    layer_sharing_repeats: int = 2

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.rope_parameters is None:
            self.rope_parameters = {"rope_type": "default", "rope_theta": 100000.0}
        if not self.layer_sharing:
            self.layer_sharing_repeats = 1

        if self.factorized_embedding and self.embedding_rank <= 0:
            raise ValueError(
                f"`embedding_rank` must be positive when `factorized_embedding=True`, got {self.embedding_rank}."
            )
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"`hidden_size` ({self.hidden_size}) must be divisible by `num_attention_heads` ({self.num_attention_heads})."
            )
        if self.layer_sharing_repeats < 1:
            raise ValueError(f"`layer_sharing_repeats` must be >= 1, got {self.layer_sharing_repeats}.")

        super().__post_init__(**kwargs)


class NandiRMSNorm(LlamaRMSNorm):
    pass


class NandiRotaryEmbedding(LlamaRotaryEmbedding):
    pass


class NandiMLP(LlamaMLP):
    pass


class NandiAttention(LlamaAttention):
    pass


class NandiDecoderLayer(LlamaDecoderLayer):
    pass


class _VirtualLayerCache:
    """Proxy that shifts cache layer indices by `offset` to give each repeat its own virtual slots."""

    def __init__(self, cache: Cache, offset: int):
        self._cache = cache
        self._offset = offset

    def __getattr__(self, name):
        return getattr(self._cache, name)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        virtual_idx = layer_idx + self._offset
        # grow the backing cache if generate() pre-allocated fewer slots than needed
        while len(self._cache.layers) <= virtual_idx:
            self._cache.layers.append(DynamicLayer())
        return self._cache.update(key_states, value_states, virtual_idx, cache_kwargs)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._cache.get_seq_length(layer_idx + self._offset)


@auto_docstring
class NandiPreTrainedModel(LlamaPreTrainedModel):
    _can_record_outputs = {
        "hidden_states": NandiDecoderLayer,
        "attentions": NandiAttention,
    }


@auto_docstring
class NandiModel(LlamaModel):
    def __init__(self, config: NandiConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.embedding_rank if config.factorized_embedding else config.hidden_size,
            self.padding_idx,
        )
        self.embedding_proj = (
            nn.Linear(config.embedding_rank, config.hidden_size, bias=False) if config.factorized_embedding else None
        )
        self.layers = nn.ModuleList(
            [NandiDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = NandiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = NandiRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self.embedding_proj is not None:
            inputs_embeds = self.embedding_proj(inputs_embeds)

        repeats = self.config.layer_sharing_repeats if self.config.layer_sharing else 1

        if use_cache and past_key_values is None:
            # Use lazy DynamicCache (no config) so it grows to accommodate
            # num_hidden_layers * repeats virtual slots for layer-sharing.
            past_key_values = DynamicCache()

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            for repeat_idx in range(repeats):
                # Each repeat gets its own virtual cache slots offset by num_hidden_layers,
                # so repeat 0 uses slots 0..N-1 and repeat 1 uses slots N..2N-1, etc.
                repeat_cache = (
                    _VirtualLayerCache(past_key_values, repeat_idx * self.config.num_hidden_layers)
                    if (past_key_values is not None and repeat_idx > 0)
                    else past_key_values
                )
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_embeddings=position_embeddings,
                    position_ids=position_ids,
                    past_key_values=repeat_cache,
                    use_cache=use_cache,
                    **kwargs,
                )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring
class NandiForCausalLM(LlamaForCausalLM):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {
        "lm_head_proj": (["hidden_states"], ["hidden_states"]),
        "lm_head": (["hidden_states"], ["logits"]),
    }

    def __init__(self, config):
        super().__init__(config)
        self.model = NandiModel(config)
        self.vocab_size = config.vocab_size

        self.lm_head_proj = (
            nn.Linear(config.hidden_size, config.embedding_rank, bias=False) if config.factorized_embedding else None
        )
        self.lm_head = nn.Linear(
            config.embedding_rank if config.factorized_embedding else config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        if self.lm_head_proj is not None:
            hidden_states = self.lm_head_proj(hidden_states)

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = ["NandiConfig", "NandiPreTrainedModel", "NandiModel", "NandiForCausalLM"]
