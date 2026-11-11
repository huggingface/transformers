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

from collections.abc import Callable

import torch
from huggingface_hub.dataclasses import strict

from ...cache_utils import Cache, DynamicCache, KVSharingDict
from ...configuration_utils import PreTrainedConfig
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import MoeModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..kolibri1.configuration_kolibri1 import Kolibri1Config
from ..kolibri1.modeling_kolibri1 import (
    Kolibri1Attention,
    Kolibri1DecoderLayer,
    Kolibri1ForCausalLM,
    Kolibri1MLP,
    Kolibri1Model,
    Kolibri1PreTrainedModel,
    Kolibri1RMSNorm,
    Kolibri1SparseMoeBlock,
    apply_rotary_pos_emb,
    eager_attention_forward,
)


_DEFAULT_KV_SOURCE_ORDINALS = (0, 0, 0, 3, 4, 5, 5, 5, 5, 0)


@auto_docstring(checkpoint="Aleph-Alpha/Kolibri-1.1")
@strict
class Kolibri1_1Config(Kolibri1Config):
    r"""
    kv_source_layers (`list[int]`, *optional*):
        For each layer, the index of the layer whose keys and values it attends to. A full-attention layer that reads
        those of an earlier layer has no key/value projections of its own. Defaults to Kolibri 1.1's layout: of its 10
        full-attention layers, the 2nd and 3rd read the keys and values of the 1st, the 7th to 9th those of the 6th
        and the 10th those of the 1st, while the 1st, 4th, 5th and 6th produce their own. Configs with more
        full-attention layers must set it explicitly. The entries of sliding-attention layers must be their own index.
    mlp_only_layers (`list[int]`, *optional*):
        Indices of the layers with a dense MLP instead of a mixture of experts. Defaults to none.
    mlp_only_intermediate_size (`int`, *optional*):
        Intermediate size of the dense MLPs of `mlp_only_layers`.

    ```python
    >>> from transformers import Kolibri1_1Model, Kolibri1_1Config

    >>> configuration = Kolibri1_1Config()
    >>> model = Kolibri1_1Model(configuration)
    >>> configuration = model.config
    ```
    """

    model_type = "kolibri1_1"

    kv_source_layers: list[int] | None = None
    mlp_only_layers: list[int] | None = None
    mlp_only_intermediate_size: int | None = None

    def __post_init__(self, **kwargs):
        # `layer_types` is needed before the parent's `__post_init__`, to derive the default K/V sharing layout
        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if (i + 1) % 5 == 0 else "sliding_attention" for i in range(self.num_hidden_layers)
            ]
        if self.mlp_only_layers is None:
            self.mlp_only_layers = []
        full_layers = [i for i, layer_type in enumerate(self.layer_types) if layer_type == "full_attention"]
        if self.kv_source_layers is None:
            if len(full_layers) > len(_DEFAULT_KV_SOURCE_ORDINALS):
                raise ValueError(
                    f"The default `kv_source_layers` covers {len(_DEFAULT_KV_SOURCE_ORDINALS)} full-attention layers, "
                    f"got {len(full_layers)}: set `kv_source_layers` explicitly."
                )
            self.kv_source_layers = list(range(len(self.layer_types)))
            for layer_idx, source_ordinal in zip(full_layers, _DEFAULT_KV_SOURCE_ORDINALS):
                self.kv_source_layers[layer_idx] = full_layers[source_ordinal]
        PreTrainedConfig.__post_init__(self, **kwargs)

    def validate_architecture(self):
        super().validate_architecture()
        if set(self.layer_types) - {"full_attention", "sliding_attention"}:
            raise ValueError(f"Unsupported layer types: {sorted(set(self.layer_types))}.")
        if len(self.kv_source_layers) != len(self.layer_types):
            raise ValueError("`kv_source_layers` needs one entry per layer.")
        for layer_idx, source in enumerate(self.kv_source_layers):
            if self.layer_types[layer_idx] == "sliding_attention" and source != layer_idx:
                raise ValueError(f"Sliding-attention layer {layer_idx} cannot read the K/V of another layer.")
            if not 0 <= source <= layer_idx or self.kv_source_layers[source] != source:
                raise ValueError(f"Layer {layer_idx} must read the K/V of an earlier layer that produces its own.")
            if self.layer_types[source] != self.layer_types[layer_idx]:
                raise ValueError(f"Layer {layer_idx} must read the K/V of a layer with the same attention type.")
        if not set(self.mlp_only_layers) <= set(range(self.num_hidden_layers)):
            raise ValueError(f"`mlp_only_layers` must be layer indices, got {self.mlp_only_layers}.")
        if self.mlp_only_layers and (self.mlp_only_intermediate_size or 0) <= 0:
            raise ValueError("`mlp_only_layers` need a positive `mlp_only_intermediate_size`.")

    @property
    def kv_sharing_modes(self) -> list[str | None]:
        """For each layer, `"store"` if another layer reads its K/V, `"reuse"` if it reads another layer's, else `None`."""
        sources = {source for layer_idx, source in enumerate(self.kv_source_layers) if source != layer_idx}
        return [
            "reuse" if source != layer_idx else "store" if layer_idx in sources else None
            for layer_idx, source in enumerate(self.kv_source_layers)
        ]


class Kolibri1_1Attention(Kolibri1Attention):
    def __init__(self, config: Kolibri1_1Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.kv_source_layer = config.kv_source_layers[layer_idx]
        self.is_kv_shared_layer = config.kv_sharing_modes[layer_idx] == "reuse"
        self.store_kv = config.kv_sharing_modes[layer_idx] == "store"
        # Layers reading the K/V of another layer don't need any key/value weights
        if self.is_kv_shared_layer:
            del self.k_proj, self.v_proj, self.k_norm

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        shared_kv_states: KVSharingDict | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2))
        if self.is_kv_shared_layer:
            key_states, value_states = shared_kv_states[str(self.kv_source_layer)]
        else:
            key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2))
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # Full-attention layers have no positional embeddings (NoPE), which layers sharing keys rely on
        if self.is_sliding:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None and not self.is_kv_shared_layer:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
        if self.store_kv:
            shared_kv_states[str(self.layer_idx)] = key_states, value_states

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
            sliding_window=self.sliding_window if self.is_sliding else None,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Kolibri1_1MLP(Kolibri1MLP):
    pass


class Kolibri1_1RMSNorm(Kolibri1RMSNorm):
    pass


class Kolibri1_1SparseMoeBlock(Kolibri1SparseMoeBlock):
    pass


class Kolibri1_1DecoderLayer(Kolibri1DecoderLayer):
    def __init__(self, config: Kolibri1_1Config, layer_idx: int):
        # Not the parent's `__init__`, which would build the experts of the dense MLP layers only to discard them
        GradientCheckpointingLayer.__init__(self)
        self.hidden_size = config.hidden_size
        self.self_attn = Kolibri1_1Attention(config=config, layer_idx=layer_idx)
        if layer_idx in config.mlp_only_layers:
            self.mlp = Kolibri1_1MLP(config, intermediate_size=config.mlp_only_intermediate_size)
        else:
            self.mlp = Kolibri1_1SparseMoeBlock(config)
        self.input_layernorm = Kolibri1_1RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Kolibri1_1RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_self_attn_layernorm = Kolibri1_1RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_mlp_layernorm = Kolibri1_1RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Kolibri1_1PreTrainedModel(Kolibri1PreTrainedModel):
    pass


class Kolibri1_1Model(Kolibri1Model):
    def __init__(self, config: Kolibri1_1Config):
        super().__init__(config)
        # Layers reading the K/V of another layer don't use their own key/value weights, if a checkpoint has them.
        # Parent models collect these patterns from their children in their own `post_init`
        self._keys_to_ignore_on_load_unexpected = {
            rf"layers\.{i}\.self_attn\.(k_proj|v_proj|k_norm)\."
            for i, layer in enumerate(self.layers)
            if layer.self_attn.is_kv_shared_layer
        }

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
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        # Keys and values shared across layers, for the current forward only
        shared_kv_states = KVSharingDict()

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                shared_kv_states=shared_kv_states,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class Kolibri1_1ForCausalLM(Kolibri1ForCausalLM):
    def forward(self, **super_kwargs):
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, Kolibri1_1ForCausalLM

        >>> model = Kolibri1_1ForCausalLM.from_pretrained("Aleph-Alpha/Kolibri-1.1")
        >>> tokenizer = AutoTokenizer.from_pretrained("Aleph-Alpha/Kolibri-1.1")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_new_tokens=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        ```"""
        return super().forward(**super_kwargs)


__all__ = [
    "Kolibri1_1Config",
    "Kolibri1_1ForCausalLM",
    "Kolibri1_1Model",
    "Kolibri1_1PreTrainedModel",
]
