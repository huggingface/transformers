# Copyright 2026 IBM and the HuggingFace Inc. team. All rights reserved.
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
"""GraniteMoeSWA: GraniteMoeShared with Sliding Window Attention and learnable attention sinks.

GraniteMoeSWA combines the mixture-of-experts blocks (with optional shared experts, disabled by
default via ``shared_intermediate_size=0``) of GraniteMoeShared with the per-layer sliding-window
attention and learnable per-head attention sinks of GraniteSWA. The eager path computes the
``sigmoid``-scaling explicitly, while the FlexAttention, FlashAttention-3 and FlashAttention-4
backends apply the same sink through the shared attention dispatch by passing ``s_aux``.
"""

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_outputs import MoeModelOutputWithPast
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..granite_swa.modeling_granite_swa import GraniteSWAAttention
from ..granitemoeshared.configuration_granitemoeshared import GraniteMoeSharedConfig
from ..granitemoeshared.modeling_granitemoeshared import (
    GraniteMoeSharedDecoderLayer,
    GraniteMoeSharedForCausalLM,
    GraniteMoeSharedModel,
    GraniteMoeSharedPreTrainedModel,
)


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="ibm-granite/granite-swash-3b-a600m")
@strict
class GraniteMoeSWAConfig(GraniteMoeSharedConfig):
    r"""
    embedding_multiplier (`float`, *optional*, defaults to 1.0):
        embedding multiplier
    logits_scaling (`float`, *optional*, defaults to 1.0):
        divisor for output logits
    residual_multiplier (`float`, *optional*, defaults to 1.0):
        residual multiplier
    attention_multiplier (`float`, *optional*, defaults to 1.0):
        attention multiplier
    shared_intermediate_size (`int`, *optional*, defaults to 0):
        intermediate size for shared experts. Defaults to `0`, which disables the shared experts.
    sliding_window (`int`, *optional*, defaults to 128):
        Size of the sliding attention window used by layers whose `layer_types` entry is
        `"sliding_attention"`.
    layer_types (`list[str]`, *optional*):
        Per-layer attention type, each either `"full_attention"` or `"sliding_attention"`. When
        `None`, every fourth layer (`i % 4 == 0`) uses full attention and the rest use sliding
        window attention.
    no_rope_layers (`list[int]`, *optional*):
        Per-layer flag for rotary position embeddings, `1` to apply RoPE and `0` for NoPE (no
        positional embedding). When `None`, defaults to all-RoPE (`1` for every layer).

    ```python
    >>> from transformers import GraniteMoeSWAModel, GraniteMoeSWAConfig

    >>> # Initializing a GraniteMoeSWA configuration
    >>> configuration = GraniteMoeSWAConfig()

    >>> # Initializing a model from the configuration
    >>> model = GraniteMoeSWAModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "granitemoe_swa"
    # Attention shards like Granite (+ per-head `sinks` colwise to track the head-sharding); the
    # routed experts shard tensor-parallel (packed gate/up colwise, down rowwise, `moe_tp_experts`)
    # with the router replicated. The optional shared expert (`shared_mlp`, off by default) is left
    # replicated -- it is small and its full output sums consistently with the all-reduced MoE output.
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.self_attn.sinks": "colwise",
        "layers.*.block_sparse_moe.experts.gate_up_proj": "packed_colwise",
        "layers.*.block_sparse_moe.experts.down_proj": "rowwise",
        "layers.*.block_sparse_moe.experts": "moe_tp_experts",
    }

    sliding_window: int | None = 128
    layer_types: list[str] | None = None
    no_rope_layers: list[int] | None = None

    def __post_init__(self, **kwargs):
        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if i % 4 == 0 else "sliding_attention" for i in range(self.num_hidden_layers)
            ]

        # Per-layer RoPE vs NoPE (1 = apply RoPE, 0 = NoPE). Default is all-RoPE;
        # set `no_rope_layers` explicitly to make specific layers NoPE.
        if self.no_rope_layers is None:
            self.no_rope_layers = [1] * self.num_hidden_layers

        super().__post_init__(**kwargs)


class GraniteMoeSWAAttention(GraniteSWAAttention):
    pass


class GraniteMoeSWADecoderLayer(GraniteMoeSharedDecoderLayer):
    def __init__(self, config: GraniteMoeSWAConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = GraniteMoeSWAAttention(config=config, layer_idx=layer_idx)


class GraniteMoeSWAPreTrainedModel(GraniteMoeSharedPreTrainedModel):
    _no_split_modules = ["GraniteMoeSWADecoderLayer"]
    _supports_sdpa = False
    _supports_flex_attn = True
    _compatible_flash_implementations = ["kernels-community/vllm-flash-attn3", "flash_attention_4"]
    _can_record_outputs = {
        "hidden_states": GraniteMoeSWADecoderLayer,
        "attentions": GraniteMoeSWAAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, GraniteMoeSWAAttention):
            init.zeros_(module.sinks)


class GraniteMoeSWAModel(GraniteMoeSharedModel):
    def __init__(self, config: GraniteMoeSWAConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [GraniteMoeSWADecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

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

        inputs_embeds = inputs_embeds * self.embedding_multiplier

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        # Create the masks once per layer type (full vs sliding window) and reuse across layers.
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

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            # NoPE layers (`no_rope_layers[i] == 0`) receive `None`, so attention skips RoPE.
            layer_position_embeddings = position_embeddings if self.config.no_rope_layers[i] else None
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=layer_position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class GraniteMoeSWAForCausalLM(GraniteMoeSharedForCausalLM):
    pass


__all__ = [
    "GraniteMoeSWAConfig",
    "GraniteMoeSWAForCausalLM",
    "GraniteMoeSWAModel",
    "GraniteMoeSWAPreTrainedModel",
]
