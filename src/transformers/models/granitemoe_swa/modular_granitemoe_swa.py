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

import copy

import torch
from huggingface_hub.dataclasses import strict
from torch import nn
from torch.nn import functional as F

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_outputs import MoeModelOutputWithPast
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..granite_swa.modeling_granite_swa import GraniteSWAAttention
from ..granitemoe.modeling_granitemoe import GraniteMoeMoE, GraniteMoeTopKRouter
from ..granitemoeshared.configuration_granitemoeshared import GraniteMoeSharedConfig
from ..granitemoeshared.modeling_granitemoeshared import (
    GraniteMoeSharedDecoderLayer,
    GraniteMoeSharedForCausalLM,
    GraniteMoeSharedModel,
    GraniteMoeSharedPreTrainedModel,
    GraniteMoeSharedRotaryEmbedding,
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
    layer_rope_theta (`list[float]`, *optional*):
        Per-layer RoPE base (`theta`) frequency. `0` sets NoPE (no positional embedding) for
        that layer. Overrides global `rope_parameters["rope_theta"]`, which is only used when
        this list is not provided or specified (`layer_rope_theta = None`).

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
    # Expert-parallel plan: shard the routed experts across ranks (each rank owns a slice of the
    # experts) with the router driving the dispatch. The optional shared expert is left replicated.
    base_model_ep_plan = {
        "layers.*.block_sparse_moe.router": "ep_router",
        "layers.*.block_sparse_moe.experts.gate_up_proj": "grouped_gemm",
        "layers.*.block_sparse_moe.experts.down_proj": "grouped_gemm",
        "layers.*.block_sparse_moe.experts": "moe_tp_experts",
    }

    sliding_window: int | None = 128
    layer_types: list[str] | None = None
    layer_rope_theta: list[float] | None = None

    def __post_init__(self, **kwargs):
        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if i % 4 == 0 else "sliding_attention" for i in range(self.num_hidden_layers)
            ]

        super().__post_init__(**kwargs)

        # Per-layer RoPE base theta (0 => NoPE). Default: global rope_theta
        if self.layer_rope_theta is None:
            self.layer_rope_theta = [self.rope_parameters["rope_theta"]] * self.num_hidden_layers


class GraniteMoeSWATopKRouter(GraniteMoeTopKRouter):
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Identical to GraniteMoeTopKRouter, but returns (router_logits, router_scores, router_indices)
        # as expected by 'base_model_ep_plan' from `ep_router``. Enables native HF EP.
        router_logits = F.linear(hidden_states, self.weight).float()  # (num_tokens, num_experts)
        top_k_logits, top_k_index = router_logits.topk(self.top_k, dim=-1)  # (num_tokens, top_k)
        top_k_weights = torch.softmax(top_k_logits, dim=-1).type_as(hidden_states)  # (num_tokens, top_k)
        return router_logits, top_k_weights, top_k_index


class GraniteMoeSWAMoE(GraniteMoeMoE):
    def forward(self, layer_input: torch.Tensor) -> torch.Tensor:
        bsz, length, emb_size = layer_input.size()
        hidden_states = layer_input.reshape(-1, emb_size)
        # Router now returns (router_logits, top_k_weights, top_k_index).
        _, top_k_weights, top_k_index = self.router(hidden_states)
        layer_output = self.experts(hidden_states, top_k_index, top_k_weights)
        return layer_output.view(bsz, length, self.input_size)


class GraniteMoeSWAAttention(GraniteSWAAttention):
    pass


class GraniteMoeSWADecoderLayer(GraniteMoeSharedDecoderLayer):
    def __init__(self, config: GraniteMoeSWAConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = GraniteMoeSWAAttention(config=config, layer_idx=layer_idx)


class GraniteMoeSWAPreTrainedModel(GraniteMoeSharedPreTrainedModel):
    _no_split_modules = ["GraniteMoeSWADecoderLayer"]
    _supports_sdpa = False
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


class GraniteMoeSWARotaryEmbedding(GraniteMoeSharedRotaryEmbedding):
    pass


class GraniteMoeSWAModel(GraniteMoeSharedModel):
    def __init__(self, config: GraniteMoeSWAConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [GraniteMoeSWADecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        # Per-layer RoPE: one rotary embedding per unique non-zero theta (`theta == 0` => NoPE).
        # (`self.rotary_emb`, built by the parent at the global theta, is left in place but unused.)
        self.rotary_embs = nn.ModuleList()
        for theta in sorted({theta for theta in config.layer_rope_theta if theta}):
            theta_config = copy.deepcopy(config)
            theta_config.rope_parameters = {**config.rope_parameters, "rope_theta": theta}
            self.rotary_embs.append(GraniteMoeSWARotaryEmbedding(theta_config))

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
        # Compute (cos, sin) once per unique non-zero base theta; layers with theta 0 (NoPE) receive
        # `None` and skip RoPE in attention.
        position_embeddings_by_theta = {
            rotary_emb.config.rope_parameters["rope_theta"]: rotary_emb(hidden_states, position_ids)
            for rotary_emb in self.rotary_embs
        }

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            theta = self.config.layer_rope_theta[i]
            layer_position_embeddings = position_embeddings_by_theta[theta] if theta else None
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
