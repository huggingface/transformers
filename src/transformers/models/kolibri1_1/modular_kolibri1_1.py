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
from torch import nn

from ...cache_utils import Cache, DynamicCache, KVSharingDict
from ...configuration_utils import PreTrainedConfig
from ...masking_utils import create_compression_mask, create_sliding_window_causal_mask
from ...modeling_compression_utils import (
    build_attention_mask_from_topk,
    build_selection_mask,
    select_compression_indices,
)
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import MoeModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.deprecation import deprecate_kwarg
from ..kolibri1.configuration_kolibri1 import Kolibri1Config
from ..kolibri1.modeling_kolibri1 import (
    Kolibri1ForCausalLM,
    Kolibri1MLP,
    Kolibri1Model,
    Kolibri1PreTrainedModel,
    Kolibri1RMSNorm,
    Kolibri1SparseMoeBlock,
    apply_rotary_pos_emb,
    eager_attention_forward,
)


@auto_docstring(checkpoint="Aleph-Alpha/Kolibri-1.1")
@strict
class Kolibri1_1Config(Kolibri1Config):
    r"""
    index_layers (`list[int]`, *optional*):
        Indices of the indexed-attention layers that select a new pool of candidate keys (indexers). Every other
        indexed-attention layer rescores the pool of the closest preceding indexer (reindexer). Must contain the first
        indexed-attention layer. Defaults to Kolibri 1.1's layout: the 1st, 3rd, 4th and 6th indexed-attention layers.
    kv_source_layers (`list[int]`, *optional*):
        For each layer, the index of the layer whose keys and values it attends to. An indexed-attention layer that reads
        those of an earlier layer has no key/value projections of its own. Defaults to Kolibri 1.1's layout: of its 10
        indexed-attention layers, the 2nd and 3rd read the keys and values of the 1st, the 7th to 9th those of the 6th
        and the 10th those of the 1st, while the 1st, 4th, 5th and 6th produce their own. Configs with more
        indexed-attention layers must set it explicitly. The entries of sliding-attention layers must be their own index.
    index_topk (`int`, *optional*, defaults to 2048):
        Number of keys each query of a sparse layer selects from its pool, besides the sink and local tokens.
    index_pool_size (`int`, *optional*, defaults to 16384):
        Number of keys in an indexer's pool. Queries that see no more keys than this select from all of them.
    index_compression (`int`, *optional*, defaults to 16):
        Number of consecutive keys an indexer averages into one summary when scoring blocks for its pool.
    index_num_heads (`int`, *optional*, defaults to 12):
        Number of leading query heads used to score keys, at most the number of query heads of key head 0.
    index_num_local_tokens (`int`, *optional*, defaults to 128):
        Number of most recent tokens, including the query itself, that every query of a sparse layer attends to.
    index_num_sink_tokens (`int`, *optional*, defaults to 8):
        Number of leading tokens that every query of a sparse layer attends to.
    mlp_only_layers (`list[int]`, *optional*):
        Indices of the layers with a dense MLP instead of a mixture of experts. Defaults to Kolibri 1.1's layout: the
        first two layers.
    mlp_only_intermediate_size (`int`, *optional*, defaults to 3584):
        Intermediate size of the dense MLPs of `mlp_only_layers`.

    ```python
    >>> from transformers import Kolibri1_1Model, Kolibri1_1Config

    >>> configuration = Kolibri1_1Config()
    >>> model = Kolibri1_1Model(configuration)
    >>> configuration = model.config
    ```
    """

    model_type = "kolibri1_1"
    # No attention entries: an indexer scores keys with the query heads of key head 0 and must select the same keys
    # on every rank, so the heads cannot be sharded
    base_model_tp_plan = {
        "layers.*.mlp.shared_experts.gate_proj": "colwise",
        "layers.*.mlp.shared_experts.up_proj": "colwise",
        "layers.*.mlp.shared_experts.down_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    index_layers: list[int] | None = None
    kv_source_layers: list[int] | None = None
    index_topk: int = 2048
    index_pool_size: int = 16384
    index_compression: int = 16
    index_num_heads: int = 12
    index_num_local_tokens: int = 128
    index_num_sink_tokens: int = 8
    mlp_only_layers: list[int] | None = None
    mlp_only_intermediate_size: int = 3584

    def __post_init__(self, **kwargs):
        # `layer_types` is needed before the parent's `__post_init__`, to derive the default sparse layout
        if self.layer_types is None:
            self.layer_types = [
                "indexed_attention" if (i + 1) % 5 == 0 else "sliding_attention" for i in range(self.num_hidden_layers)
            ]
        if self.mlp_only_layers is None:
            self.mlp_only_layers = [i for i in (0, 1) if i < self.num_hidden_layers]
        indexed_layers = [i for i, layer_type in enumerate(self.layer_types) if layer_type == "indexed_attention"]
        if self.index_layers is None:
            self.index_layers = [indexed_layers[i] for i in (0, 2, 3, 5) if i < len(indexed_layers)]
        if self.kv_source_layers is None:
            # For each indexed-attention layer, which indexed-attention layer it reads the K/V of
            default_sources = (0, 0, 0, 3, 4, 5, 5, 5, 5, 0)
            if len(indexed_layers) > len(default_sources):
                raise ValueError(
                    f"The default `kv_source_layers` covers {len(default_sources)} indexed-attention layers, "
                    f"got {len(indexed_layers)}: set `kv_source_layers` explicitly."
                )
            self.kv_source_layers = list(range(len(self.layer_types)))
            for layer_idx, source in zip(indexed_layers, default_sources):
                self.kv_source_layers[layer_idx] = indexed_layers[source]
        PreTrainedConfig.__post_init__(self, **kwargs)

    def validate_architecture(self):
        super().validate_architecture()
        if set(self.layer_types) - {"indexed_attention", "sliding_attention"}:
            raise ValueError(f"Unsupported layer types: {sorted(set(self.layer_types))}.")
        indexed_layers = [i for i, layer_type in enumerate(self.layer_types) if layer_type == "indexed_attention"]
        if sorted(set(self.index_layers)) != self.index_layers or not set(self.index_layers) <= set(indexed_layers):
            raise ValueError(f"`index_layers` must be increasing indexed-attention layers, got {self.index_layers}.")
        if indexed_layers and indexed_layers[0] not in self.index_layers:
            raise ValueError("The first indexed-attention layer must be an indexer.")
        if len(self.kv_source_layers) != len(self.layer_types):
            raise ValueError("`kv_source_layers` needs one entry per layer.")
        for layer_idx, source in enumerate(self.kv_source_layers):
            if self.layer_types[layer_idx] == "sliding_attention" and source != layer_idx:
                raise ValueError(f"Sliding-attention layer {layer_idx} cannot read the K/V of another layer.")
            if not 0 <= source <= layer_idx or self.kv_source_layers[source] != source:
                raise ValueError(f"Layer {layer_idx} must read the K/V of an earlier layer that produces its own.")
            if self.layer_types[source] != self.layer_types[layer_idx]:
                raise ValueError(f"Layer {layer_idx} must read the K/V of a layer with the same attention type.")
        for name in ("index_topk", "index_pool_size", "index_compression"):
            if getattr(self, name) <= 0:
                raise ValueError(f"`{name}` must be positive, got {getattr(self, name)}.")
        if self.index_pool_size % self.index_compression:
            raise ValueError("`index_pool_size` must be a multiple of `index_compression`.")
        if self.head_dim % 16:
            raise ValueError("NVFP4 rounding of keys and values needs a `head_dim` that is a multiple of 16.")
        if not 0 < self.index_num_heads <= self.num_attention_heads // self.num_key_value_heads:
            raise ValueError("`index_num_heads` must be between 1 and the number of query heads of key head 0.")
        if not set(self.mlp_only_layers) <= set(range(self.num_hidden_layers)):
            raise ValueError(f"`mlp_only_layers` must be layer indices, got {self.mlp_only_layers}.")
        if self.mlp_only_layers and self.mlp_only_intermediate_size <= 0:
            raise ValueError("`mlp_only_layers` need a positive `mlp_only_intermediate_size`.")

    @property
    def kv_sharing_modes(self) -> list[str | None]:
        """For each layer, `"store"` if another layer reads its K/V, `"reuse"` if it reads another layer's, else `None`."""
        sources = {source for layer_idx, source in enumerate(self.kv_source_layers) if source != layer_idx}
        return [
            "reuse" if source != layer_idx else "store" if layer_idx in sources else None
            for layer_idx, source in enumerate(self.kv_source_layers)
        ]


def nvfp4_fake_quantize(hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Round to NVFP4 and back: E2M1 values with one E4M3 scale per 16 consecutive values of the last dimension, rounded
    to nearest-even. Kolibri 1.1 was trained with its sparse attention keys and values, and its routing queries and
    keys, rounded like this.
    """
    blocks = hidden_states.float().unflatten(-1, (-1, 16))
    scale = (blocks.abs().amax(-1, keepdim=True) / 6).clamp(2**-9, 448).to(torch.float8_e4m3fn).float()
    scaled = blocks / scale
    magnitude = scaled.abs().clamp(max=6)
    # E2M1 magnitudes are spaced by 0.5 below 2, by 1 below 4 and by 2 up to 6. `round` breaks ties to even
    step = torch.where(magnitude < 2, 0.5, torch.where(magnitude < 4, 1.0, 2.0))
    rounded = torch.round(magnitude / step) * step
    return (rounded.copysign(scaled) * scale).flatten(-2).to(hidden_states.dtype)


class Kolibri1_1Indexer(nn.Module):
    """
    Two-stage sparse attention indexer that first selects a pool of candidate keys by scoring compressed key groups,
    then selects the top-k individual keys within that pool. The candidate pool is shared across indexed attention
    layers, while individual key selection is performed independently at each layer.

    Uses NVFP4-rounded attention queries and keys directly, without additional learned projections.
    Sink and local tokens are always selected outside the top-k budget.

    NOTE: Assumes left-padded or unpadded sequences.
    """

    def __init__(self, config: Kolibri1_1Config, layer_idx: int):
        super().__init__()
        self.is_pool_source = layer_idx in config.index_layers
        self.num_heads = config.index_num_heads
        self.topk = config.index_topk
        self.pool_size = config.index_pool_size
        self.compression = config.index_compression
        self.num_local_tokens = config.index_num_local_tokens
        self.num_sink_tokens = config.index_num_sink_tokens
        self.head_weight = config.head_dim**-0.5 / config.index_num_heads

    @torch.no_grad()
    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        attention_mask: torch.Tensor,
        index_pool: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        seq_length = query_states.shape[-2]
        kv_length = key_states.shape[-2]
        group_size = self.compression
        budget = self.topk + self.num_sink_tokens + self.num_local_tokens
        positions = torch.arange(kv_length, device=key_states.device)
        lengths = attention_mask[:, -seq_length:].unsqueeze(-1)

        # Remove left padding, accounting for unused static cache capacity
        start = kv_length + attention_mask[:, :1] - attention_mask[:, -1:]

        if kv_length <= budget:
            indices = start.unsqueeze(1) + positions
            return indices.expand(-1, seq_length, -1).masked_fill(positions >= lengths, -1), None

        q = nvfp4_fake_quantize(query_states[:, : self.num_heads]).transpose(1, 2).float()
        k = key_states[:, 0]
        k = k.gather(1, (start + positions).clamp_max(kv_length - 1).unsqueeze(-1).expand_as(k))

        if self.is_pool_source and kv_length > self.pool_size:
            # Mean compression
            groups = nn.functional.pad(k.float(), (0, 0, 0, (-kv_length) % group_size))
            summaries = nvfp4_fake_quantize(groups.unflatten(1, (-1, group_size)).mean(dim=2))

            # Score groups of tokens
            scores = torch.matmul(q, summaries.transpose(-1, -2).unsqueeze(1))
            scores = scores.relu().sum(dim=-2) * self.head_weight

            # Incomplete groups compete with a zero score
            blocks = torch.arange(summaries.shape[1], device=k.device)
            scores = scores.masked_fill(blocks >= lengths // group_size, 0)

            pool_indices = select_compression_indices(
                scores,
                attention_mask,
                self.pool_size // group_size,
                group_size=group_size,
                return_groups=True,
                allow_partial_groups=True,
                deterministic=True,
            )
            index_pool = build_selection_mask(pool_indices, summaries.shape[1])

        # Score individual keys
        k = nvfp4_fake_quantize(k).float()
        scores = torch.matmul(q, k.transpose(-1, -2).unsqueeze(1))
        scores = scores.relu().sum(dim=-2) * self.head_weight

        # Restrict selection to the pool, except for short contexts
        if index_pool is not None:
            pool = index_pool.repeat_interleave(group_size, dim=-1)[..., :kv_length]
            scores = scores.masked_fill(~pool & (lengths > budget), float("-inf"))

        # Always select sink and local tokens
        mandatory = (positions < self.num_sink_tokens) | (positions >= lengths - self.num_local_tokens)
        scores = scores.masked_fill(mandatory, float("inf"))

        indices = select_compression_indices(
            scores,
            attention_mask,
            budget,
            deterministic=True,
        )
        return indices, index_pool


class Kolibri1_1Attention(nn.Module):
    def __init__(self, config: Kolibri1_1Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.sliding_window = config.sliding_window

        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.kv_source_layer = config.kv_source_layers[layer_idx]
        self.is_kv_shared_layer = config.kv_sharing_modes[layer_idx] == "reuse"
        self.store_full_length_kv = config.kv_sharing_modes[layer_idx] == "store"

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.q_norm = Kolibri1_1RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        # Layers reading the K/V of another layer don't need any key/value weights
        if not self.is_kv_shared_layer:
            self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
            self.k_norm = Kolibri1_1RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        self.indexer = None if self.is_sliding else Kolibri1_1Indexer(config, layer_idx)

    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        cache: Cache | None = None,
        shared_kv_states: KVSharingDict | None = None,
        index_pool: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2))
        if self.is_kv_shared_layer:
            key_states, value_states = shared_kv_states[str(self.kv_source_layer)]
        else:
            key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2))
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # Indexed-attention layers have no positional embeddings (NoPE)
        if self.is_sliding and not self.is_kv_shared_layer:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if cache is not None and not self.is_kv_shared_layer:
            key_states, value_states = cache.update(key_states, value_states, self.layer_idx)

        if self.store_full_length_kv:
            shared_kv_states[str(self.layer_idx)] = key_states, value_states

        if self.indexer is not None:
            topk_indices, index_pool = self.indexer(query_states, key_states, attention_mask, index_pool)
            attention_mask = build_attention_mask_from_topk(
                self.config,
                topk_indices=topk_indices,
                query_states=query_states,
                kv_length=key_states.shape[-2],
            )

            # Sparse layers attend to NVFP4 keys and values, rounded with a straight-through gradient
            key_states = key_states + (nvfp4_fake_quantize(key_states) - key_states).detach()
            value_states = value_states + (nvfp4_fake_quantize(value_states) - value_states).detach()

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
        return attn_output, attn_weights, index_pool


class Kolibri1_1MLP(Kolibri1MLP):
    pass


class Kolibri1_1RMSNorm(Kolibri1RMSNorm):
    pass


class Kolibri1_1SparseMoeBlock(Kolibri1SparseMoeBlock):
    pass


class Kolibri1_1DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Kolibri1_1Config, layer_idx: int):
        super().__init__()
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

    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        hidden_states: torch.Tensor,
        shared_kv_states: KVSharingDict,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        index_pool: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Indexers return a new pool of candidate keys, the other layers pass on the one they were given
        hidden_states, _, index_pool = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            shared_kv_states=shared_kv_states,
            index_pool=index_pool,
            **kwargs,
        )
        hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_mlp_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, index_pool


class Kolibri1_1PreTrainedModel(Kolibri1PreTrainedModel):
    # Sparse layers apply their key selection as an explicit mask
    _supports_flash_attn = False
    _supports_flex_attn = False


class Kolibri1_1Model(Kolibri1Model):
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and cache is None:
            cache = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = cache.get_seq_length() if cache is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache": cache,
                "position_ids": position_ids,
            }
            mask_functions = {
                "indexed_attention": create_compression_mask,
                "sliding_attention": create_sliding_window_causal_mask,
            }
            causal_mask_mapping = {
                layer_type: mask_functions[layer_type](**mask_kwargs) for layer_type in set(self.config.layer_types)
            }

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        # Keys and values shared across layers, for the current forward only
        shared_kv_states = KVSharingDict()
        # The candidate group mask of the latest indexer
        index_pool = None

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if i in self.config.index_layers:
                # An indexer selects a new pool: drop the previous one first, so that two never exist at once
                index_pool = None
            hidden_states, index_pool = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_ids=position_ids,
                cache=cache,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                shared_kv_states=shared_kv_states,
                index_pool=index_pool,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            cache=cache,
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
