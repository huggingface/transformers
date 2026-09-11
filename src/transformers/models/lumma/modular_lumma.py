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
import torch.nn as nn
from huggingface_hub.dataclasses import strict

from ...cache_utils import Cache, DynamicCache
from ...masking_utils import create_causal_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import can_return_tuple, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..llama.configuration_llama import LlamaConfig
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)


@auto_docstring(checkpoint="FrontiersMind/Lumma-0.6B-Base")
@strict
class LummaConfig(LlamaConfig):
    r"""
    Configuration class for the Lumma model.

    factorized_embedding (`bool`, *optional*, defaults to `True`):
        Whether to use a low-rank factorized embedding table.
    embedding_rank (`int`, *optional*, defaults to 512):
        Rank of the factorized embedding when `factorized_embedding=True`.
    layer_sharing (`bool`, *optional*, defaults to `False`):
        Whether to reuse the same decoder layers multiple times in the forward pass.
    layer_sharing_repeats (`int`, *optional*, defaults to 1):
        Number of times each unique decoder layer is applied when `layer_sharing=True`.
        When layer sharing is enabled, `num_hidden_layers` is the total number of forward
        passes (unique layers × `layer_sharing_repeats`).
    qk_norm (`bool`, *optional*, defaults to `False`):
        Whether to apply RMSNorm to both query and key projections.
    q_norm (`bool`, *optional*, defaults to `True`):
        Whether to apply RMSNorm to the query projection only.
    shared_kv (`bool`, *optional*, defaults to `True`):
        Whether keys and values share the same projection (no separate `v_proj`).
    kv_cache_mode (`str`, *optional*, defaults to `"shared"`):
        KV cache mode. Either `"shared"` or `"vanilla"`.

    Example:
    ```python
    >>> from transformers import LummaConfig, LummaForCausalLM
    >>> configuration = LummaConfig()
    >>> model = LummaForCausalLM(configuration)
    >>> configuration = model.config
    ```
    """

    model_type = "lumma"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    vocab_size: int = 131072
    hidden_size: int = 1440
    intermediate_size: int = 3280
    num_hidden_layers: int = 30
    num_attention_heads: int = 16
    num_key_value_heads: int | None = 8
    head_dim: int | None = None
    hidden_act: str = "silu"
    max_position_embeddings: int = 12288
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
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

    # Lumma-specific parameters
    factorized_embedding: bool = True
    embedding_rank: int = 512
    layer_sharing: bool = False
    layer_sharing_repeats: int = 1
    qk_norm: bool = False
    q_norm: bool = True
    shared_kv: bool = True
    kv_cache_mode: str = "shared"

    def __post_init__(self, **kwargs):
        if self.rope_parameters is None:
            self.rope_parameters = {"rope_theta": 1_000_000.0}
        if not self.layer_sharing:
            self.layer_sharing_repeats = 1
        else:
            self.layer_sharing_repeats = int(self.layer_sharing_repeats)

        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Sanity-check the configuration values."""
        if self.qk_norm and self.q_norm:
            raise ValueError("Exactly one of `qk_norm` or `q_norm` may be True, not both.")
        if self.kv_cache_mode not in ("shared", "vanilla"):
            raise ValueError(f"`kv_cache_mode` must be 'shared' or 'vanilla', got {self.kv_cache_mode!r}.")
        if self.factorized_embedding and self.embedding_rank <= 0:
            raise ValueError(
                f"`embedding_rank` must be a positive integer when `factorized_embedding=True`, "
                f"got {self.embedding_rank}."
            )
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"`hidden_size` ({self.hidden_size}) must be divisible by "
                f"`num_attention_heads` ({self.num_attention_heads})."
            )
        if self.layer_sharing_repeats < 1:
            raise ValueError(f"`layer_sharing_repeats` must be >= 1, got {self.layer_sharing_repeats}.")
        if self.layer_sharing and self.num_hidden_layers % self.layer_sharing_repeats != 0:
            raise ValueError(
                f"`num_hidden_layers` ({self.num_hidden_layers}) must be divisible by "
                f"`layer_sharing_repeats` ({self.layer_sharing_repeats}) when `layer_sharing=True`."
            )


class LummaRMSNorm(LlamaRMSNorm):
    pass


class LummaRotaryEmbedding(LlamaRotaryEmbedding):
    pass


class LummaMLP(LlamaMLP):
    pass


class LummaAttention(LlamaAttention):
    def __init__(self, config: LummaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.shared_kv = config.shared_kv
        self.qk_norm = config.qk_norm
        self.q_norm_enabled = config.q_norm
        if self.shared_kv:
            self.v_proj = None
        if self.qk_norm:
            self.q_norm = LummaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = LummaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        elif self.q_norm_enabled:
            self.q_norm = LummaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = None
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache_layer_offset = kwargs.pop("cache_layer_offset", 0)
        cache_layer_idx = self.layer_idx + cache_layer_offset
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k_raw = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if self.shared_kv:
            kv_cache_mode = getattr(self.config, "kv_cache_mode", "shared")

            if self.qk_norm:
                query_states = self.q_norm(query_states)
            elif self.q_norm_enabled:
                query_states = self.q_norm(query_states)

            if kv_cache_mode == "shared":
                if past_key_values is not None:
                    empty_v = torch.empty(
                        k_raw.shape[0],
                        k_raw.shape[1],
                        0,
                        k_raw.shape[3],
                        device=k_raw.device,
                        dtype=k_raw.dtype,
                    )
                    k_raw_full, _ = past_key_values.update(k_raw, empty_v, cache_layer_idx)
                else:
                    k_raw_full = k_raw

                value_states = k_raw_full
                key_states = self.k_norm(k_raw_full) if self.qk_norm else k_raw_full

                cos, sin = position_embeddings
                q_len = query_states.shape[-2]
                cos_q = cos[..., -q_len:, :]
                sin_q = sin[..., -q_len:, :]
                query_states, _ = apply_rotary_pos_emb(query_states, query_states, cos_q, sin_q)
                _, key_states = apply_rotary_pos_emb(key_states, key_states, cos, sin)

            else:
                key_states = self.k_norm(k_raw) if self.qk_norm else k_raw
                value_states = k_raw

                cos, sin = position_embeddings
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

                if past_key_values is not None:
                    key_states, value_states = past_key_values.update(key_states, value_states, cache_layer_idx)

        else:
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = k_raw

            if self.qk_norm:
                query_states = self.q_norm(query_states)
                key_states = self.k_norm(key_states)
            elif self.q_norm_enabled:
                query_states = self.q_norm(query_states)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            if past_key_values is not None:
                key_states, value_states = past_key_values.update(key_states, value_states, cache_layer_idx)

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
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class LummaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LummaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LummaAttention(config=config, layer_idx=layer_idx)
        self.mlp = LummaMLP(config)
        self.input_layernorm = LummaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LummaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class LummaPreTrainedModel(LlamaPreTrainedModel):
    config: LummaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LummaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": LummaDecoderLayer,
        "attentions": LummaAttention,
    }


@auto_docstring
class LummaModel(LummaPreTrainedModel):
    def __init__(self, config: LummaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            # CODEPATH: FrontiersMind/Lumma-0.6B-Base uses factorized_embedding=True; no released checkpoint uses factorized_embedding=False.
            config.embedding_rank if config.factorized_embedding else config.hidden_size,
            self.padding_idx,
        )
        self.embedding_proj = (
            # CODEPATH: FrontiersMind/Lumma-0.6B-Base uses factorized_embedding=True; no released checkpoint uses factorized_embedding=False.
            nn.Linear(config.embedding_rank, config.hidden_size, bias=False) if config.factorized_embedding else None
        )
        self.layers = nn.ModuleList(
            [
                LummaDecoderLayer(config, layer_idx)
                for layer_idx in range(
                    # CODEPATH: FrontiersMind/Lumma-0.6B-Base uses layer_sharing=False (default), so repeats=1; no released checkpoint sets layer_sharing=True.
                    config.num_hidden_layers // (config.layer_sharing_repeats if config.layer_sharing else 1)
                )
            ]
        )
        self.norm = LummaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LummaRotaryEmbedding(config=config)
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
        # CODEPATH: FrontiersMind/Lumma-0.6B-Base uses layer_sharing=False (default), so the forward loop runs each layer once.
        repeats = self.config.layer_sharing_repeats if self.config.layer_sharing else 1
        actual_layers = len(self.layers)

        if use_cache and past_key_values is None:
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
        kv_cache_mode = getattr(self.config, "kv_cache_mode", "shared")
        if getattr(self.config, "shared_kv", False) and kv_cache_mode == "shared" and past_key_values is not None:
            past_len = past_key_values.get_seq_length(0)
            cur_len = inputs_embeds.shape[1]
            full_position_ids = torch.arange(past_len + cur_len, device=inputs_embeds.device).unsqueeze(0)
            position_embeddings = self.rotary_emb(hidden_states, position_ids=full_position_ids)
        else:
            position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        for decoder_layer in self.layers:
            for repeat_idx in range(repeats):
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_embeddings=position_embeddings,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_layer_offset=repeat_idx * actual_layers,
                    **kwargs,
                )
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring
class LummaForCausalLM(LlamaForCausalLM):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {
        "lm_head_proj": (["hidden_states"], ["hidden_states"]),
        "lm_head": (["hidden_states"], ["logits"]),
    }

    def __init__(self, config: LummaConfig):
        super().__init__(config)
        self.model = LummaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head_proj = (
            # CODEPATH: FrontiersMind/Lumma-0.6B-Base uses factorized_embedding=True; no released checkpoint uses factorized_embedding=False.
            nn.Linear(config.hidden_size, config.embedding_rank, bias=False) if config.factorized_embedding else None
        )
        self.lm_head = nn.Linear(
            # CODEPATH: FrontiersMind/Lumma-0.6B-Base uses factorized_embedding=True; no released checkpoint uses factorized_embedding=False.
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


__all__ = ["LummaConfig", "LummaPreTrainedModel", "LummaModel", "LummaForCausalLM"]
