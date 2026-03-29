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

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, DynamicLayer
from ...configuration_utils import PretrainedConfig
from ...generation import GenerationMixin
from ...integrations import use_kernel_forward_from_hub
from ...masking_utils import create_causal_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, RopeParameters, dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import can_return_tuple, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs


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
            self.rope_parameters = {"rope_theta": 100000.0}
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


@use_kernel_forward_from_hub("RMSNorm")
class NandiRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class NandiRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: NandiConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_type = self.config.rope_parameters.get("rope_type", "default")
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(
        config: NandiConfig | None = None,
        device: torch.device | None = None,
        seq_len: int | None = None,
    ) -> tuple[torch.Tensor, float]:
        del seq_len
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        attention_factor = 1.0
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    del position_ids
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    del kwargs
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class NandiAttention(nn.Module):
    def __init__(self, config: NandiConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

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
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class NandiMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class NandiDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: NandiConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = NandiAttention(config=config, layer_idx=layer_idx)
        self.mlp = NandiMLP(config)
        self.input_layernorm = NandiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = NandiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
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
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


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
class NandiPreTrainedModel(PreTrainedModel):
    config: NandiConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["NandiDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": NandiDecoderLayer,
        "attentions": NandiAttention,
    }

    def __init__(self, config: NandiConfig):
        super().__init__(config)


@auto_docstring
class NandiModel(NandiPreTrainedModel):
    def __init__(self, config: NandiConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        embedding_dim = config.embedding_rank if config.factorized_embedding else config.hidden_size

        self.embed_tokens = nn.Embedding(config.vocab_size, embedding_dim, self.padding_idx)
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
class NandiForCausalLM(NandiPreTrainedModel, GenerationMixin):
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

        lm_head_in_features = config.embedding_rank if config.factorized_embedding else config.hidden_size
        self.lm_head_proj = (
            nn.Linear(config.hidden_size, config.embedding_rank, bias=False) if config.factorized_embedding else None
        )
        self.lm_head = nn.Linear(lm_head_in_features, config.vocab_size, bias=False)

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
