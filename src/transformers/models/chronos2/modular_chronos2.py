# Copyright 2026 The HuggingFace Team. All rights reserved.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
"""PyTorch Chronos-2 model."""

import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...activations import ACT2FN
from ...configuration_utils import PreTrainedConfig
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.generic import maybe_autocast
from ..llama.modeling_llama import apply_rotary_pos_emb
from ..phi4_multimodal.modeling_phi4_multimodal import simple_eager_attention_forward
from ..t5.configuration_t5 import T5Config
from ..t5.modeling_t5 import T5DenseActDense, T5LayerNorm


logger = logging.get_logger(__name__)


def _default_chronos_config() -> dict:
    return {
        "context_length": 8192,
        "output_patch_size": 16,
        "input_patch_size": 16,
        "input_patch_stride": 16,
        "quantiles": [
            0.01,
            0.05,
            0.1,
            0.15,
            0.2,
            0.25,
            0.3,
            0.35,
            0.4,
            0.45,
            0.5,
            0.55,
            0.6,
            0.65,
            0.7,
            0.75,
            0.8,
            0.85,
            0.9,
            0.95,
            0.99,
        ],
        "use_reg_token": True,
        "use_arcsinh": True,
        "max_output_patches": 64,
        "time_encoding_scale": 8192,
    }


@auto_docstring(checkpoint="amazon/chronos-2")
@strict
class Chronos2Config(T5Config):
    r"""
    chronos_config (`dict`, *optional*):
        Forecasting-specific configuration. It contains the context length, input and output patch sizes, input patch
        stride, quantile levels, maximum number of output patches, time-encoding scale, and switches for the REG token
        and arcsinh transformation.
    feed_forward_proj (`str`, *optional*, defaults to `"relu"`):
        Activation used in the bias-free feed-forward layers. Chronos-2 supports non-gated activations only.
    rope_theta (`float`, *optional*, defaults to 10000.0):
        Base period used by the rotary position embeddings in time self-attention.
    """

    model_type = "chronos2"
    keys_to_ignore_at_inference = []
    vocab_size: int = 2
    initializer_factor: float = 0.05
    is_encoder_decoder: bool = False
    use_cache: bool = False
    rope_theta: float = 10000.0
    chronos_config: dict | None = None

    num_decoder_layers = AttributeError()
    relative_attention_num_buckets = AttributeError()
    relative_attention_max_distance = AttributeError()
    eos_token_id = AttributeError()
    classifier_dropout = AttributeError()
    is_decoder = AttributeError()

    @classmethod
    def get_config_dict(
        cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Normalize the released legacy T5-tagged config for explicit native Chronos-2 loads."""
        config_dict, kwargs = super().get_config_dict(pretrained_model_name_or_path, **kwargs)
        is_legacy_chronos2 = (
            config_dict.get("model_type") == "t5"
            and config_dict.get("architectures") == ["Chronos2Model"]
            and isinstance(config_dict.get("chronos_config"), dict)
        )
        if is_legacy_chronos2:
            config_dict = dict(config_dict)
            config_dict["model_type"] = cls.model_type
        return config_dict, kwargs

    def __post_init__(self, **kwargs):
        checkpoint_model_type = kwargs.pop("model_type", None)
        if checkpoint_model_type == "t5":
            is_legacy_chronos2 = self.architectures == ["Chronos2Model"] and isinstance(self.chronos_config, dict)
            if not is_legacy_chronos2:
                raise ValueError(
                    "A T5 configuration can only be migrated to `Chronos2Config` when it declares "
                    "`architectures=['Chronos2Model']` and contains a mapping-valued `chronos_config`."
                )
        elif checkpoint_model_type not in (None, self.model_type):
            raise ValueError(f"Cannot instantiate `Chronos2Config` from a `{checkpoint_model_type}` configuration.")

        self.chronos_config = _default_chronos_config() if self.chronos_config is None else dict(self.chronos_config)
        self.chronos_config.setdefault("use_reg_token", False)
        self.chronos_config.setdefault("use_arcsinh", False)
        self.chronos_config.setdefault("max_output_patches", 1)
        self.chronos_config.setdefault("time_encoding_scale", self.chronos_config.get("context_length"))

        act_info = self.feed_forward_proj.split("-")
        self.dense_act_fn = act_info[-1]
        self.is_gated_act = act_info[0] == "gated"
        PreTrainedConfig.__post_init__(self, **kwargs)
        self.tie_word_embeddings = False
        self.use_cache = False
        self.is_encoder_decoder = False

    def validate_architecture(self):
        super().validate_architecture()
        if self.is_gated_act:
            raise ValueError("Chronos-2 does not support gated feed-forward activations.")
        if self.d_kv % 2 != 0:
            raise ValueError(f"`d_kv` must be even for rotary embeddings, but is {self.d_kv}.")

        required_fields = {
            "context_length",
            "output_patch_size",
            "input_patch_size",
            "input_patch_stride",
            "quantiles",
        }
        missing_fields = required_fields.difference(self.chronos_config)
        if missing_fields:
            raise ValueError(f"`chronos_config` is missing required fields: {sorted(missing_fields)}")
        if self.chronos_config["input_patch_size"] != self.chronos_config["output_patch_size"]:
            raise ValueError(
                "`input_patch_size` and `output_patch_size` must be equal, but found "
                f"{self.chronos_config['input_patch_size']} and {self.chronos_config['output_patch_size']}."
            )
        for field_name in (
            "context_length",
            "output_patch_size",
            "input_patch_size",
            "input_patch_stride",
            "max_output_patches",
            "time_encoding_scale",
        ):
            value = self.chronos_config[field_name]
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"`chronos_config['{field_name}']` must be a positive integer.")

        quantiles = self.chronos_config["quantiles"]
        if not isinstance(quantiles, (list, tuple)) or not quantiles:
            raise ValueError("`chronos_config['quantiles']` must be a non-empty list or tuple.")
        if any(
            not isinstance(quantile, (int, float))
            or isinstance(quantile, bool)
            or not math.isfinite(quantile)
            or quantile <= 0.0
            or quantile >= 1.0
            for quantile in quantiles
        ):
            raise ValueError("All Chronos-2 quantiles must be finite and strictly between 0 and 1.")
        if any(left >= right for left, right in zip(quantiles, quantiles[1:])):
            raise ValueError("Chronos-2 quantiles must be strictly increasing.")


@dataclass
class Chronos2ForecastingSettings:
    context_length: int
    output_patch_size: int
    input_patch_size: int
    input_patch_stride: int
    quantiles: list[float]
    use_reg_token: bool
    use_arcsinh: bool
    max_output_patches: int
    time_encoding_scale: int


@auto_docstring
@dataclass
class Chronos2Output(ModelOutput):
    r"""
    loss (`torch.Tensor`, *optional*):
        Twice-pinball quantile loss, returned when `future_target` is provided.
    quantile_preds (`torch.Tensor` of shape `(batch_size, num_quantiles, prediction_length)`):
        Quantile forecasts in the original scale of each input series.
    hidden_states (`tuple[torch.Tensor, ...]`, *optional*):
        Encoder hidden states, including the input to the first block and the final normalized output.
    enc_time_self_attn_weights (`tuple[torch.Tensor, ...]`, *optional*):
        Per-layer attention weights along the time axis.
    enc_group_self_attn_weights (`tuple[torch.Tensor, ...]`, *optional*):
        Per-layer attention weights across series belonging to the same group.
    """

    loss: torch.Tensor | None = None
    quantile_preds: torch.Tensor | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    enc_time_self_attn_weights: tuple[torch.Tensor, ...] | None = None
    enc_group_self_attn_weights: tuple[torch.Tensor, ...] | None = None


@dataclass
class Chronos2AttentionOutput(ModelOutput):
    hidden_states: torch.Tensor | None = None
    attn_weights: torch.Tensor | None = None


@dataclass
class Chronos2EncoderBlockOutput(ModelOutput):
    hidden_states: torch.Tensor | None = None
    time_self_attn_weights: torch.Tensor | None = None
    group_self_attn_weights: torch.Tensor | None = None


@dataclass
class Chronos2EncoderOutput(ModelOutput):
    last_hidden_state: torch.Tensor | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    all_time_self_attn_weights: tuple[torch.Tensor, ...] | None = None
    all_group_self_attn_weights: tuple[torch.Tensor, ...] | None = None


class Chronos2Patch(nn.Module):
    def __init__(self, patch_size: int, patch_stride: int):
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        length = hidden_states.shape[-1]
        if length % self.patch_size != 0:
            padding_shape = (*hidden_states.shape[:-1], self.patch_size - length % self.patch_size)
            padding = torch.full(
                padding_shape, fill_value=torch.nan, dtype=hidden_states.dtype, device=hidden_states.device
            )
            hidden_states = torch.cat((padding, hidden_states), dim=-1)
        return hidden_states.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)


class Chronos2InstanceNorm(nn.Module):
    def __init__(self, eps: float = 1e-5, use_arcsinh: bool = False):
        super().__init__()
        self.eps = eps
        self.use_arcsinh = use_arcsinh

    def forward(
        self,
        hidden_states: torch.Tensor,
        loc_scale: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        original_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        if loc_scale is None:
            loc = torch.nan_to_num(torch.nanmean(hidden_states, dim=-1, keepdim=True), nan=0.0)
            scale = torch.nan_to_num((hidden_states - loc).square().nanmean(dim=-1, keepdim=True).sqrt(), nan=1.0)
            scale = torch.where(scale == 0, self.eps, scale)
        else:
            loc, scale = loc_scale

        hidden_states = (hidden_states - loc) / scale
        if self.use_arcsinh:
            hidden_states = torch.arcsinh(hidden_states)
        return hidden_states.to(original_dtype), (loc, scale)

    def inverse(self, hidden_states: torch.Tensor, loc_scale: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        original_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        loc, scale = loc_scale
        if self.use_arcsinh:
            hidden_states = torch.sinh(hidden_states)
        return (hidden_states * scale + loc).to(original_dtype)


class Chronos2ResidualBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: str,
        dropout: float = 0.0,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.act = ACT2FN[activation]
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.residual_layer = nn.Linear(input_dim, output_dim)
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = Chronos2LayerNorm(output_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = self.residual_layer(hidden_states)
        hidden_states = self.act(self.hidden_layer(hidden_states))
        hidden_states = self.dropout(self.output_layer(hidden_states))
        hidden_states = hidden_states + residual
        if self.use_layer_norm:
            hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class Chronos2LayerNorm(T5LayerNorm):
    pass


class Chronos2MLP(T5DenseActDense):
    pass


class Chronos2FeedForward(nn.Module):
    def __init__(self, config: Chronos2Config):
        super().__init__()
        self.mlp = Chronos2MLP(config)
        self.layer_norm = Chronos2LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.mlp(forwarded_states)
        return hidden_states + self.dropout(forwarded_states)


class Chronos2RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: Chronos2Config, device: torch.device | None = None):
        super().__init__()
        self.config = config
        self.rope_type = "default"
        inv_freq, self.attention_scaling = self.compute_default_rope_parameters(config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(
        config: Chronos2Config,
        device: torch.device | None = None,
        seq_len: int | None = None,
    ) -> tuple[torch.Tensor, float]:
        del seq_len
        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, config.d_kv, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / config.d_kv)
        )
        return inv_freq, 1.0

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(hidden_states.device)
        positions = position_ids[:, None, :].float()
        device_type = hidden_states.device.type if hidden_states.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):
            frequencies = (inv_freq.float() @ positions.float()).transpose(1, 2)
            embeddings = torch.cat((frequencies, frequencies), dim=-1)
            cos = embeddings.cos() * self.attention_scaling
            sin = embeddings.sin() * self.attention_scaling
        return cos.to(dtype=hidden_states.dtype), sin.to(dtype=hidden_states.dtype)


class Chronos2Attention(nn.Module):
    def __init__(self, config: Chronos2Config, use_rope: bool = True):
        super().__init__()
        self.d_model = config.d_model
        self.kv_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.kv_proj_dim
        self.config = config

        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        self.use_rope = use_rope
        if use_rope:
            self.rope_embed = Chronos2RotaryEmbedding(config)

    def _shape(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        return hidden_states.view(batch_size, sequence_length, self.n_heads, self.kv_proj_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> Chronos2AttentionOutput:
        if self.use_rope and position_ids is None:
            raise ValueError("`position_ids` must be provided when rotary embeddings are enabled.")

        query_states = self._shape(self.q(hidden_states))
        key_states = self._shape(self.k(hidden_states))
        value_states = self._shape(self.v(hidden_states))

        if self.use_rope:
            cos, sin = self.rope_embed(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attention_interface: Callable = (
            simple_eager_attention_forward
            if output_attentions
            else ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation, simple_eager_attention_forward
            )
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.dropout if self.training else 0.0,
            scaling=1.0,
            is_causal=False,
        )
        attn_output = self.o(attn_output.reshape(*hidden_states.shape[:-1], self.inner_dim))
        return Chronos2AttentionOutput(
            hidden_states=attn_output,
            attn_weights=attn_weights if output_attentions else None,
        )


class Chronos2TimeSelfAttention(nn.Module):
    def __init__(self, config: Chronos2Config):
        super().__init__()
        self.self_attention = Chronos2Attention(config, use_rope=True)
        self.layer_norm = Chronos2LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        output_attentions: bool = False,
    ) -> Chronos2AttentionOutput:
        attention_output = self.self_attention(
            self.layer_norm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + self.dropout(attention_output.hidden_states)
        return Chronos2AttentionOutput(hidden_states=hidden_states, attn_weights=attention_output.attn_weights)


class Chronos2GroupSelfAttention(nn.Module):
    def __init__(self, config: Chronos2Config):
        super().__init__()
        self.self_attention = Chronos2Attention(config, use_rope=False)
        self.layer_norm = Chronos2LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> Chronos2AttentionOutput:
        hidden_states = hidden_states.transpose(0, 1)
        attention_output = self.self_attention(
            self.layer_norm(hidden_states),
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + self.dropout(attention_output.hidden_states)
        hidden_states = hidden_states.transpose(0, 1)
        return Chronos2AttentionOutput(hidden_states=hidden_states, attn_weights=attention_output.attn_weights)


class Chronos2EncoderBlock(nn.Module):
    def __init__(self, config: Chronos2Config):
        super().__init__()
        self.layer = nn.ModuleList(
            [
                Chronos2TimeSelfAttention(config),
                Chronos2GroupSelfAttention(config),
                Chronos2FeedForward(config),
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        group_time_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> Chronos2EncoderBlockOutput:
        time_outputs = self.layer[0](
            hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        group_outputs = self.layer[1](
            time_outputs.hidden_states,
            attention_mask=group_time_mask,
            output_attentions=output_attentions,
        )
        hidden_states = self.layer[2](group_outputs.hidden_states)
        return Chronos2EncoderBlockOutput(
            hidden_states=hidden_states,
            time_self_attn_weights=time_outputs.attn_weights,
            group_self_attn_weights=group_outputs.attn_weights,
        )


class Chronos2Encoder(nn.Module):
    def __init__(self, config: Chronos2Config):
        super().__init__()
        self.block = nn.ModuleList([Chronos2EncoderBlock(config) for _ in range(config.num_layers)])
        self.final_layer_norm = Chronos2LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    @staticmethod
    def _expand_and_invert_time_attention_mask(
        attention_mask: torch.Tensor, floating_type: torch.dtype
    ) -> torch.Tensor:
        if attention_mask.ndim != 2:
            raise ValueError("`attention_mask` must have shape `(batch_size, sequence_length)`.")
        attention_mask = attention_mask[:, None, None, :].to(dtype=floating_type)
        return (1.0 - attention_mask) * torch.finfo(floating_type).min

    @staticmethod
    def _construct_and_invert_group_time_mask(
        group_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        floating_type: torch.dtype,
    ) -> torch.Tensor:
        group_mask = group_ids[:, None] == group_ids[None, :]
        group_time_mask = torch.einsum("qb,bt->qbt", group_mask, attention_mask)
        if torch.is_floating_point(group_time_mask):
            floating_type = group_time_mask.dtype
        group_time_mask = group_time_mask.permute(2, 0, 1).unsqueeze(1)
        return (1.0 - group_time_mask) * torch.finfo(floating_type).min

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        *,
        group_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Chronos2EncoderOutput:
        batch_size, sequence_length = inputs_embeds.shape[:-1]
        print(
            "sushmanth: encoder input",
            {
                "inputs_embeds": tuple(inputs_embeds.shape),
                "group_ids": tuple(group_ids.shape),
                "attention_mask": None if attention_mask is None else tuple(attention_mask.shape),
            },
        )
        if position_ids is None:
            position_ids = torch.arange(sequence_length, dtype=torch.long, device=inputs_embeds.device).unsqueeze(0)
        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, sequence_length, device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )

        extended_attention_mask = self._expand_and_invert_time_attention_mask(attention_mask, inputs_embeds.dtype)
        group_time_mask = self._construct_and_invert_group_time_mask(group_ids, attention_mask, inputs_embeds.dtype)

        all_hidden_states = () if output_hidden_states else None
        all_time_attentions = () if output_attentions else None
        all_group_attentions = () if output_attentions else None
        hidden_states = self.dropout(inputs_embeds)

        for layer_module in self.block:
            if output_hidden_states:
                all_hidden_states = (*all_hidden_states, hidden_states)
            layer_outputs = layer_module(
                hidden_states,
                position_ids=position_ids,
                attention_mask=extended_attention_mask,
                group_time_mask=group_time_mask,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs.hidden_states
            if output_attentions:
                all_time_attentions = (*all_time_attentions, layer_outputs.time_self_attn_weights)
                all_group_attentions = (*all_group_attentions, layer_outputs.group_self_attn_weights)

        hidden_states = self.dropout(self.final_layer_norm(hidden_states))
        if output_hidden_states:
            all_hidden_states = (*all_hidden_states, hidden_states)

        return Chronos2EncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            all_time_self_attn_weights=all_time_attentions,
            all_group_self_attn_weights=all_group_attentions,
        )


@auto_docstring
class Chronos2PreTrainedModel(PreTrainedModel):
    config_class = Chronos2Config
    main_input_name = "context"
    _no_split_modules = ["Chronos2EncoderBlock"]
    _supports_sdpa = True

    @torch.no_grad()
    def _init_weights(self, module: nn.Module):
        super()._init_weights(module)
        factor = self.config.initializer_factor
        if isinstance(module, Chronos2LayerNorm):
            init.constant_(module.weight, factor)
        elif isinstance(module, Chronos2MLP):
            init.normal_(module.wi.weight, mean=0.0, std=factor * self.config.d_model**-0.5)
            init.normal_(module.wo.weight, mean=0.0, std=factor * self.config.d_ff**-0.5)
        elif isinstance(module, Chronos2Attention):
            init.normal_(
                module.q.weight,
                mean=0.0,
                std=factor * (self.config.d_model * self.config.d_kv) ** -0.5,
            )
            init.normal_(module.k.weight, mean=0.0, std=factor * self.config.d_model**-0.5)
            init.normal_(module.v.weight, mean=0.0, std=factor * self.config.d_model**-0.5)
            init.normal_(
                module.o.weight,
                mean=0.0,
                std=factor * (self.config.num_heads * self.config.d_kv) ** -0.5,
            )
        elif isinstance(module, Chronos2ResidualBlock):
            for layer in (module.hidden_layer, module.residual_layer, module.output_layer):
                init.normal_(layer.weight, mean=0.0, std=factor * layer.weight.shape[-1] ** -0.5)
                init.zeros_(layer.bias)
        elif isinstance(module, Chronos2Model):
            init.normal_(module.shared.weight, mean=0.0, std=factor)
            quantiles = torch.tensor(
                module.chronos_config.quantiles,
                dtype=module.dtype,
                device=module.quantiles.device,
            )
            init.copy_(module.quantiles, quantiles)


@auto_docstring(
    custom_intro="""
    The Chronos-2 model for probabilistic univariate, multivariate, and covariate-informed time-series forecasting.
    """
)
class Chronos2Model(Chronos2PreTrainedModel):
    _supports_long_horizon = True
    _supports_future_covariates = True

    def __init__(self, config: Chronos2Config):
        super().__init__(config)
        self.model_dim = config.d_model
        chronos_config = dict(config.chronos_config)
        chronos_config.setdefault("time_encoding_scale", chronos_config["context_length"])
        config.chronos_config = chronos_config
        self.chronos_config = Chronos2ForecastingSettings(**chronos_config)

        if self.chronos_config.use_reg_token:
            config.reg_token_id = 1
        config.vocab_size = 2 if self.chronos_config.use_reg_token else 1
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        self.input_patch_embedding = Chronos2ResidualBlock(
            input_dim=self.chronos_config.input_patch_size * 3,
            hidden_dim=config.d_ff,
            output_dim=config.d_model,
            activation=config.feed_forward_proj.split("-")[-1],
            dropout=config.dropout_rate,
        )
        self.patch = Chronos2Patch(
            patch_size=self.chronos_config.input_patch_size,
            patch_stride=self.chronos_config.input_patch_stride,
        )
        self.instance_norm = Chronos2InstanceNorm(use_arcsinh=self.chronos_config.use_arcsinh)
        self.encoder = Chronos2Encoder(config)

        self.num_quantiles = len(self.chronos_config.quantiles)
        self.register_buffer(
            "quantiles",
            torch.tensor(self.chronos_config.quantiles, dtype=self.dtype),
            persistent=False,
        )
        self.output_patch_embedding = Chronos2ResidualBlock(
            input_dim=config.d_model,
            hidden_dim=config.d_ff,
            output_dim=self.num_quantiles * self.chronos_config.output_patch_size,
            activation=config.feed_forward_proj.split("-")[-1],
            dropout=config.dropout_rate,
        )
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.shared

    def set_input_embeddings(self, value: nn.Module):
        self.shared = value

    def _validate_input(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor | None,
        group_ids: torch.Tensor | None,
        future_covariates: torch.Tensor | None,
        future_covariates_mask: torch.Tensor | None,
        num_output_patches: int,
        future_target: torch.Tensor | None,
        future_target_mask: torch.Tensor | None,
    ):
        output_patch_size = self.chronos_config.output_patch_size
        if context.ndim != 2:
            raise ValueError(f"`context` must have shape `(batch_size, context_length)`, found {context.shape}.")
        if not torch.is_floating_point(context):
            raise ValueError("`context` must be a floating-point tensor.")
        if not isinstance(num_output_patches, int) or num_output_patches <= 0:
            raise ValueError("`num_output_patches` must be a positive integer.")
        if context_mask is not None and context_mask.shape != context.shape:
            raise ValueError(f"`context_mask` must have shape {context.shape}, found {context_mask.shape}.")
        if group_ids is not None and group_ids.shape != (context.shape[0],):
            raise ValueError(f"`group_ids` must have shape `({context.shape[0]},)`, found {group_ids.shape}.")

        if future_covariates is not None:
            if future_covariates.ndim != 2 or future_covariates.shape[0] != context.shape[0]:
                raise ValueError(
                    "`future_covariates` must have shape "
                    f"`(batch_size={context.shape[0]}, future_length)`, found {future_covariates.shape}."
                )
            if not torch.is_floating_point(future_covariates):
                raise ValueError("`future_covariates` must be a floating-point tensor.")
            if future_covariates.shape[-1] > num_output_patches * output_patch_size:
                raise ValueError(
                    f"`num_output_patches={num_output_patches}` cannot accommodate future covariates of length "
                    f"{future_covariates.shape[-1]}."
                )
            if future_target is not None and future_target.shape != future_covariates.shape:
                raise ValueError(
                    "`future_target` and `future_covariates` must have the same shape, but found "
                    f"{future_target.shape} and {future_covariates.shape}."
                )
        if future_covariates_mask is not None:
            if future_covariates is None:
                raise ValueError("`future_covariates` must be provided with `future_covariates_mask`.")
            if future_covariates_mask.shape != future_covariates.shape:
                raise ValueError(
                    "`future_covariates_mask` and `future_covariates` must have the same shape, but found "
                    f"{future_covariates_mask.shape} and {future_covariates.shape}."
                )

        if future_target is not None:
            if future_target.ndim != 2 or future_target.shape[0] != context.shape[0]:
                raise ValueError(
                    f"`future_target` must have shape `(batch_size={context.shape[0]}, future_length)`, found "
                    f"{future_target.shape}."
                )
            if not torch.is_floating_point(future_target):
                raise ValueError("`future_target` must be a floating-point tensor.")
            if future_target.shape[-1] > num_output_patches * output_patch_size:
                raise ValueError(
                    f"`num_output_patches={num_output_patches}` cannot accommodate a future target of length "
                    f"{future_target.shape[-1]}."
                )
        if future_target_mask is not None:
            if future_target is None:
                raise ValueError("`future_target` must be provided with `future_target_mask`.")
            if future_target_mask.shape != future_target.shape:
                raise ValueError(
                    "`future_target_mask` and `future_target` must have the same shape, but found "
                    f"{future_target_mask.shape} and {future_target.shape}."
                )

    def _prepare_patched_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        context_mask = (
            context_mask.to(context.dtype)
            if context_mask is not None
            else torch.isnan(context).logical_not().to(context.dtype)
        )
        batch_size, context_length = context.shape
        if context_length > self.chronos_config.context_length:
            context = context[..., -self.chronos_config.context_length :]
            context_mask = context_mask[..., -self.chronos_config.context_length :]

        context = torch.where(context_mask > 0.0, context, torch.nan)
        context, loc_scale = self.instance_norm(context)
        context = context.to(self.dtype)
        context_mask = context_mask.to(self.dtype)

        patched_context = self.patch(context)
        patched_mask = torch.nan_to_num(self.patch(context_mask), nan=0.0)
        patched_context = torch.where(patched_mask > 0.0, patched_context, 0.0)
        attention_mask = patched_mask.sum(dim=-1) > 0
        num_context_patches = attention_mask.shape[-1]

        final_context_length = num_context_patches * self.chronos_config.input_patch_size
        context_time_encoding = torch.arange(
            -final_context_length,
            0,
            device=self.device,
            dtype=torch.float32,
        )
        context_time_encoding = context_time_encoding.view(
            1, num_context_patches, self.chronos_config.input_patch_size
        ).expand(batch_size, -1, -1)
        context_time_encoding = context_time_encoding.div(self.chronos_config.time_encoding_scale).to(self.dtype)
        patched_context = torch.cat((context_time_encoding, patched_context, patched_mask), dim=-1)
        return patched_context, attention_mask, loc_scale

    def _prepare_patched_future(
        self,
        future_covariates: torch.Tensor | None,
        future_covariates_mask: torch.Tensor | None,
        loc_scale: tuple[torch.Tensor, torch.Tensor],
        num_output_patches: int,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output_patch_size = self.chronos_config.output_patch_size
        if future_covariates is not None:
            future_covariates, _ = self.instance_norm(future_covariates, loc_scale)
            future_covariates = future_covariates.to(self.dtype)
            if future_covariates_mask is None:
                future_covariates_mask = torch.isnan(future_covariates).logical_not().to(future_covariates.dtype)
            else:
                future_covariates_mask = future_covariates_mask.to(future_covariates)
            future_covariates = torch.where(future_covariates_mask > 0.0, future_covariates, 0.0)
            if torch.isnan(future_covariates).any():
                raise ValueError(
                    "`future_covariates` contains NaN values at positions not masked by `future_covariates_mask`."
                )

            final_future_length = num_output_patches * output_patch_size
            if final_future_length > future_covariates.shape[-1]:
                padding_shape = (*future_covariates.shape[:-1], final_future_length - future_covariates.shape[-1])
                future_covariates = torch.cat(
                    (future_covariates, torch.zeros(padding_shape).to(future_covariates)), dim=-1
                )
                future_covariates_mask = torch.cat(
                    (future_covariates_mask, torch.zeros(padding_shape).to(future_covariates_mask)), dim=-1
                )
            patched_future_covariates = future_covariates.reshape(batch_size, num_output_patches, output_patch_size)
            patched_future_covariates_mask = future_covariates_mask.reshape(
                batch_size, num_output_patches, output_patch_size
            )
        else:
            patched_future_covariates = torch.zeros(
                batch_size,
                num_output_patches,
                output_patch_size,
                device=self.device,
                dtype=self.dtype,
            )
            patched_future_covariates_mask = torch.zeros_like(patched_future_covariates)

        final_future_length = num_output_patches * output_patch_size
        future_time_encoding = torch.arange(
            final_future_length,
            device=self.device,
            dtype=torch.float32,
        )
        future_time_encoding = future_time_encoding.view(1, num_output_patches, output_patch_size).expand(
            batch_size, -1, -1
        )
        future_time_encoding = future_time_encoding.div(self.chronos_config.time_encoding_scale).to(self.dtype)
        patched_future = torch.cat(
            (future_time_encoding, patched_future_covariates, patched_future_covariates_mask), dim=-1
        )
        return patched_future, patched_future_covariates_mask

    def _compute_loss(
        self,
        quantile_preds: torch.Tensor,
        future_target: torch.Tensor,
        future_target_mask: torch.Tensor | None,
        patched_future_covariates_mask: torch.Tensor,
        loc_scale: tuple[torch.Tensor, torch.Tensor],
        num_output_patches: int,
    ) -> torch.Tensor:
        batch_size = future_target.shape[0]
        output_patch_size = self.chronos_config.output_patch_size
        future_target, _ = self.instance_norm(future_target, loc_scale)
        future_target = future_target.unsqueeze(1).to(self.device)
        future_target_mask = (
            future_target_mask.unsqueeze(1).to(self.device)
            if future_target_mask is not None
            else ~torch.isnan(future_target)
        )
        future_target = torch.where(future_target_mask > 0.0, future_target, 0.0)

        if quantile_preds.shape[-1] > future_target.shape[-1]:
            padding_shape = (*future_target.shape[:-1], quantile_preds.shape[-1] - future_target.shape[-1])
            future_target = torch.cat((future_target, torch.zeros(padding_shape).to(future_target)), dim=-1)
            future_target_mask = torch.cat(
                (future_target_mask, torch.zeros(padding_shape).to(future_target_mask)), dim=-1
            )

        quantiles = self.quantiles.view(1, self.num_quantiles, 1)
        quantile_loss = 2 * torch.abs(
            (future_target - quantile_preds) * ((future_target <= quantile_preds).float() - quantiles)
        )
        inverse_covariate_mask = 1 - patched_future_covariates_mask.reshape(
            batch_size, 1, num_output_patches * output_patch_size
        )
        loss_mask = future_target_mask.float() * inverse_covariate_mask
        return (quantile_loss * loss_mask).mean(dim=-1).sum(dim=-1).mean()

    def encode(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        group_ids: torch.Tensor | None = None,
        future_covariates: torch.Tensor | None = None,
        future_covariates_mask: torch.Tensor | None = None,
        num_output_patches: int = 1,
        future_target: torch.Tensor | None = None,
        future_target_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> tuple[
        Chronos2EncoderOutput,
        tuple[torch.Tensor, torch.Tensor],
        torch.Tensor,
        int,
    ]:
        """Prepare and encode context and future patches before the forecasting head.

        Returns the encoder output, normalization statistics, the patched future-covariate mask, and the number of
        context patches.
        """
        self._validate_input(
            context,
            context_mask,
            group_ids,
            future_covariates,
            future_covariates_mask,
            num_output_patches,
            future_target,
            future_target_mask,
        )
        batch_size = context.shape[0]
        patched_context, attention_mask, loc_scale = self._prepare_patched_context(context, context_mask)
        num_context_patches = attention_mask.shape[-1]
        input_embeds = self.input_patch_embedding(patched_context)

        if self.chronos_config.use_reg_token:
            reg_input_ids = torch.full(
                (batch_size, 1), self.config.reg_token_id, dtype=torch.long, device=input_embeds.device
            )
            input_embeds = torch.cat((input_embeds, self.shared(reg_input_ids)), dim=-2)
            attention_mask = torch.cat(
                (attention_mask.to(self.dtype), torch.ones_like(reg_input_ids, dtype=self.dtype)), dim=-1
            )

        patched_future, patched_future_covariates_mask = self._prepare_patched_future(
            future_covariates,
            future_covariates_mask,
            loc_scale,
            num_output_patches,
            batch_size,
        )
        future_embeds = self.input_patch_embedding(patched_future)
        future_attention_mask = torch.ones(batch_size, num_output_patches, dtype=self.dtype, device=self.device)
        input_embeds = torch.cat((input_embeds, future_embeds), dim=-2)
        attention_mask = torch.cat((attention_mask, future_attention_mask), dim=-1)

        print(
            "sushmanth: prepared model inputs",
            {
                "input_embeds": tuple(input_embeds.shape),
                "attention_mask": tuple(attention_mask.shape),
                "context_patches": num_context_patches,
                "output_patches": num_output_patches,
            },
        )

        if group_ids is None:
            group_ids = torch.arange(batch_size, dtype=torch.long, device=self.device)
        encoder_outputs = self.encoder(
            inputs_embeds=input_embeds,
            group_ids=group_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        return encoder_outputs, loc_scale, patched_future_covariates_mask, num_context_patches

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        group_ids: torch.Tensor | None = None,
        future_covariates: torch.Tensor | None = None,
        future_covariates_mask: torch.Tensor | None = None,
        num_output_patches: int = 1,
        future_target: torch.Tensor | None = None,
        future_target_mask: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Chronos2Output | tuple[torch.Tensor, ...]:
        r"""
        context (`torch.Tensor` of shape `(batch_size, context_length)`):
            Historical values. Missing values may be represented by NaNs or identified with `context_mask`.
        context_mask (`torch.Tensor` of shape `(batch_size, context_length)`, *optional*):
            Binary mask indicating observed context values.
        group_ids (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Series with equal IDs exchange information through group self-attention. By default every row is treated
            independently.
        future_covariates (`torch.Tensor` of shape `(batch_size, future_length)`, *optional*):
            Known future values for any series rows. Unknown values should be represented by NaNs.
        future_covariates_mask (`torch.Tensor` of shape `(batch_size, future_length)`, *optional*):
            Binary mask indicating known future covariate values.
        num_output_patches (`int`, *optional*, defaults to 1):
            Number of future patches to forecast in this forward pass.
        future_target (`torch.Tensor` of shape `(batch_size, future_length)`, *optional*):
            Target values used to compute the training loss.
        future_target_mask (`torch.Tensor` of shape `(batch_size, future_length)`, *optional*):
            Binary mask indicating observed target values.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        print(
            "sushmanth: starting Chronos-2 forward",
            {
                "context": tuple(context.shape),
                "future_covariates": None if future_covariates is None else tuple(future_covariates.shape),
                "num_output_patches": num_output_patches,
            },
        )
        batch_size = context.shape[0]
        encoder_outputs, loc_scale, patched_future_covariates_mask, num_context_patches = self.encode(
            context=context,
            context_mask=context_mask,
            group_ids=group_ids,
            future_covariates=future_covariates,
            future_covariates_mask=future_covariates_mask,
            num_output_patches=num_output_patches,
            future_target=future_target,
            future_target_mask=future_target_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = encoder_outputs.last_hidden_state
        if hidden_states is None:
            raise RuntimeError("The Chronos-2 encoder did not return hidden states.")
        num_special_tokens = int(self.chronos_config.use_reg_token)
        expected_shape = (
            batch_size,
            num_context_patches + num_special_tokens + num_output_patches,
            self.model_dim,
        )
        if hidden_states.shape != expected_shape:
            raise ValueError(f"Unexpected encoder output shape {hidden_states.shape}; expected {expected_shape}.")

        forecast_embeds = hidden_states[:, -num_output_patches:]
        quantile_preds = self.output_patch_embedding(forecast_embeds)
        quantile_preds = quantile_preds.view(
            batch_size,
            num_output_patches,
            self.num_quantiles,
            self.chronos_config.output_patch_size,
        )
        quantile_preds = quantile_preds.permute(0, 2, 1, 3).reshape(
            batch_size,
            self.num_quantiles,
            num_output_patches * self.chronos_config.output_patch_size,
        )

        loss = None
        if future_target is not None:
            loss = self._compute_loss(
                quantile_preds,
                future_target,
                future_target_mask,
                patched_future_covariates_mask,
                loc_scale,
                num_output_patches,
            )

        prediction_length = num_output_patches * self.chronos_config.output_patch_size
        quantile_preds = quantile_preds.reshape(batch_size, self.num_quantiles * prediction_length)
        quantile_preds = self.instance_norm.inverse(quantile_preds, loc_scale)
        quantile_preds = quantile_preds.reshape(batch_size, self.num_quantiles, prediction_length)

        print(
            "sushmanth: finished Chronos-2 forward",
            {
                "quantile_preds": tuple(quantile_preds.shape),
                "loss_computed": loss is not None,
            },
        )

        return Chronos2Output(
            loss=loss,
            quantile_preds=quantile_preds,
            hidden_states=encoder_outputs.hidden_states,
            enc_time_self_attn_weights=encoder_outputs.all_time_self_attn_weights,
            enc_group_self_attn_weights=encoder_outputs.all_group_self_attn_weights,
        )

    @staticmethod
    def _get_prob_mass_per_quantile_level(quantile_levels: torch.Tensor) -> torch.Tensor:
        if quantile_levels.ndim != 1 or quantile_levels.numel() == 0:
            raise ValueError("`quantile_levels` must be a non-empty one-dimensional tensor.")
        if not torch.isfinite(quantile_levels).all() or quantile_levels.min() <= 0.0 or quantile_levels.max() >= 1.0:
            raise ValueError("All unrolled quantiles must be finite and strictly between 0 and 1.")
        if quantile_levels.numel() > 1 and not torch.all(quantile_levels[1:] > quantile_levels[:-1]):
            raise ValueError("`quantile_levels` must be strictly increasing.")
        boundaries = torch.cat(
            (
                torch.tensor([0.0], device=quantile_levels.device),
                quantile_levels,
                torch.tensor([1.0], device=quantile_levels.device),
            )
        )
        probability_mass = (boundaries[2:] - boundaries[:-2]) / 2
        return probability_mass / probability_mass.sum()

    @staticmethod
    def _interpolate_quantiles(
        query_quantile_levels: torch.Tensor,
        original_quantile_levels: torch.Tensor,
        original_values: torch.Tensor,
    ) -> torch.Tensor:
        original_dtype = original_values.dtype
        device = original_values.device
        query_quantile_levels = query_quantile_levels.to(device)
        original_quantile_levels = original_quantile_levels.to(device)
        original_values = original_values.to(torch.float32)

        original_shape = original_values.shape
        num_original_quantiles = original_quantile_levels.shape[-1]
        original_values = original_values.reshape(-1, num_original_quantiles)
        batch_size = original_values.shape[0]
        if original_quantile_levels.ndim == 1:
            original_quantile_levels = original_quantile_levels.expand(batch_size, -1)
        else:
            original_quantile_levels = original_quantile_levels.reshape(-1, num_original_quantiles)

        sorted_levels, sorted_indices = torch.sort(original_quantile_levels, dim=-1)
        sorted_values = torch.gather(original_values, dim=-1, index=sorted_indices)
        levels = []
        values = []
        if original_quantile_levels.min() > 0.0:
            levels.append(torch.zeros((batch_size, 1), dtype=torch.float32, device=device))
            values.append(sorted_values[:, :1])
        levels.append(sorted_levels)
        values.append(sorted_values)
        if original_quantile_levels.max() < 1.0:
            levels.append(torch.ones((batch_size, 1), dtype=torch.float32, device=device))
            values.append(sorted_values[:, -1:])
        sorted_levels = torch.cat(levels, dim=-1).contiguous()
        sorted_values = torch.cat(values, dim=-1)

        queries = query_quantile_levels.unsqueeze(0).expand(batch_size, -1).contiguous()
        upper_indices = torch.searchsorted(sorted_levels, queries, right=True)
        upper_indices = torch.clamp(upper_indices, max=sorted_levels.shape[-1] - 1)
        lower_indices = upper_indices - 1
        lower_levels = torch.gather(sorted_levels, dim=1, index=lower_indices)
        upper_levels = torch.gather(sorted_levels, dim=1, index=upper_indices)
        lower_values = torch.gather(sorted_values, dim=1, index=lower_indices)
        upper_values = torch.gather(sorted_values, dim=1, index=upper_indices)
        weights = torch.nan_to_num((queries - lower_levels) / (upper_levels - lower_levels), nan=0.0)
        interpolated = lower_values + weights * (upper_values - lower_values)
        return interpolated.reshape(*original_shape[:-1], query_quantile_levels.numel()).to(original_dtype)

    @classmethod
    def _weighted_quantile(
        cls,
        query_quantile_levels: torch.Tensor,
        sample_weights: torch.Tensor,
        samples: torch.Tensor,
    ) -> torch.Tensor:
        original_dtype = samples.dtype
        device = samples.device
        query_quantile_levels = query_quantile_levels.to(device)
        sample_weights = sample_weights.to(device)
        samples = samples.to(torch.float32)
        original_shape = samples.shape
        num_samples = sample_weights.numel()
        samples = samples.reshape(-1, num_samples)
        batch_size = samples.shape[0]
        sample_weights = (sample_weights / sample_weights.sum()).expand(batch_size, -1).contiguous()
        sorted_samples, sort_indices = torch.sort(samples, dim=-1)
        sorted_weights = torch.gather(sample_weights, dim=-1, index=sort_indices)
        cumulative_weights = torch.cumsum(sorted_weights, dim=-1).clamp(min=0.0, max=1.0)
        quantiles = cls._interpolate_quantiles(
            query_quantile_levels,
            cumulative_weights,
            sorted_samples,
        )
        return quantiles.reshape(*original_shape[:-1], query_quantile_levels.numel()).to(original_dtype)

    def _predict_step(
        self,
        context: torch.Tensor,
        group_ids: torch.Tensor,
        future_covariates: torch.Tensor | None,
        num_output_patches: int,
    ) -> torch.Tensor:
        if future_covariates is not None:
            output_size = num_output_patches * self.chronos_config.output_patch_size
            if output_size > future_covariates.shape[1]:
                padding = torch.full(
                    (future_covariates.shape[0], output_size - future_covariates.shape[1]),
                    fill_value=torch.nan,
                    device=future_covariates.device,
                )
                future_covariates = torch.cat((future_covariates, padding), dim=1)
            else:
                future_covariates = future_covariates[..., :output_size]
        return self(
            context=context,
            group_ids=group_ids,
            future_covariates=future_covariates,
            num_output_patches=num_output_patches,
        ).quantile_preds.to(context)

    @staticmethod
    def _slide_context_and_future_covariates(
        context: torch.Tensor,
        future_covariates: torch.Tensor,
        slide_by: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        future_slice = future_covariates[..., :slide_by]
        context[..., -slide_by:] = torch.where(torch.isnan(future_slice), context[..., -slide_by:], future_slice)
        return context, future_covariates[..., slide_by:]

    @torch.no_grad()
    def predict(
        self,
        context: torch.Tensor,
        prediction_length: int | None = None,
        group_ids: torch.Tensor | None = None,
        future_covariates: torch.Tensor | None = None,
        max_output_patches: int | None = None,
        unrolled_quantiles: list[float] | tuple[float, ...] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forecast prepared rank-2 tensors, using quantile-path unrolling for long horizons."""
        if context.ndim != 2 or not torch.is_floating_point(context):
            raise ValueError("`context` must be a rank-2 floating-point tensor.")
        prediction_length = (
            self.chronos_config.max_output_patches * self.chronos_config.output_patch_size
            if prediction_length is None
            else prediction_length
        )
        if not isinstance(prediction_length, int) or prediction_length <= 0:
            raise ValueError("`prediction_length` must be a positive integer.")
        max_output_patches = (
            self.chronos_config.max_output_patches if max_output_patches is None else max_output_patches
        )
        if not isinstance(max_output_patches, int) or max_output_patches <= 0:
            raise ValueError("`max_output_patches` must be a positive integer.")
        direct_prediction_length = max_output_patches * self.chronos_config.output_patch_size
        if prediction_length > direct_prediction_length:
            logger.warning_once(
                f"Chronos-2 is unrolling beyond its {direct_prediction_length}-step direct forecast. "
                "Forecast quality may degrade because the model was not optimized for this longer horizon."
            )

        context = context.to(device=self.device, dtype=torch.float32)
        batch_size = context.shape[0]
        if group_ids is None:
            group_ids = torch.arange(batch_size, dtype=torch.long, device=self.device)
        else:
            if group_ids.shape != (batch_size,):
                raise ValueError(f"`group_ids` must have shape `({batch_size},)`, found {group_ids.shape}.")
            group_ids = group_ids.to(self.device)

        if future_covariates is not None:
            if future_covariates.ndim != 2 or future_covariates.shape[0] != batch_size:
                raise ValueError(f"`future_covariates` must have shape `(batch_size={batch_size}, future_length)`.")
            future_covariates = future_covariates.to(device=self.device, dtype=torch.float32)
            if future_covariates.shape[-1] < prediction_length:
                padding = torch.full(
                    (batch_size, prediction_length - future_covariates.shape[-1]),
                    fill_value=torch.nan,
                    device=self.device,
                )
                future_covariates = torch.cat((future_covariates, padding), dim=-1)

        def get_num_output_patches(remaining_horizon: int) -> int:
            num_patches = (remaining_horizon + self.chronos_config.output_patch_size - 1) // (
                self.chronos_config.output_patch_size
            )
            return min(num_patches, max_output_patches)

        remaining = prediction_length
        prediction = self._predict_step(
            context,
            group_ids,
            future_covariates,
            get_num_output_patches(remaining),
        )
        predictions = [prediction]
        remaining -= prediction.shape[-1]

        if remaining <= 0:
            return torch.cat(predictions, dim=-1)[..., :prediction_length].to(dtype=torch.float32, device="cpu")

        model_quantiles = self.chronos_config.quantiles
        if unrolled_quantiles is None:
            default_unrolled_quantiles = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
            unrolled_quantiles = (
                default_unrolled_quantiles
                if all(
                    any(abs(quantile - model_quantile) <= 1e-6 for model_quantile in model_quantiles)
                    for quantile in default_unrolled_quantiles
                )
                else model_quantiles
            )
        unrolled_quantiles_tensor = torch.as_tensor(unrolled_quantiles, dtype=torch.float32, device=self.device)
        if any(
            not any(abs(float(quantile) - model_quantile) <= 1e-6 for model_quantile in model_quantiles)
            for quantile in unrolled_quantiles_tensor.cpu()
        ):
            raise ValueError("`unrolled_quantiles` must be a subset of the quantiles in `config.chronos_config`.")
        self._get_prob_mass_per_quantile_level(unrolled_quantiles_tensor)

        num_paths = unrolled_quantiles_tensor.numel()
        context = context.unsqueeze(1).repeat(1, num_paths, 1)
        group_ids = group_ids.unsqueeze(1).repeat(1, num_paths)
        group_ids = group_ids * num_paths + torch.arange(num_paths, device=self.device).unsqueeze(0)
        if future_covariates is not None:
            future_covariates = future_covariates.unsqueeze(1).repeat(1, num_paths, 1)
        sample_weights = torch.outer(
            self._get_prob_mass_per_quantile_level(unrolled_quantiles_tensor),
            self._get_prob_mass_per_quantile_level(
                torch.tensor(self.chronos_config.quantiles, dtype=torch.float32, device=self.device)
            ),
        )

        while remaining > 0:
            prediction_unrolled = self._interpolate_quantiles(
                unrolled_quantiles_tensor,
                self.quantiles,
                prediction.transpose(1, 2),
            ).transpose(1, 2)
            context = torch.cat((context, prediction_unrolled), dim=-1)[..., -self.chronos_config.context_length :]
            if future_covariates is not None:
                context, future_covariates = self._slide_context_and_future_covariates(
                    context, future_covariates, prediction.shape[-1]
                )

            num_paths = unrolled_quantiles_tensor.numel()
            flat_context = context.reshape(batch_size * num_paths, context.shape[-1])
            flat_group_ids = group_ids.reshape(batch_size * num_paths)
            flat_future_covariates = (
                future_covariates.reshape(batch_size * num_paths, future_covariates.shape[-1])
                if future_covariates is not None
                else None
            )
            prediction = self._predict_step(
                flat_context,
                flat_group_ids,
                flat_future_covariates,
                get_num_output_patches(remaining),
            )
            step_length = prediction.shape[-1]
            prediction = prediction.reshape(batch_size, num_paths * self.num_quantiles, step_length)
            prediction = self._weighted_quantile(
                self.quantiles,
                sample_weights.reshape(-1),
                prediction.transpose(1, 2),
            ).transpose(1, 2)
            predictions.append(prediction)
            remaining -= prediction.shape[-1]

        return torch.cat(predictions, dim=-1)[..., :prediction_length].to(dtype=torch.float32, device="cpu")


__all__ = ["Chronos2Config", "Chronos2Model", "Chronos2Output", "Chronos2PreTrainedModel"]
