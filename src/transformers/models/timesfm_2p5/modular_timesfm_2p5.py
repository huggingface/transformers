# coding=utf-8
# Copyright 2025 the HuggingFace Team. All rights reserved.
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

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import PretrainedConfig
from ...integrations import use_kernel_forward_from_hub
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import logging
from ..timesfm.configuration_timesfm import TimesFmConfig
from ..timesfm.modeling_timesfm import (
    TimesFmAttention,
    TimesFmDecoderLayer,
    TimesFmMLP,
    TimesFmModel,
    TimesFmModelForPrediction,
    TimesFmOutput,
    TimesFmOutputForPrediction,
    TimesFmPreTrainedModel,
    TimesFmResidualBlock,
    simple_eager_attention_forward,
)


logger = logging.get_logger(__name__)

_TOLERANCE = 1e-6


def revin(
    x: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    reverse: bool = False,
):
    """Reversible instance normalization matching official TimesFM 2.5."""
    if len(mu.shape) == len(x.shape) - 1:
        mu = mu[..., None]
        sigma = sigma[..., None]
    elif len(mu.shape) == len(x.shape) - 2:
        mu = mu[..., None, None]
        sigma = sigma[..., None, None]

    if reverse:
        return x * sigma + mu
    else:
        return (x - mu) / torch.where(sigma < _TOLERANCE, 1.0, sigma)


class PerDimScale(nn.Module):
    """Per-dimension scaling matching official TimesFM 2.5."""

    def __init__(self, num_dims: int):
        super().__init__()
        self.num_dims = num_dims
        self.per_dim_scale = nn.Parameter(torch.zeros(num_dims))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale_factor = 1.442695041 / math.sqrt(self.num_dims) * F.softplus(self.per_dim_scale)
        return x * scale_factor


class Timesfm2P5Config(TimesFmConfig, PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`Timesfm2P5ModelForPrediction`]. It is used to
    instantiate a TimesFM 2.5 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the TimesFM 2.5
    [google/timesfm-2.5-200m-pytorch](https://huggingface.co/google/timesfm-2.5-200m-pytorch) architecture.

    Configuration objects inherit from [`TimesFmConfig`] and can be used to control the model outputs. Read the
    documentation from [`TimesFmConfig`] for more information.

    Args:
        patch_length (`int`, *optional*, defaults to 32): <fill_docstring>
        context_length (`int`, *optional*, defaults to 512): <fill_docstring>
        horizon_length (`int`, *optional*, defaults to 128): <fill_docstring>
        freq_size (`int`, *optional*, defaults to 3): <fill_docstring>
        num_hidden_layers (`int`, *optional*, defaults to 50): <fill_docstring>
        hidden_size (`int`, *optional*, defaults to 1280): <fill_docstring>
        intermediate_size (`int`, *optional*, defaults to 1280): <fill_docstring>
        head_dim (`int`, *optional*, defaults to 80): <fill_docstring>
        num_attention_heads (`int`, *optional*, defaults to 16): <fill_docstring>
        tolerance (`float`, *optional*, defaults to 1e-06): <fill_docstring>
        rms_norm_eps (`float`, *optional*, defaults to 1e-06): <fill_docstring>
        quantiles (`list`, *optional*, defaults to `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`): <fill_docstring>
        pad_val (`float`, *optional*, defaults to 1123581321.0): <fill_docstring>
        attention_dropout (`float`, *optional*, defaults to 0.0): <fill_docstring>
        use_positional_embedding (`bool`, *optional*, defaults to `False`): <fill_docstring>
        initializer_range (`float`, *optional*, defaults to 0.02): <fill_docstring>
        min_timescale (`int`, *optional*, defaults to 1): <fill_docstring>
        max_timescale (`int`, *optional*, defaults to 10000): <fill_docstring>
        use_rotary_position_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to use rotary positional embeddings instead of traditional sinusoidal embeddings.
        use_revin_normalization (`bool`, *optional*, defaults to `True`):
            Whether to use reversible instance normalization (RevIN) for input preprocessing.
    """

    model_type = "timesfm_2p5"

    def __init__(
        self,
        patch_length: int = 32,
        context_length: int = 512,
        horizon_length: int = 128,
        freq_size: int = 3,
        num_hidden_layers: int = 50,
        hidden_size: int = 1280,
        intermediate_size: int = 1280,
        head_dim: int = 80,
        num_attention_heads: int = 16,
        tolerance: float = 1e-6,
        rms_norm_eps: float = 1e-6,
        quantiles: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        pad_val: float = 1123581321.0,
        attention_dropout: float = 0.0,
        use_positional_embedding: bool = False,
        initializer_range: float = 0.02,
        min_timescale: int = 1,
        max_timescale: int = 10_000,
        use_rotary_position_embeddings: bool = True,
        use_revin_normalization: bool = True,
        **kwargs,
    ):
        # TimesFM 2.5 specific parameters
        self.use_rotary_position_embeddings = use_rotary_position_embeddings
        self.use_revin_normalization = use_revin_normalization

        # Call parent constructor with all parameters
        super().__init__(
            patch_length=patch_length,
            context_length=context_length,
            horizon_length=horizon_length,
            freq_size=freq_size,
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            head_dim=head_dim,
            num_attention_heads=num_attention_heads,
            tolerance=tolerance,
            rms_norm_eps=rms_norm_eps,
            quantiles=quantiles,
            pad_val=pad_val,
            attention_dropout=attention_dropout,
            use_positional_embedding=use_positional_embedding,
            initializer_range=initializer_range,
            min_timescale=min_timescale,
            max_timescale=max_timescale,
            **kwargs,
        )


class Timesfm2P5Output(TimesFmOutput):
    pass


class Timesfm2P5OutputForPrediction(TimesFmOutputForPrediction):
    pass


class Timesfm2P5MLP(TimesFmMLP):
    pass


class Timesfm2P5ResidualBlock(TimesFmResidualBlock):
    pass


@use_kernel_forward_from_hub("RMSNorm")
class Timesfm2P5RMSNorm(nn.Module):
    """RMS normalization for TimesFM 2.5 matching official implementation."""

    def __init__(self, num_features: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(num_features))
        self.num_features = num_features
        self.epsilon = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        input_dtype = inputs.dtype
        inputs = inputs.to(torch.float32)
        var = torch.mean(torch.square(inputs), dim=-1, keepdim=True)
        normed_inputs = inputs * torch.rsqrt(var + self.epsilon)
        normed_inputs = normed_inputs * self.scale
        return normed_inputs.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.scale.shape)}, eps={self.epsilon}"


class Timesfm2P5RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embedding matching official TimesFM 2.5."""

    def __init__(
        self,
        embedding_dims: int,
        min_timescale: float = 1.0,
        max_timescale: float = 10000.0,
    ):
        super().__init__()
        self.embedding_dims = embedding_dims
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale

    def forward(
        self,
        inputs: torch.Tensor,
        position: Optional[torch.Tensor] = None,
    ):
        """Apply rotary positional embeddings matching official implementation."""
        if self.embedding_dims != inputs.shape[-1]:
            raise ValueError(
                "The embedding dims of the rotary position embedding must match the hidden dimension of the inputs."
            )

        half_embedding_dim = self.embedding_dims // 2
        fraction = 2 * torch.arange(0, half_embedding_dim, device=inputs.device) / self.embedding_dims
        timescale = (self.min_timescale * (self.max_timescale / self.min_timescale) ** fraction).to(inputs.device)

        if position is None:
            seq_length = inputs.shape[1]
            position = torch.arange(seq_length, dtype=torch.float32, device=inputs.device)[None, :]

        if len(inputs.shape) == 4:
            position = position[..., None, None]
            timescale = timescale[None, None, None, :]
        elif len(inputs.shape) == 3:
            position = position[..., None]
            timescale = timescale[None, None, :]
        else:
            raise ValueError("Inputs must be of rank 3 or 4.")

        sinusoid_inp = position / timescale
        sin = torch.sin(sinusoid_inp)
        cos = torch.cos(sinusoid_inp)
        first_half, second_half = torch.chunk(inputs, 2, dim=-1)
        first_part = first_half * cos - second_half * sin
        second_part = second_half * cos + first_half * sin
        return torch.cat([first_part, second_part], dim=-1)


class Timesfm2P5Attention(TimesFmAttention):
    """TimesFM 2.5 attention matching official implementation with rotary embeddings, QK norm, and per-dim scaling."""

    def __init__(self, config: Timesfm2P5Config, layer_idx: int):
        super().__init__(config, layer_idx)

        # Override with no-bias projections to match official implementation
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * config.head_dim, config.hidden_size, bias=False)

        # Add QK normalization (RMS norm on query and key)
        self.query_ln = Timesfm2P5RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.key_ln = Timesfm2P5RMSNorm(config.head_dim, eps=config.rms_norm_eps)

        # Add per-dimension scaling
        self.per_dim_scale = PerDimScale(num_dims=config.head_dim)

        # Rotary positional embeddings
        if config.use_rotary_position_embeddings:
            self.rotary_emb = Timesfm2P5RotaryPositionalEmbedding(
                embedding_dims=config.head_dim,
                min_timescale=config.min_timescale,
                max_timescale=config.max_timescale,
            )
        else:
            self.rotary_emb = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_length = hidden_states.shape[:2]

        # Project to query, key, value
        query_states = self.q_proj(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim)

        # Apply rotary positional embeddings if enabled
        if self.rotary_emb is not None:
            query_states = self.rotary_emb(query_states)
            key_states = self.rotary_emb(key_states)

        # Apply QK normalization
        query_states = self.query_ln(query_states)
        key_states = self.key_ln(key_states)

        # Apply per-dimension scaling to query
        query_states = self.per_dim_scale(query_states)

        # Transpose for attention computation
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Use attention implementation
        attention_interface = simple_eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=1.0,
            **kwargs,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Timesfm2P5DecoderLayer(TimesFmDecoderLayer):
    """TimesFM 2.5 decoder layer with updated normalization."""

    def __init__(self, config: Timesfm2P5Config, layer_idx: int):
        nn.Module.__init__(self)

        self.self_attn = Timesfm2P5Attention(config, layer_idx=layer_idx)
        self.mlp = Timesfm2P5MLP(config)
        self.input_layernorm = Timesfm2P5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Timesfm2P5PreTrainedModel(TimesFmPreTrainedModel):
    config_class = Timesfm2P5Config
    base_model_prefix = "timesfm_2p5"


class Timesfm2P5Model(TimesFmModel):
    """TimesFM 2.5 model with updated normalization and rotary embeddings."""

    def __init__(self, config: Timesfm2P5Config):
        super().__init__(config)

        self.config = config
        self.input_ff_layer = Timesfm2P5ResidualBlock(
            input_dims=2 * config.patch_length,
            output_dims=config.hidden_size,
            hidden_dims=config.intermediate_size,
        )
        self.freq_emb = nn.Embedding(num_embeddings=config.freq_size, embedding_dim=config.hidden_size)
        self.layers = nn.ModuleList(
            [Timesfm2P5DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        # Use traditional positional embeddings if rotary is disabled
        if self.config.use_positional_embedding and not self.config.use_rotary_position_embeddings:
            from ..timesfm.modeling_timesfm import TimesFmPositionalEmbedding

            self.position_emb = TimesFmPositionalEmbedding(config=config)

        self.post_init()

    def _revin_transform(
        self, inputs: torch.Tensor, patched_pads: torch.Tensor, reverse: bool = False
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Reversible instance normalization (RevIN) for TimesFM 2.5."""
        if not reverse:
            # Forward normalization using masked statistics
            mu, sigma = self._timesfm_2p5_masked_mean_std(inputs, patched_pads)

            # Apply RevIN normalization
            normalized = revin(inputs, mu, sigma, reverse=False)

            # Handle padding values
            normalized = torch.where(
                torch.abs(inputs - self.config.pad_val) < self.config.tolerance,
                torch.tensor(self.config.pad_val, dtype=normalized.dtype, device=normalized.device),
                normalized,
            )
            return normalized, (mu, sigma)
        else:
            # Reverse normalization
            mu, sigma = patched_pads  # stats passed as second argument when reverse=True
            return revin(inputs, mu, sigma, reverse=True), (mu, sigma)

    def _forward_transform(
        self, inputs: torch.Tensor, patched_pads: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Choose normalization method based on config."""
        if self.config.use_revin_normalization:
            return self._revin_transform(inputs, patched_pads, reverse=False)
        else:
            # Use original TimesFM normalization
            return super()._forward_transform(inputs, patched_pads)


class Timesfm2P5ModelForPrediction(TimesFmModelForPrediction):
    """TimesFM 2.5 model for prediction with updated architecture."""

    def __init__(self, config: Timesfm2P5Config):
        super().__init__(config)

        self.config = config
        self.context_len = config.context_length
        self.horizon_len = config.horizon_length

        self.decoder = Timesfm2P5Model(config)

        # quantile and mean output
        self.horizon_ff_layer = Timesfm2P5ResidualBlock(
            input_dims=config.hidden_size,
            output_dims=config.horizon_length * (1 + len(config.quantiles)),
            hidden_dims=config.intermediate_size,
        )

        self.post_init()


__all__ = [
    "Timesfm2P5Config",
    "Timesfm2P5ModelForPrediction",
    "Timesfm2P5PreTrainedModel",
    "Timesfm2P5Model",
    "Timesfm2P5RMSNorm",
    "Timesfm2P5RotaryPositionalEmbedding",
    "Timesfm2P5Attention",
]
