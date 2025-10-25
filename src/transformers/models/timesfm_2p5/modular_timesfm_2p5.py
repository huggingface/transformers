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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...utils import auto_docstring, can_return_tuple, logging
from ..gemma2.modeling_gemma2 import (
    Gemma2Attention,
    Gemma2RotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from ..llama.modeling_llama import LlamaRMSNorm
from ..timesfm.configuration_timesfm import TimesFmConfig
from ..timesfm.modeling_timesfm import (
    TimesFmModelForPrediction,
    TimesFmOutput,
    TimesFmOutputForPrediction,
    TimesFmPreTrainedModel,
)


logger = logging.get_logger(__name__)


class Timesfm2P5Config(TimesFmConfig):
    """Configuration class for TimesFM 2.5 model."""

    model_type = "timesfm_2p5"

    def __init__(
        self,
        # TimesFM 2.5 specific parameters
        patch_length: int = 32,
        context_length: int = 16384,
        horizon_length: int = 128,
        quantiles: list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        pad_val: float = -1e9,
        freq_size: int = 10,  # Not used in 2.5, but kept for compatibility
        hidden_size: int = 1280,
        intermediate_size: int = 1280,
        head_dim: int = 80,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 16,  # Same as num_attention_heads for full attention
        tolerance: float = 1e-5,
        rms_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
        initializer_range: float = 0.02,
        min_timescale: float = 1.0,
        max_timescale: float = 10000.0,
        # Override defaults for 2.5
        num_hidden_layers: int = 20,
        output_quantile_len: int = 1024,  # From original TimesFM 2.5 config
        decode_index: int = 5,
        use_rotary_embeddings: bool = True,
        use_qk_norm: bool = True,
        use_per_dim_scale: bool = True,
        use_bias: bool = False,
        activation: str = "swish",
        use_positional_embedding: bool = False,  # TimesFM 2.5 uses rotary embeddings instead
        use_continuous_quantile_head: bool = True,
        normalize_inputs: bool = True,
        # Gemma2-compatible parameters for query scaling
        query_pre_attn_scalar: float = 256.0,  # This provides the per-dim scaling
        attn_logit_softcapping: Optional[float] = None,
        layer_types: Optional[list] = None,  # All layers are the same type
        sliding_window: Optional[int] = None,  # No sliding window
        max_position_embeddings: int = 16384,  # Should match context_length
        rope_theta: float = 10000.0,  # RoPE theta parameter
        **kwargs,
    ):
        super().__init__(
            patch_length=patch_length,
            context_length=context_length,
            horizon_length=horizon_length,
            quantiles=quantiles,
            pad_val=pad_val,
            freq_size=freq_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            head_dim=head_dim,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            tolerance=tolerance,
            rms_norm_eps=rms_norm_eps,
            attention_dropout=attention_dropout,
            attention_bias=attention_bias,
            initializer_range=initializer_range,
            min_timescale=min_timescale,
            max_timescale=max_timescale,
            num_hidden_layers=num_hidden_layers,
            use_positional_embedding=use_positional_embedding,
            layer_types=layer_types or ["attention"] * num_hidden_layers,
            sliding_window=sliding_window,
            **kwargs,
        )
        self.output_quantile_len = output_quantile_len
        self.decode_index = decode_index
        self.use_rotary_embeddings = use_rotary_embeddings
        self.use_qk_norm = use_qk_norm
        self.use_per_dim_scale = use_per_dim_scale
        self.use_bias = use_bias
        self.activation = activation
        self.use_continuous_quantile_head = use_continuous_quantile_head
        self.normalize_inputs = normalize_inputs
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.attn_logit_softcapping = attn_logit_softcapping
        self.num_key_value_heads = num_key_value_heads
        self.attention_bias = attention_bias
        self.layer_types = layer_types or ["attention"] * num_hidden_layers
        self.sliding_window = sliding_window
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta


@dataclass
@auto_docstring
class Timesfm2P5Output(TimesFmOutput):
    r"""
    context_mu (`torch.Tensor` of shape `(batch_size, num_patches)`):
        Running means computed per input patch during normalization.
    context_sigma (`torch.Tensor` of shape `(batch_size, num_patches)`):
        Running standard deviations computed per input patch during normalization.
    patch_padding (`torch.Tensor` of shape `(batch_size, num_patches)`):
        Boolean mask indicating fully padded patches.
    """

    context_mu: Optional[torch.Tensor] = None
    context_sigma: Optional[torch.Tensor] = None
    patch_padding: Optional[torch.Tensor] = None


@dataclass
@auto_docstring
class Timesfm2P5OutputForPrediction(TimesFmOutputForPrediction):
    r"""
    mean_predictions (`torch.Tensor` of shape `(batch_size, horizon_length)`):
        Deterministic forecasts after denormalization.
    full_predictions (`torch.Tensor` of shape `(batch_size, horizon_length, quantiles)`):
        Quantile forecasts including the median after denormalization.
    loss (`torch.Tensor` of shape `(1,)`, *optional*, returned when `future_values` is provided):
        Training loss combining MSE and quantile losses when targets are supplied.
    """

    pass


class Timesfm2P5MLP(nn.Module):
    """
    TimesFM 2.5 MLP layer with configurable activation.

    This is a feedforward network with two linear layers and configurable activation.
    """

    def __init__(self, config: Timesfm2P5Config):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        use_bias = config.use_bias

        self.ff0 = nn.Linear(hidden_size, intermediate_size, bias=use_bias)
        self.ff1 = nn.Linear(intermediate_size, hidden_size, bias=use_bias)

        if config.activation == "swish":
            self.activation = nn.SiLU()
        elif config.activation == "relu":
            self.activation = nn.ReLU()
        elif config.activation == "none":
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation: {config.activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.ff0(x)
        hidden = self.activation(hidden)
        output = self.ff1(hidden)
        return output


class Timesfm2P5ResidualBlock(nn.Module):
    """
    TimesFM 2.5 residual block with configurable activation and bias.

    This implements the ResidualBlock from TimesFM 2.5 which supports:
    - Configurable activation functions (relu, swish/silu, none)
    - Optional bias in linear layers
    - Residual connection from input to output
    """

    def __init__(
        self, input_dims: int, hidden_dims: int, output_dims: int, use_bias: bool = True, activation: str = "swish"
    ):
        super().__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims
        self.use_bias = use_bias
        self.activation_type = activation

        # Linear layers
        self.hidden_layer = nn.Linear(input_dims, hidden_dims, bias=use_bias)
        self.output_layer = nn.Linear(hidden_dims, output_dims, bias=use_bias)
        self.residual_layer = nn.Linear(input_dims, output_dims, bias=use_bias)

        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "swish" or activation == "silu":
            self.activation = nn.SiLU()
        elif activation == "none":
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Activation '{activation}' not supported. Choose from 'relu', 'swish', or 'none'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the residual block.

        Args:
            x: Input tensor of shape (batch_size, ..., input_dims)

        Returns:
            Output tensor of shape (batch_size, ..., output_dims)
        """
        hidden = self.hidden_layer(x)
        hidden = self.activation(hidden)
        output = self.output_layer(hidden)
        residual = self.residual_layer(x)
        return output + residual


class Timesfm2P5RMSNorm(LlamaRMSNorm):
    pass


class Timesfm2P5RotaryEmbedding(Gemma2RotaryEmbedding):
    pass


class Timesfm2P5Attention(Gemma2Attention):
    """
    TimesFM 2.5 attention extends Gemma2Attention but overrides the forward to implement
    the exact TimesFM 2.5 operations: QK normalization + per-dimension scaling
    """

    def __init__(self, config: Timesfm2P5Config, layer_idx: int):
        super().__init__(config, layer_idx)

        # Add QK normalization specific to TimesFM 2.5
        self.use_qk_norm = getattr(config, "use_qk_norm", True)
        if self.use_qk_norm:
            self.query_ln = Timesfm2P5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.key_ln = Timesfm2P5RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Add per-dimension scaling parameter (same as TimesFmAttention)
        self.use_per_dim_scale = getattr(config, "use_per_dim_scale", True)
        if self.use_per_dim_scale:
            self.scaling = nn.Parameter(torch.empty((self.head_dim,)))

    def _scale_query(self, query: torch.Tensor) -> torch.Tensor:
        """Per-dimension scaling - exact copy from TimesFmAttention."""
        if not self.use_per_dim_scale:
            return query
        scale = F.softplus(self.scaling).mul(1.442695041 / math.sqrt(self.head_dim))
        return query * scale[None, None, None, :]

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values=None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """Forward with TimesFM 2.5 specific QK normalization and per-dimension scaling."""
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Linear projections
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # Apply rotary position embeddings before normalization/scaling (matches original TimesFM)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Apply QK normalization (TimesFM 2.5 specific)
        if self.use_qk_norm:
            query_states = self.query_ln(query_states)
            key_states = self.key_ln(key_states)

        # Apply per-dimension scaling to query (TimesFM 2.5 specific)
        query_states = self._scale_query(query_states)

        # Handle past key/value for caching
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Use the standard attention computation from Gemma2
        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # Attention computation with custom scaling disabled (we use per_dim_scale instead)
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=1.0,  # No scaling - we already applied per_dim_scale to queries
            sliding_window=getattr(self, "sliding_window", None),
            softcap=getattr(self, "attn_logit_softcapping", None),
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Timesfm2P5DecoderLayer(nn.Module):
    """
    TimesFM 2.5 Transformer decoder layer.

    This layer consists of:
    - Self-attention with rotary embeddings and QK normalization
    - MLP feedforward network with configurable activation
    - RMS normalization
    """

    def __init__(self, config: Timesfm2P5Config, layer_idx: int):
        super().__init__()

        # Attention layers
        self.self_attn = Timesfm2P5Attention(config, layer_idx=layer_idx)

        # Normalization layers
        self.pre_attn_ln = Timesfm2P5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attn_ln = Timesfm2P5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_ff_ln = Timesfm2P5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_ff_ln = Timesfm2P5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # MLP
        self.mlp = Timesfm2P5MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        # Self-Attention with pre and post normalization
        residual = hidden_states
        hidden_states = self.pre_attn_ln(hidden_states)
        hidden_states, scores = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            **kwargs,
        )
        hidden_states = self.post_attn_ln(hidden_states) + residual

        # MLP with pre and post normalization
        residual = hidden_states
        hidden_states = self.pre_ff_ln(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_ff_ln(hidden_states) + residual

        return scores, hidden_states


class Timesfm2P5PreTrainedModel(TimesFmPreTrainedModel):
    config_class = Timesfm2P5Config
    base_model_prefix = "timesfm_2p5"
    _no_split_modules = ["Timesfm2P5DecoderLayer"]


class Timesfm2P5Model(Timesfm2P5PreTrainedModel):
    """
    TimesFM 2.5 model - standalone implementation (not inheriting from TimesFmModel).

    Uses TimesFM 2.5 specific architecture:
    - Timesfm2P5ResidualBlock for input projection
    - Timesfm2P5DecoderLayer for transformer layers
    - No frequency embedding (model adapts automatically)
    - No positional embedding (uses rotary embeddings)
    """

    def __init__(self, config: Timesfm2P5Config):
        super().__init__(config)
        self.config = config
        self._tolerance = 1e-6

        # Input projection with TimesFM 2.5 ResidualBlock
        # Note: tokenizer uses bias=True (different from transformer layers)
        self.input_ff_layer = Timesfm2P5ResidualBlock(
            input_dims=2 * config.patch_length,  # 64 (32*2)
            hidden_dims=config.hidden_size,  # 1280 (not intermediate_size)
            output_dims=config.hidden_size,  # 1280
            use_bias=True,  # tokenizer uses bias=True
            activation=config.activation,  # "swish"
        )

        # TimesFM 2.5 has NO frequency embedding - model adapts automatically
        # (This is a key difference from TimesFM 2.0)

        # Transformer layers with TimesFM 2.5 specific components
        self.layers = nn.ModuleList(
            [Timesfm2P5DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        # TimesFM 2.5 uses rotary embeddings - add rotary embedding component
        self.rotary_emb = Timesfm2P5RotaryEmbedding(config)

        # Initialize weights and apply final processing
        # self.post_init()  # Temporarily disabled due to initialization issue

    def _revin(self, x: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        if len(loc.shape) == len(x.shape) - 1:
            loc = loc[..., None]
            scale = scale[..., None]
        elif len(loc.shape) == len(x.shape) - 2:
            loc = loc[..., None, None]
            scale = scale[..., None, None]

        safe_scale = torch.where(scale < self._tolerance, torch.ones_like(scale), scale)

        if reverse:
            return x * scale + loc

        return (x - loc) / safe_scale

    @staticmethod
    def _update_running_stats(
        n: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        is_valid = (~mask).to(x.dtype)
        inc_n = is_valid.sum(dim=-1)

        inc_n_safe = torch.where(inc_n == 0, torch.ones_like(inc_n), inc_n)
        inc_mu = (x * is_valid).sum(dim=-1) / inc_n_safe
        inc_mu = torch.where(inc_n == 0, torch.zeros_like(inc_mu), inc_mu)

        centered = x - inc_mu.unsqueeze(-1)
        inc_var = ((centered * is_valid) ** 2).sum(dim=-1) / inc_n_safe
        inc_var = torch.where(inc_n == 0, torch.zeros_like(inc_var), inc_var)
        inc_sigma = torch.sqrt(torch.clamp(inc_var, min=0.0))

        new_n = n + inc_n
        new_n_safe = torch.where(new_n == 0, torch.ones_like(new_n), new_n)

        new_mu = (n * mu + inc_mu * inc_n) / new_n_safe
        new_mu = torch.where(new_n == 0, torch.zeros_like(new_mu), new_mu)

        term1 = n * sigma.pow(2)
        term2 = inc_n * inc_sigma.pow(2)
        term3 = n * (mu - new_mu).pow(2)
        term4 = inc_n * (inc_mu - new_mu).pow(2)

        new_var = (term1 + term2 + term3 + term4) / new_n_safe
        new_var = torch.where(new_n == 0, torch.zeros_like(new_var), new_var)
        new_sigma = torch.sqrt(torch.clamp(new_var, min=0.0))

        return new_n, new_mu, new_sigma

    def forward(
        self,
        past_values: torch.Tensor,
        past_values_padding: torch.LongTensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        batch_size, seq_len = past_values.shape
        patch_len = self.config.patch_length

        patched_inputs = past_values.view(batch_size, -1, patch_len)
        context_padding = past_values_padding[:, :seq_len]
        patched_masks = context_padding.view(batch_size, -1, patch_len)
        patched_masks_bool = patched_masks >= 0.5

        n = past_values.new_zeros(batch_size)
        mu = past_values.new_zeros(batch_size)
        sigma = past_values.new_zeros(batch_size)
        mu_history: list[torch.Tensor] = []
        sigma_history: list[torch.Tensor] = []

        for i in range(patched_inputs.shape[1]):
            n, mu, sigma = self._update_running_stats(
                n,
                mu,
                sigma,
                patched_inputs[:, i, :],
                patched_masks_bool[:, i, :],
            )
            mu_history.append(mu)
            sigma_history.append(sigma)

        if mu_history:
            context_mu = torch.stack(mu_history, dim=1)
            context_sigma = torch.stack(sigma_history, dim=1)
        else:
            context_mu = mu.unsqueeze(1)
            context_sigma = sigma.unsqueeze(1)

        normed_inputs = self._revin(patched_inputs, context_mu, context_sigma, reverse=False)
        normed_inputs = torch.where(patched_masks_bool, torch.zeros_like(normed_inputs), normed_inputs)

        tokenizer_inputs = torch.cat(
            [normed_inputs, patched_masks_bool.to(dtype=normed_inputs.dtype)],
            dim=-1,
        ).to(dtype=self.dtype)
        input_embeddings = self.input_ff_layer(tokenizer_inputs)

        patch_padding = patched_masks_bool[..., -1]
        attention_mask = self._make_attention_mask(patch_padding, input_embeddings.dtype)

        # Compute position IDs accounting for padding (matches original TimesFM implementation)
        # position = arange(n_patches) - num_masked
        # This ensures padded positions get negative values and real data starts at position 0
        sequence_length = input_embeddings.shape[1]
        num_masked = patch_padding.to(torch.int32).sum(dim=-1, keepdim=True)
        position_ids = torch.arange(sequence_length, device=input_embeddings.device).unsqueeze(0) - num_masked
        position_embeddings = self.rotary_emb(input_embeddings, position_ids)

        hidden_states = input_embeddings
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
            )

            if output_attentions:
                attention_weights, hidden_states = layer_outputs
                all_attentions = all_attentions + (attention_weights,)
            else:
                _, hidden_states = layer_outputs

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        loc = context_mu[:, -1]
        scale = torch.clamp(context_sigma[:, -1], min=self._tolerance)

        return Timesfm2P5Output(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            loc=loc,
            scale=scale,
            context_mu=context_mu,
            context_sigma=context_sigma,
            patch_padding=patch_padding,
        )

    def _make_attention_mask(self, patch_padding: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        batch_size, seq_len = patch_padding.shape
        num_masked = patch_padding.to(torch.int32).sum(dim=-1)
        q_index = torch.arange(seq_len, device=patch_padding.device)[None, None, :, None]
        kv_index = torch.arange(seq_len, device=patch_padding.device)[None, None, None, :]
        allowed = torch.logical_and(q_index >= kv_index, kv_index >= num_masked[:, None, None, None])

        mask = torch.zeros((batch_size, 1, seq_len, seq_len), dtype=dtype, device=patch_padding.device)
        mask = mask.masked_fill(~allowed, torch.finfo(dtype).min)
        return mask


class Timesfm2P5ModelForPrediction(TimesFmModelForPrediction):
    """
    TimesFM 2.5 model for quantile and mean prediction.

    Inherits from TimesFmModelForPrediction but uses:
    - Timesfm2P5Model as the decoder
    - Separate output projections for point and quantile predictions (matching original TimesFM 2.5)
    """

    def __init__(self, config: Timesfm2P5Config):
        # Call the parent's __init__ first to get the basic structure
        super().__init__(config)

        # Now override with TimesFM 2.5 specific components
        self.config = config
        self.context_len = config.context_length
        self.horizon_len = config.horizon_length

        # Override decoder with TimesFM 2.5 model
        self.decoder = Timesfm2P5Model(config)

        # Replace the parent's horizon_ff_layer with TimesFM 2.5 separate output projections
        # Point prediction projection: hidden_size -> horizon_length * num_quantiles
        # Example: 1280 -> 128 * 10 = 1280 (for original config)
        num_quantiles = len(config.quantiles) + 1
        point_output_size = config.horizon_length * num_quantiles
        self.output_projection_point = Timesfm2P5ResidualBlock(
            input_dims=config.hidden_size,
            hidden_dims=config.hidden_size,
            output_dims=point_output_size,
            use_bias=config.use_bias,  # False
            activation=config.activation,  # "swish"
        )

        # Quantile prediction projection: hidden_size -> output_quantile_len * (num_quantiles + 1)
        # Original: 1024 * 10 = 10240 (9 quantiles + 1 extra)
        output_quantile_len = getattr(config, "output_quantile_len", 1024)  # Default from original
        quantile_output_size = output_quantile_len * (len(config.quantiles) + 1)
        self.output_projection_quantiles = Timesfm2P5ResidualBlock(
            input_dims=config.hidden_size,
            hidden_dims=config.hidden_size,
            output_dims=quantile_output_size,  # Dynamic based on horizon_length
            use_bias=config.use_bias,  # False
            activation=config.activation,  # "swish"
        )

        # Keep the parent's horizon_ff_layer for compatibility but we'll use separate projections
        # Note: parent's horizon_ff_layer will exist but we won't use it in our _postprocess_output

        # Initialize weights and apply final processing
        self.post_init()

    def _preprocess(
        self, inputs: Sequence[torch.Tensor], freq: Sequence[int], context_len: Optional[int] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Override parent's _preprocess to support custom context_len for forecasting.

        Args:
          inputs: A list of 1d Tensors. Each Tensor is the context time series.
          freq: list of frequencies
          context_len: Optional context length to pad to. If None, uses self.context_len.

        Returns:
          Tuple of (padded inputs, padding mask, frequencies)
        """
        if context_len is None:
            context_len = self.context_len

        input_ts, input_padding, inp_freq = [], [], []

        for i, ts in enumerate(inputs):
            input_len = ts.shape[0]
            padding = torch.zeros(input_len + self.horizon_len, dtype=ts.dtype, device=ts.device)
            if input_len < context_len:
                num_front_pad = context_len - input_len
                ts = torch.cat([torch.zeros(num_front_pad, dtype=ts.dtype, device=ts.device), ts], dim=0)
                padding = torch.cat([torch.ones(num_front_pad, dtype=ts.dtype, device=padding.device), padding], dim=0)
            elif input_len > context_len:
                ts = ts[-context_len:]
                padding = padding[-(context_len + self.horizon_len) :]

            input_ts.append(ts)
            input_padding.append(padding)
            inp_freq.append(freq[i])

        return (
            torch.stack(input_ts, dim=0),
            torch.stack(input_padding, dim=0),
            torch.tensor(inp_freq, dtype=torch.int32).reshape(-1, 1),
        )

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        past_values: Sequence[torch.Tensor],
        window_size: Optional[int] = None,
        future_values: Optional[torch.Tensor] = None,
        forecast_context_len: Optional[int] = None,
        return_forecast_on_context: bool = False,
        truncate_negative: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Timesfm2P5OutputForPrediction:
        """
        TimesFM 2.5 forward method matching original forecast operations.

        TimesFM 2.5 simplified API - no frequency parameter needed as the model
        automatically adapts to different time series frequencies.

        Args:
            past_values (`Sequence[torch.Tensor]`):
                Past values of the time series that serves as input to the model.
                Each tensor is a 1D time series of variable length.
            window_size (`int`, *optional*):
                Window size of trend + residual decomposition. If None then we do not do decomposition.
            future_values (`torch.Tensor`, *optional*):
                Optional future time series values to be used for loss computation.
            forecast_context_len (`int`, *optional*):
                Optional max context length.
            return_forecast_on_context (`bool`, *optional*):
                True to return the forecast on the context when available.
            truncate_negative (`bool`, *optional*):
                Truncate to only non-negative values if any contexts have non-negative values.
            return_dict (`bool`, *optional*):
                Whether or not to return a ModelOutput instead of a plain tuple.
            output_attentions (`bool`, *optional*):
                Whether to output the attentions.
            output_hidden_states (`bool`, *optional*):
                Whether to output the hidden states.

        Returns:
            Timesfm2P5OutputForPrediction: Output with mean_predictions, full_predictions, and optional loss.
        """
        if forecast_context_len is None:
            fcontext_len = self.context_len
        else:
            fcontext_len = forecast_context_len

        device = past_values[0].device

        if return_forecast_on_context:
            raise NotImplementedError("`return_forecast_on_context` is not supported for TimesFM 2.5 conversion yet.")

        inputs = [ts[-fcontext_len:] for ts in past_values]
        inp_min = torch.min(torch.stack([torch.min(ts) for ts in inputs]))

        if window_size is not None:
            new_inputs: list[torch.Tensor] = []
            for ts in inputs:
                new_inputs.extend(self._timesfm_moving_average(ts, window_size))
            inputs = new_inputs

        freq = [0] * len(inputs)

        if output_attentions is None:
            output_attentions = self.config.output_attentions
        if output_hidden_states is None:
            output_hidden_states = self.config.output_hidden_states

        # Pass fcontext_len to _preprocess so it pads to the correct length
        input_ts, input_padding, _ = self._preprocess(inputs, freq, context_len=fcontext_len)
        input_ts = input_ts.to(device)
        input_padding = input_padding.to(device)

        mu_global = input_ts.mean(dim=1, keepdim=True)
        sigma_global = input_ts.std(dim=1, keepdim=True)
        sigma_safe = torch.where(
            sigma_global < self.decoder._tolerance,
            torch.ones_like(sigma_global),
            sigma_global,
        )

        normalized_ts = (input_ts - mu_global) / sigma_safe
        normalized_ts = normalized_ts.to(self.decoder.dtype)

        model_outputs = self.decoder(
            past_values=normalized_ts,
            past_values_padding=input_padding,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = model_outputs.last_hidden_state
        context_mu = model_outputs.context_mu
        context_sigma = model_outputs.context_sigma

        point_output_norm = self.output_projection_point(hidden_states)
        quantile_output_norm = self.output_projection_quantiles(hidden_states)

        point_output = self.decoder._revin(point_output_norm, context_mu, context_sigma, reverse=True)
        quantile_output = self.decoder._revin(quantile_output_norm, context_mu, context_sigma, reverse=True)

        # Reshape point output: [batch, num_patches, horizon_len * num_quantiles] -> [batch, num_patches, horizon_len, num_quantiles]
        batch_size, num_patches, _ = point_output.shape
        num_quantiles = len(self.config.quantiles) + 1
        point_output = point_output.view(batch_size, num_patches, self.config.horizon_length, num_quantiles)
        pf_outputs = point_output[:, -1, :, :]  # Take last patch: [batch, horizon_len, num_quantiles]

        # Reshape quantile output: [batch, num_patches, output_quantile_len * num_quantiles] -> [batch, num_patches, output_quantile_len, num_quantiles]
        output_quantile_len = getattr(self.config, "output_quantile_len", 1024)
        quantile_output = quantile_output.view(batch_size, num_patches, output_quantile_len, num_quantiles)
        quantile_spreads = quantile_output[:, -1, :, :]  # Take last patch: [batch, output_quantile_len, num_quantiles]

        horizon = min(self.horizon_len, pf_outputs.shape[1])
        full_forecast = pf_outputs[:, :horizon, :].clone()

        median_index = min(self.config.decode_index, full_forecast.shape[-1] - 1)
        if self.config.use_continuous_quantile_head:
            # Use up to horizon worth of quantile spread data (matching max_horizon behavior in original)
            # quantile_spreads has shape [batch, quantile_patch_len, num_quantiles] where quantile_patch_len=1024
            # We use the first 'horizon' elements to match full_forecast shape
            max_quantile_horizon = min(horizon, quantile_spreads.shape[1])
            for idx, _ in enumerate(self.config.quantiles, start=1):
                if idx == median_index or idx >= full_forecast.shape[-1]:
                    continue
                # Apply continuous quantile head formula
                full_forecast[:, :max_quantile_horizon, idx] = (
                    quantile_spreads[:, :max_quantile_horizon, idx]
                    - quantile_spreads[:, :max_quantile_horizon, median_index]
                    + full_forecast[:, :max_quantile_horizon, median_index]
                )

        full_predictions = self.decoder._revin(full_forecast, mu_global, sigma_global, reverse=True)
        decode_index = min(self.config.decode_index, full_predictions.shape[-1] - 1)
        mean_predictions = full_predictions[:, :, decode_index]

        if window_size is not None:
            mean_predictions = mean_predictions[0::2, ...] + mean_predictions[1::2, ...]
            full_predictions = full_predictions[0::2, ...] + full_predictions[1::2, ...]

        if inp_min >= 0 and truncate_negative:
            zero = torch.zeros(1, device=mean_predictions.device, dtype=mean_predictions.dtype)
            mean_predictions = torch.maximum(mean_predictions, zero)
            full_predictions = torch.maximum(full_predictions, zero)

        loss = None
        if future_values is not None:
            mse_loss = F.mse_loss(mean_predictions, future_values)
            quantile_indices = [i for i in range(full_predictions.shape[-1]) if i != decode_index]
            if quantile_indices:
                index_tensor = torch.tensor(quantile_indices, device=full_predictions.device, dtype=torch.long)
                quantile_tensor = torch.index_select(full_predictions, dim=-1, index=index_tensor)
                quantile_loss = self._quantile_loss(quantile_tensor, future_values)
                loss = mse_loss + quantile_loss
            else:
                loss = mse_loss

        return Timesfm2P5OutputForPrediction(
            last_hidden_state=model_outputs.last_hidden_state,
            attentions=model_outputs.attentions if output_attentions else None,
            hidden_states=model_outputs.hidden_states if output_hidden_states else None,
            mean_predictions=mean_predictions,
            full_predictions=full_predictions,
            loss=loss,
        )

    @staticmethod
    def _revin(
        x: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        reverse: bool = False,
    ) -> torch.Tensor:
        """Reversible instance normalization - exact copy from original TimesFM."""
        _TOLERANCE = 1e-6

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

    @staticmethod
    def _timesfm_moving_average(arr: torch.Tensor, window_size: int) -> list[torch.Tensor]:
        """Calculates the moving average using PyTorch's convolution function."""
        # Pad with zeros to handle initial window positions
        arr_padded = F.pad(arr, (window_size - 1, 0), "constant", 0)
        # Create a convolution kernel
        kernel = torch.ones(window_size, dtype=arr.dtype, device=arr.device) / window_size
        # Apply convolution to calculate the moving average
        smoothed_arr = F.conv1d(arr_padded.view(1, 1, -1), kernel.view(1, 1, -1)).squeeze()
        return [smoothed_arr, arr - smoothed_arr]


__all__ = [
    "Timesfm2P5Config",
    "Timesfm2P5ModelForPrediction",
    "Timesfm2P5PreTrainedModel",
    "Timesfm2P5Model",
]
