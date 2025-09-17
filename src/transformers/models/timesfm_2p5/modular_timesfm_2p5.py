# coding=utf-8
# Copyright 2025 Google LLC and HuggingFace Inc. team.
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
"""PyTorch TimesFM 2.5 model."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring, can_return_tuple, logging
from ..timesfm.configuration_timesfm import TimesFmConfig


logger = logging.get_logger(__name__)


@dataclass
class Timesfm2P5Output(BaseModelOutput):
    """
    Base class for TimesFM 2.5 model outputs.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            The last hidden state from the model.
        loc (`torch.FloatTensor` of shape `(batch_size, num_patches)`):
            The location parameters (means) for each patch.
        scale (`torch.FloatTensor` of shape `(batch_size, num_patches)`):
            The scale parameters (standard deviations) for each patch.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loc: torch.FloatTensor = None
    scale: torch.FloatTensor = None


@dataclass
class Timesfm2P5OutputForPrediction(BaseModelOutput):
    """
    Output class for TimesFM 2.5 prediction model.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            The last hidden state from the model.
        point_predictions (`torch.FloatTensor` of shape `(batch_size, prediction_length)`):
            The point forecasts (mean prediction).
        quantile_predictions (`torch.FloatTensor` of shape `(batch_size, prediction_length, num_quantiles)`):
            The quantile forecasts.
        mean_predictions (`torch.FloatTensor` of shape `(batch_size, prediction_length)`):
            Alias for point_predictions for API compatibility.
        full_predictions (`torch.FloatTensor` of shape `(batch_size, prediction_length, num_quantiles + 1)`):
            Combined predictions including mean and all quantiles.
        loss (`torch.FloatTensor`, *optional*):
            The loss (when labels/future_values are provided).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden states from all layers.
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            Attention weights from all layers.
    """

    point_predictions: torch.FloatTensor = None
    quantile_predictions: torch.FloatTensor = None
    mean_predictions: torch.FloatTensor = None  # For compatibility with tests
    full_predictions: torch.FloatTensor = None  # For compatibility with tests
    loss: Optional[torch.FloatTensor] = None


def strip_leading_nans(arr):
    """Removes contiguous NaN values from the beginning of a NumPy array.

    Args:
      arr: The input NumPy array.

    Returns:
      A new NumPy array with leading NaN values removed.
      If the array is all NaNs or empty, returns an empty array.
    """

    isnan = np.isnan(arr)
    first_valid_index = np.argmax(~isnan)
    return arr[first_valid_index:]


def linear_interpolation(arr):
    """Performs linear interpolation to fill NaN values in a 1D numpy array.

    Args:
        arr: The 1D numpy array containing NaN values.

    Returns:
        A new numpy array with NaN values filled using linear interpolation,
        or the original array if no NaNs are present.
        Returns None if the input is not a 1D array.
        Returns the original array if there are no NaN values.
    """

    nans = np.isnan(arr)
    if not np.any(nans):  # Check if there are any NaNs
        return arr

    def x(z):
        return z.nonzero()[0]

    nans_indices = x(nans)
    non_nans_indices = x(~nans)
    non_nans_values = arr[~nans]

    try:
        arr[nans] = np.interp(nans_indices, non_nans_indices, non_nans_values)
    except ValueError:
        if non_nans_values:
            mu = np.nanmean(arr)
        else:
            mu = 0.0
        arr = np.where(np.isfinite(arr), arr, mu)
    return arr


class Timesfm2P5Config(TimesFmConfig):
    r"""
    This is the configuration class to store the configuration of a [`Timesfm2P5ModelForPrediction`]. It is used to
    instantiate a TimesFM 2.5 model according to the specified arguments, defining the model architecture.
    Inherits from [`TimesFmConfig`] but with different defaults for TimesFM 2.5.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs.

    Arguments:
        patch_length (`int`, *optional*, defaults to 32):
            The length of one patch in the input sequence.
        context_length (`int`, *optional*, defaults to 512):
            The maximum length of the input context.
        horizon_length (`int`, *optional*, defaults to 128):
            The length of the prediction horizon.
        output_patch_length (`int`, *optional*, defaults to 128):
            The length of the output patches for point predictions.
        output_quantile_len (`int`, *optional*, defaults to 1024):
            The length of the output for quantile predictions.
        num_hidden_layers (`int`, *optional*, defaults to 20):
            Number of Transformer layers.
        hidden_size (`int`, *optional*, defaults to 1280):
            Size of the hidden layers in the feed-forward networks.
        intermediate_size (`int`, *optional*, defaults to 1280):
            Dimension of the MLP representations.
        head_dim (`int`, *optional*, defaults to 80):
            Size of the key, query, value projections per attention head.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        quantiles (`list[float]`, *optional*, defaults to `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`):
            The quantiles to predict.
        decode_index (`int`, *optional*, defaults to 5):
            The index used for decoding quantiles (corresponds to median, quantile 0.5).
        use_rotary_position_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to use rotary position embeddings.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the RMS normalization layers.
    """

    model_type = "timesfm_2p5"

    def __init__(
        self,
        patch_length: int = 32,
        context_length: int = 512,
        horizon_length: int = 128,
        output_patch_length: int = 128,
        output_quantile_len: int = 1024,
        num_hidden_layers: int = 20,
        hidden_size: int = 1280,
        intermediate_size: int = 1280,
        head_dim: int = 80,
        num_attention_heads: int = 16,
        quantiles: Optional[list[float]] = None,
        decode_index: int = 5,
        use_rotary_position_embeddings: bool = True,
        rms_norm_eps: float = 1e-6,
        freq_size: int = 3,
        tolerance: float = 1e-6,
        pad_val: float = 1123581321.0,
        attention_dropout: float = 0.0,
        use_positional_embedding: bool = False,
        initializer_range: float = 0.02,
        min_timescale: int = 1,
        max_timescale: int = 10000,
        **kwargs,
    ):
        if quantiles is None:
            quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        super().__init__(
            patch_length=patch_length,
            context_length=context_length,
            horizon_length=horizon_length,
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            head_dim=head_dim,
            num_attention_heads=num_attention_heads,
            quantiles=quantiles,
            rms_norm_eps=rms_norm_eps,
            freq_size=freq_size,
            tolerance=tolerance,
            pad_val=pad_val,
            attention_dropout=attention_dropout,
            use_positional_embedding=use_positional_embedding,
            initializer_range=initializer_range,
            min_timescale=min_timescale,
            max_timescale=max_timescale,
            **kwargs,
        )

        # TimesFM 2.5 specific parameters
        self.output_patch_length = output_patch_length
        self.output_quantile_len = output_quantile_len
        self.decode_index = decode_index
        self.use_rotary_position_embeddings = use_rotary_position_embeddings


class Timesfm2P5ResidualBlock(nn.Module):
    """TimesFM 2.5 residual block matching original implementation."""

    def __init__(
        self, config: Timesfm2P5Config, input_dims: int, hidden_dims: int, output_dims: int, use_bias: bool = True
    ):
        super().__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims

        # Use hidden_layer naming to match original
        self.hidden_layer = nn.Linear(input_dims, hidden_dims, bias=use_bias)
        self.activation = nn.SiLU()  # Swish activation
        self.output_layer = nn.Linear(hidden_dims, output_dims, bias=use_bias)
        self.residual_layer = nn.Linear(input_dims, output_dims, bias=use_bias)

    def forward(self, x):
        # Match original forward logic exactly
        # Ensure dtype consistency for mixed precision training
        x = x.to(self.hidden_layer.weight.dtype)
        return self.output_layer(self.activation(self.hidden_layer(x))) + self.residual_layer(x)


class Timesfm2P5RMSNorm(nn.Module):
    """RMS normalization matching the original TimesFM 2.5 implementation."""

    def __init__(self, num_features: int, epsilon: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(num_features))  # Initialize to zeros like original
        self.num_features = num_features
        self.epsilon = epsilon

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        var = torch.mean(torch.square(inputs), dim=-1, keepdim=True)
        normed_inputs = inputs * torch.rsqrt(var + self.epsilon)
        normed_inputs = normed_inputs * self.scale
        return normed_inputs


class Timesfm2P5RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embedding for TimesFM 2.5."""

    def __init__(self, embedding_dims: int, min_timescale: float = 1.0, max_timescale: float = 10000.0):
        super().__init__()
        self.embedding_dims = embedding_dims
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale

    def forward(self, inputs: torch.Tensor, position: Optional[torch.Tensor] = None):
        """Apply rotary positional embeddings."""
        if self.embedding_dims != inputs.shape[-1]:
            raise ValueError(
                "The embedding dims of the rotary position embedding must match the hidden dimension of the inputs."
            )

        half_embedding_dim = self.embedding_dims // 2
        fraction = 2 * torch.arange(0, half_embedding_dim, device=inputs.device) / self.embedding_dims
        timescale = (self.min_timescale * (self.max_timescale / self.min_timescale) ** fraction).to(inputs.device, inputs.dtype)

        if position is None:
            seq_length = inputs.shape[1]
            position = torch.arange(seq_length, dtype=inputs.dtype, device=inputs.device)[None, :]

        if len(inputs.shape) == 4:
            position = position[..., None, None]
            timescale = timescale[None, None, None, :]
        elif len(inputs.shape) == 3:
            position = position[..., None]
            timescale = timescale[None, None, :]
        else:
            raise ValueError("Inputs must be of rank 3 or 4.")

        sinusoid_inp = position / timescale
        sin = torch.sin(sinusoid_inp).to(inputs.dtype)
        cos = torch.cos(sinusoid_inp).to(inputs.dtype)
        first_half, second_half = torch.chunk(inputs, 2, dim=-1)
        first_part = first_half * cos - second_half * sin
        second_part = second_half * cos + first_half * sin
        return torch.cat([first_part, second_part], dim=-1)


class Timesfm2P5PerDimScale(nn.Module):
    """Per-dimension scaling matching the original TimesFM 2.5 implementation."""

    def __init__(self, num_dims: int):
        super().__init__()
        self.num_dims = num_dims
        self.per_dim_scale = nn.Parameter(torch.zeros(num_dims))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Exact formula from original implementation
        scale_factor = 1.442695041 / math.sqrt(self.num_dims) * F.softplus(self.per_dim_scale)
        return x * scale_factor


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Apply rotary position embedding to query and key tensors."""
    # Reshape cos and sin to match q, k dims
    cos = cos[..., : q.shape[-2], :]
    sin = sin[..., : q.shape[-2], :]

    # Apply rotary embedding
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """Standard eager attention implementation."""
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


def _dot_product_attention(query, key, value, mask=None, output_attentions=False):
    """Dot-product attention matching the original TimesFM 2.5 implementation."""
    attn_weights = torch.einsum("...qhd,...khd->...hqk", query, key)
    if mask is not None:
        attn_weights = torch.where(mask, attn_weights, -torch.finfo(attn_weights.dtype).max / 2)

    attn_weights = F.softmax(attn_weights, dim=-1)
    attn_output = torch.einsum("...hqk,...khd->...qhd", attn_weights, value)

    if output_attentions:
        return attn_output, attn_weights
    else:
        return attn_output


def make_attn_mask(
    query_length: int,
    num_all_masked_kv: torch.Tensor,
    query_index_offset: Optional[torch.Tensor] = None,
    kv_length: int = 0,
) -> torch.Tensor:
    """Makes attention mask matching the original TimesFM 2.5 implementation."""
    if kv_length == 0:
        kv_length = query_length

    q_index = torch.arange(query_length, device=num_all_masked_kv.device)[None, None, :, None]
    if query_index_offset is not None:
        q_index = q_index + query_index_offset[:, None, None, None]
    kv_index = torch.arange(kv_length, device=num_all_masked_kv.device)[None, None, None, :]
    return torch.logical_and(
        q_index >= kv_index,
        kv_index >= num_all_masked_kv[:, None, None, None],
    )


class Timesfm2P5Attention(nn.Module):
    """Multi-headed attention following TimesFM style with RoPE and query scaling."""

    def __init__(self, config: Timesfm2P5Config, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout

        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_heads * self.head_dim

        # Query scaling parameter like original TimesFM
        self.scaling = nn.Parameter(torch.empty((self.head_dim,)))

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # QK normalization like in original TimesFM 2.5
        self.q_norm = Timesfm2P5RMSNorm(self.head_dim, epsilon=config.rms_norm_eps)
        self.k_norm = Timesfm2P5RMSNorm(self.head_dim, epsilon=config.rms_norm_eps)

        # Rotary positional embeddings
        if config.use_rotary_position_embeddings:
            self.rotary_emb = Timesfm2P5RotaryPositionalEmbedding(
                embedding_dims=self.head_dim,
                min_timescale=config.min_timescale,
                max_timescale=config.max_timescale,
            )
        else:
            self.rotary_emb = None

    def _scale_query(self, query: torch.Tensor) -> torch.Tensor:
        """Apply per-dimension query scaling like original TimesFM."""
        scale = F.softplus(self.scaling.to(query.dtype)).mul(1.442695041 / math.sqrt(self.head_dim))
        return query * scale[None, None, None, :]

    def forward(
        self,
        inputs_q: torch.Tensor,
        patch_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        b, n_patches, _ = inputs_q.shape
        if patch_mask is None:
            patch_mask = torch.zeros(b, n_patches, dtype=torch.bool, device=inputs_q.device)

        input_shape = inputs_q.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Project and reshape
        query_states = self.q_norm(self.q_proj(inputs_q).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(inputs_q).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(inputs_q).view(hidden_shape).transpose(1, 2)

        # Apply query scaling
        query_states = self._scale_query(query_states)

        # Apply RoPE if enabled
        if self.rotary_emb is not None:
            num_masked = torch.sum(patch_mask.to(torch.int32), dim=-1)
            position = torch.arange(n_patches, device=inputs_q.device)[None, :] - num_masked[:, None]

            # Use original RoPE implementation
            query_states_orig = query_states.transpose(1, 2).view(b, n_patches, self.num_heads, self.head_dim)
            key_states_orig = key_states.transpose(1, 2).view(b, n_patches, self.num_heads, self.head_dim)

            query_states_orig = self.rotary_emb(query_states_orig, position)
            key_states_orig = self.rotary_emb(key_states_orig, position)

            query_states = query_states_orig.view(b, n_patches, -1).view(hidden_shape).transpose(1, 2)
            key_states = key_states_orig.view(b, n_patches, -1).view(hidden_shape).transpose(1, 2)

        # Create attention mask
        num_masked = torch.sum(patch_mask.to(torch.int32), dim=-1)
        attn_mask = make_attn_mask(query_length=n_patches, num_all_masked_kv=num_masked)

        # Convert to standard attention mask format (4D additive) that works with ALL_ATTENTION_FUNCTIONS
        if attn_mask is not None:
            # attn_mask has shape [batch, 1, n_patches, n_patches]
            # Convert True (allowed) -> 0.0, False (masked) -> -inf
            attention_mask = torch.where(attn_mask, 0.0, -torch.finfo(query_states.dtype).max)
            # Keep 4D shape [batch, 1, n_patches, n_patches] for compatibility with simple_eager_attention_forward
        else:
            attention_mask = None

        # Use native attention implementations that handle fp16 properly
        if self.config._attn_implementation == "sdpa":
            # Use PyTorch's scaled dot product attention for better fp16 support
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=0.0 if not self.training else self.attention_dropout,
                is_causal=False,  # We handle causal masking through attention_mask
            )
            attn_weights = None
        else:
            # Fallback to manual implementation with proper dtype handling for fp16
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            # Keep computation in fp32 for stability, then convert back
            attn_weights = F.softmax(attn_weights.float(), dim=-1)
            if self.training and self.attention_dropout > 0:
                attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=True)
            # Ensure both tensors have same dtype for matmul
            attn_weights = attn_weights.to(value_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


# Keep the original multi-head attention for compatibility during conversion
class Timesfm2P5MultiHeadAttention(nn.Module):
    """Multi-head attention matching the original TimesFM 2.5 implementation."""

    def __init__(self, config: Timesfm2P5Config):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim

        # Linear projections without bias (original uses bias=False)
        self.query = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.key = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.value = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.out = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # QK normalization
        self.query_ln = Timesfm2P5RMSNorm(self.head_dim, epsilon=config.rms_norm_eps)
        self.key_ln = Timesfm2P5RMSNorm(self.head_dim, epsilon=config.rms_norm_eps)

        # Per-dimension scaling
        self.per_dim_scale = Timesfm2P5PerDimScale(self.head_dim)

        # Rotary positional embeddings
        if config.use_rotary_position_embeddings:
            self.rotary_emb = Timesfm2P5RotaryPositionalEmbedding(
                embedding_dims=self.head_dim,
                min_timescale=config.min_timescale,
                max_timescale=config.max_timescale,
            )
        else:
            self.rotary_emb = None

    def forward(
        self,
        inputs_q: torch.Tensor,
        patch_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ):
        b, n_patches, _ = inputs_q.shape
        if patch_mask is None:
            patch_mask = torch.zeros(b, n_patches, dtype=torch.bool, device=inputs_q.device)

        query = self.query(inputs_q).view(b, n_patches, self.num_heads, self.head_dim)
        key = self.key(inputs_q).view(b, n_patches, self.num_heads, self.head_dim)
        value = self.value(inputs_q).view(b, n_patches, self.num_heads, self.head_dim)

        num_masked = torch.sum(patch_mask.to(torch.int32), dim=-1)

        if self.rotary_emb is not None:
            position = torch.arange(n_patches, device=inputs_q.device)[None, :] - num_masked[:, None]
            query = self.rotary_emb(query, position)
            key = self.rotary_emb(key, position)

        query = self.query_ln(query)
        key = self.key_ln(key)

        query = self.per_dim_scale(query)

        attn_mask = make_attn_mask(query_length=n_patches, num_all_masked_kv=num_masked)

        x, attn_weights = _dot_product_attention(query, key, value, mask=attn_mask, output_attentions=True)
        x = x.reshape(b, n_patches, self.hidden_size)
        out = self.out(x)

        if output_attentions:
            return out, attn_weights
        else:
            return out, None


class Timesfm2P5DecoderLayer(nn.Module):
    """Transformer layer matching the original TimesFM 2.5 implementation."""

    def __init__(self, config: Timesfm2P5Config, layer_idx: int):
        super().__init__()
        self.config = config

        # Layer normalizations
        self.pre_attn_ln = Timesfm2P5RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)
        self.post_attn_ln = Timesfm2P5RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)
        self.pre_ff_ln = Timesfm2P5RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)
        self.post_ff_ln = Timesfm2P5RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)

        # Attention
        self.attn = Timesfm2P5Attention(config, layer_idx)

        # Feed-forward layers
        self.ff0 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.ff1 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.activation = nn.SiLU()

    def forward(
        self,
        input_embeddings: torch.Tensor,
        patch_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        # Self Attention
        residual = input_embeddings
        hidden_states = self.pre_attn_ln(input_embeddings)
        hidden_states, scores = self.attn(
            inputs_q=hidden_states,
            patch_mask=patch_mask,
        )
        hidden_states = residual + hidden_states
        hidden_states = self.post_attn_ln(hidden_states)

        # Feed-forward block (matching original order)
        hidden_states = (
            self.post_ff_ln(self.ff1(self.activation(self.ff0(self.pre_ff_ln(hidden_states))))) + hidden_states
        )

        return scores, hidden_states


@auto_docstring
class Timesfm2P5PreTrainedModel(PreTrainedModel):
    config_class = Timesfm2P5Config
    base_model_prefix = "timesfm_2p5"
    _no_split_modules = ["Timesfm2P5DecoderLayer"]
    main_input_name = "past_values"
    _supports_sdpa = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, Timesfm2P5RMSNorm):
            # Initialize scale to zeros like original
            nn.init.zeros_(module.scale)
        elif isinstance(module, Timesfm2P5PerDimScale):
            nn.init.zeros_(module.per_dim_scale)
        elif isinstance(module, Timesfm2P5Attention):
            # Initialize scaling parameter like original TimesFM
            nn.init.ones_(module.scaling)


@auto_docstring
class Timesfm2P5Model(Timesfm2P5PreTrainedModel):
    def __init__(self, config: Timesfm2P5Config):
        super().__init__(config)
        self.config = config
        self.tolerance = 1e-6

        # Input tokenizer (residual block) - matches original naming
        self.tokenizer = Timesfm2P5ResidualBlock(
            config,
            input_dims=2 * config.patch_length,
            hidden_dims=config.hidden_size,
            output_dims=config.hidden_size,
            use_bias=True,
        )

        # Transformer layers using original naming
        self.stacked_xf = nn.ModuleList(
            [Timesfm2P5DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        # Initialize weights and apply final processing
        self.post_init()

    def _revin(self, x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, reverse: bool = False):
        """Reversible instance normalization matching original implementation."""
        if len(mu.shape) == len(x.shape) - 1:
            mu = mu[..., None]
            sigma = sigma[..., None]
        elif len(mu.shape) == len(x.shape) - 2:
            mu = mu[..., None, None]
            sigma = sigma[..., None, None]

        if reverse:
            return x * sigma + mu
        else:
            return (x - mu) / torch.where(sigma < self.tolerance, 1.0, sigma)

    def _update_running_stats(
        self,
        n: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update running statistics matching original implementation."""
        is_legit = torch.logical_not(mask)
        inc_n = torch.sum(is_legit.to(x.dtype), dim=-1)

        inc_mu_numerator = torch.sum(x * is_legit, dim=-1)
        inc_n_safe = torch.where(inc_n == 0, 1.0, inc_n)
        inc_mu = inc_mu_numerator / inc_n_safe
        inc_mu = torch.where(inc_n == 0, 0.0, inc_mu)

        inc_var_numerator = torch.sum(((x - inc_mu.unsqueeze(-1)) ** 2) * is_legit, dim=-1)
        inc_var = inc_var_numerator / inc_n_safe
        inc_var = torch.where(inc_n == 0, 0.0, inc_var)
        inc_sigma = torch.sqrt(inc_var)

        new_n = n + inc_n
        new_n_safe = torch.where(new_n == 0, 1.0, new_n)

        new_mu = (n * mu + inc_mu * inc_n) / new_n_safe
        new_mu = torch.where(new_n == 0, 0.0, new_mu)

        term1 = n * sigma.pow(2)
        term2 = inc_n * inc_sigma.pow(2)
        term3 = n * (mu - new_mu).pow(2)
        term4 = inc_n * (inc_mu - new_mu).pow(2)

        new_var = (term1 + term2 + term3 + term4) / new_n_safe
        new_var = torch.where(new_n == 0, 0.0, new_var)
        new_sigma = torch.sqrt(torch.clamp(new_var, min=0.0))

        return new_n, new_mu, new_sigma

    def forward(
        self,
        past_values: torch.Tensor,
        past_values_mask: torch.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, Timesfm2P5Output]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, seq_length = past_values.shape[:2]

        # Compute patch-wise statistics for normalization
        num_patches = seq_length // self.config.patch_length
        patched_values = past_values.view(batch_size, num_patches, self.config.patch_length)
        patched_masks = past_values_mask.view(batch_size, num_patches, self.config.patch_length)

        # Compute running statistics per patch (matching original exactly)
        n = torch.zeros(batch_size, device=past_values.device)
        mu = torch.zeros(batch_size, device=past_values.device)
        sigma = torch.zeros(batch_size, device=past_values.device)
        patch_mu = []
        patch_sigma = []
        for i in range(num_patches):
            n, mu, sigma = self._update_running_stats(n, mu, sigma, patched_values[:, i], patched_masks[:, i])
            patch_mu.append(mu)
            patch_sigma.append(sigma)

        context_mu = torch.stack(patch_mu, dim=1)
        context_sigma = torch.stack(patch_sigma, dim=1)

        # Apply RevIN normalization
        normed_inputs = self._revin(patched_values, context_mu, context_sigma, reverse=False)
        normed_inputs = torch.where(patched_masks, 0.0, normed_inputs)

        # Concatenate values and mask for tokenizer input (matching original)
        tokenizer_inputs = torch.cat([normed_inputs, patched_masks.to(normed_inputs.dtype)], dim=-1)

        # Apply tokenizer
        input_embeddings = self.tokenizer(tokenizer_inputs)

        # Pass through transformer layers
        hidden_states = input_embeddings
        all_attentions = []
        all_hidden_states = []

        for layer in self.stacked_xf:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            scores, hidden_states = layer(
                input_embeddings=hidden_states,
                patch_mask=patched_masks[..., -1],
                output_attentions=output_attentions,
            )
            if output_attentions:
                all_attentions.append(scores)

        if output_hidden_states:
            all_hidden_states = [input_embeddings] + all_hidden_states + [hidden_states]
        else:
            all_hidden_states = None

        output_embeddings = hidden_states

        if not return_dict:
            outputs = (output_embeddings, context_mu, context_sigma)
            if output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            if output_attentions:
                outputs = outputs + (all_attentions,)
            return outputs

        return Timesfm2P5Output(
            last_hidden_state=output_embeddings,
            loc=context_mu,
            scale=context_sigma,
            hidden_states=all_hidden_states,
            attentions=all_attentions if output_attentions else None,
        )


@auto_docstring
class Timesfm2P5ModelForPrediction(Timesfm2P5PreTrainedModel):
    """TimesFM 2.5 model for time series prediction."""

    def __init__(self, config: Timesfm2P5Config):
        super().__init__(config)
        self.config = config

        # Base model
        self.model = Timesfm2P5Model(config)

        # Output projection for point predictions
        self.output_projection_point = Timesfm2P5ResidualBlock(
            config,
            input_dims=config.hidden_size,
            hidden_dims=config.hidden_size,
            output_dims=config.hidden_size,
            use_bias=False,  # Original uses bias=False for output projections
        )

        # Output projection for quantiles
        num_quantiles = len(config.quantiles) + 1  # +1 for mean
        self.output_projection_quantiles = Timesfm2P5ResidualBlock(
            config,
            input_dims=config.hidden_size,
            hidden_dims=config.hidden_size,
            output_dims=config.output_quantile_len * num_quantiles,
            use_bias=False,  # Original uses bias=False for output projections
        )

        # Initialize weights and apply final processing
        self.post_init()

    def _quantile_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate quantile loss for TimesFM 2.5."""
        losses = []
        for i, q in enumerate(self.config.quantiles):
            errors = targets - predictions[..., i]
            loss = torch.max((q - 1) * errors, q * errors)
            losses.append(loss.mean())
        return torch.stack(losses).mean()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        past_values: Union[torch.Tensor, Sequence[torch.Tensor]],
        past_values_mask: Optional[torch.Tensor] = None,
        freq: Optional[torch.Tensor] = None,
        horizon: Optional[int] = None,
        future_values: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, Timesfm2P5OutputForPrediction]:
        r"""
        past_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)` or `Sequence[torch.Tensor]`):
            Past values of the time series that serves as input to the model. Can be either a single tensor
            or a sequence of tensors for variable-length inputs.
        past_values_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Boolean mask to indicate which values in `past_values` are padding. Only used when past_values is a tensor.
        freq (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Frequency indices for the time series data. Note: TimesFM 2.5 does not use frequency information,
            so this parameter is accepted for API compatibility but ignored.
        horizon (`int`, *optional*):
            The prediction horizon. If not provided, uses the model's default horizon.
        future_values (`torch.Tensor`, *optional*):
            Optional future time series values to be used for loss computation.
        output_attentions (`bool`, *optional*):
            Whether to output the attentions.
        output_hidden_states (`bool`, *optional*):
            Whether to output the hidden states.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        horizon = horizon if horizon is not None else self.config.horizon_length

        # Handle both tensor and sequence inputs (like original TimesFM)
        if isinstance(past_values, (list, tuple)):
            # Apply exact preprocessing from official TimesFM 2.5
            values = []
            masks = []
            context = self.config.context_length

            for each_input in past_values:
                # Convert to numpy for preprocessing
                if isinstance(each_input, torch.Tensor):
                    inp_array = each_input.detach().cpu().numpy()
                else:
                    inp_array = np.array(each_input)

                # Official preprocessing steps
                value = linear_interpolation(strip_leading_nans(inp_array))

                if (w := len(value)) >= context:
                    value = value[-context:]
                    mask = np.zeros_like(value, dtype=bool)
                else:
                    mask = np.array([True] * (context - w) + [False] * w)
                    value = np.pad(value, (context - w, 0), "constant", constant_values=0.0)

                values.append(value)
                masks.append(mask)

            # Convert to tensors
            past_values = torch.tensor(np.stack(values), dtype=torch.float32)
            past_values_mask = torch.tensor(np.stack(masks), dtype=torch.bool)
        else:
            # Handle tensor input - ensure it matches context_length
            if past_values.shape[1] != self.config.context_length:
                batch_size = past_values.shape[0]
                current_length = past_values.shape[1]
                context = self.config.context_length

                if current_length >= context:
                    # Truncate to context_length
                    past_values = past_values[:, -context:]
                    if past_values_mask is not None:
                        past_values_mask = past_values_mask[:, -context:]
                    else:
                        past_values_mask = torch.zeros_like(past_values, dtype=torch.bool)
                else:
                    # Pad to context_length
                    pad_length = context - current_length
                    past_values = torch.cat(
                        [
                            torch.zeros(batch_size, pad_length, device=past_values.device, dtype=past_values.dtype),
                            past_values,
                        ],
                        dim=1,
                    )

                    # Create/update mask
                    if past_values_mask is not None:
                        past_values_mask = torch.cat(
                            [
                                torch.ones(batch_size, pad_length, device=past_values.device, dtype=torch.bool),
                                past_values_mask,
                            ],
                            dim=1,
                        )
                    else:
                        past_values_mask = torch.cat(
                            [
                                torch.ones(batch_size, pad_length, device=past_values.device, dtype=torch.bool),
                                torch.zeros(batch_size, current_length, device=past_values.device, dtype=torch.bool),
                            ],
                            dim=1,
                        )
            elif past_values_mask is None:
                past_values_mask = torch.zeros_like(past_values, dtype=torch.bool)

        # Get model outputs
        model_outputs = self.model(
            past_values=past_values,
            past_values_mask=past_values_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        # Apply output projections (matching original logic)
        output_embeddings = model_outputs.last_hidden_state
        output_ts = self.output_projection_point(output_embeddings)
        output_quantile_spread = self.output_projection_quantiles(output_embeddings)

        # Reshape outputs (matching original logic)
        batch_size, num_patches = output_embeddings.shape[:2]
        num_quantiles = len(self.config.quantiles) + 1

        output_ts = output_ts.view(batch_size, num_patches, self.config.output_patch_length, num_quantiles)
        output_quantile_spread = output_quantile_spread.view(
            batch_size, num_patches, self.config.output_quantile_len, num_quantiles
        )

        # Apply reverse normalization
        context_mu = model_outputs.loc
        context_sigma = model_outputs.scale
        renormed_outputs = self.model._revin(output_ts, context_mu, context_sigma, reverse=True)
        renormed_quantile_spread = self.model._revin(output_quantile_spread, context_mu, context_sigma, reverse=True)[
            :, -1, ...
        ]  # Take last patch

        # Extract predictions for the requested horizon
        point_predictions = renormed_outputs[:, -1, :horizon, self.config.decode_index]
        quantile_predictions = renormed_quantile_spread[:, :horizon, :]

        # Calculate loss if future_values provided
        loss = None
        if future_values is not None:
            # Ensure future_values matches the prediction shape
            if future_values.shape != point_predictions.shape:
                if len(future_values.shape) == 1:
                    future_values = future_values.unsqueeze(0)
                if future_values.shape[1] != horizon:
                    future_values = future_values[:, :horizon]
                if future_values.shape[0] != point_predictions.shape[0]:
                    # Repeat for batch size if needed
                    future_values = future_values.repeat(point_predictions.shape[0], 1)

            # Calculate MSE loss for point predictions
            mse_loss = F.mse_loss(point_predictions, future_values)

            # Calculate quantile loss
            quantile_loss = self._quantile_loss(quantile_predictions, future_values.unsqueeze(-1))

            # Combine losses
            loss = mse_loss + quantile_loss

        if not return_dict:
            return (point_predictions, quantile_predictions)

        # Create full predictions by concatenating point and quantile predictions
        full_predictions = torch.cat(
            [
                point_predictions.unsqueeze(-1),  # Add quantile dimension
                quantile_predictions,
            ],
            dim=-1,
        )

        return Timesfm2P5OutputForPrediction(
            point_predictions=point_predictions,
            quantile_predictions=quantile_predictions,
            mean_predictions=point_predictions,  # Alias for API compatibility
            full_predictions=full_predictions,
            last_hidden_state=model_outputs.last_hidden_state,
            hidden_states=model_outputs.hidden_states,
            attentions=model_outputs.attentions,
            loss=loss,
        )


__all__ = [
    "Timesfm2P5Config",
    "Timesfm2P5ModelForPrediction",
    "Timesfm2P5PreTrainedModel",
    "Timesfm2P5Model",
    "Timesfm2P5Output",
    "Timesfm2P5OutputForPrediction",
]
