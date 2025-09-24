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
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, logging
from ..gemma2.modeling_gemma2 import Gemma2Attention
from ..llama.modeling_llama import LlamaRMSNorm
from ..timesfm.modeling_timesfm import (
    TimesFmDecoderLayer,
    TimesFmMLP,
    TimesFmModel,
    TimesFmModelForPrediction,
    TimesFmOutput,
    TimesFmOutputForPrediction,
    TimesFmPreTrainedModel,
)
from ..timesfm.configuration_timesfm import TimesFmConfig


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
        rms_norm_eps: float = 1e-5,
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
        initializer_range: float = 0.02,
        min_timescale: float = 1.0,
        max_timescale: float = 10000.0,
        # Override defaults for 2.5
        num_hidden_layers: int = 20,
        output_quantile_len: int = 1024,
        decode_index: int = 5,
        use_rotary_embeddings: bool = True,
        use_qk_norm: bool = True,
        use_per_dim_scale: bool = True,
        use_bias: bool = False,
        activation: str = "swish",
        use_positional_embedding: bool = False,  # TimesFM 2.5 uses rotary embeddings instead
        # Gemma2-compatible parameters for query scaling
        query_pre_attn_scalar: float = 256.0,  # This provides the per-dim scaling
        attn_logit_softcapping: Optional[float] = None,
        layer_types: list = None,  # All layers are the same type
        sliding_window: int = None,  # No sliding window
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
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.attn_logit_softcapping = attn_logit_softcapping
        self.num_key_value_heads = num_key_value_heads
        self.attention_bias = attention_bias
        self.layer_types = layer_types or ["attention"] * num_hidden_layers
        self.sliding_window = sliding_window


class Timesfm2P5Output(TimesFmOutput):
    pass


class Timesfm2P5OutputForPrediction(TimesFmOutputForPrediction):
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

        # Activation function
        if config.activation == "relu":
            self.activation = nn.ReLU()
        elif config.activation == "swish" or config.activation == "silu":
            self.activation = nn.SiLU()
        elif config.activation == "none":
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Activation '{config.activation}' not supported.")

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
        self,
        input_dims: int,
        hidden_dims: int,
        output_dims: int,
        use_bias: bool = True,
        activation: str = "swish"
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




class Timesfm2P5Attention(Gemma2Attention):
    """
    TimesFM 2.5 attention inherits from Gemma2Attention which provides:
    - Rotary position embeddings
    - Query scaling (per-dimension scaling equivalent)
    - Efficient attention implementation

    We only add QK normalization on top of Gemma2's implementation.
    """

    def __init__(self, config: Timesfm2P5Config, layer_idx: int):
        # Convert TimesFM config to Gemma2-like config for compatibility
        super().__init__(config, layer_idx)

        # Add QK normalization specific to TimesFM 2.5
        self.use_qk_norm = getattr(config, 'use_qk_norm', True)
        if self.use_qk_norm:
            self.query_ln = Timesfm2P5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.key_ln = Timesfm2P5RMSNorm(self.head_dim, eps=config.rms_norm_eps)


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
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        # Self-Attention with pre and post normalization
        residual = hidden_states
        hidden_states = self.pre_attn_ln(hidden_states)
        hidden_states, scores = self.self_attn(
            hidden_states=hidden_states,
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

        # Input projection with TimesFM 2.5 ResidualBlock
        # Note: tokenizer uses bias=True (different from transformer layers)
        self.input_ff_layer = Timesfm2P5ResidualBlock(
            input_dims=2 * config.patch_length,  # 64 (32*2)
            hidden_dims=config.hidden_size,      # 1280 (not intermediate_size)
            output_dims=config.hidden_size,      # 1280
            use_bias=True,                       # tokenizer uses bias=True
            activation=config.activation         # "swish"
        )

        # TimesFM 2.5 has NO frequency embedding - model adapts automatically
        # (This is a key difference from TimesFM 2.0)

        # Transformer layers with TimesFM 2.5 specific components
        self.layers = nn.ModuleList([
            Timesfm2P5DecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])

        # TimesFM 2.5 uses rotary embeddings in attention, not separate positional embeddings
        # So we don't need the position_emb layer that TimesFM 2.0 uses

        # Initialize weights and apply final processing
        # self.post_init()  # Temporarily disabled due to initialization issue

    def forward(
        self,
        past_values: torch.Tensor,
        past_values_padding: torch.LongTensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        """
        TimesFM 2.5 forward pass - matches original TimesFM 2.5 preprocessing.

        Args:
            past_values: Input tensor of shape (batch_size, sequence_length)
            past_values_padding: Padding tensor of shape (batch_size, sequence_length)
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states
        """
        batch_size = past_values.shape[0]

        # Step 1: Patch the inputs (reshape to [B, N, P] where P=patch_length=32)
        patched_inputs = past_values.view(batch_size, -1, self.config.patch_length)
        patched_masks = past_values_padding.view(batch_size, -1, self.config.patch_length)

        # Step 2: TimesFM 2.5 preprocessing - concatenate inputs and masks
        # inputs: [B, N, P], masks: [B, N, P] -> tokenizer_inputs: [B, N, 2*P]
        tokenizer_inputs = torch.cat([
            patched_inputs,
            patched_masks.to(patched_inputs.dtype)
        ], dim=-1)

        # Step 3: Input embedding through tokenizer (ResidualBlock: 64 -> 1280)
        input_embeddings = self.input_ff_layer(tokenizer_inputs)

        # Step 4: No frequency embedding in TimesFM 2.5 - model adapts automatically
        # freq parameter is ignored (kept only for API compatibility with parent class)
        # Step 5: Pass through transformer layers
        hidden_states = input_embeddings
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=None,  # TimesFM 2.5 doesn't use attention mask in base forward
                position_ids=None,
                output_attentions=output_attentions,
            )

            if output_attentions:
                attention_weights, hidden_states = layer_outputs
                all_attentions = all_attentions + (attention_weights,)
            else:
                _, hidden_states = layer_outputs

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # Return in TimesFmOutput format
        # Note: TimesFM 2.5 doesn't compute loc/scale stats in base model
        return Timesfm2P5Output(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            loc=None,  # Not computed in TimesFM 2.5 base model
            scale=None,
        )


class Timesfm2P5ModelForPrediction(TimesFmModelForPrediction):
    """
    TimesFM 2.5 model for quantile and mean prediction.

    Inherits from TimesFmModelForPrediction but uses:
    - Timesfm2P5Model as the decoder
    - Separate output projections for point and quantile predictions (matching original TimesFM 2.5)
    """

    def __init__(self, config: Timesfm2P5Config):
        # Skip the parent's __init__ to avoid creating the wrong decoder and projections
        Timesfm2P5PreTrainedModel.__init__(self, config)

        self.config = config
        self.context_len = config.context_length
        self.horizon_len = config.horizon_length

        # Override decoder with TimesFM 2.5 model
        self.decoder = Timesfm2P5Model(config)

        # TimesFM 2.5 has separate output projections (matching original architecture)
        # Point prediction projection: 1280 -> 1280
        self.output_projection_point = Timesfm2P5ResidualBlock(
            input_dims=config.hidden_size,     # 1280
            hidden_dims=config.hidden_size,    # 1280
            output_dims=config.hidden_size,    # 1280
            use_bias=config.use_bias,          # False
            activation=config.activation       # "swish"
        )

        # Quantile prediction projection: 1280 -> 10240 (1024 * 10)
        self.output_projection_quantiles = Timesfm2P5ResidualBlock(
            input_dims=config.hidden_size,                                    # 1280
            hidden_dims=config.hidden_size,                                   # 1280
            output_dims=config.output_quantile_len * len(config.quantiles),   # 1024 * 9 = 9216
            use_bias=config.use_bias,                                          # False
            activation=config.activation                                       # "swish"
        )

        # Initialize weights and apply final processing
        # self.post_init()  # Temporarily disabled due to initialization issue

    def _postprocess_output(
        self, model_output: torch.Tensor, stats: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        Postprocess output of stacked transformer - TimesFM 2.5 version with separate projections.

        Args:
            model_output: Output from decoder [B, N, D] where D=hidden_size=1280
            stats: Tuple of (mu, sigma) for normalization

        Returns:
            output_ts: [B, N, horizon_length, quantiles+1] tensor
        """
        # Apply separate output projections (matching original TimesFM 2.5)
        point_output = self.output_projection_point(model_output)      # [B, N, 1280]
        quantile_output = self.output_projection_quantiles(model_output)  # [B, N, 9216]

        # Reshape outputs to match expected format
        b, n, _ = model_output.shape

        # Point predictions: [B, N, 1280] -> [B, N, horizon_length, 1]
        # Since point output is 1280 and we need horizon_length patches
        num_patches = point_output.shape[-1] // self.config.horizon_length  # 1280 / 128 = 10
        point_reshaped = point_output.view(b, n, num_patches, self.config.horizon_length)
        # Take the mean prediction (index 5 in original TimesFM 2.5)
        point_final = point_reshaped[:, :, self.config.decode_index:self.config.decode_index+1, :]  # [B, N, 1, 128]
        point_final = point_final.permute(0, 1, 3, 2)  # [B, N, 128, 1]

        # Quantile predictions: [B, N, 9216] -> [B, N, horizon_length, quantiles]
        quantile_reshaped = quantile_output.view(
            b, n, self.config.output_quantile_len, len(self.config.quantiles)
        )  # [B, N, 1024, 9]
        # Take the first horizon_length entries
        quantile_final = quantile_reshaped[:, :, :self.config.horizon_length, :]  # [B, N, 128, 9]

        # Combine point and quantile predictions: [B, N, 128, 1] + [B, N, 128, 9] -> [B, N, 128, 10]
        output_ts = torch.cat([point_final, quantile_final], dim=-1)

        # Apply normalization (same as parent)
        mu, sigma = stats
        return output_ts * sigma[:, None, None, None] + mu[:, None, None, None]

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
        return_dict: bool = True,
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

        # Get device from first input tensor
        device = past_values[0].device

        # Truncate inputs to forecast_context_len
        inputs = [ts[-fcontext_len:] for ts in past_values]
        inp_min = torch.min(torch.stack([torch.min(ts) for ts in inputs]))

        if window_size is not None:
            new_inputs = []
            for ts in inputs:
                new_inputs.extend(self._timesfm_moving_average(ts, window_size))
            inputs = new_inputs

        # TimesFM 2.5 doesn't use frequency - set dummy freq for internal compatibility
        freq = [0] * len(inputs)  # TimesFM 2.5 simplified API

        if output_attentions is None:
            output_attentions = self.config.output_attentions
        if output_hidden_states is None:
            output_hidden_states = self.config.output_hidden_states

        input_ts, input_padding, inp_freq = self._preprocess(inputs, freq)
        # Move tensors to the same device as input
        input_ts = input_ts.to(device)
        input_padding = input_padding.to(device)
        inp_freq = inp_freq.to(device)

        final_out = input_ts
        context_len = final_out.shape[1]
        full_outputs = []

        if input_padding.shape[1] != final_out.shape[1] + self.horizon_len:
            raise ValueError(
                "Length of paddings must match length of input + horizon_len:"
                f" {input_padding.shape[1]} != {final_out.shape[1]} + {self.horizon_len}"
            )
        output_patch_len = self.config.horizon_length

        num_decode_patches = (self.horizon_len + output_patch_len - 1) // output_patch_len

        for step_index in range(num_decode_patches):
            current_padding = input_padding[:, 0 : final_out.shape[1]]
            input_ts = final_out[:, -fcontext_len:]
            input_padding = current_padding[:, -fcontext_len:]

            # Use TimesFM 2.5 decoder (no freq parameter)
            decoder_output = self.decoder(
                past_values=input_ts,
                past_values_padding=input_padding,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            # TimesFM 2.5 specific postprocessing with separate projections
            fprop_outputs = self._postprocess_output(
                decoder_output.last_hidden_state,
                (decoder_output.loc, decoder_output.scale),
            )

            if return_forecast_on_context and step_index == 0:
                # For the first decoding step, collect the model forecast on the
                # context except the unavailable first input batch forecast.
                new_full_ts = fprop_outputs[:, :-1, : self.config.patch_length, :]
                # We have to use reshape and not view for non-contiguous memory
                new_full_ts = new_full_ts.reshape(new_full_ts.size(0), -1, new_full_ts.size(3))
                full_outputs.append(new_full_ts)

            # (full batch, last patch, output_patch_len, index of mean forecast = 0)
            new_ts = fprop_outputs[:, -1, :output_patch_len, 0]
            new_full_ts = fprop_outputs[:, -1, :output_patch_len, :]
            # (full batch, last patch, output_patch_len, all output indices)
            full_outputs.append(new_full_ts)
            final_out = torch.concatenate([final_out, new_ts], axis=-1)

        if return_forecast_on_context:
            # `full_outputs` indexing starts at after the first input patch.
            full_outputs = torch.concatenate(full_outputs, axis=1)[
                :, : (context_len - self.config.patch_length + self.horizon_len), :
            ]
        else:
            # `full_outputs` indexing starts at the forecast horizon.
            full_outputs = torch.concatenate(full_outputs, axis=1)[:, 0 : self.horizon_len, :]

        mean_outputs = full_outputs[:, :, 0]
        if window_size is not None:
            mean_outputs = mean_outputs[0::2, ...] + mean_outputs[1::2, ...]
            full_outputs = full_outputs[0::2, ...] + full_outputs[1::2, ...]
        if inp_min >= 0 and truncate_negative:
            mean_outputs = torch.maximum(mean_outputs, 0.0)
            full_outputs = torch.maximum(full_outputs, 0.0)

        loss = None
        if future_values is not None:
            mse_loss = F.mse_loss(mean_outputs, future_values)
            quantile_loss = self._quantile_loss(full_outputs[:, :, 1:], future_values)
            loss = mse_loss + quantile_loss

        return Timesfm2P5OutputForPrediction(
            last_hidden_state=decoder_output.last_hidden_state,
            attentions=decoder_output.attentions if output_attentions else None,
            hidden_states=decoder_output.hidden_states if output_hidden_states else None,
            mean_predictions=mean_outputs,
            full_predictions=full_outputs,
            loss=loss,
        )

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