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
from typing import Optional, Tuple

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
        # Override defaults for 2.5
        context_length: int = 16384,
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
        **kwargs,
    ):
        super().__init__(
            context_length=context_length,
            num_hidden_layers=num_hidden_layers,
            use_positional_embedding=use_positional_embedding,
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
        self.post_init()

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
    - Timesfm2P5ResidualBlock for output projection
    """

    def __init__(self, config: Timesfm2P5Config):
        super().__init__(config)

        # Override decoder with TimesFM 2.5 model
        self.decoder = Timesfm2P5Model(config)

        # Override output projection with TimesFM 2.5 ResidualBlock
        self.horizon_ff_layer = Timesfm2P5ResidualBlock(
            input_dims=config.hidden_size,
            hidden_dims=config.intermediate_size,
            output_dims=config.horizon_length * (1 + len(config.quantiles)),
            use_bias=config.use_bias,
            activation=config.activation
        )

        # Initialize weights and apply final processing
        self.post_init()


__all__ = [
    "Timesfm2P5Config",
    "Timesfm2P5ModelForPrediction",
    "Timesfm2P5PreTrainedModel",
    "Timesfm2P5Model",
]