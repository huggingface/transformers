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
    TimesFmPositionalEmbedding,
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
        # Gemma2-compatible parameters for query scaling
        query_pre_attn_scalar: float = 256.0,  # This provides the per-dim scaling
        attn_logit_softcapping: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            context_length=context_length,
            num_hidden_layers=num_hidden_layers,
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


class Timesfm2P5PositionalEmbedding(TimesFmPositionalEmbedding):
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


class Timesfm2P5Model(TimesFmModel):
    pass


class Timesfm2P5ModelForPrediction(TimesFmModelForPrediction):
    pass


__all__ = [
    "Timesfm2P5Config",
    "Timesfm2P5ModelForPrediction",
    "Timesfm2P5PreTrainedModel",
    "Timesfm2P5Model",
]