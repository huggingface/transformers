# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Modular PyTorch FastConformer model."""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from ...activations import ACT2FN
from ...configuration_utils import PretrainedConfig
from ...modeling_outputs import BaseModelOutput, CausalLMOutput
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from ...utils.generic import can_return_tuple


logger = logging.get_logger(__name__)


# Configuration
class FastConformerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FastConformerModel`]. It is used to instantiate a
    FastConformer model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 1024):
            Vocabulary size of the FastConformer model. Note: This parameter is not used in the FastConformer
            audio encoder but is required for HuggingFace framework compatibility.
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimension of the layers and the hidden states.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 4096):
            Dimension of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the encoder and pooler.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        conv_kernel_size (`int`, *optional*, defaults to 9):
            The kernel size of the convolution layers in the Conformer block.
        subsampling_factor (`int`, *optional*, defaults to 8):
            The factor by which the input sequence is subsampled.
        subsampling_conv_channels (`int`, *optional*, defaults to 256):
            The number of channels in the subsampling convolution layers.
        num_mel_bins (`int`, *optional*, defaults to 128):
            Number of mel features.
        xscaling (`bool`, *optional*, defaults to `False`):
            Whether to apply input scaling to the model inputs.
        dropout_emb (`float`, *optional*, defaults to 0.0):
            The dropout ratio for embedding layers.
        encoder_layerdrop (`float`, *optional*, defaults to 0.1):
            The LayerDrop probability for the encoder.
        activation_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for activations inside the fully connected layer.
        use_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the linear layers.
        pad_token_id (`int`, *optional*, defaults to 0):
            The id of the padding token.
        bos_token_id (`int`, *optional*, defaults to 1):
            The id of the beginning-of-sequence token.
        eos_token_id (`int`, *optional*, defaults to 2):
            The id of the end-of-sequence token.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie word embeddings.
        use_cache (`bool`, *optional*, defaults to `False`):
            Whether to use cache. Not used in FastConformer but kept for compatibility.
        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether to output attention weights.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether to output hidden states.

    Example:
        ```python
        >>> from transformers import FastConformerModel, FastConformerConfig

        >>> # Initializing a FastConformer configuration
        >>> configuration = FastConformerConfig()

        >>> # Initializing a model from the configuration
        >>> model = FastConformerModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
        ```

    This configuration class is based on the FastConformer architecture from NVIDIA NeMo. You can find more details
    and pre-trained models at [nvidia/parakeet-ctc-1.1b](https://huggingface.co/nvidia/parakeet-ctc-1.1b).
    """

    model_type = "fastconformer"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=1024,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=8,
        intermediate_size=4096,
        hidden_act="silu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        conv_kernel_size=9,
        subsampling_factor=8,
        subsampling_conv_channels=256,
        num_mel_bins=128,
        xscaling=False,
        dropout_emb=0.0,
        encoder_layerdrop=0.1,
        activation_dropout=0.1,
        use_bias=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        # Core architecture parameters
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range

        # Dropout parameters
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.activation_dropout = activation_dropout
        self.dropout_emb = dropout_emb
        self.encoder_layerdrop = encoder_layerdrop

        # FastConformer-specific parameters
        self.conv_kernel_size = conv_kernel_size
        self.subsampling_factor = subsampling_factor
        self.subsampling_conv_channels = subsampling_conv_channels
        self.num_mel_bins = num_mel_bins
        self.xscaling = xscaling
        self.use_bias = use_bias

        # Output control
        self.use_cache = use_cache
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states

        # For compatibility with existing code
        self.d_model = hidden_size
        self.encoder_layers = num_hidden_layers
        self.encoder_attention_heads = num_attention_heads
        self.encoder_ffn_dim = intermediate_size
        self.dropout = hidden_dropout_prob
        self.attention_dropout = attention_probs_dropout_prob
        self.activation_function = hidden_act


class ParakeetCTCConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ParakeetCTC`]. It is used to instantiate a
    Parakeet CTC model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 1024):
            Vocabulary size of the CTC head. Defines the number of different tokens that can be predicted by the model.
        blank_token_id (`int`, *optional*, defaults to 0):
            The id of the blank token used in CTC. Typically 0.
        pad_token_id (`int`, *optional*, defaults to 0):
            The id of the padding token.
        bos_token_id (`int`, *optional*, defaults to 1):
            The id of the beginning-of-sequence token.
        eos_token_id (`int`, *optional*, defaults to 2):
            The id of the end-of-sequence token.
        ctc_loss_reduction (`str`, *optional*, defaults to `"mean"`):
            The reduction method for CTC loss. Can be "mean", "sum", or "none".
        ctc_zero_infinity (`bool`, *optional*, defaults to `True`):
            Whether to set infinite losses to zero in CTC loss computation.
        fastconformer_config (`FastConformerConfig`, *optional*):
            Configuration for the FastConformer encoder.

    Example:
        ```python
        >>> from transformers import ParakeetCTC, ParakeetCTCConfig

        >>> # Initializing a ParakeetCTC configuration
        >>> configuration = ParakeetCTCConfig()

        >>> # Initializing a model from the configuration
        >>> model = ParakeetCTC(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
        ```

    This configuration class is based on the Parakeet CTC architecture from NVIDIA NeMo. You can find more details
    and pre-trained models at [nvidia/parakeet-ctc-1.1b](https://huggingface.co/nvidia/parakeet-ctc-1.1b).
    """

    model_type = "parakeet_ctc"
    keys_to_ignore_at_inference = ["past_key_values"]
    sub_configs = {
        "fastconformer_config": FastConformerConfig,
    }

    def __init__(
        self,
        vocab_size=1024,
        blank_token_id=0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        fastconformer_config=None,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        # CTC-specific parameters
        self.vocab_size = vocab_size
        self.blank_token_id = blank_token_id
        self.ctc_loss_reduction = ctc_loss_reduction
        self.ctc_zero_infinity = ctc_zero_infinity

        # FastConformer encoder configuration
        if fastconformer_config is None:
            self.fastconformer_config = FastConformerConfig()
            logger.info("fastconformer_config is None, using default FastConformer config.")
        elif isinstance(fastconformer_config, dict):
            self.fastconformer_config = FastConformerConfig(**fastconformer_config)
        elif isinstance(fastconformer_config, FastConformerConfig):
            self.fastconformer_config = fastconformer_config
        else:
            raise ValueError(
                f"fastconformer_config must be a dict, FastConformerConfig, or None, got {type(fastconformer_config)}"
            )


# Future decoder configurations - placeholders for later implementation
# class ParakeetTDTConfig(PretrainedConfig):
#     """Configuration for Parakeet TDT models (FastConformer + TDT decoder)"""
#     model_type = "parakeet_tdt"
#
# class ParakeetRNNTConfig(PretrainedConfig):
#     """Configuration for Parakeet RNNT models (FastConformer + RNN-T decoder)"""
#     model_type = "parakeet_rnnt"
#
# class CanaryAEDConfig(PretrainedConfig):
#     """Configuration for Canary models (FastConformer + AED decoder)"""
#     model_type = "canary"


__all__ = [
    "FastConformerConfig",
    "ParakeetCTCConfig",
    "FastConformerEncoder",
    "FastConformerModel",
    "ParakeetCTC",
    "FastConformerPreTrainedModel",
]


def calc_length(lengths, all_paddings, kernel_size, stride, ceil_mode, repeat_num=1):
    """Calculates the output length of a Tensor passed through a convolution or max pooling layer"""
    add_pad: float = all_paddings - kernel_size
    one: float = 1.0
    for i in range(repeat_num):
        lengths = torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + one
        if ceil_mode:
            lengths = torch.ceil(lengths)
        else:
            lengths = torch.floor(lengths)
    return lengths.to(dtype=torch.int)


# Positional Encoding - keeping the FastConformer specific implementation
class FastConformerRelPositionalEncoding(nn.Module):
    """Relative positional encoding for FastConformer."""

    def __init__(self, config: FastConformerConfig):
        super().__init__()
        self.d_model = config.hidden_size
        self.scale_input = config.xscaling
        self.max_len = 5000

        if self.scale_input:
            self.xscale = math.sqrt(config.hidden_size)
        else:
            self.xscale = None

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if config.dropout_emb > 0.0:
            self.dropout_emb = nn.Dropout(config.dropout_emb)
        else:
            self.dropout_emb = None

        self.pe = None

    def extend_pe(self, length: int, device: "torch.device", dtype: "torch.dtype"):
        """Reset and extend the positional encodings if needed."""
        needed_size = 2 * length - 1
        if hasattr(self, "pe") and self.pe is not None and self.pe.size(1) >= needed_size:
            return

        positions = torch.arange(length - 1, -length, -1, dtype=torch.float32, device=device).unsqueeze(1)
        self.create_pe(positions=positions, dtype=dtype)

    def create_pe(self, positions: "torch.Tensor", dtype: "torch.dtype"):
        """Create positional encoding matrix."""
        d_model = self.d_model
        pe = torch.zeros(positions.size(0), d_model, dtype=dtype, device=positions.device)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32, device=positions.device) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)

        # Register as buffer and add batch dimension
        pe_tensor = pe.unsqueeze(0)  # (1, T, D)
        try:
            self.register_buffer("pe", pe_tensor, persistent=False)
        except KeyError:
            del self.pe
            self.register_buffer("pe", pe_tensor, persistent=False)

    def forward(self, hidden_states: "torch.Tensor", cache_len: int = 0) -> Tuple["torch.Tensor", "torch.Tensor"]:
        batch_size, seq_len, _ = hidden_states.shape
        input_len = seq_len + cache_len
        self.extend_pe(input_len, hidden_states.device, hidden_states.dtype)

        # Apply input scaling if enabled
        if self.xscale is not None:
            hidden_states = hidden_states * self.xscale

        center_pos = self.pe.size(1) // 2 + 1
        start_pos = center_pos - input_len
        end_pos = center_pos + input_len - 1
        pos_emb = self.pe[:, start_pos:end_pos]

        # Apply dropout to positional embeddings if configured
        if self.dropout_emb is not None:
            pos_emb = self.dropout_emb(pos_emb)

        return self.dropout(hidden_states), pos_emb


# Attention - using original FastConformer attention with NeMo-compatible parameter names
class FastConformerAttention(nn.Module):
    """FastConformer attention with relative positional encoding."""

    def __init__(self, config: FastConformerConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        # Use NeMo-compatible parameter names for weight loading
        self.linear_q = nn.Linear(config.hidden_size, config.hidden_size, bias=config.use_bias)
        self.linear_k = nn.Linear(config.hidden_size, config.hidden_size, bias=config.use_bias)
        self.linear_v = nn.Linear(config.hidden_size, config.hidden_size, bias=config.use_bias)
        self.linear_out = nn.Linear(config.hidden_size, config.hidden_size, bias=config.use_bias)

        # FastConformer-specific components
        self.linear_pos = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.pos_bias_u = nn.Parameter(torch.zeros(config.num_attention_heads, self.head_dim))
        self.pos_bias_v = nn.Parameter(torch.zeros(config.num_attention_heads, self.head_dim))

        # Override scaling factor
        self.s_d_k = math.sqrt(self.head_dim)

    def rel_shift(self, attention_scores):
        """Relative position shift for Shaw et al. style attention."""
        batch_size, num_heads, query_length, position_length = attention_scores.size()
        attention_scores = torch.nn.functional.pad(attention_scores, pad=(1, 0))
        attention_scores = attention_scores.view(batch_size, num_heads, -1, query_length)
        attention_scores = attention_scores[:, :, 1:].view(batch_size, num_heads, query_length, position_length)
        return attention_scores

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        pos_emb: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        FastConformer attention forward pass with relative positional encoding.

        Args:
            hidden_states: Input hidden states
            attention_mask: Attention mask
            position_embeddings: Not used in FastConformer (kept for compatibility)
            pos_emb: Relative positional embeddings
            output_attentions: Whether to return attention weights
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Use the original FastConformer attention projections
        query_states = (
            self.linear_q(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        )
        key_states = (
            self.linear_k(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        )
        value_states = (
            self.linear_v(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        )

        # FastConformer relative positional attention
        if pos_emb is not None:
            # pos_emb has shape [1, pos_len, hidden_size]
            n_batch_pos = pos_emb.size(0)  # This will be 1
            p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.num_heads, self.head_dim)
            p = p.transpose(1, 2)  # (1, n_heads, pos_len, head_dim)

            # Shaw et al. relative attention computation
            q_with_bias_u = query_states + self.pos_bias_u.unsqueeze(0).unsqueeze(2)
            q_with_bias_v = query_states + self.pos_bias_v.unsqueeze(0).unsqueeze(2)

            # Content-based attention
            matrix_ac = torch.matmul(q_with_bias_u, key_states.transpose(-2, -1))

            # Position-based attention
            matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
            matrix_bd = self.rel_shift(matrix_bd)

            # Truncate to match sequence length
            matrix_bd = matrix_bd[:, :, :, : matrix_ac.size(-1)]
            scores = (matrix_ac + matrix_bd) / self.s_d_k
        else:
            # Standard attention without relative positions
            scores = torch.matmul(query_states, key_states.transpose(-2, -1)) / self.s_d_k

        # Apply attention mask
        if attention_mask is not None:
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            # Use a safe mask value for different dtypes (FP16-safe)
            if scores.dtype == torch.float16:
                mask_value = -65504.0
            else:
                mask_value = -1e9
            scores = scores.masked_fill(attention_mask, mask_value)

        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # Apply attention to values
        context = torch.matmul(attn_weights, value_states)
        context = context.transpose(1, 2).contiguous()
        context = context.reshape(batch_size, seq_len, hidden_size)

        # Apply output projection
        context = self.linear_out(context)

        return context, attn_weights if output_attentions else None


# Feed Forward - using standard implementation
class FastConformerFeedForward(nn.Module):
    def __init__(self, config: FastConformerConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.use_bias)
        self.activation = ACT2FN[config.hidden_act]
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.use_bias)
        self.activation_dropout = config.activation_dropout

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.linear2(hidden_states)
        return hidden_states


# Convolution Module - FastConformer specific
class FastConformerConvModule(nn.Module):
    def __init__(self, config: FastConformerConfig):
        super().__init__()
        hidden_size = config.hidden_size
        kernel_size = config.conv_kernel_size
        use_bias = config.use_bias

        assert (kernel_size - 1) % 2 == 0
        self.padding = (kernel_size - 1) // 2

        self.pointwise_conv1 = nn.Conv1d(hidden_size, hidden_size * 2, kernel_size=1, bias=use_bias)
        self.depthwise_conv = nn.Conv1d(
            hidden_size, hidden_size, kernel_size=kernel_size, padding=0, groups=hidden_size, bias=use_bias
        )
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.activation = ACT2FN[config.hidden_act]
        self.pointwise_conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1, bias=use_bias)

    def forward(self, hidden_states: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.pointwise_conv1(hidden_states)
        hidden_states = F.glu(hidden_states, dim=1)

        # Apply padding mask before convolution
        if pad_mask is not None:
            hidden_states = hidden_states.masked_fill(pad_mask.unsqueeze(1), 0.0)

        hidden_states = F.pad(hidden_states, (self.padding, self.padding))
        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.pointwise_conv2(hidden_states)
        return hidden_states.transpose(1, 2)


# Conformer Block
class FastConformerBlock(nn.Module):
    def __init__(self, config: FastConformerConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.gradient_checkpointing = False

        self.feed_forward1 = FastConformerFeedForward(config)
        self.self_attn = FastConformerAttention(config, layer_idx)
        self.conv = FastConformerConvModule(config)
        self.feed_forward2 = FastConformerFeedForward(config)

        self.norm_feed_forward1 = nn.LayerNorm(config.hidden_size)
        self.norm_self_att = nn.LayerNorm(config.hidden_size)
        self.norm_conv = nn.LayerNorm(config.hidden_size)
        self.norm_feed_forward2 = nn.LayerNorm(config.hidden_size)
        self.norm_out = nn.LayerNorm(config.hidden_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pos_emb: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Store the original device for consistency
        original_device = hidden_states.device

        # First feed forward with 0.5 scaling
        ff1_output = self.feed_forward1(self.norm_feed_forward1(hidden_states))
        hidden_states = hidden_states + 0.5 * ff1_output.to(original_device)

        # Self attention
        normalized_hidden_states = self.norm_self_att(hidden_states)
        attn_output, attn_weights = self.self_attn(
            normalized_hidden_states,
            attention_mask=attention_mask,
            pos_emb=pos_emb,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + attn_output.to(original_device)

        # Convolution
        conv_output = self.conv(self.norm_conv(hidden_states), pad_mask=pad_mask)
        hidden_states = hidden_states + conv_output.to(original_device)

        # Second feed forward with 0.5 scaling
        ff2_output = self.feed_forward2(self.norm_feed_forward2(hidden_states))
        hidden_states = hidden_states + 0.5 * ff2_output.to(original_device)

        # Final layer norm
        hidden_states = self.norm_out(hidden_states)

        return hidden_states, attn_weights


# Subsampling - reusing existing implementation
class FastConformerSubsamplingConv2D(nn.Module):
    def __init__(self, config: FastConformerConfig, feat_in: int):
        super().__init__()

        self.subsampling_factor = config.subsampling_factor
        self.conv_channels = config.subsampling_conv_channels

        self.num_layers = int(math.log2(self.subsampling_factor))
        self.stride = 2
        self.kernel_size = 3

        self.left_padding = (self.kernel_size - 1) // 2
        self.right_padding = (self.kernel_size - 1) // 2
        self.padding = self.left_padding
        self.ceil_mode = False

        layers = []
        in_channels = 1
        use_bias = True

        for i in range(self.num_layers):
            if i == 0:
                conv = nn.Conv2d(
                    in_channels,
                    self.conv_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    bias=use_bias,
                )
                layers.append(conv)
                layers.append(nn.ReLU())
            else:
                depthwise_conv = nn.Conv2d(
                    self.conv_channels,
                    self.conv_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    groups=self.conv_channels,
                    bias=use_bias,
                )
                pointwise_conv = nn.Conv2d(self.conv_channels, self.conv_channels, kernel_size=1, bias=use_bias)
                layers.extend([depthwise_conv, pointwise_conv, nn.ReLU()])

        self.conv = nn.Sequential(*layers)

        in_length = torch.tensor(feat_in, dtype=torch.float)
        out_length = calc_length(
            lengths=in_length,
            all_paddings=self.left_padding + self.right_padding,
            kernel_size=self.kernel_size,
            stride=self.stride,
            ceil_mode=self.ceil_mode,
            repeat_num=self.num_layers,
        )

        if out_length.is_meta:
            out_length_val = feat_in // (self.stride**self.num_layers)
        else:
            out_length_val = int(out_length)

        self.out = nn.Linear(self.conv_channels * out_length_val, config.hidden_size, bias=True)

    def forward(self, input_features: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lengths = calc_length(
            lengths,
            all_paddings=self.left_padding + self.right_padding,
            kernel_size=self.kernel_size,
            stride=self.stride,
            ceil_mode=self.ceil_mode,
            repeat_num=self.num_layers,
        )

        hidden_states = input_features.unsqueeze(1)
        hidden_states = self.conv(hidden_states)

        batch_size, conv_channels, time_steps, freq_bins = hidden_states.size()
        hidden_states = self.out(hidden_states.transpose(1, 2).reshape(batch_size, time_steps, -1))

        return hidden_states, lengths


# Base Model Classes
class FastConformerPreTrainedModel(PreTrainedModel):
    config_class = FastConformerConfig
    base_model_prefix = "model"
    main_input_name = "input_features"
    supports_gradient_checkpointing = True
    _no_split_modules = ["FastConformerBlock"]
    _skip_keys_device_placement = []

    def _init_weights(self, module):
        # Get initializer_range from the appropriate config
        if hasattr(self.config, "initializer_range"):
            std = self.config.initializer_range
        elif hasattr(self.config, "fastconformer_config"):
            std = self.config.fastconformer_config.initializer_range
        else:
            std = 0.02  # default fallback
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            module.weight.data.fill_(1.0)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, FastConformerAttention):
            # Initialize positional bias parameters
            module.pos_bias_u.data.normal_(mean=0.0, std=std)
            module.pos_bias_v.data.normal_(mean=0.0, std=std)


class FastConformerEncoder(FastConformerPreTrainedModel):
    def __init__(self, config: FastConformerConfig):
        super().__init__(config)
        self.config = config
        self.gradient_checkpointing = False

        self.subsampling = FastConformerSubsamplingConv2D(config, config.num_mel_bins)
        self.pos_enc = FastConformerRelPositionalEncoding(config)

        self.layers = nn.ModuleList(
            [FastConformerBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

    @can_return_tuple
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        input_lengths: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Use input_lengths from feature extractor if available
        if input_lengths is not None:
            lengths = input_lengths
        elif attention_mask is not None:
            lengths = attention_mask.sum(-1)
        else:
            lengths = torch.full(
                (input_features.size(0),), input_features.size(1), dtype=torch.long, device=input_features.device
            )

        hidden_states, lengths = self.subsampling(input_features, lengths)
        hidden_states, pos_emb = self.pos_enc(hidden_states)

        max_audio_length = hidden_states.size(1)

        # Create masks
        pad_mask_valid = torch.arange(max_audio_length, device=hidden_states.device)[None, :] < lengths[:, None]
        pad_mask_for_att = pad_mask_valid.unsqueeze(1).expand(-1, max_audio_length, -1)
        pad_mask_for_att = pad_mask_for_att & pad_mask_for_att.transpose(1, 2)
        attention_mask = ~pad_mask_for_att
        pad_mask = ~pad_mask_valid

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # Layer dropping
            dropout_probability = torch.rand([])
            skip_the_layer = True if self.training and (dropout_probability < self.config.encoder_layerdrop) else False
            if not skip_the_layer:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        layer.__call__,
                        hidden_states,
                        attention_mask,
                        pos_emb,
                        pad_mask,
                        output_attentions,
                    )
                else:
                    layer_outputs = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        pos_emb=pos_emb,
                        pad_mask=pad_mask,
                        output_attentions=output_attentions,
                    )
                hidden_states = layer_outputs[0]

                if output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)
            else:
                if output_attentions:
                    all_attentions = all_attentions + (None,)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class FastConformerModel(FastConformerPreTrainedModel):
    def __init__(self, config: FastConformerConfig):
        super().__init__(config)
        self.gradient_checkpointing = False
        self.encoder = FastConformerEncoder(config)
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        input_lengths: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        return self.encoder(
            input_features=input_features,
            attention_mask=attention_mask,
            input_lengths=input_lengths,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )


class ParakeetCTC(FastConformerPreTrainedModel):
    config_class = ParakeetCTCConfig

    def __init__(self, config: ParakeetCTCConfig):
        # Initialize with the encoder config for the PreTrainedModel
        super().__init__(config.fastconformer_config)
        # Override self.config with the CTC config so it gets saved correctly
        self.config = config
        # Store encoder config separately for internal use
        self.encoder_config = config.fastconformer_config

        # Create the FastConformer encoder using the correct sub-config
        self.encoder = FastConformerEncoder(config.fastconformer_config)

        # CTC head uses vocab_size from the CTC config
        self.ctc_head = nn.Linear(config.fastconformer_config.hidden_size, config.vocab_size)

        # Store CTC-specific parameters
        self.blank_token_id = config.blank_token_id
        self.ctc_loss_reduction = config.ctc_loss_reduction
        self.ctc_zero_infinity = config.ctc_zero_infinity

        # Initialize weights
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        input_lengths: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutput]:
        return_dict = return_dict if return_dict is not None else self.encoder_config.use_return_dict

        # Forward through encoder
        encoder_outputs = self.encoder(
            input_features=input_features,
            attention_mask=attention_mask,
            input_lengths=input_lengths,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden_states = encoder_outputs.last_hidden_state

        # Apply CTC head
        logits = self.ctc_head(hidden_states)

        loss = None
        if labels is not None:
            # Calculate encoder output lengths
            if input_lengths is not None:
                encoder_lengths = calc_length(
                    input_lengths.float(),
                    all_paddings=2,
                    kernel_size=3,
                    stride=2,
                    ceil_mode=False,
                    repeat_num=int(math.log2(self.encoder_config.subsampling_factor)),
                )
                encoder_lengths = encoder_lengths.long()
            elif attention_mask is not None:
                input_lens = attention_mask.sum(-1).float()
                encoder_lengths = calc_length(
                    input_lens,
                    all_paddings=2,
                    kernel_size=3,
                    stride=2,
                    ceil_mode=False,
                    repeat_num=int(math.log2(self.encoder_config.subsampling_factor)),
                )
                encoder_lengths = encoder_lengths.long()
            else:
                encoder_lengths = torch.full((logits.size(0),), logits.size(1), dtype=torch.long, device=logits.device)

            # Calculate CTC loss using the configured parameters
            label_lengths = torch.sum((labels != -100) & (labels != self.blank_token_id), dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            log_probs = log_probs.transpose(0, 1)

            targets = []
            for i, label_length in enumerate(label_lengths):
                label = labels[i, :label_length]
                label = label[label != -100]
                targets.append(label)

            targets = torch.cat(targets)

            loss = F.ctc_loss(
                log_probs=log_probs,
                targets=targets,
                input_lengths=encoder_lengths,
                target_lengths=label_lengths,
                blank=self.blank_token_id,
                reduction=self.ctc_loss_reduction,
                zero_infinity=self.ctc_zero_infinity,
            )

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def generate_speech_recognition_outputs(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        input_lengths: Optional[torch.Tensor] = None,
    ) -> List[List[int]]:
        """
        Generate CTC decoded token sequences using greedy decoding.

        Args:
            input_features: Input mel-spectrogram features
            attention_mask: Attention mask
            input_lengths: Sequence lengths

        Returns:
            List of decoded token sequences (one per batch item)
        """
        with torch.no_grad():
            # Forward pass to get logits
            outputs = self.forward(
                input_features=input_features,
                attention_mask=attention_mask,
                input_lengths=input_lengths,
                return_dict=True,
            )

            logits = outputs.logits  # (batch, time, vocab)

            # Greedy CTC decoding
            predicted_ids = torch.argmax(logits, dim=-1)  # (batch, time)

            batch_size = predicted_ids.size(0)
            decoded_sequences = []

            for batch_idx in range(batch_size):
                sequence = predicted_ids[batch_idx]

                # Get actual sequence length if available
                if input_lengths is not None:
                    # Calculate the actual output length after subsampling
                    actual_length = calc_length(
                        input_lengths[batch_idx : batch_idx + 1].float(),
                        all_paddings=2,
                        kernel_size=3,
                        stride=2,
                        ceil_mode=False,
                        repeat_num=int(math.log2(self.encoder_config.subsampling_factor)),
                    ).item()
                    sequence = sequence[:actual_length]
                elif attention_mask is not None:
                    # Use attention mask to determine length
                    input_len = attention_mask[batch_idx].sum().float()
                    actual_length = calc_length(
                        input_len.unsqueeze(0),
                        all_paddings=2,
                        kernel_size=3,
                        stride=2,
                        ceil_mode=False,
                        repeat_num=int(math.log2(self.encoder_config.subsampling_factor)),
                    ).item()
                    sequence = sequence[:actual_length]

                # CTC collapse: remove blanks and repeated tokens
                decoded_tokens = []
                prev_token = None

                for token_id in sequence.tolist():
                    # Skip blank tokens (using the configured blank token ID)
                    if token_id == self.blank_token_id:
                        prev_token = token_id
                        continue

                    # Skip repeated tokens (CTC collapse)
                    if token_id != prev_token:
                        decoded_tokens.append(token_id)

                    prev_token = token_id

                decoded_sequences.append(decoded_tokens)

            return decoded_sequences


# Future model classes - placeholders for later implementation
# class ParakeetTDT(FastConformerPreTrainedModel):
#     """Parakeet model for TDT-based speech recognition"""
#     config_class = ParakeetTDTConfig
#
# class ParakeetRNNT(FastConformerPreTrainedModel):
#     """Parakeet model for RNN-T-based speech recognition"""
#     config_class = ParakeetRNNTConfig
#
# class CanaryAED(FastConformerPreTrainedModel):
#     """Canary model for AED-based multilingual speech recognition"""
#     config_class = CanaryAEDConfig


# Remove the commented examples since we now have actual implementations
# # Future model-specific classes (examples):
