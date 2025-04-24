# coding=utf-8
# Copyright 2025 HuggingFace Inc. team. All rights reserved.
#
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
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from ...cache_utils import Cache, DynamicCache, EncoderDecoderCache, StaticCache
from ...generation import GenerationMixin
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import (
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    logging,
    replace_return_docstrings,
)
from ..auto import AutoConfig, AutoModel
from ..emu3.configuration_emu3 import (
    Emu3Config,
    Emu3TextConfig,
    Emu3VQVAEConfig,
)
from ..emu3.modeling_emu3 import (
    Emu3MLP,
    Emu3PreTrainedModel,
    Emu3RMSNorm,
    Emu3VQVAE,
    Emu3VQVAEAttentionBlock,
    eager_attention_forward,
)
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaModel,
    rotate_half,
)


if is_flash_attn_2_available():
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


_CONFIG_FOR_DOC = "CosmosConfig"
_CHECKPOINT_FOR_DOC = "NVIDIA/Cosmos-4B-hf"

logger = logging.get_logger(__name__)


class CosmosVQVAEConfig(Emu3VQVAEConfig):
    r"""
    This is the configuration class to store the configuration of a [`CosmosVQVAE`]. It is used to instantiate an VQ-VAE
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a configuration to the VQ model presented in Cosmos paper.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        embed_dim (`int`, *optional*, defaults to 6):
            Dimension of the quantized vector in codebook.
        latent_channels (`int`, *optional*, defaults to 16):
            Dimension of the output channel of encoder and the input channel of decoder
        double_latent (`bool`, *optional*, defaults to `False`):
            Whether double the output dim of the encoder.
        in_channels (`int`, *optional*, defaults to 3):
            Input channel of encoder.
        out_channels (`int`, *optional*, defaults to 3):
            Output channel of decoder.
        temporal_downsample_factor (`int`, *optional*, defaults to 8):
            Temporal downsample factor.
        base_channels (`int`, *optional*, defaults to 128):
            Basic channel number of the intermediate blocks.
        channel_multiplier (`List[int]`, *optional*, defaults to `[2, 4, 4]`):
            Channel scaling factor of the intermediate blocks.
        num_res_blocks (`int`, *optional*, defaults to 2):
            Residual block number in each stage.
        attn_resolutions (`List[int]`, *optional*, defaults to `[3]`):
            Stage indices to apply attention.
        hidden_size (`int`, *optional*, defaults to 512):
            Dimension of the hidden representations in the attention layer.
        num_attention_heads (`int`, *optional*, defaults to 1):
            Number of attention heads for each attention layer.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        patch_size (`int`, *optional*, defaults to 4):
            VAE patch size
        levels (`List`, *optional*, defaults to `[8, 8, 8, 5, 5, 5]`):
            Levels used by the quantizer
        dropout (`float`, *optional*, defaults to 0.0):
            Dropout to apply.

    ```python
    >>> from transformers import CosmosVQVAE, CosmosVQVAEConfig

    >>> # Initializing a video VQ model of Cosmos configuration
    >>> configuration = CosmosVQVAEConfig()

    >>> # Initializing a model from the Cosmos VQ model style configuration
    >>> model = CosmosVQVAE(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    def __init__(
        self,
        embed_dim: int = 6,
        latent_channels: int = 16,
        temporal_downsample_factor: int = 8,
        attn_resolutions: List[int] = None,
        base_channels: int = 128,
        channel_multiplier: List[int] = [2, 4, 4],
        num_res_blocks: int = 2,
        hidden_size: int = 512,
        patch_size: int = 4,
        levels: List[int] = [8, 8, 8, 5, 5, 5],
        dropout: float = 0.0,
        double_latent: bool = False,
        in_channels: int = 3,
        out_channels: int = 3,
        num_attention_heads: int = 1,
        attention_dropout: float = 0.0,
    ):
        super().__init__(
            embed_dim=embed_dim,
            latent_channels=latent_channels,
            temporal_downsample_factor=temporal_downsample_factor,
            attn_resolutions=attn_resolutions,
            base_channels=base_channels,
            channel_multiplier=channel_multiplier,
            num_res_blocks=num_res_blocks,
            hidden_size=hidden_size,
            double_latent=double_latent,
            in_channels=in_channels,
            out_channels=out_channels,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
        )
        self.patch_size = patch_size
        self.levels = levels
        self.dropout = dropout


class CosmosTextConfig(Emu3TextConfig):
    r"""
    This is the configuration class to store the configuration of a [`CosmosTextModel`]. It is used to instantiate a
    cosmos model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the
    [Cosmos-community/Cosmos-Chat-hf](https://huggingface.co/Cosmos-community/Cosmos-Chat-hf).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 64000):
            Vocabulary size of the Cosmos model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`CosmosModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 14336):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 16):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 12800):
            The maximum sequence length that this model might ever be used with. Emu supports up to 9216 tokens,
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*, defaults to 151643):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 64000):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 64001):
            End of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 500000.0):
            The base period of the RoPE embeddings.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in up_proj, down_proj and gate_proj layers in the MLP layers.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rope_latent_shape (`List`, *optional*):
            Shapes of time, height and width grids.
        apply_abs_pos_emb (`bool`, *optional*, defaults to `False`):
            Whether to apply absolute positional embedding or not.
        cross_attn_hidden_size (`int`, *optional*, defaults to 1024):
            Cross attention hidden size.
        insert_cross_attn_layers (`List`, *optional*):
            Layer indices where to insert cross attention modules.
        is_video_to_world (`bool`, *optional*, defaults to `False`):
            Whether model is used in video-2-world setting.


    ```python
    >>> from transformers import CosmosModel, CosmosConfig

    >>> # Initializing a Cosmos-community/Cosmos-Chat-hf style configuration
    >>> configuration = CosmosConfig()

    >>> # Initializing a model from the Cosmos-community/Cosmos-Chat-hf style configuration
    >>> model = CosmosModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    def __init__(
        self,
        vocab_size: int = 64000,
        hidden_size: int = 4096,
        intermediate_size: int = 14336,
        num_hidden_layers: int = 16,
        num_attention_heads: int = 32,
        num_key_value_heads: Optional[int] = 8,
        hidden_act: str = "silu",
        max_position_embeddings: int = 12800,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        pad_token_id: int = None,
        bos_token_id: int = 64000,
        eos_token_id: int = 64001,
        tie_word_embeddings: bool = False,
        rope_theta: float = 500000.0,
        rope_scaling: Optional = None,
        mlp_bias=False,
        attention_bias=False,
        attention_dropout: float = 0.1,
        initializer_range: float = 0.02,
        rope_latent_shape: List[int] = None,
        apply_abs_pos_emb: bool = False,
        cross_attn_hidden_size: int = 1024,
        insert_cross_attn_layers: List[int] = None,
        is_video_to_world: bool = False,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            mlp_bias=mlp_bias,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            initializer_range=initializer_range,
        )

        self.rope_latent_shape = rope_latent_shape
        self.apply_abs_pos_emb = apply_abs_pos_emb
        self.cross_attn_hidden_size = cross_attn_hidden_size
        self.insert_cross_attn_layers = insert_cross_attn_layers or []
        self.is_video_to_world = is_video_to_world


class CosmosConfig(Emu3Config):
    """
    This is the configuration class to store the configuration of a [`CosmosModel`]. It is used to instantiate a
    cosmos model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the
    [Cosmos-community/Cosmos-Chat-hf](https://huggingface.co/Cosmos-community/Cosmos-Chat-hf).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vq_config (`Union[Dict, CosmosVQVAEConfig]`, *optional*):
            CosmosVQVAEConfig instance containing the configuration for the VQ-VAE model.
        text_config (`Union[Dict, CosmosTextConfig]``, *optional*):
            CosmosTextConfig instance containing the configuration for the language model.
        prompt_encoder_config (`Union[Dict, PreTrainedConfig]``, *optional*):
            PreTrainedConfig instance containing the configuration for the prompt encoder. Used only for
            video-text generation models.
        image_token_id (`dict`, *optional*, defaults to 64000):
            An image placeholder token index.
    """

    sub_configs = {
        "text_config": CosmosTextConfig,
        "vq_config": CosmosVQVAEConfig,
        "prompt_encoder_config": AutoConfig,
    }

    def __init__(
        self,
        vq_config: Union[Dict, CosmosVQVAEConfig] = None,
        text_config: Union[Dict, CosmosTextConfig] = None,
        prompt_encoder_config: Union[Dict, AutoConfig] = None,
        image_token_id: int = 64000,
        **kwargs,
    ):
        super().__init__(text_config=text_config, vq_config=vq_config)

        del self.vocabulary_map

        if prompt_encoder_config is None:
            prompt_encoder_config = {
                "d_ff": 65536,
                "d_kv": 128,
                "d_model": 1024,
                "dropout_rate": 0.1,
                "eos_token_id": 1,
                "layer_norm_epsilon": 1e-06,
                "n_positions": 512,
                "num_heads": 128,
                "num_layers": 24,
                "pad_token_id": 0,
                "relative_attention_num_buckets": 32,
                "vocab_size": 32128,
            }
            prompt_encoder_config = AutoConfig.for_model("t5", **prompt_encoder_config)
        elif isinstance(prompt_encoder_config, dict):
            prompt_encoder_config = AutoConfig.for_model(**prompt_encoder_config)

        self.prompt_encoder_config = prompt_encoder_config
        self.image_token_id = image_token_id


@dataclass
class CosmosBaseModelOutputWithPast(BaseModelOutputWithPastAndCrossAttentions):
    """
    Base class for Cosmos Model's outputs.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        encoder_last_hidden_state (`torch.FloatTensor`, *optional*):
            A `torch.FloatTensor` of size (batch_size, prommp_length, hidden_size)`.
            Last hidden states of the prompt encoder, obtianed when `encoder_input_ids is not None`.
        encoder_hidden_states (`torch.FloatTensor`, *optional*):
            Hidden states of the prompt encoder, obtianed when `encoder_input_ids is not None`.
        encoder_attentions (`torch.FloatTensor`, *optional*):
            Attentions of the prompt encoder, obatained when `encoder_input_ids is not None`.
    """

    decoder_hidden_states: Optional[torch.FloatTensor] = None
    decoder_attentions: Optional[torch.FloatTensor] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[torch.FloatTensor] = None
    encoder_attentions: Optional[torch.FloatTensor] = None


@dataclass
class CosmosCausalLMOutputWithPast(Seq2SeqLMOutput):
    """
    Base class for Cosmos model outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """


class CosmosVQVAEVectorQuantizer(nn.Module):
    """
    A module for vector quantization using learned embedding vectors.

    This module implements the quantization process similar to the one described in
    the [Finite Scalar Quantization: VQ-VAE Made Simple paper](https://arxiv.org/abs/2309.15505). It quantizes continuous
    input vectors into discrete codebook vectors, which are learned during training.

    Adapted from: https://github.com/lucidrains/vector-quantize-pytorch/blob/9502a1f447876d53fd37685b226bf28f250dc4a3/
    vector_quantize_pytorch/finite_scalar_quantization.py. [Copyright (c) 2020 Phil Wang]
    """

    def __init__(self, config: CosmosVQVAEConfig):
        super().__init__()

        levels = config.levels

        _levels = torch.tensor(levels, dtype=torch.int32)
        self.register_buffer("_levels", _levels, persistent=False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=torch.int32)
        self.register_buffer("_basis", _basis, persistent=False)

        codebook_size = self._levels.prod().item()
        implicit_codebook = self.indices_to_codes(torch.arange(codebook_size))
        self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)

    def forward(self, hidden_states: torch.Tensor):
        batch_size, temporal, channels, height, width = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 3, 4, 1).contiguous()
        hidden_states_flattened = hidden_states.view(batch_size, -1, temporal).unsqueeze(-2)

        codes = self.quantize(hidden_states_flattened)

        indices = self.codes_to_indices(codes)
        indices = indices.view(batch_size, channels, height, width)

        codes = codes.view(batch_size, channels, height, width, -1)
        codes = codes.permute(0, 4, 1, 2, 3)
        return indices, codes

    def bound(self, hidden_state: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (hidden_state + shift).tanh() * half_l - offset

    def quantize(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.bound(hidden_state)
        quantized = hidden_state.round()
        quantized = hidden_state + (quantized - hidden_state).detach()
        half_width = self._levels // 2
        return quantized / half_width

    def codes_to_indices(self, codes: torch.Tensor) -> torch.Tensor:
        half_width = self._levels // 2
        codes = (codes * half_width) + half_width
        indices = (codes.float() * self._basis).sum(dim=-1).to(torch.int64)
        return indices

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        indices = indices.unsqueeze(-1)
        codes_non_centered = (indices // self._basis) % self._levels
        half_width = self._levels // 2
        codes = (codes_non_centered - half_width) / half_width
        if codes.ndim == 5:
            codes = codes.permute(0, 4, 1, 2, 3)
        return codes


class CosmosCausalConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int],
        stride: int = 1,
        time_stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.time_pad = (kernel_size[0] - 1) + (1 - time_stride)
        self.padding = (padding,) * 4 + (0, 0)
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride=(time_stride, stride, stride))

    def forward(self, hidden_states: torch.Tensor):
        hidden_states_prev = hidden_states[:, :, :1, ...].repeat(1, 1, self.time_pad, 1, 1)
        hidden_states = torch.cat([hidden_states_prev, hidden_states], dim=2)

        hidden_states = F.pad(hidden_states, self.padding, mode="constant")
        hidden_states = self.conv3d(hidden_states)
        return hidden_states


class CosmosVQVAETemporalNorm(nn.Module):
    def __init__(self, in_channels, num_groups=1):
        super().__init__()
        self.norm = torch.nn.GroupNorm(num_channels=in_channels, num_groups=1, eps=1e-6, affine=True)

    def forward(self, hidden_states: torch.Tensor):
        # group time and batch dims, then ungroup back
        batch_size, channels, temporal, height, width = hidden_states.shape
        hidden_states = hidden_states.transpose(1, 2).reshape(-1, channels, height, width)
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.view(batch_size, temporal, channels, height, width).transpose(1, 2)
        return hidden_states


class CosmosVQVAEEncoderDownsample(nn.Module):
    def __init__(self, in_channels, temporal_down: bool = True):
        super().__init__()
        self.conv1 = CosmosCausalConv3d(
            in_channels, in_channels, kernel_size=(1, 3, 3), stride=2, time_stride=1, padding=0
        )
        self.conv2 = (
            CosmosCausalConv3d(in_channels, in_channels, kernel_size=(3, 1, 1), stride=1, time_stride=2, padding=0)
            if temporal_down
            else nn.Identity()
        )
        self.conv3 = CosmosCausalConv3d(
            in_channels, in_channels, kernel_size=(1, 1, 1), stride=1, time_stride=1, padding=0
        )
        self.temporal_down = temporal_down

    def forward(self, hidden_states):
        # hybrid downsample spatially
        hidden_states = F.pad(hidden_states, pad=(0, 1, 0, 1, 0, 0), mode="constant", value=0)
        hidden_states_1 = self.conv1(hidden_states)
        hidden_states_2 = F.avg_pool3d(hidden_states, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        hidden_states = hidden_states_1 + hidden_states_2

        # hybrid downsample temporally
        if self.temporal_down:
            hidden_states = torch.cat([hidden_states[:, :, :1, ...], hidden_states], dim=2)
            hidden_states_1 = self.conv2(hidden_states)
            hidden_states_2 = F.avg_pool3d(hidden_states, kernel_size=(2, 1, 1), stride=(2, 1, 1))
            hidden_states = hidden_states_1 + hidden_states_2

        # final 1x1x1 conv
        hidden_states = self.conv3(hidden_states)
        return hidden_states


class CosmosVQVAEDecoderUpsample(nn.Module):
    def __init__(self, in_channels, temporal_up: bool = True):
        super().__init__()
        self.conv1 = (
            CosmosCausalConv3d(in_channels, in_channels, kernel_size=(3, 1, 1), stride=1, time_stride=1, padding=0)
            if temporal_up
            else nn.Identity()
        )
        self.conv2 = CosmosCausalConv3d(
            in_channels, in_channels, kernel_size=(1, 3, 3), stride=1, time_stride=1, padding=1
        )
        self.conv3 = CosmosCausalConv3d(
            in_channels, in_channels, kernel_size=(1, 1, 1), stride=1, time_stride=1, padding=0
        )
        self.temporal_up = temporal_up

    def forward(self, hidden_states):
        # hybrid upsample temporally
        if self.temporal_up:
            time_factor = int(hidden_states.shape[2] > 1)
            hidden_states = hidden_states.repeat_interleave((time_factor + 1), dim=2)
            hidden_states = hidden_states[..., time_factor:, :, :]
            hidden_states = self.conv1(hidden_states) + hidden_states

        # hybrid upsample spatially
        hidden_states = hidden_states.repeat_interleave(2, dim=3).repeat_interleave(2, dim=4)
        hidden_states = self.conv2(hidden_states) + hidden_states

        # final 1x1x1 conv
        hidden_states = self.conv3(hidden_states)
        return hidden_states


class CosmosPatch3D(nn.Module):
    """A 3D discrete wavelet transform for video data."""

    def __init__(self, patch_size: int = 1):
        super().__init__()
        self.patch_size = patch_size
        wavelets = torch.tensor([0.7071067811865476, 0.7071067811865476])

        self.range = range(int(torch.log2(torch.tensor(self.patch_size)).item()))
        self.register_buffer("_arange", torch.arange(2), persistent=False)
        self.register_buffer("wavelets", wavelets, persistent=False)
        self.register_buffer("patch_size_buffer", patch_size * torch.ones([1], dtype=torch.int32), persistent=False)

    def _dwt(self, hidden_states, mode="reflect", rescale=False):
        dtype = hidden_states.dtype
        wavelets = self.wavelets

        wavelet_len = wavelets.shape[0]
        seq_len = hidden_states.shape[1]
        wavelets_low = wavelets.flip(0).reshape(1, 1, -1).repeat(seq_len, 1, 1).to(dtype=dtype)
        wavelets_high = (wavelets * ((-1) ** self._arange)).reshape(1, 1, -1).repeat(seq_len, 1, 1).to(dtype=dtype)

        # Handles temporal axis.
        hidden_states = F.pad(
            hidden_states,
            pad=(
                max(0, wavelet_len - 2),
                wavelet_len - 1,
                wavelet_len - 2,
                wavelet_len - 1,
                wavelet_len - 2,
                wavelet_len - 1,
            ),
            mode=mode,
        ).to(dtype)
        hidden_states_low = F.conv3d(
            hidden_states, wavelets_low.unsqueeze(3).unsqueeze(4), groups=seq_len, stride=(2, 1, 1)
        )
        hidden_states_high = F.conv3d(
            hidden_states, wavelets_high.unsqueeze(3).unsqueeze(4), groups=seq_len, stride=(2, 1, 1)
        )

        # Handles spatial axes.
        hidden_low_low = F.conv3d(
            hidden_states_low, wavelets_low.unsqueeze(2).unsqueeze(4), groups=seq_len, stride=(1, 2, 1)
        )
        hidden_low_hight = F.conv3d(
            hidden_states_low, wavelets_high.unsqueeze(2).unsqueeze(4), groups=seq_len, stride=(1, 2, 1)
        )
        hidden_high_low = F.conv3d(
            hidden_states_high, wavelets_low.unsqueeze(2).unsqueeze(4), groups=seq_len, stride=(1, 2, 1)
        )
        hidden_high_high = F.conv3d(
            hidden_states_high, wavelets_high.unsqueeze(2).unsqueeze(4), groups=seq_len, stride=(1, 2, 1)
        )

        hidden_lll = F.conv3d(hidden_low_low, wavelets_low.unsqueeze(2).unsqueeze(3), groups=seq_len, stride=(1, 1, 2))
        hidden_llh = F.conv3d(
            hidden_low_low, wavelets_high.unsqueeze(2).unsqueeze(3), groups=seq_len, stride=(1, 1, 2)
        )
        hidden_lhl = F.conv3d(
            hidden_low_hight, wavelets_low.unsqueeze(2).unsqueeze(3), groups=seq_len, stride=(1, 1, 2)
        )
        hidden_lhh = F.conv3d(
            hidden_low_hight, wavelets_high.unsqueeze(2).unsqueeze(3), groups=seq_len, stride=(1, 1, 2)
        )
        hidden_hll = F.conv3d(
            hidden_high_low, wavelets_low.unsqueeze(2).unsqueeze(3), groups=seq_len, stride=(1, 1, 2)
        )
        hidden_hlh = F.conv3d(
            hidden_high_low, wavelets_high.unsqueeze(2).unsqueeze(3), groups=seq_len, stride=(1, 1, 2)
        )
        hidden_hhl = F.conv3d(
            hidden_high_high, wavelets_low.unsqueeze(2).unsqueeze(3), groups=seq_len, stride=(1, 1, 2)
        )
        hidden_hhh = F.conv3d(
            hidden_high_high, wavelets_high.unsqueeze(2).unsqueeze(3), groups=seq_len, stride=(1, 1, 2)
        )

        out = torch.cat(
            [hidden_lll, hidden_llh, hidden_lhl, hidden_lhh, hidden_hll, hidden_hlh, hidden_hhl, hidden_hhh], dim=1
        )
        if rescale:
            out = out / (2 * torch.sqrt(torch.tensor(2.0)))
        return out

    def forward(self, hidden_states: torch.FloatTensor):
        hidden_states_i, hidden_states_v = torch.split(hidden_states, [1, hidden_states.shape[2] - 1], dim=2)
        hidden_states = torch.cat([hidden_states_i.repeat_interleave(self.patch_size, dim=2), hidden_states_v], dim=2)
        for _ in self.range:
            hidden_states = self._dwt(hidden_states, rescale=True)
        return hidden_states


class CosmosUnpatch3D(nn.Module):
    """A 3D inverse discrete wavelet transform for video wavelet decompositions."""

    def __init__(self, patch_size=1):
        super().__init__()
        self.patch_size = patch_size
        wavelets = torch.tensor([0.7071067811865476, 0.7071067811865476])

        self.register_buffer("wavelets", wavelets, persistent=False)
        self.range = range(int(torch.log2(torch.tensor(self.patch_size)).item()))
        self.register_buffer("_arange", torch.arange(2), persistent=False)

    def _idwt(self, hidden_states, rescale=False):
        dtype = hidden_states.dtype
        wavelets = self.wavelets

        groups = hidden_states.shape[1] // 8  # split into 8 spatio-temporal filtered tenors.
        hl = wavelets.flip([0]).reshape(1, 1, -1).repeat([groups, 1, 1]).to(dtype=dtype)
        hh = (wavelets * ((-1) ** self._arange)).reshape(1, 1, -1).repeat(groups, 1, 1).to(dtype=dtype)
        xlll, xllh, xlhl, xlhh, xhll, xhlh, xhhl, xhhh = torch.chunk(hidden_states, 8, dim=1)

        # Height height transposed convolutions.
        xll = F.conv_transpose3d(xlll, hl.unsqueeze(2).unsqueeze(3), groups=groups, stride=(1, 1, 2))
        xll += F.conv_transpose3d(xllh, hh.unsqueeze(2).unsqueeze(3), groups=groups, stride=(1, 1, 2))

        xlh = F.conv_transpose3d(xlhl, hl.unsqueeze(2).unsqueeze(3), groups=groups, stride=(1, 1, 2))
        xlh += F.conv_transpose3d(xlhh, hh.unsqueeze(2).unsqueeze(3), groups=groups, stride=(1, 1, 2))

        xhl = F.conv_transpose3d(xhll, hl.unsqueeze(2).unsqueeze(3), groups=groups, stride=(1, 1, 2))
        xhl += F.conv_transpose3d(xhlh, hh.unsqueeze(2).unsqueeze(3), groups=groups, stride=(1, 1, 2))

        xhh = F.conv_transpose3d(xhhl, hl.unsqueeze(2).unsqueeze(3), groups=groups, stride=(1, 1, 2))
        xhh += F.conv_transpose3d(xhhh, hh.unsqueeze(2).unsqueeze(3), groups=groups, stride=(1, 1, 2))

        # Handles width transposed convolutions.
        xl = F.conv_transpose3d(xll, hl.unsqueeze(2).unsqueeze(4), groups=groups, stride=(1, 2, 1))
        xl += F.conv_transpose3d(xlh, hh.unsqueeze(2).unsqueeze(4), groups=groups, stride=(1, 2, 1))
        xh = F.conv_transpose3d(xhl, hl.unsqueeze(2).unsqueeze(4), groups=groups, stride=(1, 2, 1))
        xh += F.conv_transpose3d(xhh, hh.unsqueeze(2).unsqueeze(4), groups=groups, stride=(1, 2, 1))

        # Handles time axis transposed convolutions.
        hidden_states = F.conv_transpose3d(xl, hl.unsqueeze(3).unsqueeze(4), groups=groups, stride=(2, 1, 1))
        hidden_states += F.conv_transpose3d(xh, hh.unsqueeze(3).unsqueeze(4), groups=groups, stride=(2, 1, 1))

        if rescale:
            hidden_states = hidden_states * (2 * torch.sqrt(torch.tensor(2.0)))
        return hidden_states

    def forward(self, hidden_states: torch.FloatTensor):
        for _ in self.range:
            hidden_states = self._idwt(hidden_states, rescale=True)
        hidden_states = hidden_states[:, :, self.patch_size - 1 :, ...]
        return hidden_states


# NOTE: Copy from Emu3 fails for all subsequent modules, because layers init under condition or
# for loop aren't overwritten/skipped correctly. Modular cannot handle it yet!
# A bigger refactor to make VAE a model of its own, reusable by all VLMs is a better long-term solution (@raushan TODO)
class CosmosVQVAEResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = CosmosVQVAETemporalNorm(in_channels)
        self.norm2 = CosmosVQVAETemporalNorm(out_channels)

        self.conv1 = nn.Sequential(
            CosmosCausalConv3d(in_channels, out_channels, kernel_size=(1, 3, 3), stride=1, padding=1),
            CosmosCausalConv3d(out_channels, out_channels, kernel_size=(3, 1, 1), stride=1, padding=0),
        )
        self.conv2 = nn.Sequential(
            CosmosCausalConv3d(out_channels, out_channels, kernel_size=(1, 3, 3), stride=1, padding=1),
            CosmosCausalConv3d(out_channels, out_channels, kernel_size=(3, 1, 1), stride=1, padding=0),
        )
        self.dropout = torch.nn.Dropout(dropout)

        if self.in_channels != self.out_channels:
            self.nin_shortcut = CosmosCausalConv3d(
                in_channels, out_channels, kernel_size=(1, 1, 1), stride=1, padding=0
            )

    def forward(self, hidden_states: torch.Tensor):
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states *= torch.sigmoid(hidden_states)
        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states)
        hidden_states *= torch.sigmoid(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.in_channels != self.out_channels:
            residual = self.nin_shortcut(residual)

        return residual + hidden_states


class CosmosVQVAETemporalAttentionBlock(Emu3VQVAEAttentionBlock):
    pass


class CosmosVQVAEAttentionBlock(Emu3VQVAEAttentionBlock):
    pass


class CosmosVQVAEAttention(nn.Module):
    def __init__(self, config, in_channels):
        super().__init__()

        self.attn_1 = CosmosVQVAEAttentionBlock(config)
        self.attn_2 = CosmosVQVAETemporalAttentionBlock(config)
        self.attn_norm_1 = CosmosVQVAETemporalNorm(in_channels)
        self.attn_norm_2 = CosmosVQVAETemporalNorm(in_channels)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        # Apply attn norm + attn in spatial dim
        residual = hidden_states
        hidden_states = self.attn_norm_1(hidden_states)

        # b c t h w -> (b t) c (h w)
        batch_size, channels, temporal, height, width = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).contiguous()
        hidden_states = hidden_states.view(batch_size * temporal, channels, height * width).transpose(1, 2)
        hidden_states = self.attn_1(hidden_states)[0]
        hidden_states = hidden_states.reshape(batch_size, temporal, height, width, channels).permute(0, 4, 1, 2, 3)
        hidden_states += residual

        # Apply attn norm + attn in temporal dim
        residual = hidden_states
        hidden_states = self.attn_norm_2(hidden_states)

        # b c t h w -> (b h w) c t
        batch_size, channels, temporal, height, width = hidden_states.shape
        hidden_states = hidden_states.permute(0, 3, 4, 1, 2).contiguous()
        hidden_states = hidden_states.view(batch_size * height * width, channels, temporal).transpose(1, 2)
        hidden_states = self.attn_2(hidden_states, attention_mask=attention_mask)[0]
        hidden_states = hidden_states.reshape(batch_size, height, width, temporal, channels).permute(0, 4, 3, 1, 2)
        hidden_states += residual
        return hidden_states


class CosmosVQVAEMiddleBlock(nn.Module):
    def __init__(self, config, in_channels):
        super().__init__()
        self.block_1 = CosmosVQVAEResnetBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            dropout=config.dropout,
        )
        self.attn = CosmosVQVAEAttention(config, in_channels)
        self.block_2 = CosmosVQVAEResnetBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            dropout=config.dropout,
        )

    def forward(self, hidden_states: torch.FloatTensor, attention_mask: torch.Tensor):
        hidden_states = self.block_1(hidden_states)
        hidden_states = self.attn(hidden_states, attention_mask=attention_mask)
        hidden_states = self.block_2(hidden_states)
        return hidden_states


class CosmosVQVAEDownBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_resolutions = len(config.channel_multiplier)
        self.num_res_blocks = config.num_res_blocks

        base_channels = config.base_channels
        channel_multiplier = config.channel_multiplier
        self.num_temporal_downs = int(math.log2(config.temporal_downsample_factor)) - int(math.log2(config.patch_size))

        in_channel_multiplier = (1,) + tuple(channel_multiplier)
        self.in_channel_multiplier = in_channel_multiplier
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = base_channels * in_channel_multiplier[i_level]
            block_out = base_channels * channel_multiplier[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    CosmosVQVAEResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=config.dropout,
                    )
                )
                block_in = block_out
                if config.attn_resolutions is not None and i_level in config.attn_resolutions:
                    attn.append(CosmosVQVAEAttention(config, block_in))

            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                temporal_down = i_level < self.num_temporal_downs
                down.downsample = CosmosVQVAEEncoderDownsample(block_in, temporal_down=temporal_down)

            self.down.append(down)

    def forward(self, hidden_states: torch.FloatTensor):
        for i_level, blocks in enumerate(self.down):
            for i_block in range(self.num_res_blocks):
                hidden_states = blocks.block[i_block](hidden_states)
                if len(blocks.attn) > 0:
                    residual = hidden_states
                    hidden_states = blocks.attn_norms[i_block](hidden_states)

                    batch_size, channels, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(batch_size, channels, height * width).transpose(1, 2)
                    hidden_states = blocks.attn[i_block](hidden_states)[0]

                    hidden_states = hidden_states.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
                    hidden_states = residual + hidden_states

            if i_level != self.num_resolutions - 1:
                hidden_states = blocks.downsample(hidden_states)

        return hidden_states


class CosmosVQVAEUpBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_resolutions = len(config.channel_multiplier)
        self.num_res_blocks = config.num_res_blocks
        self.num_temporal_ups = int(math.log2(config.temporal_downsample_factor)) - int(math.log2(config.patch_size))

        block_in = config.base_channels * config.channel_multiplier[-1]

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = config.base_channels * config.channel_multiplier[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    CosmosVQVAEResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=config.dropout,  # DIFF HERE
                    )
                )
                block_in = block_out
                if config.attn_resolutions is not None and i_level in config.attn_resolutions:
                    attn.append(CosmosVQVAEAttention(config, block_in))

            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                i_level_reverse = self.num_resolutions - i_level - 1
                temporal_up = 0 < i_level_reverse < self.num_temporal_ups + 1
                up.upsample = CosmosVQVAEDecoderUpsample(block_in, temporal_up=temporal_up)

            self.up.insert(0, up)

    def forward(self, hidden_states: torch.FloatTensor):
        for i_level, blocks in enumerate(self.up[::-1]):
            for i_block in range(self.num_res_blocks + 1):
                hidden_states = blocks.block[i_block](hidden_states)
                if len(blocks.attn) > 0:
                    residual = hidden_states
                    hidden_states = blocks.attn_norms[i_block](hidden_states)

                    batch_size, channels, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(batch_size, channels, height * width).transpose(1, 2)
                    hidden_states = blocks.attn[i_block](hidden_states)[0]

                    hidden_states = hidden_states.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
                    hidden_states = residual + hidden_states
            if i_level != len(self.up) - 1:
                hidden_states = blocks.upsample(hidden_states)

        return hidden_states


class CosmosVQVAEEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        base_channels = config.base_channels
        in_channels = config.in_channels * config.patch_size * config.patch_size * config.patch_size
        double_latent = config.double_latent
        latent_channels = config.latent_channels
        channel_multiplier = config.channel_multiplier
        block_in = base_channels * channel_multiplier[-1]
        out_channels = 2 * latent_channels if double_latent else latent_channels

        self.patch = CosmosPatch3D(config.patch_size)
        self.conv_in = nn.Sequential(
            CosmosCausalConv3d(in_channels, base_channels, kernel_size=(1, 3, 3), stride=1, padding=1),
            CosmosCausalConv3d(base_channels, base_channels, kernel_size=(3, 1, 1), stride=1, padding=0),
        )
        self.down_block = CosmosVQVAEDownBlock(config)
        self.middle_block = CosmosVQVAEMiddleBlock(config, block_in)
        self.norm_out = CosmosVQVAETemporalNorm(block_in)
        self.conv_out = nn.Sequential(
            CosmosCausalConv3d(block_in, out_channels, kernel_size=(1, 3, 3), stride=1, padding=1),
            CosmosCausalConv3d(out_channels, out_channels, kernel_size=(3, 1, 1), stride=1, padding=0),
        )

    def forward(self, pixel_values: torch.LongTensor, attention_mask: torch.Tensor):
        hidden_states = self.patch(pixel_values)

        # downsampling & middle
        hidden_states = self.conv_in(hidden_states)
        hidden_states = self.down_block(hidden_states)
        hidden_states = self.middle_block(hidden_states, attention_mask=attention_mask)

        # end
        hidden_states = self.norm_out(hidden_states)
        hidden_states *= torch.sigmoid(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        return hidden_states


class CosmosVQVAEDecoder(nn.Module):
    def __init__(self, config: CosmosVQVAEConfig):
        super().__init__()

        block_in = config.base_channels * config.channel_multiplier[-1]
        self.conv_in = nn.Sequential(
            CosmosCausalConv3d(config.latent_channels, block_in, kernel_size=(1, 3, 3), stride=1, padding=1),
            CosmosCausalConv3d(block_in, block_in, kernel_size=(3, 1, 1), stride=1, padding=0),
        )
        self.middle_block = CosmosVQVAEMiddleBlock(config, block_in)
        self.up_block = CosmosVQVAEUpBlock(config)

        block_in = config.base_channels * config.channel_multiplier[0]
        self.norm_out = CosmosVQVAETemporalNorm(block_in)
        out_channels = config.out_channels * config.patch_size * config.patch_size * config.patch_size
        self.conv_out = nn.Sequential(
            CosmosCausalConv3d(block_in, out_channels, kernel_size=(1, 3, 3), stride=1, padding=1),
            CosmosCausalConv3d(out_channels, out_channels, kernel_size=(3, 1, 1), stride=1, padding=0),
        )
        self.unpatch = CosmosUnpatch3D(config.patch_size)

    def forward(self, hidden_states: torch.FloatTensor, attention_mask: torch.Tensor):
        hidden_states = self.conv_in(hidden_states)

        hidden_states = self.middle_block(hidden_states, attention_mask=attention_mask)
        hidden_states = self.up_block(hidden_states)

        hidden_states = self.norm_out(hidden_states)
        hidden_states *= torch.sigmoid(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        hidden_states = self.unpatch(hidden_states)

        return hidden_states


class CosmosVQVAE(Emu3VQVAE):
    base_model_prefix = "cosmosvideovq"
    _no_split_modules = [
        "CosmosVQVAEAttentionBlock",
        "CosmosVQVAETemporalAttentionBlock",
        "CosmosVQVAEResnetBlock",
        "CosmosVQVAEVectorQuantizer",
    ]

    def __init__(self, config: CosmosVQVAEConfig):
        super().__init__(config)

        self.config = config

        self.encoder = CosmosVQVAEEncoder(config)
        self.decoder = CosmosVQVAEDecoder(config)
        self.quantize = CosmosVQVAEVectorQuantizer(config)

        self.quant_conv = CosmosCausalConv3d(
            config.latent_channels, config.embed_dim, kernel_size=(1, 1, 1), padding=0
        )
        self.post_quant_conv = CosmosCausalConv3d(
            config.embed_dim, config.latent_channels, kernel_size=(1, 1, 1), padding=0
        )

        self.eval()  # VQ model is frozen and not implemented for training
        self.post_init()

    def encode(self, pixel_values: torch.Tensor):
        # b t c h w -> b c t h w
        pixel_values = pixel_values.permute(0, 2, 1, 3, 4)
        causal_mask = self._update_causal_mask(pixel_values)
        hidden_states = self.encoder(pixel_values, attention_mask=causal_mask)
        hidden_states = self.quant_conv(hidden_states)
        codes, quant_indices = self.quantize(hidden_states)
        return codes, quant_indices

    def decode(self, indices: torch.Tensor):
        codes = self.quantize.indices_to_codes(indices)

        # dequantization returns an fp32 tensor, we need to cast to VAE dtype if needed
        codes = codes.to(self.post_quant_conv.conv3d.weight.dtype)
        hidden_states = self.post_quant_conv(codes)
        causal_mask = self._update_causal_mask(hidden_states)
        video = self.decoder(hidden_states, attention_mask=causal_mask)
        return video

    def forward(self, pixel_values):
        quant_info, quant_codes = self.encode(pixel_values)
        reconstructions = self.decode(quant_info)
        return reconstructions, quant_info

    def _update_causal_mask(self, input_tensor: torch.Tensor):
        # For SDPA, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in order to dispatch on Flash Attention 2
        if self.config._attn_implementation == "flash_attention_2" or self.config._attn_implementation:
            return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = self.config.base_channels * self.config.channel_multiplier[-1]
        batch_size = input_tensor.shape[0] * 40 * 64
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            sequence_length=sequence_length,
            dtype=dtype,
            device=device,
            batch_size=batch_size,
        )

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        sequence_length: int,
        dtype: torch.dtype,
        device: torch.device,
        batch_size: int,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length).

        Args:
            sequence_length (`int`):
                The sequence length being processed. Tthe mask will be as long as the sequence length,
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        min_dtype = torch.finfo(dtype).min
        causal_mask = torch.full((sequence_length, sequence_length), fill_value=min_dtype, dtype=dtype, device=device)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)

        return causal_mask


class CosmosAbsolutePositionEmbedding(nn.Module):
    def __init__(self, config: CosmosTextConfig):
        super().__init__()
        hidden_size = config.hidden_size
        dim_spatial = hidden_size // 6 * 2
        dim_temporal = hidden_size - 2 * dim_spatial
        self.latent_shape = config.rope_latent_shape
        num_temporal_grid, num_height_grid, num_width_grid = self.latent_shape
        self.pos_emb_h = self.get_1d_sincos_pos_embed_from_grid(dim_spatial, pos=num_height_grid)
        self.pos_emb_w = self.get_1d_sincos_pos_embed_from_grid(dim_spatial, pos=num_width_grid)
        self.pos_emb_t = self.get_1d_sincos_pos_embed_from_grid(dim_temporal, pos=num_temporal_grid)

        pos_embeddings = self._create_absolute_embeddings()
        self.register_buffer("abs_pos_embeddings", pos_embeddings, persistent=False)

    @staticmethod
    def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
        pos = torch.arange(pos, dtype=torch.float32)
        omega = torch.arange(embed_dim // 2, dtype=torch.float32)
        omega = 1.0 / (10000 ** (omega / (embed_dim / 2.0)))

        out = pos[:, None] * omega[None, :]
        emb_sin = torch.sin(out)
        emb_cos = torch.cos(out)
        emb = torch.cat([emb_sin, emb_cos], dim=1)
        return emb

    def _create_absolute_embeddings(self, training_type=None) -> torch.Tensor:
        num_temporal_grid, num_height_grid, num_width_grid = self.latent_shape
        pos_embeddings = torch.cat(
            [
                self.pos_emb_t[:, None, None, :].repeat(1, num_height_grid, num_width_grid, 1),
                self.pos_emb_h[None, :, None, :].repeat(num_temporal_grid, 1, num_width_grid, 1),
                self.pos_emb_w[None, None, :, :].repeat(num_temporal_grid, num_height_grid, 1, 1),
            ],
            dim=-1,
        )

        pos_embeddings = pos_embeddings.flatten(0, 2)
        bov_embedding = torch.zeros(
            (1, *pos_embeddings.shape[1:]), device=pos_embeddings.device, dtype=pos_embeddings.dtype
        )
        pos_embeddings = torch.cat((bov_embedding, pos_embeddings), dim=0)
        return pos_embeddings.unsqueeze(0)

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor):
        device_type = hidden_states.device.type
        position_ids = position_ids.squeeze(0)
        position_embeddings = self.abs_pos_embeddings[:, position_ids].to(dtype=hidden_states.dtype)
        position_embeddings = position_embeddings.squeeze(0)
        with torch.autocast(device_type=device_type, enabled=False):
            hidden_states = hidden_states + position_embeddings
        return hidden_states


class CosmosTextRotaryEmbedding(nn.Module):
    def __init__(self, config: CosmosTextConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = self._compute_3d_rope_parameters
        spatial_inv_freq, temporal_inv_freq = self.rope_init_fn(self.config, device)
        self.register_buffer("spatial_inv_freq", spatial_inv_freq, persistent=False)
        self.register_buffer("temporal_inv_freq", temporal_inv_freq, persistent=False)

    @staticmethod
    def _compute_3d_rope_parameters(
        config: Optional[CosmosConfig] = None,
        device: Optional["torch.device"] = None,
    ) -> Tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original Cosmos RoPE implementation
        Args:
            config ([`~transformers.PretrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
        Returns:
            Tuple of (`torch.Tensor`, `torch.Tensor`), containing the inverse frequencies for the RoPE embeddings for
            spatial and temporal positions.
        """
        base = config.rope_theta
        latent_shape = getattr(config, "rope_latent_shape", None)
        partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        dim = int(head_dim * partial_rotary_factor)

        # Compute the inverse frequencies
        if latent_shape is None:
            raise ValueError("RoPE latent shape is required for 3D RoPE, but not found in `self.config`")

        dim_spatial = dim // 6 * 2
        dim_temporal = dim - 2 * dim_spatial
        dim_spatial_range = torch.arange(0, dim_spatial, 2)[: (dim_spatial // 2)].float().to(device) / dim_spatial
        spatial_inv_freq = 1.0 / (base**dim_spatial_range)
        dim_temporal_range = torch.arange(0, dim_temporal, 2)[: (dim_temporal // 2)].float().to(device) / dim_temporal
        temporal_inv_freq = 1.0 / (base**dim_temporal_range)
        return spatial_inv_freq, temporal_inv_freq

    @torch.no_grad()
    def forward(self, x, position_ids):
        # Core RoPE block
        # NOTE: Position ids are a 3D tensors of shape [bs, 3, seq_len] with different positions for THW grids
        spatial_freq_expanded = self.spatial_inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        temporal_freq_expanded = self.temporal_inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_width = position_ids[:, 2:3, :].float()
        position_height = position_ids[:, 1:2, :].float()
        position_temporal = position_ids[:, 0:1, :].float()

        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            width_freqs = (spatial_freq_expanded.float() @ position_width.float()).transpose(1, 2)
            height_freqs = (spatial_freq_expanded.float() @ position_height.float()).transpose(1, 2)
            temporal_freqs = (temporal_freq_expanded.float() @ position_temporal.float()).transpose(1, 2)
            emb = torch.cat((temporal_freqs, height_freqs, width_freqs) * 2, dim=-1)

            if self.config.is_video_to_world and emb.shape[1] != 1:
                # since we added <bov> token at the beginning of the video for text2world
                # we also extend the position embedding by one token in the beginning. Only in prefill stage,
                # at decoding time we don't add more <bov> token
                bov_pos_emb = torch.zeros((1, 1, emb.shape[-1]), device=emb.device)
                emb = torch.cat((bov_pos_emb, emb), dim=-2)

            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class CosmosTextRMSNorm(Emu3RMSNorm):
    pass


class CosmosTextMLP(Emu3MLP):
    pass


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


class CosmosTextAttention(LlamaAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: CosmosTextConfig, layer_idx: int, is_self_attention: bool = True):
        # define `kv_hidden_size` before `init` to get it prepended to copied code, not appended
        kv_hidden_size = config.hidden_size if is_self_attention else config.cross_attn_hidden_size
        super().__init__()

        self.is_causal = is_self_attention
        self.q_norm = CosmosTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = CosmosTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_proj = nn.Linear(kv_hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(kv_hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_embeddings: Tuple[torch.Tensor, torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # if encoder_hidden_states are provided this layer is used as a cross-attention layer
        is_cross_attention = encoder_hidden_states is not None
        current_states = encoder_hidden_states if is_cross_attention else hidden_states

        input_shape = hidden_states.shape[:-1]
        kv_shape = current_states.shape[:-1]
        hidden_shape = (*kv_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim).transpose(1, 2)
        query_states = self.q_norm(query_states)

        if past_key_value is not None and isinstance(past_key_value, EncoderDecoderCache):
            is_updated = past_key_value.is_updated.get(self.layer_idx)
            if is_cross_attention:
                # after the first generated id, we can subsequently re-use all key/value_states from cache
                past_key_value.is_updated[self.layer_idx] = True
                past_key_value = past_key_value.cross_attention_cache
            else:
                past_key_value = past_key_value.self_attention_cache

        if is_cross_attention and past_key_value is not None and is_updated:
            # reuse k,v, cross_attentions
            key_states = past_key_value.key_cache[self.layer_idx]
            value_states = past_key_value.value_cache[self.layer_idx]
        else:
            key_states = self.k_proj(current_states).view(hidden_shape).transpose(1, 2)
            value_states = self.v_proj(current_states).view(hidden_shape).transpose(1, 2)

            key_states = self.k_norm(key_states)

            if not is_cross_attention:
                cos, sin = position_embeddings
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            if past_key_value is not None:
                # save all key/value_states to cache to be re-used for fast auto-regressive generation
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            is_causal=False if is_cross_attention else None,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class CosmosTextDecoderLayer(nn.Module):
    def __init__(self, config: CosmosTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config

        self.self_attn = CosmosTextAttention(config=config, layer_idx=layer_idx)

        if layer_idx in config.insert_cross_attn_layers:
            self.cross_attn = CosmosTextAttention(config=config, layer_idx=layer_idx, is_self_attention=False)
            self.cross_input_layernorm = CosmosTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.mlp = CosmosTextMLP(config)
        self.input_layernorm = CosmosTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = CosmosTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        cross_attn_weights = None
        if self.self_attn.layer_idx in self.config.insert_cross_attn_layers:
            residual = hidden_states
            hidden_states = self.cross_input_layernorm(hidden_states)

            # NOTE: orig impl overrides mask with all ones for all cases, but why?
            if cross_attention_mask is not None:
                cross_attention_mask = torch.zeros_like(cross_attention_mask)
            hidden_states, cross_attn_weights = self.cross_attn(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=cross_attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )
            hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs


COSMOS_TEXT_INPUTS_DOCSTRING = None


class CosmosTextModel(LlamaModel):
    def __init__(self, config: CosmosTextConfig):
        super().__init__(config)
        self.absolute_position_emb = CosmosAbsolutePositionEmbedding(config=config)

    def _calculate_position_ids(self, seq_length: int, device: str = "cpu"):
        """
        Calculates positions ids for each grid separately for 3D RoPE. Given a sequence length,
        the position ids will be calculated as follows:

        For width grids, positions are constructed in vanilla way up to `num_width_grid`, after which
        the counter startes from 0 again. As such with `num_width_grid=2` the width positions are `[0, 1, 0, 1, 0, 1, ...]`

        For height grids, positions are constructed by keeping the same height position until the whole row of `num_width_grid`
        is exhausted. After that height positions increases by `1`. For example with `num_width_grid=2`,
        the height position ids will be `[0, 0, 1, 1, 2, 2, ..., `num_height_grid-1`]`

        For temporal grids, positions are very much similar, but this time the counter is increased by `1` only when a new video frame
        is reached. For example with `num_width_grid=2` and `num_height_grid=3`, the temporal position will be `[0, 0, 0, 0, 0, 0, 1, 1, ...]`.
        In other words, the ids are updted every `num_width_grid * num_height_grid` positions.
        """
        if self.config.is_video_to_world:
            seq_length -= 1  # `bov` token not counted to positions

        num_temporal_grid, num_height_grid, num_width_grid = self.config.rope_latent_shape
        one_frame_len = num_height_grid * num_width_grid
        w_grids = math.ceil(seq_length / num_width_grid)
        num_frames = math.ceil(seq_length / one_frame_len)
        position_width = torch.tensor(list(range(64)) * w_grids)[:seq_length].unsqueeze(0)
        position_height = torch.arange(w_grids).repeat_interleave(num_width_grid)[:seq_length].unsqueeze(0)
        position_height = position_height % num_height_grid
        position_temporal = torch.arange(num_frames).repeat_interleave(one_frame_len)[:seq_length].unsqueeze(0)
        position_ids = torch.stack([position_temporal, position_height, position_width], dim=1).to(device)
        return position_ids

    @add_start_docstrings_to_model_forward(COSMOS_TEXT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_ids_rope: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            if not self.config.insert_cross_attn_layers:
                past_key_values = DynamicCache()
            else:
                past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids_rope is None:
            seq_length_with_past = cache_position[-1] + 1
            position_ids_rope = self._calculate_position_ids(seq_length_with_past, device=cache_position.device)
            position_ids_rope = position_ids_rope[..., -inputs_embeds.shape[1] :]

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        self_attn_cache = (
            past_key_values.self_attention_cache
            if isinstance(past_key_values, EncoderDecoderCache)
            else past_key_values
        )
        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values=self_attn_cache,
            output_attentions=output_attentions,
        )
        if encoder_attention_mask is not None:
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)

        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.to(inputs_embeds.dtype)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids_rope)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.config.apply_abs_pos_emb:
                hidden_states = self.absolute_position_emb(hidden_states, position_ids)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    encoder_hidden_states,
                    causal_mask,
                    encoder_attention_mask,
                    position_ids_rope,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=causal_mask,
                    cross_attention_mask=encoder_attention_mask,
                    position_ids=position_ids_rope,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                all_cross_attns += (layer_outputs[2],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attns,
        )
        return output if return_dict else output.to_tuple()


COSMOS_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, max_num_images, max_num_tiles, channels, image_size, image_size)):
            The tensors corresponding to the input videos. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CosmosVideoProcessor.__call__`] for details ([]`CosmosProcessor`] uses
            [`CosmosVideoProcessor`] for processing images).
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Has to be an instance of [`~cache_utils.Cache`] instance, see our
            [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


class CosmosPreTrainedModel(Emu3PreTrainedModel):
    _no_split_modules = [
        "CosmosVQVAEAttentionBlock",
        "CosmosVQVAETemporalAttentionBlock",
        "CosmosVQVAEResnetBlock",
        "CosmosVQVAEVectorQuantizer",
        "CosmosTextDecoderLayer",
    ]


class CosmosModel(CosmosPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.language_model = CosmosTextModel._from_config(config.text_config)
        self.vqmodel = CosmosVQVAE._from_config(config.vq_config)
        if config.text_config.is_video_to_world:
            self.prompt_encoder = AutoModel.from_config(config.prompt_encoder_config).encoder
            self._keep_in_fp32_modules = ["prompt_encoder"]

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_video_tokens(self, pixel_values_videos: torch.FloatTensor):
        """
        Tokenizes images into discrete tokens with Vector Quantizer module.

        Args:
            pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input videos.
        """
        vq_tokens, _ = self.vqmodel.encode(pixel_values_videos)
        vq_tokens = vq_tokens.flatten(1)  # (batch_size, seq_length)

        # Only supported lengths are 1 for image and 2 for video conditioning (i.e. 1 or 2 time grids of input context)
        # VAE can support only 5 time grids, so we'll generate the rest 3 or 4 grids
        latent_context_size = 2 if pixel_values_videos.shape[1] > 1 else 1
        time, height, width = self.config.text_config.rope_latent_shape
        num_gen_tokens = int(np.prod([time - latent_context_size, height, width]))
        vq_tokens = vq_tokens[:, :-num_gen_tokens]  # remove repeated pad tokens

        # Decoder start token is always added by `generate()`. We need to handle it in `forward()` manually though
        if self.config.text_config.is_video_to_world:
            bov_tokens = [[self.config.get_text_config().bos_token_id]] * vq_tokens.shape[0]
            bov_tokens = torch.tensor(bov_tokens, device=vq_tokens.device, dtype=vq_tokens.dtype)
            vq_tokens = torch.cat([bov_tokens, vq_tokens], dim=-1)

        return vq_tokens

    @torch.no_grad
    def decode_video_tokens(self, video_tokens: torch.LongTensor):
        """
        Decodes generated image tokens from language model to continuous pixel values
        with VQGAN module via upsampling.

        Args:
            video_tokens (`torch.LongTensor` of shape `(batch_size, num_of_tokens)`):
                The tensors corresponding to the input video.
        """
        video_tokens = video_tokens.view(-1, 5, 40, 64)
        video = self.vqmodel.decode(video_tokens)
        return video

    @add_start_docstrings_to_model_forward(COSMOS_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values_videos: torch.FloatTensor = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_ids_rope: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CosmosBaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (decoder_input_ids is None) ^ (pixel_values_videos is not None):
            raise ValueError("You must specify exactly one of decoder_input_ids or pixel_values_videos")

        if pixel_values_videos is not None:
            decoder_input_ids = self.get_video_tokens(pixel_values_videos)

        encoder_hidden_states = None
        if self.config.is_encoder_decoder:
            if encoder_outputs is None:
                encoder_outputs = self.prompt_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    output_hidden_states=True,
                    output_attentions=True,
                )
            encoder_hidden_states = encoder_outputs.last_hidden_state
            if attention_mask is not None:
                lengths = attention_mask.sum(dim=1)
                for batch_id in range(encoder_hidden_states.shape[0]):
                    encoder_hidden_states[batch_id][lengths[batch_id] :] = 0

        outputs = self.language_model(
            input_ids=decoder_input_ids,
            inputs_embeds=decoder_inputs_embeds,
            attention_mask=decoder_attention_mask,
            encoder_attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            position_ids=position_ids,
            position_ids_rope=position_ids_rope,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
        )

        output = CosmosBaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.hidden_states,
            decoder_attentions=outputs.attentions,
            encoder_last_hidden_state=encoder_hidden_states,
            encoder_hidden_states=encoder_outputs.hidden_states if encoder_outputs is not None else None,
            encoder_attentions=encoder_outputs.attentions if encoder_outputs is not None else None,
            cross_attentions=outputs.cross_attentions,
        )
        return output if return_dict else output.to_tuple()

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


CosmosImageVocabularyMapping = None
CosmosForCausalLM = None


class CosmosForConditionalGeneration(CosmosPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = CosmosModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    @add_start_docstrings_to_model_forward(COSMOS_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CosmosCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values_videos: torch.FloatTensor = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_ids_rope: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[Tuple, CosmosCausalLMOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> import torch
        >>> import imageio
        >>> from transformers.image_utils import load_video
        >>> from transformers import CosmosProcessor, CosmosForConditionalGeneration

        >>> model_id = "NVIDIA/Cosmos-5B-hf"
        >>> processor = CosmosProcessor.from_pretrained(model_id)

        >>> model = CosmosForConditionalGeneration.from_pretrained(
        ...     model_id,
        ...     torch_dtype="bfloat16",
        ...     low_cpu_mem_usage=True,
        ...     device_map="auto",
        ... )

        >>> # Generate from last 9 frames of the video
        >>> video, _ = load_video("cosmos1/models/autoregressive/assets/v1p0/input.mp4", backend="decord")[-9:]
        >>> text = "A video recorded from a moving vehicle's perspective, capturing roads, buildings, landscapes, and changing weather and lighting conditions."
        >>> inputs = processor(videos=video, text=text, return_tensors="pt").to(model.device, dtype=torch.bfloat16)

        >>> out = model.generate(**inputs, max_new_tokens=7680)

        >>> # Remove the first token which is `BOS`. Decode the video and save.
        >>> video_decoded = model.model.decode_video_tokens(out[:, 1:])
        >>> video_decoded = video_decoded.permute(0, 2, 1, 3, 4).float()
        >>> video_processed = proc.postprocess([video_decoded[0]], return_tensors="np")
        >>> imageio.mimsave("generated_video.mp4", video_processed['pixel_values'].squeeze(0), fps=25)
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            pixel_values_videos=pixel_values_videos,
            position_ids=position_ids,
            position_ids_rope=position_ids_rope,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

        # FIXME: @raushan why this is not loaded in bf16 wehn asked?
        # self.lm_head.weight.data = self.lm_head.weight.data.to(torch.bfloat16)
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        output = CosmosCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
        return output if return_dict else output.to_tuple()

    @torch.no_grad()
    def generate(self, input_ids=None, pixel_values_videos=None, **kwargs):
        # Generation from video input only, so we obtain video input ids and pass to generate
        decoder_input_ids = self.model.get_video_tokens(pixel_values_videos)
        kwargs["cache_implementation"] = "static"
        if input_ids is None:
            return super().generate(decoder_input_ids, **kwargs)

        # Else we are in video2world generation. We need to encode the prompt
        attention_mask = kwargs.pop("attention_mask", None)
        encoder_outputs = self.model.prompt_encoder(
            input_ids, attention_mask=attention_mask, output_hidden_states=True, output_attentions=True
        )

        output = super().generate(
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            **kwargs,
        )
        return output

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        decoder_attention_mask=None,
        decoder_inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values_videos=None,
        encoder_outputs=None,
        attention_mask=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = super().prepare_inputs_for_generation(
            decoder_input_ids,
            past_key_values=past_key_values,
            decoder_attention_mask=decoder_attention_mask,
            decoder_inputs_embeds=decoder_inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values_videos=pixel_values_videos,
            use_cache=use_cache,
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            **kwargs,
        )

        # Cosmos needs 3D positions constructed in custom way. DO NOT overwrite 1D positions, which are used in `AbsolutePosEmbLayer`
        seq_length = cache_position[-1] + 1
        position_ids_rope = self.model.language_model._calculate_position_ids(seq_length, device=cache_position.device)
        input_ids_key = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
        input_length = model_inputs[input_ids_key].shape[1]
        model_inputs["position_ids_rope"] = position_ids_rope[..., -input_length:]

        # little hack to support encoder-decoder and decoder-only from one model class
        if not self.config.is_encoder_decoder:
            for input_name in ["input_ids", "attention_mask", "inputs_embeds"]:
                model_inputs[f"decoder_{input_name}"] = model_inputs.get(input_name, None)
                model_inputs.pop(input_name, None)

        if cache_position[0] != 0:
            model_inputs["pixel_values_videos"] = None

        return model_inputs


__all__ = [
    "CosmosForConditionalGeneration",
    "CosmosTextModel",
    "CosmosModel",
    "CosmosVQVAE",
    "CosmosConfig",
    "CosmosVQVAEConfig",
    "CosmosTextConfig",
    "CosmosTextPreTrainedModel",  # noqa: F822
    "CosmosPreTrainedModel",
]
