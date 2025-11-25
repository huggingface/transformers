# coding=utf-8
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Qwen3Omni model (Audio, Image, Video)."""

import math
import re
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ...activations import ACT2FN
from ...audio_utils import AudioInput
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PretrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...generation import GenerationMixin
from ...image_utils import ImageInput
from ...masking_utils import create_causal_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
)
from ...processing_utils import ProcessorMixin, Unpack
from ...tokenization_utils_base import TextInput
from ...utils import auto_docstring, can_return_tuple, logging
from ...utils.generic import OutputRecorder, TransformersKwargs, check_model_inputs
from ...video_utils import VideoInput, make_batched_videos
from ..mimi.modeling_mimi import MimiLayerScale
from ..qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniAudioEncoderConfig,
    Qwen2_5OmniThinkerConfig,
)
from ..qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniAudioAttention,
    Qwen2_5OmniAudioEncoder,
    Qwen2_5OmniPreTrainedModel,
    Qwen2_5OmniPreTrainedModelForConditionalGeneration,
    Qwen2_5OmniThinkerForConditionalGeneration,
    SnakeBeta,
)
from ..qwen2_5_omni.processing_qwen2_5_omni import Qwen2_5OmniProcessor, Qwen2_5OmniProcessorKwargs
from ..qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock
from ..qwen3.configuration_qwen3 import Qwen3Config
from ..qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3MLP,
    Qwen3Model,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)
from ..qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from ..qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
    Qwen3MoeMLP,
    Qwen3MoePreTrainedModel,
    Qwen3MoeSparseMoeBlock,
    load_balancing_loss_func,
)
from ..qwen3_vl_moe.configuration_qwen3_vl_moe import Qwen3VLMoeVisionConfig
from ..qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeTextModel,
    Qwen3VLMoeTextRotaryEmbedding,
    Qwen3VLMoeVisionAttention,
    Qwen3VLMoeVisionModel,
)


logger = logging.get_logger(__name__)


def _get_feat_extract_output_lengths(input_lengths):
    """
    Computes the output length of the convolutional layers and the output length of the audio encoder
    """

    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    return output_lengths


class Qwen3OmniMoeAudioEncoderConfig(Qwen2_5OmniAudioEncoderConfig):
    def __init__(
        self,
        num_mel_bins=128,
        encoder_layers=32,
        encoder_attention_heads=20,
        encoder_ffn_dim=5120,
        d_model=1280,
        dropout=0,
        attention_dropout=0,
        activation_function="gelu",
        activation_dropout=0,
        scale_embedding=False,
        initializer_range=0.02,
        max_source_positions=1500,
        n_window=100,
        output_dim=3584,
        n_window_infer=400,
        conv_chunksize=500,
        downsample_hidden_size=480,
        **kwargs,
    ):
        super().__init__(
            num_mel_bins,
            encoder_layers,
            encoder_attention_heads,
            encoder_ffn_dim,
            d_model,
            dropout,
            attention_dropout,
            activation_function,
            activation_dropout,
            scale_embedding,
            initializer_range,
            max_source_positions,
            n_window,
            output_dim,
            **kwargs,
        )
        self.n_window_infer = n_window_infer
        self.conv_chunksize = conv_chunksize
        self.downsample_hidden_size = downsample_hidden_size


class Qwen3OmniMoeVisionEncoderConfig(Qwen3VLMoeVisionConfig):
    pass


class Qwen3OmniMoeTextConfig(Qwen3MoeConfig):
    def __init__(
        self,
        vocab_size=3584,
        hidden_size=2048,
        intermediate_size=18944,
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        rope_scaling=None,
        attention_bias=False,
        sliding_window=None,
        attention_dropout=0,
        decoder_sparse_step=1,
        moe_intermediate_size=768,
        num_experts_per_tok=8,
        num_experts=128,
        norm_topk_prob=True,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        mlp_only_layers=None,
        **kwargs,
    ):
        super().__init__(
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            hidden_act,
            max_position_embeddings,
            initializer_range,
            rms_norm_eps,
            use_cache,
            tie_word_embeddings,
            rope_theta,
            rope_scaling,
            attention_bias,
            False,
            sliding_window,
            attention_dropout,
            decoder_sparse_step,
            moe_intermediate_size,
            num_experts_per_tok,
            num_experts,
            norm_topk_prob,
            output_router_logits,
            router_aux_loss_coef,
            mlp_only_layers,
            **kwargs,
        )
        del self.use_sliding_window
        self.sliding_window = sliding_window


class Qwen3OmniMoeThinkerConfig(Qwen2_5OmniThinkerConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen3OmniMoeThinker`]. It is used to instantiate a
    Qwen3-Omni-Thinker model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the thinker component of the Qwen3-Omni
    architecture.

    e.g. [Qwen/Qwen3-Omni-7B](https://huggingface.co/Qwen/Qwen3-Omni-7B)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        audio_config (`dict`, *optional*):
            The config dictionary of the audio backbone.
        vision_config (`dict`, *optional*):
            The config dictionary of the vision backbone.
        text_config (`dict`, *optional*):
            The config dictionary of the text backbone.
        audio_token_id (`int`, *optional*, defaults to 151646):
            The audio token id to encode the audio prompt.
        image_token_id (`int`, *optional*, defaults to 151655):
            The image token id to encode the image prompt.
        video_token_id (`int`, *optional*, defaults to 151656):
            The video token id to encode the video prompt.
        position_id_per_seconds (`int`, *optional*, defaults to 25):
            The increment of position id per second.
        audio_start_token_id (`int`, *optional*, defaults to 151647):
            The audio start token id to encode the audio prompt.
        user_token_id (`int`, *optional*, defaults to 872):
            The user token id to encode the user token.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

    Example:

    ```python
    >>> from transformers import Qwen3OmniMoeThinkerModel, Qwen3OmniMoeThinkerConfig

    >>> # Initializing a default Qwen3OmniMoeThinkerConfig
    >>> configuration = Qwen3OmniMoeThinkerConfig()

    >>> # Initializing a model (with random weights) from the default configuration
    >>> model = Qwen3OmniMoeThinkerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen3_omni_moe_thinker"
    # Override parent's attribute_map as we use audio_token_id directly, not audio_token_index
    attribute_map = {}

    def __init__(
        self,
        audio_config=None,
        vision_config=None,
        text_config=None,
        audio_token_id=151646,
        image_token_id=151655,
        video_token_id=151656,
        position_id_per_seconds=25,
        audio_start_token_id=151647,
        user_token_id=872,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(
            audio_config,
            vision_config,
            text_config,
            None,
            None,
            None,
            position_id_per_seconds,
            None,
            audio_start_token_id,
            None,
            user_token_id,
            initializer_range,
            **kwargs,
        )
        del self.seconds_per_chunk
        del self.audio_token_index
        del self.image_token_index
        del self.video_token_index
        del self.audio_end_token_id
        self.audio_token_id = audio_token_id
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id


class Qwen3OmniMoeTalkerCodePredictorConfig(Qwen3Config):
    def __init__(
        self,
        vocab_size=2048,
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=5,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=0.000001,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000,
        rope_scaling=None,
        attention_bias=False,
        sliding_window=None,
        layer_types=None,
        attention_dropout=0,
        num_code_groups=32,
        **kwargs,
    ):
        super().__init__(
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            hidden_act,
            max_position_embeddings,
            initializer_range,
            rms_norm_eps,
            use_cache,
            tie_word_embeddings,
            rope_theta,
            rope_scaling,
            attention_bias,
            False,
            sliding_window,
            None,
            layer_types,
            attention_dropout,
            **kwargs,
        )
        del self.use_sliding_window
        del self.max_window_layers
        self.sliding_window = sliding_window
        self.num_code_groups = num_code_groups


class Qwen3OmniMoeTalkerTextConfig(Qwen3MoeConfig):
    def __init__(
        self,
        vocab_size=3072,
        hidden_size=1024,
        intermediate_size=2048,
        num_hidden_layers=20,
        num_attention_heads=16,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=0.000001,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000,
        rope_scaling=None,
        attention_bias=False,
        sliding_window=None,
        attention_dropout=0,
        decoder_sparse_step=1,
        moe_intermediate_size=384,
        num_experts_per_tok=8,
        num_experts=128,
        norm_topk_prob=False,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        mlp_only_layers=None,
        **kwargs,
    ):
        super().__init__(
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            hidden_act,
            max_position_embeddings,
            initializer_range,
            rms_norm_eps,
            use_cache,
            tie_word_embeddings,
            rope_theta,
            rope_scaling,
            attention_bias,
            False,
            sliding_window,
            attention_dropout,
            decoder_sparse_step,
            moe_intermediate_size,
            num_experts_per_tok,
            num_experts,
            norm_topk_prob,
            output_router_logits,
            router_aux_loss_coef,
            mlp_only_layers,
            **kwargs,
        )
        del self.use_sliding_window
        self.sliding_window = sliding_window


class Qwen3OmniMoeTalkerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen3OmniMoeTalker`]. It is used to instantiate a
    Qwen3-Omni multi-modal talker model capable of handling text, audio, and vision modalities in a unified architecture.
    The model integrates a text decoder with a code predictor for autoregressive generation of both semantic and acoustic
    tokens, enabling speech and multimodal content generation. This configuration wraps sub-configurations for the text and
    code predictor components, allowing modular setup and initialization.

    e.g. [Qwen/Qwen3-Omni-7B](https://huggingface.co/Qwen/Qwen3-Omni-7B)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        code_predictor_config (`dict`, *optional*):
            A dictionary of configuration parameters used to initialize a [`Qwen3OmniMoeTalkerCodePredictorConfig`].
            If not provided, defaults will be used.
        text_config (`dict`, *optional*):
            A dictionary of configuration parameters used to initialize a [`Qwen3OmniMoeTalkerTextConfig`].
            If not provided, defaults will be used.
        num_code_groups (`int`, *optional*, defaults to 32):
            Number of codebook groups used in the predicted acoustic token sequence, corresponding to multi-codebook VQ representation.
        thinker_hidden_size (`int`, *optional*, defaults to 2048):
            Hidden dimension size of the thinker module used for intermediate reasoning or latent planning before audio generation.
        codec_eos_token_id (`int`, *optional*, defaults to 4198):
            Token ID representing the end-of-speech token in the codec-generated sequence.
        accept_hidden_layer (`int`, *optional*, defaults to 18):
            Index of the hidden layer whose output is used for accepting or refining generated tokens during think-and-speak process.
        codec_nothink_id (`int`, *optional*, defaults to 4203):
            Token ID indicating no thinking step is required during generation.
        codec_think_bos_id (`int`, *optional*, defaults to 4204):
            Token ID marking the beginning of a thinking sequence.
        codec_think_eos_id (`int`, *optional*, defaults to 4205):
            Token ID marking the end of a thinking sequence.
        codec_pad_id (`int`, *optional*, defaults to 4196):
            Padding token ID used in codec input sequences.
        codec_bos_id (`int`, *optional*, defaults to 4197):
            Beginning-of-speech token ID in codec sequences.
        audio_token_id (`int`, *optional*, defaults to 151646):
            Special token ID used to indicate the position of audio tokens in the input sequence.
        image_token_id (`int`, *optional*, defaults to 151655):
            Special token ID used to represent image inputs in the multimodal context.
        video_token_id (`int`, *optional*, defaults to 151656):
            Special token ID used to represent video inputs.
        vision_start_token_id (`int`, *optional*, defaults to 151652):
            Token ID indicating the start of a visual input sequence (e.g., image or video embeddings).
        position_id_per_seconds (`int`, *optional*, defaults to 25):
            Number of position IDs allocated per second of audio content, used for temporal alignment in generation.
        audio_start_token_id (`int`, *optional*, defaults to 151669):
            Token ID that indicates the start of an audio generation segment in the output.
        speaker_id (`dict`, *optional*):
            Speaker name to speaker id dict.

    Example:

    ```python
    >>> from transformers import Qwen3OmniMoeTalkerConfig, Qwen3OmniMoeTalker

    >>> # Initialize a Qwen3OmniMoeTalkerConfig with default sub-configurations
    >>> config = Qwen3OmniMoeTalkerConfig(
    ...     num_code_groups=32,
    ...     thinker_hidden_size=2048,
    ... )

    >>> # Initialize the full Qwen3-Omni Talker model
    >>> model = Qwen3OmniMoeTalker(config)

    >>> # Access the model configuration
    >>> config = model.config
    >>> print(config.text_config)  # Access text decoder configuration
    >>> print(config.code_predictor_config)  # Access code predictor configuration
    ```"""

    sub_configs = {
        "code_predictor_config": Qwen3OmniMoeTalkerCodePredictorConfig,
        "text_config": Qwen3OmniMoeTalkerTextConfig,
    }

    def __init__(
        self,
        code_predictor_config=None,
        text_config=None,
        num_code_groups=32,
        thinker_hidden_size=2048,
        codec_eos_token_id=4198,
        accept_hidden_layer=18,
        codec_nothink_id=4203,
        codec_think_bos_id=4204,
        codec_think_eos_id=4205,
        codec_pad_id=4196,
        codec_bos_id=4197,
        audio_token_id=151646,
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        position_id_per_seconds=25,
        audio_start_token_id=151669,
        speaker_id=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if code_predictor_config is None:
            code_predictor_config = {}
            self.code_predictor_config = Qwen3OmniMoeTalkerCodePredictorConfig()
            logger.info("code_predictor_config is None. Initializing code_predictor_config model with default values")
        elif isinstance(code_predictor_config, Qwen3OmniMoeTalkerCodePredictorConfig):
            self.code_predictor_config = code_predictor_config
        else:
            self.code_predictor_config = Qwen3OmniMoeTalkerCodePredictorConfig(**code_predictor_config)

        if text_config is None:
            text_config = {}
            self.text_config = Qwen3OmniMoeTalkerTextConfig()
            logger.info("talker text_config is None. Initializing talker text model with default values")
        elif isinstance(text_config, Qwen3OmniMoeTalkerTextConfig):
            self.text_config = text_config
        else:
            self.text_config = Qwen3OmniMoeTalkerTextConfig(**text_config)
        self.num_code_groups = num_code_groups
        self.thinker_hidden_size = thinker_hidden_size
        self.codec_eos_token_id = codec_eos_token_id
        self.accept_hidden_layer = accept_hidden_layer
        self.codec_nothink_id = codec_nothink_id
        self.codec_think_bos_id = codec_think_bos_id
        self.codec_think_eos_id = codec_think_eos_id
        self.codec_pad_id = codec_pad_id
        self.codec_bos_id = codec_bos_id
        self.audio_token_id = audio_token_id
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.position_id_per_seconds = position_id_per_seconds
        self.audio_start_token_id = audio_start_token_id
        self.vision_start_token_id = vision_start_token_id
        self.speaker_id = speaker_id


class Qwen3OmniMoeCode2WavConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen3OmniMoeCode2WavConfig`]. It is used to instantiate a
    Qwen3-Omni code-to-waveform decoder, responsible for converting discrete audio codes into high-fidelity waveforms.
    The configuration defines the architecture of the decoder, including parameters for vector quantization, autoregressive modeling,
    and upsampling layers.

    e.g. [Qwen/Qwen3-Omni-7B](https://huggingface.co/Qwen/Qwen3-Omni-7B)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        codebook_size (`int`, *optional*, defaults to 2048):
            Number of entries in each residual codebook used for acoustic token quantization.
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the hidden states and embeddings in the autoregressive transformer decoder.
        max_position_embeddings (`int`, *optional*, defaults to 8000):
            Maximum sequence length that the autoregressive decoder can handle. Determines positional embedding size.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period for rotary position embeddings (RoPE) applied to attention layers.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the decoder.
        num_key_value_heads (`int`, *optional*, defaults to 16):
            Number of key and value attention heads used in grouped-query attention (if applicable).
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the attention projection layers.
        sliding_window (`int`, *optional*, defaults to 72):
            Window size for local attention mechanism, limiting attention context to improve efficiency.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the feed-forward (intermediate) layer in each transformer block.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function used in the feed-forward layers. Supports `"silu"`, `"relu"`, `"gelu"`, etc.
        layer_scale_initial_scale (`float`, *optional*, defaults to 0.01):
            Initial value for LayerScale applied in transformer blocks, helping stabilize training.
        rms_norm_eps (`float`, *optional*, defaults to 1e-5):
            Epsilon value for RMS normalization layers to prevent division by zero.
        num_hidden_layers (`int`, *optional*, defaults to 8):
            Number of transformer blocks in the autoregressive decoder.
        num_quantizers (`int`, *optional*, defaults to 16):
            Number of residual vector quantizers used in the vocoder for fine-grained audio reconstruction.
        upsample_rates (`Tuple[int]`, *optional*, defaults to `(8, 5, 4, 3)`):
            Rate at which features are upsampled in the final waveform synthesis stage.
        upsampling_ratios (`Tuple[int]`, *optional*, defaults to `(2, 2)`):
            Ratios used in transposed convolutional layers to progressively upsample feature maps to waveform.
        decoder_dim (`int`, *optional*, defaults to 1536):
            Final dimensionality of the decoder's output before waveform generation.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability applied to attention weights in the decoder.

    Example:

    ```python
    >>> from transformers import Qwen3OmniMoeCode2WavConfig, Qwen3OmniMoeCode2WavModel

    >>> # Initializing a default Qwen3OmniMoeCode2WavConfig
    >>> config = Qwen3OmniMoeCode2WavConfig()

    >>> # Initializing the Code2Wav model with the configuration
    >>> model = Qwen3OmniMoeCode2WavModel(config)

    >>> # Accessing configuration
    >>> config = model.config
    ```"""

    def __init__(
        self,
        codebook_size=2048,
        hidden_size=1024,
        max_position_embeddings=8000,
        rope_theta=10000,
        num_attention_heads=16,
        num_key_value_heads=16,
        attention_bias=False,
        sliding_window=72,
        intermediate_size=3072,
        hidden_act="silu",
        layer_scale_initial_scale=0.01,
        rms_norm_eps=1e-5,
        num_hidden_layers=8,
        num_quantizers=16,
        upsample_rates=(8, 5, 4, 3),
        upsampling_ratios=(2, 2),
        decoder_dim=1536,
        attention_dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.codebook_size = codebook_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.attention_bias = attention_bias
        self.sliding_window = sliding_window
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.layer_scale_initial_scale = layer_scale_initial_scale
        self.rms_norm_eps = rms_norm_eps
        self.num_hidden_layers = num_hidden_layers
        self.num_quantizers = num_quantizers
        self.upsample_rates = upsample_rates
        self.upsampling_ratios = upsampling_ratios
        self.decoder_dim = decoder_dim
        self.attention_dropout = attention_dropout

    @property
    def layer_types(self):
        """
        All layer in code2wav should be sliding attention
        """
        return ["sliding_attention"] * self.num_hidden_layers


class Qwen3OmniMoeConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`Qwen3OmniMoeForConditionalGeneration`]. It is used to instantiate a Qwen3Omni
    model according to the specified sub-models configurations, defining the model architecture.

    Instantiating a configuration with the defaults will yield a similar configuration to that of the
    [Qwen/Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        thinker_config (`dict`, *optional*): Configuration of the underlying thinker sub-model.
        talker_config (`dict`, *optional*): Configuration of the underlying talker sub-model.
        code2wav_config (`dict`, *optional*): Configuration of the underlying code2wav sub-model.
        enable_audio_output (`bool`, *optional*, defaults to `True`): Whether enable audio output and load talker and code2wav module.

    Example:

    ```python
    >>> from transformers import (
    ...     Qwen3OmniMoeThinkerConfig,
    ...     Qwen3OmniMoeTalkerConfig,
    ...     Qwen3OmniMoeCode2WavConfig,
    ...     Qwen3OmniMoeForConditionalGeneration,
    ...     Qwen3OmniMoeConfig,
    ... )

    >>> # Initializing a Qwen3OmniMoe style configuration
    >>> configuration = Qwen3OmniMoeConfig()

    >>> # Initializing a model from the configuration
    >>> model = Qwen3OmniMoeForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen3_omni_moe"
    sub_configs = {
        "thinker_config": Qwen3OmniMoeThinkerConfig,
        "talker_config": Qwen3OmniMoeTalkerConfig,
        "code2wav_config": Qwen3OmniMoeCode2WavConfig,
    }

    def __init__(
        self,
        thinker_config=None,
        talker_config=None,
        code2wav_config=None,
        enable_audio_output=True,
        im_start_token_id=151644,
        im_end_token_id=151645,
        tts_pad_token_id=151671,
        tts_bos_token_id=151672,
        tts_eos_token_id=151673,
        system_token_id=8948,
        user_token_id=872,
        assistant_token_id=77091,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if thinker_config is None:
            thinker_config = {}
            logger.info("thinker_config is None. Initializing thinker model with default values")

        if talker_config is None:
            talker_config = {}
            logger.info("talker_config is None. Initializing talker model with default values")

        if code2wav_config is None:
            code2wav_config = {}
            logger.info("code2wav_config is None. Initializing code2wav model with default values")

        self.thinker_config = Qwen3OmniMoeThinkerConfig(**thinker_config)
        self.talker_config = Qwen3OmniMoeTalkerConfig(**talker_config)
        self.code2wav_config = Qwen3OmniMoeCode2WavConfig(**code2wav_config)
        self.enable_audio_output = enable_audio_output
        self.im_start_token_id = im_start_token_id
        self.im_end_token_id = im_end_token_id
        self.tts_pad_token_id = tts_pad_token_id
        self.tts_bos_token_id = tts_bos_token_id
        self.tts_eos_token_id = tts_eos_token_id
        self.system_token_id = system_token_id
        self.user_token_id = user_token_id
        self.assistant_token_id = assistant_token_id

    def get_text_config(self, decoder=False) -> "PretrainedConfig":
        """
        Returns the config that is meant to be used with text IO. On most models, it is the original config instance
        itself. On specific composite models, it is under a set of valid names.

        Args:
            decoder (`Optional[bool]`, *optional*, defaults to `False`):
                If set to `True`, then only search for decoder config names.
        """
        # Overridden for deeply nested config like Qwen2-Omni. We don't have any omni model
        # except for Qwen yet. This has to be generalized if more deeply nested configs are
        # added. NOTE: currently method used only by vLLM
        return self.thinker_config.get_text_config()


class Qwen3OmniMoePreTrainedModel(Qwen2_5OmniPreTrainedModel):
    pass


class Qwen3OmniMoePreTrainedModelForConditionalGeneration(Qwen2_5OmniPreTrainedModelForConditionalGeneration):
    def get_llm_pos_ids_for_vision(
        self,
        start_idx: int,
        vision_idx: int,
        spatial_merge_size: int,
        t_index: list[torch.Tensor],
        grid_hs: list[torch.Tensor],
        grid_ws: list[torch.Tensor],
    ):
        llm_pos_ids_list = []
        llm_grid_h = grid_hs[vision_idx] // spatial_merge_size
        llm_grid_w = grid_ws[vision_idx] // spatial_merge_size
        h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(len(t_index), -1, llm_grid_w).flatten().float()
        w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(len(t_index), llm_grid_h, -1).flatten().float()
        t_index = torch.Tensor(t_index).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten().float()
        _llm_pos_ids = torch.stack([t_index, h_index, w_index])
        llm_pos_ids_list.append(_llm_pos_ids + start_idx)
        llm_pos_ids = torch.cat(llm_pos_ids_list, dim=1)
        return llm_pos_ids

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_audio_in_video: bool = False,
        audio_seqlens: Optional[torch.LongTensor] = None,
        second_per_grids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embedding for text part.
            Examples:
                Temporal (Time): 3 patches, representing different segments of the video in time.
                Height: 2 patches, dividing each frame vertically.
                Width: 2 patches, dividing each frame horizontally.
                We also have some important parameters:
                fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
                tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
                temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
                interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [101, 102, 103, 104, 105]
                text height position_ids: [101, 102, 103, 104, 105]
                text width position_ids: [101, 102, 103, 104, 105]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            use_audio_in_video (`bool`, *optional*):
                 If set to `True`, use the audio in video.
            audio_seqlens (`torch.LongTensor` of shape `(num_audios)`, *optional*):
                The length of feature shape of each audio in LLM.
            second_per_grids (`torch.LongTensor` of shape `(num_videos)`, *optional*):
                The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = self.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        audio_token_id = self.config.audio_token_id
        vision_start_token_id = self.config.vision_start_token_id
        audio_start_token_id = self.config.audio_start_token_id
        position_id_per_seconds = self.config.position_id_per_seconds

        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is not None:
                attention_mask = attention_mask == 1
            position_ids = torch.zeros(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=torch.float,
                device=input_ids.device,
            )
            image_idx, video_idx, audio_idx = 0, 0, 0
            for i, input_ids in enumerate(total_input_ids):
                if attention_mask is not None:
                    input_ids = input_ids[attention_mask[i]]
                image_nums, video_nums, audio_nums = 0, 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                audio_nums = torch.sum(input_ids == audio_start_token_id)
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (
                    (vision_tokens == audio_start_token_id).sum()
                    if use_audio_in_video
                    else (vision_tokens == video_token_id).sum()
                )
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos, remain_audios = image_nums, video_nums, audio_nums
                multimodal_nums = (
                    image_nums + audio_nums if use_audio_in_video else image_nums + video_nums + audio_nums
                )
                for _ in range(multimodal_nums):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    if (image_token_id in input_tokens or video_token_id in input_tokens) and (
                        remain_videos > 0 or remain_images > 0
                    ):
                        ed_vision_start = input_tokens.index(vision_start_token_id, st)
                    else:
                        ed_vision_start = len(input_tokens) + 1
                    if audio_token_id in input_tokens and remain_audios > 0:
                        ed_audio_start = input_tokens.index(audio_start_token_id, st)
                    else:
                        ed_audio_start = len(input_tokens) + 1
                    min_ed = min(ed_vision_start, ed_audio_start)

                    text_len = min_ed - st
                    if text_len != 0:
                        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                        st_idx += text_len
                    # Audio in Video
                    if min_ed == ed_vision_start and ed_vision_start + 1 == ed_audio_start:
                        bos_len, eos_len = 2, 2
                    else:
                        bos_len, eos_len = 1, 1
                    llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)
                    st_idx += bos_len
                    # Audio Only
                    if min_ed == ed_audio_start:
                        audio_len = _get_feat_extract_output_lengths(audio_seqlens[audio_idx])
                        llm_pos_ids = torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                        llm_pos_ids_list.append(llm_pos_ids)

                        st += int(text_len + bos_len + audio_len + eos_len)
                        audio_idx += 1
                        remain_audios -= 1

                    # Image Only
                    elif min_ed == ed_vision_start and input_ids[ed_vision_start + 1] == image_token_id:
                        grid_t = image_grid_thw[image_idx][0]
                        grid_hs = image_grid_thw[:, 1]
                        grid_ws = image_grid_thw[:, 2]
                        t_index = (torch.arange(grid_t) * 1 * position_id_per_seconds).float()
                        llm_pos_ids = self.get_llm_pos_ids_for_vision(
                            st_idx, image_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                        )
                        image_len = image_grid_thw[image_idx].prod() // (spatial_merge_size**2)
                        llm_pos_ids_list.append(llm_pos_ids)

                        st += int(text_len + bos_len + image_len + eos_len)
                        image_idx += 1
                        remain_images -= 1

                    # Video Only
                    elif min_ed == ed_vision_start and input_ids[ed_vision_start + 1] == video_token_id:
                        grid_t = video_grid_thw[video_idx][0]
                        grid_hs = video_grid_thw[:, 1]
                        grid_ws = video_grid_thw[:, 2]
                        t_index = (
                            torch.arange(grid_t) * second_per_grids[video_idx].cpu().float() * position_id_per_seconds
                        ).float()
                        llm_pos_ids = self.get_llm_pos_ids_for_vision(
                            st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                        )
                        video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)
                        llm_pos_ids_list.append(llm_pos_ids)

                        st += int(text_len + bos_len + video_len + eos_len)
                        video_idx += 1
                        remain_videos -= 1

                    # Audio in Video
                    elif min_ed == ed_vision_start and ed_vision_start + 1 == ed_audio_start:
                        audio_len = _get_feat_extract_output_lengths(audio_seqlens[audio_idx])
                        audio_llm_pos_ids = torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                        grid_t = video_grid_thw[video_idx][0]
                        grid_hs = video_grid_thw[:, 1]
                        grid_ws = video_grid_thw[:, 2]

                        t_index = (
                            torch.arange(grid_t) * second_per_grids[video_idx].cpu().float() * position_id_per_seconds
                        ).float()
                        video_llm_pos_ids = self.get_llm_pos_ids_for_vision(
                            st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                        )
                        video_data_index, audio_data_index = 0, 0
                        while (
                            video_data_index < video_llm_pos_ids.shape[-1]
                            and audio_data_index < audio_llm_pos_ids.shape[-1]
                        ):
                            if video_llm_pos_ids[0][video_data_index] <= audio_llm_pos_ids[0][audio_data_index]:
                                llm_pos_ids_list.append(video_llm_pos_ids[:, video_data_index : video_data_index + 1])
                                video_data_index += 1
                            else:
                                llm_pos_ids_list.append(audio_llm_pos_ids[:, audio_data_index : audio_data_index + 1])
                                audio_data_index += 1
                        if video_data_index < video_llm_pos_ids.shape[-1]:
                            llm_pos_ids_list.append(
                                video_llm_pos_ids[:, video_data_index : video_llm_pos_ids.shape[-1]]
                            )
                        if audio_data_index < audio_llm_pos_ids.shape[-1]:
                            llm_pos_ids_list.append(
                                audio_llm_pos_ids[:, audio_data_index : audio_llm_pos_ids.shape[-1]]
                            )
                        video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)

                        st += int(text_len + bos_len + audio_len + video_len + eos_len)

                        audio_idx += 1
                        video_idx += 1
                        remain_videos -= 1
                        remain_audios -= 1
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat([item.float() for item in llm_pos_ids_list], dim=1).reshape(3, -1)

                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(input_ids))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)

            return position_ids, mrope_position_deltas
        else:
            position_ids = attention_mask.float().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - torch.sum(attention_mask, dim=-1, keepdim=True)

            return position_ids, mrope_position_deltas


class Qwen3OmniMoeAudioAttention(Qwen2_5OmniAudioAttention):
    def __init__(self, config):
        super().__init__(config)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)


class Qwen3OmniMoeAudioEncoder(Qwen2_5OmniAudioEncoder):
    def __init__(self, config: Qwen3OmniMoeAudioEncoderConfig):
        super().__init__(config)
        del self.proj
        del self.avg_pooler
        del self.audio_bos_eos_token
        del self.conv1
        del self.conv2
        self.conv2d1 = nn.Conv2d(1, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv2d2 = nn.Conv2d(config.downsample_hidden_size, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv2d3 = nn.Conv2d(config.downsample_hidden_size, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv_out = nn.Linear(
            config.downsample_hidden_size * ((((config.num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2),
            config.d_model,
            bias=False,
        )
        self.proj1 = nn.Linear(config.d_model, config.d_model)
        self.act = ACT2FN[config.activation_function]
        self.proj2 = nn.Linear(config.d_model, config.output_dim)
        self.n_window_infer = self.config.n_window_infer
        self.conv_chunksize = self.config.conv_chunksize

    def forward(
        self,
        input_features,
        feature_lens=None,
        aftercnn_lens=None,
    ):
        aftercnn_lens = _get_feat_extract_output_lengths(feature_lens)
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()

        chunk_lengths = torch.tensor(
            [self.n_window * 2] * chunk_num.sum(),
            dtype=torch.long,
            device=feature_lens.device,
        )
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths[chunk_lengths == 0] = self.n_window * 2

        chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
        padded_feature = nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
        feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths)
        padded_mask_after_cnn = nn.utils.rnn.pad_sequence(
            [torch.ones(length, dtype=torch.bool, device=padded_feature.device) for length in feature_lens_after_cnn],
            batch_first=True,
        )
        padded_feature = padded_feature.unsqueeze(1)
        # Split to chunk to avoid OOM during convolution
        padded_embeds = []
        for chunk in padded_feature.split(self.conv_chunksize, dim=0):
            padded_embed = F.gelu(self.conv2d1(chunk))
            padded_embed = F.gelu(self.conv2d2(padded_embed))
            padded_embed = F.gelu(self.conv2d3(padded_embed))
            padded_embeds.append(padded_embed)
        padded_embed = torch.cat(padded_embeds, dim=0)
        b, c, f, t = padded_embed.size()
        padded_embed = self.conv_out(padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))

        positional_embedding = (
            self.positional_embedding.positional_embedding[: padded_embed.shape[1], :]
            .unsqueeze(0)
            .to(padded_embed.dtype)
        )
        padded_embed = padded_embed + positional_embedding
        hidden_states = padded_embed[padded_mask_after_cnn]
        cu_chunk_lens = [0]
        window_aftercnn = padded_mask_after_cnn.shape[-1] * (self.n_window_infer // (self.n_window * 2))
        for cnn_len in aftercnn_lens:
            cu_chunk_lens += [window_aftercnn] * (cnn_len // window_aftercnn)
            remainder = cnn_len % window_aftercnn
            if remainder != 0:
                cu_chunk_lens += [remainder]
        cu_seqlens = torch.tensor(cu_chunk_lens, device=aftercnn_lens.device).cumsum(-1, dtype=torch.int32)

        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(
                hidden_states,
                cu_seqlens,
            )

            hidden_states = layer_outputs[0]

        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj2(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states)


class Qwen3OmniMoeVisionAttention(Qwen3VLMoeVisionAttention):
    def __init__(self, config: Qwen3OmniMoeVisionEncoderConfig):
        super().__init__(config)


class Qwen3OmniMoeVisionPatchMerger(nn.Module):
    def __init__(self, config: Qwen3OmniMoeVisionEncoderConfig, use_postshuffle_norm=False) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.ln_q = nn.LayerNorm(self.hidden_size if use_postshuffle_norm else config.hidden_size, eps=1e-6)
        self.mlp = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, config.out_hidden_size),
            ]
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = self.ln_q(hidden.view(-1, self.hidden_size) if self.use_postshuffle_norm else hidden).view(
            -1, self.hidden_size
        )
        for layer in self.mlp:
            hidden = layer(hidden)
        return hidden


class Qwen3OmniMoeVisionEncoder(Qwen3VLMoeVisionModel):
    config: Qwen3OmniMoeVisionEncoderConfig
    _no_split_modules = ["Qwen3OmniMoeVisionBlock"]

    def __init__(self, config, *inputs, **kwargs):
        self.merger_list = nn.ModuleList(
            [
                Qwen3OmniMoeVisionPatchMerger(
                    config=config,
                    use_postshuffle_norm=True,
                )
                for _ in range(len(config.deepstack_visual_indexes))
            ]
        )
        super().__init__(config, *inputs, **kwargs)
        del self.deepstack_merger_list

    @property
    def deepstack_merger_list(self):
        return self.merger_list


class Qwen3OmniMoeThinkerTextRotaryEmbedding(Qwen3VLMoeTextRotaryEmbedding):
    pass


class Qwen3OmniMoeThinkerTextSparseMoeBlock(Qwen3MoeSparseMoeBlock):
    pass


class Qwen3OmniMoeThinkerTextAttention(Qwen3MoeAttention):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.sliding_window = None


class Qwen3OmniMoeThinkerTextDecoderLayer(Qwen3MoeDecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.self_attn = Qwen3OmniMoeThinkerTextAttention(config, layer_idx)


class Qwen3OmniMoeThinkerTextPreTrainedModel(Qwen3MoePreTrainedModel):
    config_class = Qwen3OmniMoeTextConfig
    config = Qwen3OmniMoeTextConfig


class Qwen3OmniMoeThinkerTextModel(Qwen3VLMoeTextModel):
    config_class = Qwen3OmniMoeTextConfig
    _can_record_outputs = {
        "hidden_states": Qwen3OmniMoeThinkerTextDecoderLayer,
        "attentions": Qwen3OmniMoeThinkerTextAttention,
        "router_logits": OutputRecorder(Qwen3OmniMoeThinkerTextSparseMoeBlock, index=1),
    }

    def __init__(self, config: Qwen3OmniMoeTextConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Qwen3OmniMoeThinkerTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.rotary_emb = Qwen3OmniMoeThinkerTextRotaryEmbedding(config)

    def _deepstack_process(self, hidden_states, visual_pos_masks, visual_embeds):
        visual_pos_masks = visual_pos_masks[..., 0]
        return super()._deepstack_process(hidden_states, visual_pos_masks, visual_embeds)


@dataclass
class Qwen3OmniMoeThinkerCausalLMOutputWithPast(MoeCausalLMOutputWithPast):
    r"""
    Args:
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
    """

    rope_deltas: Optional[torch.LongTensor] = None


class Qwen3OmniMoeThinkerForConditionalGeneration(Qwen2_5OmniThinkerForConditionalGeneration):
    _no_split_modules = [
        "Qwen3OmniMoeAudioEncoderLayer",
        "Qwen3OmniMoeThinkerTextDecoderLayer",
    ]
    _can_record_outputs = {
        "hidden_states": Qwen3OmniMoeThinkerTextDecoderLayer,
        "attentions": Qwen3OmniMoeThinkerTextAttention,
        "router_logits": OutputRecorder(Qwen3OmniMoeThinkerTextSparseMoeBlock, index=1),
    }

    def __init__(self, config):
        super().__init__(config)
        self.num_experts = config.text_config.num_experts
        self.num_experts_per_tok = config.text_config.num_experts_per_tok

    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        feature_attention_mask: Optional[torch.LongTensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
    ):
        """
        Encodes audios into continuous embeddings that can be forwarded to the language model.

        Args:
            input_features (`torch.FloatTensor`):
                The tensors corresponding to the input audios.
            feature_attention_mask (`torch.LongTensor`, *optional*):
                Mask to avoid performing attention on padding feature indices. Mask values selected in `[0, 1]`:
            audio_feature_lengths (`torch.LongTensor` of shape `(num_audios)`, *optional*):
                The length of feature shape of each audio in LLM.
        """
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
        else:
            audio_feature_lengths = None

        feature_lens = audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
        audio_outputs = self.audio_tower(
            input_features,
            feature_lens=feature_lens,
        )
        audio_features = audio_outputs.last_hidden_state

        return audio_features

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids=None,
        input_features=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
        feature_attention_mask=None,
        audio_feature_lengths=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        rope_deltas=None,
        labels=None,
        use_cache=None,
        output_router_logits: Optional[bool] = None,
        use_audio_in_video=None,
        cache_position=None,
        video_second_per_grid=None,
        **kwargs,
    ) -> Union[tuple, Qwen3OmniMoeThinkerCausalLMOutputWithPast]:
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.text_config.output_router_logits
        )

        if inputs_embeds is None:
            # 1. Extract the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

        visual_embeds_multiscale = None
        visual_pos_masks = None
        # 2. Merge text , audios , image and video
        if input_features is not None:
            audio_features = self.get_audio_features(
                input_features,
                feature_attention_mask=feature_attention_mask,
                audio_feature_lengths=audio_feature_lengths,
            )
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            _, _, audio_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

        if pixel_values is not None:
            image_embeds, image_embeds_multiscale = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            visual_pos_masks = image_mask
            visual_embeds_multiscale = image_embeds_multiscale

        if pixel_values_videos is not None:
            video_embeds, video_embeds_multiscale = self.get_video_features(pixel_values_videos, video_grid_thw)

            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if visual_embeds_multiscale is None:
                visual_embeds_multiscale = video_embeds_multiscale
                visual_pos_masks = video_mask
            else:
                visual_pos_masks = video_mask | image_mask
                visual_embeds_multiscale_joint = ()
                image_mask_joint = image_mask[visual_pos_masks]
                video_mask_joint = video_mask[visual_pos_masks]
                for img_embed, vid_embed in zip(visual_embeds_multiscale, video_embeds_multiscale):
                    embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1])
                    embed_joint[image_mask_joint, :] = img_embed
                    embed_joint[video_mask_joint, :] = vid_embed
                    visual_embeds_multiscale_joint = visual_embeds_multiscale_joint + (embed_joint,)
                visual_embeds_multiscale = visual_embeds_multiscale_joint

        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        else:
            audio_feature_lengths = None

        if attention_mask is not None and position_ids is None:
            if (
                cache_position is None
                or (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
            ):
                delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask,
                    use_audio_in_video,
                    audio_feature_lengths,
                    video_second_per_grid,
                )
                rope_deltas = rope_deltas - delta0
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length = input_ids.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            cache_position=cache_position,
            deepstack_visual_embeds=visual_embeds_multiscale,
            visual_pos_masks=visual_pos_masks,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.get_text_config().vocab_size
            )

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

        return Qwen3OmniMoeThinkerCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            aux_loss=aux_loss,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )


class Qwen3OmniMoeTalkerResizeMLP(nn.Module):
    def __init__(self, config: Qwen3OmniMoeTalkerConfig):
        super().__init__()
        self.linear_fc1 = nn.Linear(config.thinker_hidden_size, config.text_config.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(config.text_config.intermediate_size, config.text_config.hidden_size, bias=True)
        self.act_fn = ACT2FN[config.text_config.hidden_act]

    def forward(self, hidden_state):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


@dataclass
class Qwen3OmniMoeTalkerCodePredictorOutputWithPast(CausalLMOutputWithPast):
    r"""
    generation_steps (`int`, *optional*)
        Current generation step of code predictor model.
    """

    generation_steps: Optional[int] = None


class Qwen3OmniMoeTalkerCodePredictorAttention(Qwen3Attention):
    pass


class Qwen3OmniMoeTalkerCodePredictorDecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.self_attn = Qwen3OmniMoeTalkerCodePredictorAttention(config=config, layer_idx=layer_idx)


class Qwen3OmniMoeTalkerCodePredictorModel(Qwen3Model):
    config_class = Qwen3OmniMoeTalkerCodePredictorConfig
    base_model_prefix = "talker.code_predictor.model"
    _can_record_outputs = {
        "attentions": Qwen3OmniMoeTalkerCodePredictorAttention,
        "hidden_states": Qwen3OmniMoeTalkerCodePredictorDecoderLayer,
    }

    def __init__(self, config: Qwen3OmniMoeTalkerCodePredictorConfig):
        super().__init__(config)
        del self.embed_tokens
        self.layers = nn.ModuleList(
            [
                Qwen3OmniMoeTalkerCodePredictorDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.codec_embedding = nn.ModuleList(
            [nn.Embedding(config.vocab_size, config.hidden_size) for _ in range(config.num_code_groups - 1)]
        )

    def get_input_embeddings(self):
        return self.codec_embedding

    @check_model_inputs()
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if input_ids is not None:
            raise ValueError("`input_ids` is expected to be `None`")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration(Qwen3ForCausalLM):
    config_class = Qwen3OmniMoeTalkerCodePredictorConfig
    base_model_prefix = "talker.code_predictor"
    _can_record_outputs = {
        "attentions": Qwen3OmniMoeTalkerCodePredictorAttention,
        "hidden_states": Qwen3OmniMoeTalkerCodePredictorDecoderLayer,
    }

    def __init__(self, config: Qwen3OmniMoeTalkerCodePredictorConfig):
        super().__init__(config)
        self.model = Qwen3OmniMoeTalkerCodePredictorModel._from_config(config)
        self.lm_head = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.vocab_size, bias=False) for _ in range(config.num_code_groups - 1)]
        )

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        cache_position=None,
        generation_steps=None,
        **kwargs,
    ):
        r"""
        Args:
            generation_steps (`int`):
                generation step of code predictor, 0..num_code_groups-1
        """

        # Prefill stage
        if inputs_embeds is not None and inputs_embeds.shape[1] > 1:
            generation_steps = inputs_embeds.shape[1] - 2  # hidden & layer 0
        # Generation stage
        else:
            inputs_embeds = self.model.get_input_embeddings()[generation_steps - 1](input_ids)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head[generation_steps](hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return Qwen3OmniMoeTalkerCodePredictorOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            generation_steps=generation_steps + 1,
        )

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder=False, num_new_tokens=1):
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder, num_new_tokens
        )
        model_kwargs["generation_steps"] = outputs.generation_steps
        return model_kwargs


@dataclass
class Qwen3OmniMoeTalkerOutputWithPast(MoeCausalLMOutputWithPast):
    r"""
    Args:
        generation_step (`int`, *optional*):
            Current generation step, used to track which `trailing_text_hidden` should be used.
    """

    generation_step: Optional[int] = None


class Qwen3OmniMoeTalkerRotaryEmbedding(Qwen3OmniMoeThinkerTextRotaryEmbedding):
    pass


class Qwen3OmniMoeTalkerTextMLP(Qwen3MoeMLP):
    pass


class Qwen3OmniMoeTalkerTextSparseMoeBlock(Qwen2MoeSparseMoeBlock):
    pass


class Qwen3OmniMoeTalkerDecoderLayer(Qwen3MoeDecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.self_attn = Qwen3OmniMoeThinkerTextAttention(config, layer_idx)
        self.mlp = Qwen3OmniMoeTalkerTextSparseMoeBlock(config)


class Qwen3OmniMoeTalkerModel(Qwen3VLMoeTextModel):
    config_class = Qwen3OmniMoeTalkerTextConfig
    base_model_prefix = "talker.model"
    _no_split_modules = ["Qwen3OmniMoeTalkerDecoderLayer"]
    _can_record_outputs = {
        "hidden_states": Qwen3OmniMoeTalkerDecoderLayer,
        "attentions": Qwen3OmniMoeThinkerTextAttention,
        "router_logits": OutputRecorder(Qwen3OmniMoeTalkerTextSparseMoeBlock, index=1),
    }

    def __init__(self, config: Qwen3OmniMoeTalkerTextConfig):
        super().__init__(config)
        del self.embed_tokens
        self.codec_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3OmniMoeTalkerDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.rotary_emb = Qwen3OmniMoeTalkerRotaryEmbedding(config)

    def get_input_embeddings(self):
        return self.codec_embedding


class Qwen3OmniMoeTalkerForConditionalGeneration(Qwen3MoeForCausalLM):
    config_class = Qwen3OmniMoeTalkerConfig
    base_model_prefix = "talker"
    _no_split_modules = ["Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration"]
    _can_record_outputs = {
        "attentions": Qwen3OmniMoeThinkerTextAttention,
        "router_logits": OutputRecorder(Qwen3OmniMoeTalkerTextSparseMoeBlock, index=1),
    }

    def __init__(self, config: Qwen3OmniMoeTalkerConfig):
        super().__init__(config)
        del self.lm_head
        self.model = Qwen3OmniMoeTalkerModel._from_config(config.text_config)
        self.text_projection = Qwen3OmniMoeTalkerResizeMLP(config)
        self.hidden_projection = Qwen3OmniMoeTalkerResizeMLP(config)
        self.codec_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.code_predictor = Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration._from_config(
            config=config.code_predictor_config
        )
        self.rope_deltas = None
        self.spatial_merge_size = self.config.spatial_merge_size
        self.vocab_size = config.text_config.vocab_size
        self.router_aux_loss_coef = config.text_config.router_aux_loss_coef
        self.num_experts = config.text_config.num_experts
        self.num_experts_per_tok = config.text_config.num_experts_per_tok

    # Should inherit from PretrainedModel, but cannot inherit multiple classes in modular
    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_audio_in_video: bool = False,
        audio_seqlens: Optional[torch.LongTensor] = None,
        second_per_grids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return Qwen3OmniMoePreTrainedModelForConditionalGeneration.get_rope_index(
            self,
            input_ids,
            image_grid_thw,
            video_grid_thw,
            attention_mask,
            use_audio_in_video,
            audio_seqlens,
            second_per_grids,
        )

    def get_llm_pos_ids_for_vision(
        self,
        start_idx: int,
        vision_idx: int,
        spatial_merge_size: int,
        t_index: list[torch.Tensor],
        grid_hs: list[torch.Tensor],
        grid_ws: list[torch.Tensor],
    ):
        return Qwen3OmniMoePreTrainedModelForConditionalGeneration.get_llm_pos_ids_for_vision(
            self, start_idx, vision_idx, spatial_merge_size, t_index, grid_hs, grid_ws
        )

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        use_audio_in_video=None,
        audio_feature_lengths=None,
        video_second_per_grid=None,
        image_grid_thw=None,
        video_grid_thw=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_router_logits=None,
        cache_position=None,
        residual_codes=None,
        trailing_text_hidden=None,
        tts_pad_embed=None,
        generation_step=None,
        talker_input_ids=None,
        **kwargs,
    ):
        r"""
        Args:
            use_audio_in_video (`bool`, *optional*):
                If set to `True`, use the audio in video.
            audio_feature_lengths (`torch.LongTensor` of shape `(num_audios)`, *optional*):
                The length of feature shape of each audio in LLM.
            video_second_per_grid (`torch.LongTensor` of shape `(num_videos)`, *optional*):
                Number of seconds per grid for each video, used for temporal feature mapping.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            residual_codes (`torch.Tensor`):
                The predicted residual codes of previous step.
            trailing_text_hidden (`torch.Tensor`):
                Text hidden states from thinker after the first token.
            tts_pad_embed (`torch.Tensor`):
                Embedding tensor of `tts_pad_token_id`.
            generation_step (`int`):
                Generation step since prefill, used to sync with `trailing_text_hidden`.
            talker_input_ids (`torch.Tensor`):
                Input ids from thinker, used to compute 3d RoPE.
        """
        # Prefill
        if inputs_embeds is not None and inputs_embeds.shape[1] > 1:
            generation_step = -1
            residual_codes = None
        if attention_mask is not None:
            if (
                cache_position is None
                or (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
            ):
                delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                position_ids, rope_deltas = self.get_rope_index(
                    talker_input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask,
                    use_audio_in_video,
                    audio_feature_lengths,
                    video_second_per_grid,
                )
                rope_deltas = rope_deltas - delta0
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length = input_ids.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs: MoeModelOutputWithPast = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.codec_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

        return Qwen3OmniMoeTalkerOutputWithPast(
            loss=loss,
            logits=logits,
            aux_loss=aux_loss,
            past_key_values=outputs.past_key_values,
            hidden_states=(
                outputs.hidden_states,
                residual_codes,
            ),  # TODO: hack here to take residual codes out, need refactor.
            generation_step=generation_step + 1,
        )

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder=False, num_new_tokens=1):
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder, num_new_tokens
        )
        model_kwargs["hidden_states"] = outputs.hidden_states
        model_kwargs["generation_step"] = outputs.generation_step
        return model_kwargs

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None, **kwargs
    ):
        hidden_states = kwargs.pop("hidden_states", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values, attention_mask, inputs_embeds, cache_position, **kwargs
        )
        # Decode stage
        # TODO(raushan, gante): Refactor this part to a utility function
        if cache_position[0] != 0:
            input_ids = input_ids[:, -1:]
            generation_step = kwargs.get("generation_step")
            trailing_text_hidden = kwargs.get("trailing_text_hidden")
            tts_pad_embed = kwargs.get("tts_pad_embed")
            last_id_hidden = self.get_input_embeddings()(input_ids)

            past_hidden = hidden_states[0][-1][:, -1:].to(last_id_hidden.device)  # hidden, last layer, last token
            predictor_result = self.code_predictor.generate(
                inputs_embeds=torch.cat((past_hidden, last_id_hidden), dim=1),
                max_new_tokens=self.config.num_code_groups - 1,
                do_sample=True,
                top_k=50,
                top_p=0.8,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            residual_codes = torch.cat((input_ids, predictor_result.sequences.to(input_ids.device)), dim=-1)

            mid_residual_hiddens = [hid[0].to(last_id_hidden.device) for hid in predictor_result.hidden_states[1:]]
            last_residual_hidden = self.code_predictor.get_input_embeddings()[-1](
                predictor_result.sequences[..., -1:]
            ).to(last_id_hidden.device)
            codec_hiddens = torch.cat(
                [last_id_hidden] + mid_residual_hiddens + [last_residual_hidden],
                dim=1,
            )
            inputs_embeds = codec_hiddens.sum(1, keepdim=True)

            if generation_step < trailing_text_hidden.shape[1]:
                inputs_embeds = inputs_embeds + trailing_text_hidden[:, generation_step].unsqueeze(1).to(
                    inputs_embeds.device
                )
            else:
                inputs_embeds = inputs_embeds + tts_pad_embed.to(inputs_embeds.device)
            inputs["inputs_embeds"] = inputs_embeds
            inputs["residual_codes"] = residual_codes
        return inputs


class Qwen3OmniMoeCausalConvNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        stride=1,
        groups=1,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
        )
        self.stride = stride
        self.kernel_size = (kernel_size - 1) * dilation + 1
        self.dilation = dilation
        self.padding = self.kernel_size - self.stride

    def _get_extra_padding_for_conv1d(self, hidden_state: torch.Tensor) -> int:
        length = hidden_state.shape[-1]
        n_frames = (length - self.kernel_size + self.padding) / self.stride + 1
        ideal_length = (math.ceil(n_frames) - 1) * self.stride + (self.kernel_size - self.padding)
        return ideal_length - length

    def forward(self, hidden_state):
        extra_padding = self._get_extra_padding_for_conv1d(hidden_state)
        hidden_state = F.pad(hidden_state, (self.padding, extra_padding), mode="constant", value=0)
        return self.conv(hidden_state).contiguous()


class Qwen3OmniMoeCausalTransConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride)

        pad = kernel_size - stride
        self.left_pad = math.ceil(pad)
        self.right_pad = pad = self.left_pad

    def forward(self, hidden_state):
        hidden_state = self.conv(hidden_state)
        hidden_state = hidden_state[..., self.left_pad : hidden_state.shape[-1] - self.right_pad]
        return hidden_state.contiguous()


class Qwen3OmniMoeConvNeXtBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = Qwen3OmniMoeCausalConvNet(
            dim,
            dim,
            kernel_size=7,
            groups=dim,
            dilation=1,
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, hidden_states):
        input = hidden_states

        hidden_states = self.dwconv(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.norm(hidden_states)
        hidden_states = self.pwconv1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.pwconv2(hidden_states)

        hidden_states = self.gamma * hidden_states

        hidden_states = hidden_states.permute(0, 2, 1)

        hidden_states = input + hidden_states

        return hidden_states


class Qwen3OmniMoeCode2WavRotatoryEmbedding(Qwen3RotaryEmbedding):
    pass


class Qwen3OmniMoeCode2WavAttention(Qwen3Attention):
    def __init__(self, config: Qwen3OmniMoeCode2WavConfig, layer_idx):
        super().__init__(config, layer_idx)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.sliding_window = config.sliding_window


class Qwen3OmniMoeCode2WavMlp(Qwen3MLP):
    pass


class Qwen3OmniMoeCode2WavRMSNorm(Qwen3RMSNorm):
    pass


class Qwen3OmniMoeCode2WavLayerScale(MimiLayerScale):
    pass


class Qwen3OmniMoeCode2WavTransformerLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3OmniMoeCode2WavConfig, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3OmniMoeCode2WavAttention(config, layer_idx)
        self.mlp = Qwen3OmniMoeCode2WavMlp(config)
        self.input_layernorm = Qwen3OmniMoeCode2WavRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3OmniMoeCode2WavRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn_layer_scale = Qwen3OmniMoeCode2WavLayerScale(config)
        self.mlp_layer_scale = Qwen3OmniMoeCode2WavLayerScale(config)
        self.attention_type = "sliding_attention"

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_values (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + self.self_attn_layer_scale(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.mlp_layer_scale(hidden_states)

        return hidden_states


class Qwen3OmniMoeCode2WavTransformerModel(Qwen3Model):
    _can_record_outputs = {
        "hidden_states": Qwen3OmniMoeCode2WavTransformerLayer,
        "attentions": Qwen3OmniMoeCode2WavAttention,
    }

    def __init__(self, config: Qwen3OmniMoeCode2WavConfig):
        super().__init__(config)
        del self.vocab_size
        del self.padding_idx
        del self.embed_tokens
        self.window_size = config.sliding_window
        self.layers = nn.ModuleList(
            [Qwen3OmniMoeCode2WavTransformerLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        cache_position=None,
        **kwargs,
    ):
        if input_ids is not None:
            raise ValueError("input_ids is not expected")
        return super().forward(
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
            inputs_embeds,
            use_cache,
            cache_position,
            **kwargs,
        )


class SnakeBeta(SnakeBeta):
    pass


class Qwen3OmniMoeCode2WavDecoderResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()

        self.act1 = SnakeBeta(dim)
        self.conv1 = Qwen3OmniMoeCausalConvNet(dim, dim, kernel_size=7, dilation=dilation)
        self.act2 = SnakeBeta(dim)
        self.conv2 = Qwen3OmniMoeCausalConvNet(dim, dim, kernel_size=1)

    def forward(self, hidden_state):
        residual = hidden_state

        hidden_state = self.act1(hidden_state)
        hidden_state = self.conv1(hidden_state)
        hidden_state = self.act2(hidden_state)
        hidden_state = self.conv2(hidden_state)
        return hidden_state + residual


class Qwen3OmniMoeCode2WavDecoderBlock(Qwen3OmniMoePreTrainedModel):
    def __init__(self, config: Qwen3OmniMoeCode2WavConfig, layer_idx):
        super().__init__(config)
        in_dim = config.decoder_dim // 2**layer_idx
        out_dim = config.decoder_dim // 2 ** (layer_idx + 1)
        upsample_rate = config.upsample_rates[layer_idx]

        block = [
            SnakeBeta(in_dim),
            Qwen3OmniMoeCausalTransConvNet(in_dim, out_dim, 2 * upsample_rate, upsample_rate),
        ]

        for dilation in (1, 3, 9):
            block.append(Qwen3OmniMoeCode2WavDecoderResidualUnit(out_dim, dilation))

        self.block = nn.ModuleList(block)

    def forward(self, hidden):
        for block in self.block:
            hidden = block(hidden)
        return hidden


class Qwen3OmniMoeCode2Wav(Qwen3OmniMoePreTrainedModel):
    def __init__(self, config: Qwen3OmniMoeCode2WavConfig):
        super().__init__(config)
        self.total_upsample = np.prod(config.upsample_rates + config.upsampling_ratios)
        self.pre_transformer = Qwen3OmniMoeCode2WavTransformerModel._from_config(config)
        self.code_embedding = nn.Embedding(config.codebook_size * config.num_quantizers, config.hidden_size)
        self.register_buffer(
            "code_offset", torch.arange(config.num_quantizers).view(1, -1, 1) * config.codebook_size, persistent=False
        )

        upsample = []
        for factor in config.upsampling_ratios:
            upsample.append(
                nn.ModuleList(
                    [
                        Qwen3OmniMoeCausalTransConvNet(config.hidden_size, config.hidden_size, factor, factor),
                        Qwen3OmniMoeConvNeXtBlock(config.hidden_size),
                    ]
                )
            )
        self.upsample = nn.ModuleList(upsample)

        decoder = [Qwen3OmniMoeCausalConvNet(config.hidden_size, config.decoder_dim, 7)]
        for i in range(len(config.upsample_rates)):
            decoder.append(Qwen3OmniMoeCode2WavDecoderBlock(config, i))
        output_dim = config.decoder_dim // 2 ** len(config.upsample_rates)
        decoder += [
            SnakeBeta(output_dim),
            Qwen3OmniMoeCausalConvNet(output_dim, 1, 7),
        ]
        self.decoder = nn.ModuleList(decoder)

        self.post_init()

    def forward(self, codes):
        if codes.shape[1] != self.config.num_quantizers:
            raise ValueError(f"Expected {self.config.num_quantizers} layer of codes, got {codes.shape[1]}")
        hidden = self.code_embedding(codes + self.code_offset).mean(1)
        hidden = self.pre_transformer(inputs_embeds=hidden).last_hidden_state
        hidden = hidden.permute(0, 2, 1)
        for blocks in self.upsample:
            for block in blocks:
                hidden = block(hidden)
        wav = hidden
        for block in self.decoder:
            wav = block(wav)
        return wav.clamp(min=-1, max=1)

    def chunked_decode(self, codes, chunk_size=300, left_context_size=25):
        wavs = []
        start_index = 0
        while start_index < codes.shape[-1]:
            end_index = min(start_index + chunk_size, codes.shape[-1])
            context_size = left_context_size if start_index - left_context_size > 0 else start_index
            codes_chunk = codes[..., start_index - context_size : end_index]
            wav_chunk = self(codes_chunk)
            wavs.append(wav_chunk[..., context_size * self.total_upsample :])
            start_index = end_index
        return torch.cat(wavs, dim=-1)


class Qwen3OmniMoeForConditionalGeneration(Qwen3OmniMoePreTrainedModel, GenerationMixin):
    config_class = Qwen3OmniMoeConfig

    def __init__(self, config: Qwen3OmniMoeConfig):
        super().__init__(config)

        self.thinker = Qwen3OmniMoeThinkerForConditionalGeneration._from_config(config.thinker_config)
        self.has_talker = config.enable_audio_output
        if self.has_talker:
            self.enable_talker()
        self.post_init()

    def enable_talker(self):
        self.talker = Qwen3OmniMoeTalkerForConditionalGeneration._from_config(self.config.talker_config)
        self.code2wav = Qwen3OmniMoeCode2Wav._from_config(self.config.code2wav_config)

    def disable_talker(self):
        if hasattr(self, "talker"):
            del self.talker
        if hasattr(self, "code2wav"):
            del self.code2wav
        self.has_talker = False

    def _get_talker_user_parts(
        self, im_start_index, segment_end_index, multimodal_mask, thinker_hidden, thinker_embed
    ):
        user_talker_part = torch.empty(
            (1, segment_end_index - im_start_index, self.config.talker_config.text_config.hidden_size),
            device=self.talker.device,
            dtype=self.talker.dtype,
        )

        user_mm_mask = multimodal_mask[:, im_start_index:segment_end_index]

        # Multimodal data exists
        if user_mm_mask.any():
            user_thinker_hidden_mm = thinker_hidden[:, im_start_index:segment_end_index][user_mm_mask]
            mm_hidden = self.talker.hidden_projection(user_thinker_hidden_mm).to(self.talker.device)
            user_talker_part[user_mm_mask] = mm_hidden
        user_thinker_embed = thinker_embed[:, im_start_index:segment_end_index][~user_mm_mask]
        user_text_hidden = self.talker.text_projection(user_thinker_embed).to(self.talker.device)
        user_talker_part[~user_mm_mask] = user_text_hidden
        return user_talker_part

    def _get_talker_assistant_parts(
        self, im_start_index, segment_end_index, speaker_id, thinker_embed, tts_pad_embed, tts_bos_embed, tts_eos_embed
    ):
        assistant_hidden = self.talker.text_projection(thinker_embed[:, im_start_index:segment_end_index]).to(
            self.talker.device
        )  # [1 t d]
        assistant_text_hidden = torch.cat(
            (
                assistant_hidden[:, :3],
                tts_pad_embed.expand(-1, 4, -1),
                tts_bos_embed,
                assistant_hidden[:, 3:4],  # First text
            ),
            dim=1,
        )
        codec_special_tokens = torch.tensor(
            [
                [
                    self.config.talker_config.codec_nothink_id,
                    self.config.talker_config.codec_think_bos_id,
                    self.config.talker_config.codec_think_eos_id,
                    speaker_id,
                    self.config.talker_config.codec_pad_id,
                    self.config.talker_config.codec_bos_id,
                ]
            ],
            device=self.talker.device,
            dtype=torch.long,
        )
        assistant_codec_hidden = torch.cat(
            (
                torch.zeros(
                    (1, 3, self.config.talker_config.text_config.hidden_size),
                    device=self.talker.device,
                    dtype=self.talker.dtype,
                ),
                self.talker.get_input_embeddings()(codec_special_tokens).to(self.talker.device),
            ),
            dim=1,
        )
        trailing_text_hidden = torch.cat(
            (
                assistant_hidden[:, 4:],
                tts_eos_embed,
            ),
            dim=1,
        )

        input_embeds = assistant_text_hidden + assistant_codec_hidden
        input_ids = torch.full(
            (1, assistant_text_hidden.shape[1]),
            fill_value=self.config.tts_pad_token_id,
            dtype=torch.long,
            device=assistant_text_hidden.device,
        )
        return input_embeds, input_ids, trailing_text_hidden

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        speaker: str = "Ethan",
        use_audio_in_video: bool = False,
        return_audio: Optional[bool] = None,
        thinker_max_new_tokens: int = 1024,
        thinker_eos_token_id: int = 151645,
        talker_max_new_tokens: int = 4096,
        talker_do_sample: bool = True,
        talker_top_k: int = 50,
        talker_top_p: float = 1.0,
        talker_temperature: float = 0.9,
        talker_repetition_penalty: float = 1.05,
        **kwargs,
    ):
        if return_audio and not self.has_talker:
            raise ValueError(
                "Cannot use talker when talker module not initialized. Use `enable_talker` method or set enable_talker in config to enable talker."
            )
        if return_audio is None:
            return_audio = self.has_talker

        shared_kwargs = {"use_audio_in_video": use_audio_in_video}
        thinker_kwargs = {
            "max_new_tokens": thinker_max_new_tokens,
            "eos_token_id": thinker_eos_token_id,
        }

        talker_kwargs = {}
        token2wav_kwargs = {}
        if return_audio:
            speaker_id = self.config.talker_config.speaker_id.get(speaker.lower())
            if speaker_id is None:
                raise NotImplementedError(f"Speaker {speaker} not implemented")
            if input_ids.shape[0] != 1:
                raise NotImplementedError("Qwen3-Omni currently does not support batched inference with audio output")
            talker_supppressed_tokens = [
                i
                for i in range(
                    self.config.talker_config.text_config.vocab_size - 1024,
                    self.config.talker_config.text_config.vocab_size,
                )
                if i != self.config.talker_config.codec_eos_token_id
            ]  # Suppress additional special tokens, should not be predicted
            talker_kwargs = {
                "max_new_tokens": talker_max_new_tokens,
                "do_sample": talker_do_sample,
                "top_k": talker_top_k,
                "top_p": talker_top_p,
                "temperature": talker_temperature,
                "eos_token_id": self.config.talker_config.codec_eos_token_id,
                "repetition_penalty": talker_repetition_penalty,
                "suppress_tokens": talker_supppressed_tokens,
                "output_hidden_states": True,
                "return_dict_in_generate": True,
            }
            token2wav_kwargs = {}

        for key, value in kwargs.items():
            if key.startswith("thinker_"):
                thinker_kwargs[key[len("thinker_") :]] = value
            elif key.startswith("talker_"):
                talker_kwargs[key[len("talker_") :]] = value
            elif key.startswith("token2wav_"):
                token2wav_kwargs[key[len("token2wav_") :]] = value
            # Process special input values
            elif key == "feature_attention_mask":
                thinker_kwargs[key] = value
                talker_kwargs["audio_feature_lengths"] = torch.sum(value, dim=1)
            elif key in ("input_features", "attention_mask"):
                thinker_kwargs[key] = value
            # Put other key to shared kwargs
            else:
                shared_kwargs[key] = value

        # Merge kwargs
        for key, value in shared_kwargs.items():
            if key not in thinker_kwargs:
                thinker_kwargs[key] = value
            if key not in talker_kwargs and key in ["image_grid_thw", "video_grid_thw", "video_second_per_grid"]:
                talker_kwargs[key] = value
            if key not in token2wav_kwargs:
                token2wav_kwargs[key] = value

        # 1. Generate from thinker module
        generate_audio = return_audio and self.has_talker
        if generate_audio:
            thinker_kwargs["output_hidden_states"] = True
            thinker_kwargs["return_dict_in_generate"] = True

        thinker_result = self.thinker.generate(input_ids=input_ids, **thinker_kwargs)

        if not generate_audio:
            return thinker_result, None

        # 2. Prepare talker input
        thinker_embed = torch.cat([hidden_states[0] for hidden_states in thinker_result.hidden_states], dim=1).to(
            self.talker.device
        )  # [1 t d]
        thinker_hidden = torch.cat(
            [
                hidden_states[self.config.talker_config.accept_hidden_layer]
                for hidden_states in thinker_result.hidden_states
            ],
            dim=1,
        ).to(self.talker.device)  # [1 t d]
        im_start_indexes = torch.cat(
            (
                torch.nonzero(input_ids[0] == self.config.im_start_token_id).squeeze(),
                torch.tensor([thinker_result.sequences.shape[-1]], device=input_ids.device, dtype=input_ids.dtype),
            ),
            dim=-1,
        ).to(self.talker.device)  # Shape [n_starts + 1]; Take batch 0 since batched inference is not supported here.
        multimodal_mask = (
            (thinker_result.sequences == self.config.thinker_config.audio_token_id) |
            (thinker_result.sequences == self.config.thinker_config.image_token_id) |
            (thinker_result.sequences == self.config.thinker_config.video_token_id)
        ).to(self.talker.device)  # [1 t] # fmt: skip

        talker_special_tokens = torch.tensor(
            [[self.config.tts_bos_token_id, self.config.tts_eos_token_id, self.config.tts_pad_token_id]],
            device=self.thinker.device,
            dtype=input_ids.dtype,
        )
        tts_bos_embed, tts_eos_embed, tts_pad_embed = (
            self.talker.text_projection(self.thinker.get_input_embeddings()(talker_special_tokens))
            .to(self.talker.device)
            .chunk(3, dim=1)
        )  # 3 * [1 1 d]

        talker_input_embeds = []  # [1 t d]
        talker_input_ids = []
        # For every chatml parts
        for i in range(len(im_start_indexes) - 1):
            im_start_index = im_start_indexes[i]
            segment_end_index = im_start_indexes[i + 1]
            role_token = input_ids[0][im_start_index + 1]
            # Talker should ignore thinker system prompt
            if role_token == self.config.system_token_id:
                continue
            # Talker takes word embeddings for tokens and hidden state from `accept_hidden_layer` for multimodal inputs
            elif role_token == self.config.user_token_id:
                talker_user_part = self._get_talker_user_parts(
                    im_start_index, segment_end_index, multimodal_mask, thinker_hidden, thinker_embed
                )
                talker_input_embeds.append(talker_user_part)
                talker_input_ids.append(thinker_result.sequences[:, im_start_index:segment_end_index])
            # Take assistant output (for now)
            elif role_token == self.config.assistant_token_id and i == len(im_start_indexes) - 2:
                talker_assistant_embeds, talker_assistant_ids, trailing_text_hidden = self._get_talker_assistant_parts(
                    im_start_index,
                    segment_end_index,
                    speaker_id,
                    thinker_embed,
                    tts_pad_embed,
                    tts_bos_embed,
                    tts_eos_embed,
                )
                talker_input_embeds.append(talker_assistant_embeds)
                talker_input_ids.append(talker_assistant_ids)
            # History assistant output (ignore for now)
            elif role_token == self.config.assistant_token_id and i != len(im_start_indexes) - 2:
                continue
            else:
                raise AssertionError("Expect role id after <|im_start|> (assistant, user, system)")
        talker_input_embed = torch.cat([embed.to(self.talker.device) for embed in talker_input_embeds], dim=1)
        talker_input_id = torch.cat([embed.to(self.talker.device) for embed in talker_input_ids], dim=1)
        talker_result = self.talker.generate(
            inputs_embeds=talker_input_embed,
            trailing_text_hidden=trailing_text_hidden,
            tts_pad_embed=tts_pad_embed,
            talker_input_ids=talker_input_id,  # Not use input_ids to prevent repetation penalty out of bound
            **talker_kwargs,
        )
        talker_codes = (
            torch.stack([hid[-1] for hid in talker_result.hidden_states if hid[-1] is not None], dim=1)
            .transpose(1, 2)
            .to(self.code2wav.device)
        )
        talker_wavs = self.code2wav.chunked_decode(talker_codes, chunk_size=300, left_context_size=25)

        return thinker_result, talker_wavs.float()


class Qwen3OmniMoeProcessorKwargs(Qwen2_5OmniProcessorKwargs):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "padding_side": "left",
        },
        "videos_kwargs": {
            "seconds_per_chunk": 2.0,
            "position_id_per_seconds": 13.0,
            "use_audio_in_video": False,
            "size": {
                "shortest_edge": 128 * 32 * 32,
                "longest_edge": 768 * 32 * 32,
            },
        },
        "audio_kwargs": {
            "sampling_rate": 16000,
            "padding": True,
            "return_attention_mask": True,
        },
    }


class Qwen3OmniMoeProcessor(Qwen2_5OmniProcessor, ProcessorMixin):
    def replace_multimodal_special_tokens(
        self,
        text,
        audio_lengths,
        image_grid_thw,
        video_grid_thw,
        video_second_per_grid,
        use_audio_in_video,
        position_id_per_seconds,
        seconds_per_chunk,
    ):
        # Extend mm token length
        merge_length_image = self.image_processor.merge_size**2
        merge_length_video = self.video_processor.merge_size**2

        processed_text = []
        for sample in text:
            positions = []
            special_tokens = [re.escape(tok) for tok in [self.audio_token, self.image_token, self.video_token]]
            pattern = "|".join(special_tokens)
            positions = sorted([(match.start(), match.group()) for match in re.finditer(pattern, sample)])
            positions.sort(key=lambda x: x[0])

            for _, special_token in positions:
                if special_token == self.audio_token:
                    sample = sample.replace(self.audio_token, "<|audio_placeholder|>" * next(audio_lengths), 1)
                elif special_token == self.image_token:
                    image_seq_length = next(image_grid_thw).prod() // merge_length_image
                    sample = sample.replace(self.image_token, "<|image_placeholder|>" * image_seq_length, 1)
                elif special_token == self.video_token:
                    if not use_audio_in_video:
                        video_seq_length = next(video_grid_thw).prod() // merge_length_video
                        sample = sample.replace(self.video_token, "<|video_placeholder|>" * video_seq_length, 1)
                    else:
                        audio_token_indices = np.arange(next(audio_lengths))
                        curr_video_grid_thw = next(video_grid_thw)
                        height = curr_video_grid_thw[1] // self.video_processor.merge_size
                        width = curr_video_grid_thw[2] // self.video_processor.merge_size
                        video_token_indices = np.arange(curr_video_grid_thw[0]).reshape(-1, 1, 1)
                        video_token_indices = np.broadcast_to(
                            video_token_indices, (video_token_indices.shape[0], height, width)
                        ).reshape(-1)
                        video_token_indices = (
                            video_token_indices * next(video_second_per_grid) * position_id_per_seconds
                        )

                        video_data_index, audio_data_index = 0, 0
                        placeholder_string = self.vision_bos_token + self.audio_bos_token
                        while video_data_index < len(video_token_indices) and audio_data_index < len(
                            audio_token_indices
                        ):
                            if video_token_indices[video_data_index] <= audio_token_indices[audio_data_index]:
                                placeholder_string += "<|video_placeholder|>"
                                video_data_index += 1
                            else:
                                placeholder_string += "<|audio_placeholder|>"
                                audio_data_index += 1
                        if video_data_index < len(video_token_indices):
                            placeholder_string += "<|video_placeholder|>" * (
                                len(video_token_indices) - video_data_index
                            )
                        if audio_data_index < len(audio_token_indices):
                            placeholder_string += "<|audio_placeholder|>" * (
                                len(audio_token_indices) - audio_data_index
                            )
                        placeholder_string += self.audio_eos_token + self.vision_eos_token
                        sample = sample.replace(
                            self.vision_bos_token + self.video_token + self.vision_eos_token,
                            placeholder_string,
                            1,
                        )

            sample = sample.replace("<|audio_placeholder|>", self.audio_token)
            sample = sample.replace("<|image_placeholder|>", self.image_token)
            sample = sample.replace("<|video_placeholder|>", self.video_token)
            processed_text.append(sample)
        return processed_text

    def __call__(
        self,
        text: TextInput = None,
        images: ImageInput = None,
        videos: VideoInput = None,
        audio: AudioInput = None,
        **kwargs,
    ):
        """
        Main method to prepare for the model one or several sequences(s) and audio(s). This method forwards the `text`
        and `kwargs` arguments to Qwen2TokenizerFast's [`~Qwen2TokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the audio(s), this method forwards the `audio` and `kwargs` arguments to
        WhisperFeatureExtractor's [`~WhisperFeatureExtractor.__call__`] if `audio` is not `None`. To prepare the vision inputs,
        this method forwards the `vision_infos` and `kwargs` arguments to Qwen2VLImageProcessor's [`~Qwen2VLImageProcessor.__call__`]
        if `vision_infos` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            videos (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of videos to be prepared. Each video can be a 4D NumPy array or PyTorch
                tensor, or a nested list of 3D frames. Both channels-first and channels-last formats are supported.
            audio (`np.ndarray`, `List[np.ndarray]`):
                The audio or batch of audio to be prepared. Each audio can be a NumPy array.
        """

        if text is None:
            raise ValueError("You need to specify either a `text` input to process.")

        output_kwargs = self._merge_kwargs(
            Qwen3OmniMoeProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        seconds_per_chunk = output_kwargs["videos_kwargs"].pop("seconds_per_chunk")
        position_id_per_seconds = output_kwargs["videos_kwargs"].pop("position_id_per_seconds")
        use_audio_in_video = output_kwargs["videos_kwargs"].pop("use_audio_in_video")
        fps = output_kwargs["videos_kwargs"].get("fps", 1.0)

        if audio is not None:
            output_kwargs["audio_kwargs"]["padding"] = True  # Setting to True to avoid default truncation
            audio_inputs = self.feature_extractor(audio, **output_kwargs["audio_kwargs"])
            audio_inputs["feature_attention_mask"] = audio_inputs.pop(
                "attention_mask"
            )  # rename feature_attention_mask to prevent conflicts later on
            audio_inputs["input_features"] = audio_inputs.pop(
                "input_features"
            )  # rename input_features to prevent conflicts later on
            audio_lengths = iter(_get_feat_extract_output_lengths(audio_inputs["feature_attention_mask"].sum(-1)))
        else:
            audio_inputs = {}
            audio_lengths = iter([])

        if images is not None:
            images_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = iter(images_inputs["image_grid_thw"])
        else:
            images_inputs = {}
            image_grid_thw = iter([])

        if videos is not None:
            videos = make_batched_videos(videos)
            videos_inputs = self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])
            fps = [fps] * len(videos)
            videos_inputs["video_second_per_grid"] = [
                self.video_processor.temporal_patch_size / fps[i] for i in range(len(fps))
            ]
            video_grid_thw = iter(videos_inputs["video_grid_thw"])
            video_second_per_grid = iter(videos_inputs["video_second_per_grid"])
        else:
            videos_inputs = {}
            video_grid_thw = iter([])
            video_second_per_grid = iter([])

        if not isinstance(text, list):
            text = [text]

        text = self.replace_multimodal_special_tokens(
            text,
            audio_lengths,
            image_grid_thw,
            video_grid_thw,
            video_second_per_grid=video_second_per_grid,
            use_audio_in_video=use_audio_in_video,
            position_id_per_seconds=position_id_per_seconds,
            seconds_per_chunk=seconds_per_chunk,
        )

        texts_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        return BatchFeature(
            data={**texts_inputs, **images_inputs, **videos_inputs, **audio_inputs},
            tensor_type=kwargs.get("return_tensors"),
        )

    def apply_chat_template(self, conversations, chat_template=None, **kwargs):
        return ProcessorMixin.apply_chat_template(self, conversations, chat_template, **kwargs)


__all__ = [
    "Qwen3OmniMoeConfig",
    "Qwen3OmniMoeThinkerConfig",
    "Qwen3OmniMoeTalkerConfig",
    "Qwen3OmniMoeForConditionalGeneration",
    "Qwen3OmniMoeThinkerTextModel",
    "Qwen3OmniMoeThinkerForConditionalGeneration",
    "Qwen3OmniMoeTalkerForConditionalGeneration",
    "Qwen3OmniMoePreTrainedModel",
    "Qwen3OmniMoePreTrainedModelForConditionalGeneration",
    "Qwen3OmniMoeTalkerModel",
    "Qwen3OmniMoeThinkerTextPreTrainedModel",
    "Qwen3OmniMoeProcessor",
    "Qwen3OmniMoeCode2Wav",
    "Qwen3OmniMoeCode2WavDecoderBlock",
    "Qwen3OmniMoeCode2WavTransformerModel",
    "Qwen3OmniMoeTalkerCodePredictorModel",
    "Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration",
]
