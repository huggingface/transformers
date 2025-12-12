# coding=utf-8
# Copyright 2025 OpenMOSS and HuggingFace Inc. teams. All rights reserved.
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
"""XY-Tokenizer model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class XYTokenizerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`XYTokenizer`]. It is used to instantiate a
    XY-Tokenizer model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the XY-Tokenizer
    [fnlp/XY_Tokenizer_TTSD_V0_hf](https://huggingface.co/fnlp/XY_Tokenizer_TTSD_V0_hf) architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        input_sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate of the input audio. Alias: `input_sample_rate` (deprecated).
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate of the output audio. Alias: `output_sample_rate` (deprecated).
        encoder_downsample_rate (`int`, *optional*, defaults to 1280):
            The total downsampling factor of the encoder part.
        decoder_upsample_rate (`int`, *optional*, defaults to 1920):
            The total upsampling factor of the decoder part.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation for weight initialization.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).

        Semantic Encoder parameters:
        num_mel_bins (`int`, *optional*, defaults to 128):
            Number of mel-spectrogram bins for audio feature extraction.
        hop_length (`int`, *optional*, defaults to 160):
            Hop length for STFT operations.
        semantic_encoder_stride_size (`int`, *optional*, defaults to 2):
            Stride size for the semantic encoder convolution layers.
        semantic_encoder_kernel_size (`int`, *optional*, defaults to 3):
            Kernel size for the semantic encoder convolution layers.
        semantic_encoder_d_model (`int`, *optional*, defaults to 1280):
            Hidden size for the semantic encoder.
        semantic_encoder_scale_embedding (`bool`, *optional*, defaults to `True`):
            Whether to scale embeddings in the semantic encoder.
        max_audio_seconds (`int`, *optional*, defaults to 30):
            Maximum audio duration in seconds.
        semantic_encoder_layers (`int`, *optional*, defaults to 32):
            Number of transformer layers in the semantic encoder.
        semantic_encoder_attention_heads (`int`, *optional*, defaults to 20):
            Number of attention heads in the semantic encoder.
        semantic_encoder_ffn_dim (`int`, *optional*, defaults to 5120):
            FFN dimension in the semantic encoder.
        activation_function (`str`, *optional*, defaults to `"gelu"`):
            Activation function used throughout the model.
        semantic_encoder_attn_type (`str`, *optional*, defaults to `"varlen"`):
            Attention type for the semantic encoder.

        Semantic Encoder Adapter parameters:
        semantic_adapter_input_dim (`int`, *optional*):
            Input dimension for the semantic encoder adapter. If None, uses semantic_encoder_d_model.
        semantic_adapter_d_model (`int`, *optional*):
            Hidden size for the semantic encoder adapter. If None, uses semantic_encoder_d_model.
        semantic_adapter_output_dim (`int`, *optional*):
            Output dimension for the semantic encoder adapter. If None, uses semantic_encoder_d_model.
        semantic_adapter_max_source_positions (`int`, *optional*, defaults to 1500):
            Maximum source positions for the semantic adapter.
        semantic_adapter_layers (`int`, *optional*, defaults to 32):
            Number of transformer layers in the semantic encoder adapter.
        semantic_adapter_attention_heads (`int`, *optional*, defaults to 20):
            Number of attention heads in the semantic encoder adapter.
        semantic_adapter_ffn_dim (`int`, *optional*, defaults to 5120):
            FFN dimension in the semantic encoder adapter.
        semantic_adapter_attn_type (`str`, *optional*, defaults to `"varlen"`):
            Attention type for the semantic encoder adapter.

        Acoustic Encoder parameters:
        acoustic_encoder_d_model (`int`, *optional*, defaults to 1280):
            Hidden size for the acoustic encoder.
        acoustic_encoder_stride_size (`int`, *optional*, defaults to 2):
            Stride size for the acoustic encoder convolution layers.
        acoustic_encoder_kernel_size (`int`, *optional*, defaults to 3):
            Kernel size for the acoustic encoder convolution layers.
        acoustic_encoder_scale_embedding (`bool`, *optional*, defaults to `True`):
            Whether to scale embeddings in the acoustic encoder.
        acoustic_encoder_layers (`int`, *optional*, defaults to 32):
            Number of transformer layers in the acoustic encoder.
        acoustic_encoder_attention_heads (`int`, *optional*, defaults to 20):
            Number of attention heads in the acoustic encoder.
        acoustic_encoder_ffn_dim (`int`, *optional*, defaults to 5120):
            FFN dimension in the acoustic encoder.
        acoustic_encoder_attn_type (`str`, *optional*, defaults to `"varlen"`):
            Attention type for the acoustic encoder.

        Pre-RVQ Adapter parameters:
        pre_rvq_adapter_input_dim (`int`, *optional*):
            Input dimension for the pre-RVQ adapter. If None, computed as semantic_encoder_d_model + acoustic_encoder_d_model.
        pre_rvq_adapter_d_model (`int`, *optional*):
            Hidden size for the pre-RVQ adapter. If None, uses code_dim.
        pre_rvq_adapter_output_dim (`int`, *optional*):
            Output dimension for the pre-RVQ adapter. If None, uses code_dim.
        pre_rvq_adapter_max_source_positions (`int`, *optional*, defaults to 1500):
            Maximum source positions for the pre-RVQ adapter.
        pre_rvq_adapter_layers (`int`, *optional*, defaults to 32):
            Number of transformer layers in the pre-RVQ adapter.
        pre_rvq_adapter_attention_heads (`int`, *optional*, defaults to 20):
            Number of attention heads in the pre-RVQ adapter.
        pre_rvq_adapter_ffn_dim (`int`, *optional*, defaults to 5120):
            FFN dimension in the pre-RVQ adapter.
        pre_rvq_adapter_attn_type (`str`, *optional*, defaults to `"varlen"`):
            Attention type for the pre-RVQ adapter.

        Downsample parameters:
        downsample_d_model (`int`, *optional*):
            Hidden size for downsampling. If None, uses code_dim.
        downsample_avg_pooler (`int`, *optional*, defaults to 4):
            Average pooling factor for downsampling.

        Quantizer parameters:
        num_quantizers (`int`, *optional*, defaults to 32):
            Number of quantizers in the residual VQ.
        quantizer_input_dim (`int`, *optional*):
            Input dimension for the quantizer. If None, computed as code_dim * downsample_avg_pooler.
        quantizer_rvq_dim (`int`, *optional*):
            RVQ dimension for the quantizer. If None, uses code_dim.
        quantizer_output_dim (`int`, *optional*):
            Output dimension for the quantizer. If None, uses code_dim.
        codebook_size (`int`, *optional*, defaults to 1024):
            Number of codes in each codebook.
        codebook_dim (`int`, *optional*, defaults to 8):
            Dimension of the codebook vectors.
        quantizer_dropout (`float`, *optional*, defaults to 0.5):
            Dropout rate for quantizer during training.
        skip_rvq_ratio (`float`, *optional*, defaults to 0.0):
            Ratio of samples to skip RVQ during training.
        vq_commitment (`float`, *optional*, defaults to 1.0):
            Commitment loss weight for vector quantization.
        vq_decay (`float`, *optional*, defaults to 0.99):
            EMA decay rate for codebook updates.
        vq_epsilon (`float`, *optional*, defaults to 1e-5):
            Epsilon value for numerical stability in VQ.
        vq_threshold_ema_dead (`int`, *optional*, defaults to 2):
            Threshold for dead code replacement in VQ.
        vq_kmeans_init (`bool`, *optional*, defaults to `True`):
            Whether to use K-means initialization for codebooks.
        vq_kmeans_iters (`int`, *optional*, defaults to 10):
            Number of K-means iterations for initialization.

        Post-RVQ Adapter parameters:
        post_rvq_adapter_input_dim (`int`, *optional*):
            Input dimension for the post-RVQ adapter. If None, uses code_dim.
        post_rvq_adapter_d_model (`int`, *optional*):
            Hidden size for the post-RVQ adapter. If None, uses code_dim.
        post_rvq_adapter_output_dim (`int`, *optional*):
            Output dimension for the post-RVQ adapter. If None, computed as code_dim * upsample_stride.
        post_rvq_adapter_max_source_positions (`int`, *optional*, defaults to 1500):
            Maximum source positions for the post-RVQ adapter.
        post_rvq_adapter_layers (`int`, *optional*, defaults to 32):
            Number of transformer layers in the post-RVQ adapter.
        post_rvq_adapter_attention_heads (`int`, *optional*, defaults to 20):
            Number of attention heads in the post-RVQ adapter.
        post_rvq_adapter_ffn_dim (`int`, *optional*, defaults to 5120):
            FFN dimension in the post-RVQ adapter.
        post_rvq_adapter_attn_type (`str`, *optional*, defaults to `"varlen"`):
            Attention type for the post-RVQ adapter.

        Upsample parameters:
        upsample_d_model (`int`, *optional*):
            Hidden size for upsampling. If None, uses code_dim.
        upsample_stride (`int`, *optional*, defaults to 4):
            Upsampling stride factor.

        Acoustic Decoder parameters:
        acoustic_decoder_d_model (`int`, *optional*):
            Hidden size for the acoustic decoder. If None, uses code_dim.
        acoustic_decoder_stride_size (`int`, *optional*, defaults to 2):
            Stride size for the acoustic decoder convolution layers.
        acoustic_decoder_kernel_size (`int`, *optional*, defaults to 3):
            Kernel size for the acoustic decoder convolution layers.
        acoustic_decoder_scale_embedding (`bool`, *optional*, defaults to `True`):
            Whether to scale embeddings in the acoustic decoder.
        acoustic_decoder_layers (`int`, *optional*, defaults to 32):
            Number of transformer layers in the acoustic decoder.
        acoustic_decoder_attention_heads (`int`, *optional*, defaults to 20):
            Number of attention heads in the acoustic decoder.
        acoustic_decoder_ffn_dim (`int`, *optional*, defaults to 5120):
            FFN dimension in the acoustic decoder.
        acoustic_decoder_attn_type (`str`, *optional*, defaults to `"varlen"`):
            Attention type for the acoustic decoder.

        Vocos parameters:
        vocos_input_channels (`int`, *optional*):
            Number of input channels for Vocos. If None, uses num_mel_bins.
        vocos_dim (`int`, *optional*, defaults to 512):
            Hidden dimension for Vocos backbone.
        vocos_intermediate_dim (`int`, *optional*, defaults to 4096):
            Intermediate dimension for Vocos ConvNeXt blocks.
        vocos_num_layers (`int`, *optional*, defaults to 30):
            Number of ConvNeXt layers in Vocos.
        vocos_n_fft (`int`, *optional*, defaults to 640):
            FFT size for Vocos ISTFT.
        vocos_hop_size (`int`, *optional*, defaults to 160):
            Hop size for Vocos ISTFT.
        vocos_padding (`str`, *optional*, defaults to `"same"`):
            Padding mode for Vocos ISTFT.

        Feature Extractor parameters:
        feature_extractor_feature_size (`int`, *optional*, defaults to 80):
            Feature size for the feature extractor.
        feature_extractor_n_fft (`int`, *optional*, defaults to 400):
            FFT size for the feature extractor.
        feature_extractor_chunk_length (`int`, *optional*, defaults to 30):
            Chunk length in seconds for processing long audio.

        Code dimension:
        code_dim (`int`, *optional*, defaults to 1280):
            The base code dimension used across various modules.
    """

    model_type = "xy_tokenizer"

    def __init__(
        self,
        # Basic parameters
        input_sampling_rate: int = 16000,
        sampling_rate: int = 16000,
        encoder_downsample_rate: int = 1280,
        decoder_upsample_rate: int = 1920,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        # Shared parameters
        num_mel_bins: int = 128,
        hop_length: int = 160,
        max_audio_seconds: int = 30,
        activation_function: str = "gelu",
        code_dim: int = 1280,
        # Semantic Encoder
        semantic_encoder_stride_size: int = 2,
        semantic_encoder_kernel_size: int = 3,
        semantic_encoder_d_model: int = 1280,
        semantic_encoder_scale_embedding: bool = True,
        semantic_encoder_layers: int = 32,
        semantic_encoder_attention_heads: int = 20,
        semantic_encoder_ffn_dim: int = 5120,
        semantic_encoder_attn_type: str = "varlen",
        # Semantic Encoder Adapter
        semantic_adapter_input_dim: int = None,
        semantic_adapter_d_model: int = None,
        semantic_adapter_output_dim: int = None,
        semantic_adapter_max_source_positions: int = 1500,
        semantic_adapter_layers: int = 32,
        semantic_adapter_attention_heads: int = 20,
        semantic_adapter_ffn_dim: int = 5120,
        semantic_adapter_attn_type: str = "varlen",
        # Acoustic Encoder
        acoustic_encoder_d_model: int = 1280,
        acoustic_encoder_stride_size: int = 2,
        acoustic_encoder_kernel_size: int = 3,
        acoustic_encoder_scale_embedding: bool = True,
        acoustic_encoder_layers: int = 32,
        acoustic_encoder_attention_heads: int = 20,
        acoustic_encoder_ffn_dim: int = 5120,
        acoustic_encoder_attn_type: str = "varlen",
        # Pre-RVQ Adapter
        pre_rvq_adapter_input_dim: int = None,
        pre_rvq_adapter_d_model: int = None,
        pre_rvq_adapter_output_dim: int = None,
        pre_rvq_adapter_max_source_positions: int = 1500,
        pre_rvq_adapter_layers: int = 32,
        pre_rvq_adapter_attention_heads: int = 20,
        pre_rvq_adapter_ffn_dim: int = 5120,
        pre_rvq_adapter_attn_type: str = "varlen",
        # Downsample
        downsample_d_model: int = None,
        downsample_avg_pooler: int = 4,
        # Quantizer
        num_quantizers: int = 32,
        quantizer_input_dim: int = None,
        quantizer_rvq_dim: int = None,
        quantizer_output_dim: int = None,
        codebook_size: int = 1024,
        codebook_dim: int = 8,
        quantizer_dropout: float = 0.5,
        skip_rvq_ratio: float = 0.0,
        vq_commitment: float = 1.0,
        vq_decay: float = 0.99,
        vq_epsilon: float = 1e-5,
        vq_threshold_ema_dead: int = 2,
        vq_kmeans_init: bool = True,
        vq_kmeans_iters: int = 10,
        # Post-RVQ Adapter
        post_rvq_adapter_input_dim: int = None,
        post_rvq_adapter_d_model: int = None,
        post_rvq_adapter_output_dim: int = None,
        post_rvq_adapter_max_source_positions: int = 1500,
        post_rvq_adapter_layers: int = 32,
        post_rvq_adapter_attention_heads: int = 20,
        post_rvq_adapter_ffn_dim: int = 5120,
        post_rvq_adapter_attn_type: str = "varlen",
        # Upsample
        upsample_d_model: int = None,
        upsample_stride: int = 4,
        # Acoustic Decoder
        acoustic_decoder_d_model: int = None,
        acoustic_decoder_stride_size: int = 2,
        acoustic_decoder_kernel_size: int = 3,
        acoustic_decoder_scale_embedding: bool = True,
        acoustic_decoder_layers: int = 32,
        acoustic_decoder_attention_heads: int = 20,
        acoustic_decoder_ffn_dim: int = 5120,
        acoustic_decoder_attn_type: str = "varlen",
        # Vocos
        vocos_input_channels: int = None,
        vocos_dim: int = 512,
        vocos_intermediate_dim: int = 4096,
        vocos_num_layers: int = 30,
        vocos_n_fft: int = 640,
        vocos_hop_size: int = 160,
        vocos_padding: str = "same",
        # Feature Extractor
        feature_extractor_feature_size: int = 80,
        feature_extractor_n_fft: int = 400,
        feature_extractor_chunk_length: int = 30,
        **kwargs,
    ):
        # Backward-compatible alias handling
        if "input_sample_rate" in kwargs and input_sampling_rate == 16000:
            input_sampling_rate = kwargs.pop("input_sample_rate")
        if "output_sample_rate" in kwargs and sampling_rate == 16000:
            sampling_rate = kwargs.pop("output_sample_rate")

        # Basic parameters
        self.input_sampling_rate = input_sampling_rate
        self.sampling_rate = sampling_rate
        self.input_sample_rate = input_sampling_rate  # Deprecated alias
        self.output_sample_rate = sampling_rate  # Deprecated alias
        self.encoder_downsample_rate = encoder_downsample_rate
        self.decoder_upsample_rate = decoder_upsample_rate
        self.initializer_range = initializer_range
        self.use_cache = use_cache

        # Shared parameters
        self.num_mel_bins = num_mel_bins
        self.hop_length = hop_length
        self.max_audio_seconds = max_audio_seconds
        self.activation_function = activation_function
        self.code_dim = code_dim

        # Semantic Encoder
        self.semantic_encoder_stride_size = semantic_encoder_stride_size
        self.semantic_encoder_kernel_size = semantic_encoder_kernel_size
        self.semantic_encoder_d_model = semantic_encoder_d_model
        self.semantic_encoder_scale_embedding = semantic_encoder_scale_embedding
        self.semantic_encoder_layers = semantic_encoder_layers
        self.semantic_encoder_attention_heads = semantic_encoder_attention_heads
        self.semantic_encoder_ffn_dim = semantic_encoder_ffn_dim
        self.semantic_encoder_attn_type = semantic_encoder_attn_type

        # Semantic Encoder Adapter
        self.semantic_adapter_input_dim = semantic_adapter_input_dim or semantic_encoder_d_model
        self.semantic_adapter_d_model = semantic_adapter_d_model or semantic_encoder_d_model
        self.semantic_adapter_output_dim = semantic_adapter_output_dim or semantic_encoder_d_model
        self.semantic_adapter_max_source_positions = semantic_adapter_max_source_positions
        self.semantic_adapter_layers = semantic_adapter_layers
        self.semantic_adapter_attention_heads = semantic_adapter_attention_heads
        self.semantic_adapter_ffn_dim = semantic_adapter_ffn_dim
        self.semantic_adapter_attn_type = semantic_adapter_attn_type

        # Acoustic Encoder
        self.acoustic_encoder_d_model = acoustic_encoder_d_model
        self.acoustic_encoder_stride_size = acoustic_encoder_stride_size
        self.acoustic_encoder_kernel_size = acoustic_encoder_kernel_size
        self.acoustic_encoder_scale_embedding = acoustic_encoder_scale_embedding
        self.acoustic_encoder_layers = acoustic_encoder_layers
        self.acoustic_encoder_attention_heads = acoustic_encoder_attention_heads
        self.acoustic_encoder_ffn_dim = acoustic_encoder_ffn_dim
        self.acoustic_encoder_attn_type = acoustic_encoder_attn_type

        # Pre-RVQ Adapter
        self.pre_rvq_adapter_input_dim = pre_rvq_adapter_input_dim or (
            semantic_encoder_d_model + acoustic_encoder_d_model
        )
        self.pre_rvq_adapter_d_model = pre_rvq_adapter_d_model or code_dim
        self.pre_rvq_adapter_output_dim = pre_rvq_adapter_output_dim or code_dim
        self.pre_rvq_adapter_max_source_positions = pre_rvq_adapter_max_source_positions
        self.pre_rvq_adapter_layers = pre_rvq_adapter_layers
        self.pre_rvq_adapter_attention_heads = pre_rvq_adapter_attention_heads
        self.pre_rvq_adapter_ffn_dim = pre_rvq_adapter_ffn_dim
        self.pre_rvq_adapter_attn_type = pre_rvq_adapter_attn_type

        # Downsample
        self.downsample_d_model = downsample_d_model or code_dim
        self.downsample_avg_pooler = downsample_avg_pooler

        # Quantizer
        self.num_quantizers = num_quantizers
        self.quantizer_input_dim = quantizer_input_dim or (code_dim * downsample_avg_pooler)
        self.quantizer_rvq_dim = quantizer_rvq_dim or code_dim
        self.quantizer_output_dim = quantizer_output_dim or code_dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer_dropout = quantizer_dropout
        self.skip_rvq_ratio = skip_rvq_ratio
        self.vq_commitment = vq_commitment
        self.vq_decay = vq_decay
        self.vq_epsilon = vq_epsilon
        self.vq_threshold_ema_dead = vq_threshold_ema_dead
        self.vq_kmeans_init = vq_kmeans_init
        self.vq_kmeans_iters = vq_kmeans_iters

        # Post-RVQ Adapter
        self.post_rvq_adapter_input_dim = post_rvq_adapter_input_dim or code_dim
        self.post_rvq_adapter_d_model = post_rvq_adapter_d_model or code_dim
        self.post_rvq_adapter_output_dim = post_rvq_adapter_output_dim or (code_dim * upsample_stride)
        self.post_rvq_adapter_max_source_positions = post_rvq_adapter_max_source_positions
        self.post_rvq_adapter_layers = post_rvq_adapter_layers
        self.post_rvq_adapter_attention_heads = post_rvq_adapter_attention_heads
        self.post_rvq_adapter_ffn_dim = post_rvq_adapter_ffn_dim
        self.post_rvq_adapter_attn_type = post_rvq_adapter_attn_type

        # Upsample
        self.upsample_d_model = upsample_d_model or code_dim
        self.upsample_stride = upsample_stride

        # Acoustic Decoder
        self.acoustic_decoder_d_model = acoustic_decoder_d_model or code_dim
        self.acoustic_decoder_stride_size = acoustic_decoder_stride_size
        self.acoustic_decoder_kernel_size = acoustic_decoder_kernel_size
        self.acoustic_decoder_scale_embedding = acoustic_decoder_scale_embedding
        self.acoustic_decoder_layers = acoustic_decoder_layers
        self.acoustic_decoder_attention_heads = acoustic_decoder_attention_heads
        self.acoustic_decoder_ffn_dim = acoustic_decoder_ffn_dim
        self.acoustic_decoder_attn_type = acoustic_decoder_attn_type

        # Vocos
        self.vocos_input_channels = vocos_input_channels or num_mel_bins
        self.vocos_dim = vocos_dim
        self.vocos_intermediate_dim = vocos_intermediate_dim
        self.vocos_num_layers = vocos_num_layers
        self.vocos_n_fft = vocos_n_fft
        self.vocos_hop_size = vocos_hop_size
        self.vocos_padding = vocos_padding

        # Feature Extractor
        self.feature_extractor_feature_size = feature_extractor_feature_size
        self.feature_extractor_n_fft = feature_extractor_n_fft
        self.feature_extractor_chunk_length = feature_extractor_chunk_length

        super().__init__(**kwargs)


__all__ = ["XYTokenizerConfig"]
