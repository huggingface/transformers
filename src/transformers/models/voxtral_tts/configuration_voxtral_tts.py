# Copyright 2025 Mistral AI and The HuggingFace Inc. team. All rights reserved.
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

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="mistralai/Voxtral-4B-TTS-2603")
@strict
class VoxtralTtsCodecConfig(PreTrainedConfig):
    r"""
    semantic_codebook_size (`int`, *optional*, defaults to 8192):
        Number of entries in the semantic VQ codebook.
    semantic_dim (`int`, *optional*, defaults to 256):
        Dimensionality of the semantic codebook embeddings.
    acoustic_codebook_size (`int`, *optional*, defaults to 21):
        Number of quantization levels for each acoustic FSQ codebook.
    acoustic_dim (`int`, *optional*, defaults to 36):
        Number of acoustic FSQ codebooks.
    hidden_size (`int`, *optional*, defaults to 1024):
        Dimension of the transformer hidden states in the codec.
    intermediate_size (`int`, *optional*, defaults to 4096):
        Dimension of the MLP intermediate layer in the codec transformer blocks.
    num_attention_heads (`int`, *optional*, defaults to 8):
        Number of attention heads in the codec transformer blocks.
    num_key_value_heads (`int`, *optional*, defaults to 8):
        Number of key-value heads for grouped-query attention.
    head_dim (`int`, *optional*, defaults to 128):
        Dimension of each attention head.
    sampling_rate (`int`, *optional*, defaults to 24000):
        Audio sampling rate in Hz.
    patch_size (`int`, *optional*, defaults to 240):
        Patch size for the pretransform waveform patching.
    patch_proj_kernel_size (`int`, *optional*, defaults to 7):
        Kernel size for the patch projection convolution.
    conv_weight_norm (`bool`, *optional*, defaults to `True`):
        Whether to use weight normalization on convolutions.
    causal (`bool`, *optional*, defaults to `True`):
        Whether to use causal attention in the codec transformer.
    attn_sliding_window_size (`int`, *optional*, defaults to 16):
        Sliding window size for the codec attention.
    half_attn_window_upon_downsampling (`bool`, *optional*, defaults to `True`):
        Whether to halve the attention window upon downsampling.
    qk_norm (`bool`, *optional*, defaults to `True`):
        Whether to apply QK normalization.
    qk_norm_eps (`float`, *optional*, defaults to 1e-6):
        Epsilon for QK normalization.
    rms_norm_eps (`float`, *optional*, defaults to 0.01):
        Epsilon for RMS normalization in codec layers.
    layer_scale (`bool`, *optional*, defaults to `True`):
        Whether to use learnable layer scale parameters.
    layer_scale_init (`float`, *optional*, defaults to 0.01):
        Initial value for layer scale parameters.
    decoder_transformer_lengths (`list[int]`, *optional*, defaults to `[2, 2, 2, 2]`):
        Number of transformer layers in each decoder block.
    decoder_conv_kernels (`list[int]`, *optional*, defaults to `[3, 4, 4, 4]`):
        Kernel sizes for each decoder convolution block.
    decoder_conv_strides (`list[int]`, *optional*, defaults to `[1, 2, 2, 2]`):
        Stride sizes for each decoder convolution block.
    channels (`int`, *optional*, defaults to 1):
        Number of audio channels (1 for mono).

    Example:

    ```python
    >>> from transformers import VoxtralTtsCodecConfig

    >>> configuration = VoxtralTtsCodecConfig()
    ```"""

    model_type = "voxtral_tts_codec"
    base_config_key = "codec_config"

    semantic_codebook_size: int = 8192
    semantic_dim: int = 256
    acoustic_codebook_size: int = 21
    acoustic_dim: int = 36
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_attention_heads: int = 8
    num_key_value_heads: int = 8
    head_dim: int = 128
    sampling_rate: int = 24000
    patch_size: int = 240
    patch_proj_kernel_size: int = 7
    conv_weight_norm: bool = True
    causal: bool = True
    attn_sliding_window_size: int = 16
    half_attn_window_upon_downsampling: bool = True
    qk_norm: bool = True
    qk_norm_eps: float = 1e-6
    rms_norm_eps: float = 0.01
    layer_scale: bool = True
    layer_scale_init: float = 0.01
    decoder_transformer_lengths: list[int] | None = None
    decoder_conv_kernels: list[int] | None = None
    decoder_conv_strides: list[int] | None = None
    channels: int = 1
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if self.decoder_transformer_lengths is None:
            self.decoder_transformer_lengths = [2, 2, 2, 2]
        if self.decoder_conv_kernels is None:
            self.decoder_conv_kernels = [3, 4, 4, 4]
        if self.decoder_conv_strides is None:
            self.decoder_conv_strides = [1, 2, 2, 2]
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="mistralai/Voxtral-4B-TTS-2603")
@strict
class VoxtralTtsFlowMatchingConfig(PreTrainedConfig):
    r"""
    input_dim (`int`, *optional*, defaults to 3072):
        Input dimensionality from the backbone hidden states.
    hidden_size (`int`, *optional*, defaults to 3072):
        Dimension of the flow-matching transformer hidden states.
    intermediate_size (`int`, *optional*, defaults to 9216):
        Dimension of the MLP intermediate layer.
    num_hidden_layers (`int`, *optional*, defaults to 3):
        Number of bidirectional transformer layers.
    num_attention_heads (`int`, *optional*, defaults to 32):
        Number of attention heads.
    num_key_value_heads (`int`, *optional*, defaults to 8):
        Number of key-value heads for grouped-query attention.
    head_dim (`int`, *optional*, defaults to 128):
        Dimension of each attention head.
    rope_theta (`float`, *optional*, defaults to 10000.0):
        Base frequency for RoPE embeddings.
    sigma (`float`, *optional*, defaults to 1e-5):
        Minimum noise level for flow matching.
    sigma_max (`float`, *optional*, defaults to 1.0):
        Maximum noise level for flow matching.
    acoustic_dim (`int`, *optional*, defaults to 36):
        Number of acoustic codebooks (dimensionality of the acoustic input).

    Example:

    ```python
    >>> from transformers import VoxtralTtsFlowMatchingConfig

    >>> configuration = VoxtralTtsFlowMatchingConfig()
    ```"""

    model_type = "voxtral_tts_flow_matching"
    base_config_key = "flow_matching_config"

    input_dim: int = 3072
    hidden_size: int = 3072
    intermediate_size: int = 9216
    num_hidden_layers: int = 3
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    sigma: float = 1e-5
    sigma_max: float = 1.0
    acoustic_dim: int = 36
    hidden_act: str = "silu"
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        self.head_dim = self.head_dim if self.head_dim is not None else self.hidden_size // self.num_attention_heads
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="mistralai/Voxtral-4B-TTS-2603")
@strict
class VoxtralTtsConfig(PreTrainedConfig):
    r"""
    codec_config (`VoxtralTtsCodecConfig`, *optional*):
        Configuration for the codec decoder.
    flow_matching_config (`VoxtralTtsFlowMatchingConfig`, *optional*):
        Configuration for the flow-matching transformer.
    audio_token_id (`int`, *optional*, defaults to 24):
        Token ID used to represent audio placeholder tokens in the text input.
    begin_audio_token_id (`int`, *optional*, defaults to 25):
        Token ID marking the beginning of an audio segment.
    condition_dropped_token_id (`int`, *optional*, defaults to 42):
        Token ID used when conditioning is dropped (classifier-free guidance).
    num_codebooks (`int`, *optional*, defaults to 37):
        Total number of codebooks (1 semantic + 36 acoustic).
    semantic_codebook_size (`int`, *optional*, defaults to 8192):
        Number of entries in the semantic codebook.
    acoustic_codebook_size (`int`, *optional*, defaults to 21):
        Number of quantization levels per acoustic codebook.
    n_acoustic_codebook (`int`, *optional*, defaults to 36):
        Number of acoustic codebooks.
    rope_theta (`float`, *optional*, defaults to 1000000.0):
        Base frequency for the backbone RoPE embeddings.
    audio_vocab_size (`int`, *optional*, defaults to 9088):
        Total number of entries in the audio codebook embedding table. Covers 1 semantic codebook (8192 entries)
        plus 36 acoustic codebooks (21 levels each, laid out with stride 25).
    sampling_rate (`int`, *optional*, defaults to 24000):
        Audio sampling rate in Hz.
    frame_rate (`float`, *optional*, defaults to 12.5):
        Audio frame rate in Hz.

    Example:

    ```python
    >>> from transformers import VoxtralTtsConfig, VoxtralTtsForTextToSpeech

    >>> configuration = VoxtralTtsConfig()
    >>> model = VoxtralTtsForTextToSpeech(configuration)
    >>> configuration = model.config
    ```
    """

    model_type = "voxtral_tts"
    keys_to_ignore_at_inference = ["past_key_values"]
    sub_configs = {
        "codec_config": VoxtralTtsCodecConfig,
        "flow_matching_config": VoxtralTtsFlowMatchingConfig,
    }
    base_model_tp_plan = {
        "backbone_model.layers.*.self_attn.q_proj": "colwise",
        "backbone_model.layers.*.self_attn.k_proj": "colwise",
        "backbone_model.layers.*.self_attn.v_proj": "colwise",
        "backbone_model.layers.*.self_attn.o_proj": "rowwise",
        "backbone_model.layers.*.mlp.gate_proj": "colwise",
        "backbone_model.layers.*.mlp.up_proj": "colwise",
        "backbone_model.layers.*.mlp.down_proj": "rowwise",
    }

    vocab_size: int = 131072
    hidden_size: int = 3072
    intermediate_size: int = 9216
    num_hidden_layers: int = 26
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int | None = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 128000
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    tie_word_embeddings: bool = True
    rope_parameters: RopeParameters | dict | None = None
    rope_theta: float = 1000000.0
    sliding_window: int | None = None
    attention_dropout: float | int = 0.0

    audio_token_id: int = 24
    begin_audio_token_id: int = 25
    condition_dropped_token_id: int = 42
    num_codebooks: int = 37
    semantic_codebook_size: int = 8192
    acoustic_codebook_size: int = 21
    n_acoustic_codebook: int = 36
    audio_vocab_size: int = 9088
    sampling_rate: int = 24000
    frame_rate: float = 12.5

    codec_config: dict | PreTrainedConfig | None = None
    flow_matching_config: dict | PreTrainedConfig | None = None

    def __post_init__(self, **kwargs):
        if self.codec_config is None:
            self.codec_config = VoxtralTtsCodecConfig()
            logger.info("codec_config is None, using default codec config.")
        elif isinstance(self.codec_config, dict):
            self.codec_config = VoxtralTtsCodecConfig(**self.codec_config)

        if self.flow_matching_config is None:
            self.flow_matching_config = VoxtralTtsFlowMatchingConfig()
            logger.info("flow_matching_config is None, using default flow matching config.")
        elif isinstance(self.flow_matching_config, dict):
            self.flow_matching_config = VoxtralTtsFlowMatchingConfig(**self.flow_matching_config)

        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        self.head_dim = self.head_dim if self.head_dim is not None else self.hidden_size // self.num_attention_heads
        super().__post_init__(**kwargs)


__all__ = [
    "VoxtralTtsCodecConfig",
    "VoxtralTtsFlowMatchingConfig",
    "VoxtralTtsConfig",
]
