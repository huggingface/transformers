# Copyright 2026 Mistral AI and The HuggingFace Inc. team. All rights reserved.
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

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...utils import auto_docstring
from ..auto import AutoConfig


@auto_docstring(checkpoint="mistralai/Voxtral-4B-TTS-2603")
@strict
class VoxtralTtsAcousticTransformerConfig(PreTrainedConfig):
    r"""
    semantic_codebook_size (`int`, *optional*, defaults to 8192):
        Number of entries in the semantic codebook (excluding the special `[EMPTY_AUDIO]` and `[END_AUDIO]` tokens).
    acoustic_codebook_size (`int`, *optional*, defaults to 21):
        Number of quantization levels of each acoustic codebook (finite scalar quantization).
    num_acoustic_codebooks (`int`, *optional*, defaults to 36):
        Number of acoustic codebooks, i.e. the dimension of the continuous acoustic latent predicted by flow matching.
    num_special_audio_tokens (`int`, *optional*, defaults to 2):
        Number of special tokens (`[EMPTY_AUDIO]`, `[END_AUDIO]`) prepended to every audio codebook.
    num_flow_matching_steps (`int`, *optional*, defaults to 7):
        Number of Euler steps used to integrate the flow-matching ODE for each audio frame.
    time_embedding_theta (`float`, *optional*, defaults to 10000.0):
        Base of the sinusoidal flow-matching time embedding.
    llm_hidden_size (`int`, *optional*, defaults to 3072):
        Hidden size of the language model whose hidden states condition the acoustic transformer.

    Example:

    ```python
    >>> from transformers import VoxtralTtsAcousticTransformerConfig

    >>> configuration = VoxtralTtsAcousticTransformerConfig()
    ```"""

    model_type = "voxtral_tts_acoustic_transformer"

    hidden_size: int = 3072
    intermediate_size: int = 9216
    num_hidden_layers: int = 3
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-5
    attention_dropout: float | int = 0.0
    initializer_range: float = 0.02
    semantic_codebook_size: int = 8192
    acoustic_codebook_size: int = 21
    num_acoustic_codebooks: int = 36
    num_special_audio_tokens: int = 2
    num_flow_matching_steps: int = 7
    time_embedding_theta: float = 10000.0
    llm_hidden_size: int = 3072

    @property
    def padded_semantic_vocab_size(self) -> int:
        size = self.semantic_codebook_size + self.num_special_audio_tokens
        return 128 * math.ceil(size / 128)


@auto_docstring(checkpoint="mistralai/Voxtral-4B-TTS-2603")
@strict
class VoxtralTtsCodecConfig(PreTrainedConfig):
    r"""
    sampling_rate (`int`, *optional*, defaults to 24000):
        Sampling rate of the generated waveform.
    patch_size (`int`, *optional*, defaults to 240):
        Number of waveform samples produced per latent step by the output projection.
    patch_proj_kernel_size (`int`, *optional*, defaults to 7):
        Kernel size of the output projection convolution.
    semantic_codebook_size (`int`, *optional*, defaults to 8192):
        Number of entries in the semantic (vector-quantized) codebook.
    semantic_dim (`int`, *optional*, defaults to 256):
        Dimension of the semantic codebook entries.
    acoustic_codebook_size (`int`, *optional*, defaults to 21):
        Number of quantization levels of each acoustic codebook (finite scalar quantization).
    acoustic_dim (`int`, *optional*, defaults to 36):
        Number of acoustic codebooks.
    qk_norm_eps (`float`, *optional*, defaults to 1e-06):
        Epsilon of the RMS norms applied to the queries and keys.
    layer_scale_init (`float`, *optional*, defaults to 0.01):
        Initial value of the per-channel residual scales.
    sliding_window (`int`, *optional*, defaults to 16):
        Attention window (number of past steps) at the waveform-patch resolution. The window is halved for every
        2x downsampling, so earlier (lower frame rate) decoder stages attend over shorter windows.
    decoder_num_layers (`list[int]`, *optional*, defaults to `[2, 2, 2, 2]`):
        Number of transformer layers in each decoder stage.
    decoder_kernel_sizes (`list[int]`, *optional*, defaults to `[3, 4, 4, 4]`):
        Kernel size of the convolution that opens each decoder stage.
    decoder_strides (`list[int]`, *optional*, defaults to `[1, 2, 2, 2]`):
        Upsampling factor of the convolution that opens each decoder stage.

    Example:

    ```python
    >>> from transformers import VoxtralTtsCodecConfig, VoxtralTtsCodecModel

    >>> configuration = VoxtralTtsCodecConfig()
    >>> model = VoxtralTtsCodecModel(configuration)
    ```"""

    model_type = "voxtral_tts_codec"

    sampling_rate: int = 24000
    patch_size: int = 240
    patch_proj_kernel_size: int = 7
    semantic_codebook_size: int = 8192
    semantic_dim: int = 256
    acoustic_codebook_size: int = 21
    acoustic_dim: int = 36
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_attention_heads: int = 8
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-2
    qk_norm_eps: float = 1e-6
    layer_scale_init: float = 0.01
    sliding_window: int = 16
    attention_dropout: float | int = 0.0
    decoder_num_layers: list[int] | tuple[int, ...] = (2, 2, 2, 2)
    decoder_kernel_sizes: list[int] | tuple[int, ...] = (3, 4, 4, 4)
    decoder_strides: list[int] | tuple[int, ...] = (1, 2, 2, 2)
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        self.decoder_num_layers = list(self.decoder_num_layers)
        self.decoder_kernel_sizes = list(self.decoder_kernel_sizes)
        self.decoder_strides = list(self.decoder_strides)
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        if not len(self.decoder_num_layers) == len(self.decoder_kernel_sizes) == len(self.decoder_strides):
            raise ValueError(
                "`decoder_num_layers`, `decoder_kernel_sizes` and `decoder_strides` must have the same length, got "
                f"{self.decoder_num_layers}, {self.decoder_kernel_sizes} and {self.decoder_strides}."
            )

    @property
    def hop_length(self) -> int:
        """Number of waveform samples per audio frame."""
        return self.patch_size * math.prod(self.decoder_strides)

    @property
    def frame_rate(self) -> float:
        return self.sampling_rate / self.hop_length


@auto_docstring(checkpoint="mistralai/Voxtral-4B-TTS-2603")
@strict
class VoxtralTtsConfig(PreTrainedConfig):
    r"""
    acoustic_transformer_config (`Union[VoxtralTtsAcousticTransformerConfig, dict]`, *optional*):
        Configuration of the flow-matching acoustic transformer that turns language model hidden states into audio
        codes.
    codec_config (`Union[VoxtralTtsCodecConfig, dict]`, *optional*):
        Configuration of the audio codec decoder that turns audio codes into a waveform.
    audio_token_id (`int`, *optional*, defaults to 24):
        Id of the `[AUDIO]` placeholder token, replaced by voice embeddings in the prompt and by audio frame
        embeddings during generation.
    begin_audio_token_id (`int`, *optional*, defaults to 25):
        Id of the `[BEGIN_AUDIO]` token.
    audio_vocab_size (`int`, *optional*, defaults to 9088):
        Size of the (padded) audio embedding table shared by all codebooks.

    Example:

    ```python
    >>> from transformers import VoxtralTtsConfig, VoxtralTtsForTextToSpeech

    >>> configuration = VoxtralTtsConfig()
    >>> model = VoxtralTtsForTextToSpeech(configuration)
    ```"""

    model_type = "voxtral_tts"
    sub_configs_defaults = {
        "text_config": SubConfigSpec(
            config_class=AutoConfig,
            model_type="mistral",
            init_kwargs={
                "vocab_size": 131072,
                "hidden_size": 3072,
                "intermediate_size": 9216,
                "num_hidden_layers": 26,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "max_position_embeddings": 128000,
                "rms_norm_eps": 1e-05,
                "rope_theta": 1000000.0,
                "sliding_window": None,
                "tie_word_embeddings": True,
            },
        ),
        "acoustic_transformer_config": SubConfigSpec(
            config_class=AutoConfig, model_type="voxtral_tts_acoustic_transformer"
        ),
        "codec_config": SubConfigSpec(config_class=AutoConfig, model_type="voxtral_tts_codec"),
    }

    text_config: dict | PreTrainedConfig | None = None
    acoustic_transformer_config: dict | PreTrainedConfig | None = None
    codec_config: dict | PreTrainedConfig | None = None
    audio_token_id: int = 24
    begin_audio_token_id: int = 25
    audio_vocab_size: int = 9088
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        self.hidden_size = self.text_config.hidden_size

    @property
    def num_codebooks(self) -> int:
        return 1 + self.acoustic_transformer_config.num_acoustic_codebooks

    @property
    def codebook_sizes(self) -> list[int]:
        """Size of each codebook, including the special `[EMPTY_AUDIO]` and `[END_AUDIO]` tokens."""
        at = self.acoustic_transformer_config
        n_special = at.num_special_audio_tokens
        return [at.semantic_codebook_size + n_special] + [
            at.acoustic_codebook_size + n_special
        ] * at.num_acoustic_codebooks


__all__ = ["VoxtralTtsAcousticTransformerConfig", "VoxtralTtsCodecConfig", "VoxtralTtsConfig"]
