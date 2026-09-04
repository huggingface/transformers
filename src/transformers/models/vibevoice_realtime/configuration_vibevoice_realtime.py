# Copyright 2026 The Microsoft Team and The HuggingFace Inc. team. All rights reserved.
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
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig


@auto_docstring(checkpoint="microsoft/VibeVoice-Realtime-0.5B")
@strict
class VibeVoiceRealTimeAcousticDecoderConfig(PreTrainedConfig):
    r"""
    channels (`int`, *optional*, defaults to 1):
        Number of audio channels produced by the decoder.
    hidden_size (`int`, *optional*, defaults to 64):
        Dimensionality of the acoustic latents that are decoded into audio.
    kernel_size (`int`, *optional*, defaults to 7):
        Kernel size for convolutional layers.
    num_filters (`int`, *optional*, defaults to 32):
        Number of filters of the last convolutional layer. The number of channels starts at
        `num_filters * 2 ** (len(depths) - 1)` and is halved after each upsampling stage.
    upsampling_ratios (`list[int]`, *optional*, defaults to `[8, 5, 5, 4, 2, 2]`):
        Upsampling ratios for each layer.
    depths (`list[int]`, *optional*, defaults to `[8, 3, 3, 3, 3, 3, 3]`):
        Number of ConvNeXt blocks at each stage.
    ffn_expansion (`int`, *optional*, defaults to 4):
        Expansion factor for feed-forward networks.

    Example:

    ```python
    >>> from transformers import VibeVoiceRealTimeAcousticDecoder, VibeVoiceRealTimeAcousticDecoderConfig

    >>> # Initializing a VibeVoice real-time acoustic decoder configuration
    >>> configuration = VibeVoiceRealTimeAcousticDecoderConfig()

    >>> # Initializing a model (with random weights)
    >>> model = VibeVoiceRealTimeAcousticDecoder(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "vibevoice_realtime_acoustic_decoder"

    channels: int = 1
    hidden_size: int = 64
    kernel_size: int = 7
    rms_norm_eps: float = 1e-5
    layer_scale_init_value: float = 1e-6
    initializer_range: float = 1e-2
    num_filters: int = 32
    upsampling_ratios: list[int] | tuple[int, ...] = (8, 5, 5, 4, 2, 2)
    depths: list[int] | tuple[int, ...] = (8, 3, 3, 3, 3, 3, 3)
    hidden_act: str = "gelu"
    ffn_expansion: int = 4

    @property
    def decoder_config(self) -> PreTrainedConfig:
        # The real-time model reuses the decoder of the (non-streaming) VibeVoice acoustic tokenizer, which stores
        # the ratios in encoder order.
        config_dict = self.to_dict()
        config_dict["downsampling_ratios"] = list(reversed(config_dict.pop("upsampling_ratios")))
        config_dict["model_type"] = "vibevoice_acoustic_tokenizer_decoder"
        return CONFIG_MAPPING["vibevoice_acoustic_tokenizer_decoder"](**config_dict)


@auto_docstring(checkpoint="microsoft/VibeVoice-Realtime-0.5B")
@strict
class VibeVoiceRealTimeDiffusionHeadConfig(PreTrainedConfig):
    r"""
    latent_size (`int`, *optional*, defaults to 64):
        Dimensionality of the acoustic latents the head denoises.
    frequency_embedding_size (`int`, *optional*, defaults to 256):
        The size of the sinusoidal frequency embedding for timestep encoding in the diffusion head.
    diffusion_max_period (`int`, *optional*, defaults to 10000):
        The maximum period for the sinusoidal frequency embedding in the diffusion head.
    """

    hidden_size: int = 896
    latent_size: int = 64
    num_hidden_layers: int = 4
    intermediate_size: int = 2688
    rms_norm_eps: float = 1e-5
    hidden_act: str = "silu"
    frequency_embedding_size: int = 256
    diffusion_max_period: int = 10000
    mlp_bias: bool = False


@auto_docstring(checkpoint="microsoft/VibeVoice-Realtime-0.5B")
@strict
class VibeVoiceRealTimeConfig(PreTrainedConfig):
    r"""
    audio_config (`Union[VibeVoiceRealTimeAcousticDecoderConfig, dict]`, *optional*):
        The config object or dictionary of the acoustic decoder, which decodes acoustic latents into audio.
    tts_text_config (`Union[AutoConfig, dict]`, *optional*):
        The config object or dictionary of the TTS language model, which takes the hidden states of the language
        model and conditions the diffusion head.
    diffusion_head_config (`Union[VibeVoiceRealTimeDiffusionHeadConfig, dict]`, *optional*):
        The config object or dictionary of the diffusion head used to synthesize acoustic latents.

    ```python
    >>> from transformers import VibeVoiceRealTimeForConditionalGeneration, VibeVoiceRealTimeConfig

    >>> # Initializing a VibeVoiceRealTime configuration
    >>> configuration = VibeVoiceRealTimeConfig()

    >>> # Initializing a 0.5B model with random weights
    >>> model = VibeVoiceRealTimeForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "vibevoice_realtime"
    sub_configs = {
        "audio_config": AutoConfig,
        "text_config": AutoConfig,
        "tts_text_config": AutoConfig,
        "diffusion_head_config": VibeVoiceRealTimeDiffusionHeadConfig,
    }

    audio_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    tts_text_config: dict | PreTrainedConfig | None = None
    diffusion_head_config: dict | PreTrainedConfig | None = None
    pad_token_id: int = 151655
    initializer_range: float = 1e-2
    use_cache: bool = True

    def __post_init__(self, **kwargs):
        if isinstance(self.audio_config, dict):
            self.audio_config["model_type"] = self.audio_config.get(
                "model_type", "vibevoice_realtime_acoustic_decoder"
            )
            self.audio_config = CONFIG_MAPPING[self.audio_config["model_type"]](**self.audio_config)
        elif self.audio_config is None:
            self.audio_config = CONFIG_MAPPING["vibevoice_realtime_acoustic_decoder"]()

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "qwen2")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            # Qwen2.5-0.5B but with 4 hidden layers
            self.text_config = CONFIG_MAPPING["qwen2"](
                hidden_size=896,
                intermediate_size=4864,
                num_hidden_layers=4,
                num_attention_heads=14,
                num_key_value_heads=2,
                max_position_embeddings=8192,
                max_window_layers=24,
                sliding_window=None,
                rope_parameters={"rope_theta": 1_000_000.0},
            )

        if isinstance(self.tts_text_config, dict):
            self.tts_text_config["model_type"] = self.tts_text_config.get("model_type", "qwen2")
            self.tts_text_config = CONFIG_MAPPING[self.tts_text_config["model_type"]](**self.tts_text_config)
        elif self.tts_text_config is None:
            # Qwen2.5-0.5B but with 20 hidden layers
            self.tts_text_config = CONFIG_MAPPING["qwen2"](
                hidden_size=896,
                intermediate_size=4864,
                num_hidden_layers=20,
                num_attention_heads=14,
                num_key_value_heads=2,
                max_position_embeddings=8192,
                max_window_layers=24,
                sliding_window=None,
                rope_parameters={"rope_theta": 1_000_000.0},
            )

        if isinstance(self.diffusion_head_config, dict):
            self.diffusion_head_config = VibeVoiceRealTimeDiffusionHeadConfig(**self.diffusion_head_config)
        elif self.diffusion_head_config is None:
            self.diffusion_head_config = VibeVoiceRealTimeDiffusionHeadConfig(
                hidden_size=self.text_config.hidden_size,
                intermediate_size=3 * self.text_config.hidden_size,
                latent_size=self.audio_config.hidden_size,
            )

        self.vocab_size = self.text_config.vocab_size
        self.tie_word_embeddings = getattr(self.text_config, "tie_word_embeddings", False)
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.tts_text_config.hidden_size != self.text_config.hidden_size:
            raise ValueError(
                f"`tts_text_config.hidden_size` ({self.tts_text_config.hidden_size}) must match "
                f"`text_config.hidden_size` ({self.text_config.hidden_size})."
            )
        if self.diffusion_head_config.hidden_size != self.text_config.hidden_size:
            raise ValueError(
                f"`diffusion_head_config.hidden_size` ({self.diffusion_head_config.hidden_size}) must match "
                f"`text_config.hidden_size` ({self.text_config.hidden_size})."
            )
        if self.diffusion_head_config.latent_size != self.audio_config.hidden_size:
            raise ValueError(
                f"`diffusion_head_config.latent_size` ({self.diffusion_head_config.latent_size}) must match "
                f"`audio_config.hidden_size` ({self.audio_config.hidden_size})."
            )


__all__ = [
    "VibeVoiceRealTimeAcousticDecoderConfig",
    "VibeVoiceRealTimeConfig",
    "VibeVoiceRealTimeDiffusionHeadConfig",
]
