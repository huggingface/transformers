# Copyright 2026 Alibaba DAMO Academy and the HuggingFace Inc. team. All rights reserved.
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


@auto_docstring(checkpoint="FunAudioLLM/Fun-ASR-Nano-2512-hf")
@strict
class FunAsrNanoEncoderConfig(PreTrainedConfig):
    r"""
    num_stacked_frames (`int`, *optional*, defaults to 7):
        Number of consecutive mel frames stacked by low-frame-rate feature extraction.
    num_timestamp_prediction_blocks (`int`, *optional*, defaults to 20):
        Number of timestamp prediction encoder blocks.
    kernel_size (`int`, *optional*, defaults to 11):
        Kernel size for the feedforward sequential memory network (FSMN) convolution.
    """

    model_type = "fun_asr_nano_encoder"
    attribute_map = {
        "num_attention_heads": "encoder_attention_heads",
        "intermediate_size": "encoder_ffn_dim",
    }

    num_mel_bins: int = 80
    d_model: int = 512
    encoder_attention_heads: int = 4
    encoder_ffn_dim: int = 2048
    encoder_layers: int = 50
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.1
    activation_function: str = "relu"
    max_position_embeddings: int = 2049
    num_stacked_frames: int = 7
    num_timestamp_prediction_blocks: int = 20
    kernel_size: int = 11

    @property
    def input_size(self) -> int:
        return self.num_mel_bins * self.num_stacked_frames

    def __post_init__(self, **kwargs):
        legacy_input_size = kwargs.pop("input_size", None)
        if legacy_input_size is not None and legacy_input_size != self.input_size:
            raise ValueError(
                f"`input_size={legacy_input_size}` does not match `num_mel_bins * num_stacked_frames={self.input_size}`."
            )
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="FunAudioLLM/Fun-ASR-Nano-2512-hf")
@strict
class FunAsrNanoAdaptorConfig(PreTrainedConfig):
    attribute_map = {
        "num_attention_heads": "encoder_attention_heads",
        "num_hidden_layers": "encoder_layers",
    }

    d_model: int = 1024
    encoder_attention_heads: int = 8
    encoder_ffn_dim: int = 256
    encoder_layers: int = 2
    dropout: float = 0.0
    attention_dropout: float = 0.0
    activation_dropout: float = 0.0
    activation_function: str = "relu"


@auto_docstring(checkpoint="FunAudioLLM/Fun-ASR-Nano-2512-hf")
@strict
class FunAsrNanoConfig(PreTrainedConfig):
    r"""
    encoder_config (`dict` or `PreTrainedConfig`, *optional*):
        Configuration for the audio encoder.
    adaptor_config (`dict` or `FunAsrNanoAdaptorConfig`, *optional*):
        Configuration for the bidirectional audio adaptor.
    """

    model_type = "fun_asr_nano"
    attribute_map = {
        "audio_config": "encoder_config",
    }
    sub_configs = {
        "adaptor_config": FunAsrNanoAdaptorConfig,
        "encoder_config": AutoConfig,
        "text_config": AutoConfig,
    }

    adaptor_config: dict | PreTrainedConfig | None = None
    encoder_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    audio_token_id: int = 151646
    projector_hidden_act: str = "relu"
    projector_hidden_size: int = 2048
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        audio_config = kwargs.pop("audio_config", None)
        if self.encoder_config is None and audio_config is not None:
            self.encoder_config = audio_config

        if isinstance(self.encoder_config, dict):
            self.encoder_config["model_type"] = self.encoder_config.get("model_type", "fun_asr_nano_encoder")
            self.encoder_config = CONFIG_MAPPING[self.encoder_config["model_type"]](**self.encoder_config)
        elif self.encoder_config is None:
            self.encoder_config = CONFIG_MAPPING["fun_asr_nano_encoder"]()

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "qwen3")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["qwen3"]()

        if isinstance(self.adaptor_config, dict):
            self.adaptor_config = FunAsrNanoAdaptorConfig(**self.adaptor_config)
        elif self.adaptor_config is None:
            self.adaptor_config = FunAsrNanoAdaptorConfig(
                d_model=self.text_config.hidden_size,
                encoder_ffn_dim=self.text_config.hidden_size // 4,
            )

        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.adaptor_config.d_model != self.text_config.hidden_size:
            raise ValueError(
                f"`adaptor_config.d_model` ({self.adaptor_config.d_model}) must match "
                f"`text_config.hidden_size` ({self.text_config.hidden_size})."
            )
        if self.adaptor_config.encoder_ffn_dim != self.text_config.hidden_size // 4:
            raise ValueError(
                f"`adaptor_config.encoder_ffn_dim` ({self.adaptor_config.encoder_ffn_dim}) must equal "
                f"`text_config.hidden_size // 4` ({self.text_config.hidden_size // 4})."
            )


__all__ = ["FunAsrNanoAdaptorConfig", "FunAsrNanoConfig", "FunAsrNanoEncoderConfig"]
