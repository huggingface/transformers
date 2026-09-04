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
    num_hidden_layers (`int`, *optional*, defaults to 70):
        Total number of encoder layers, including both the transcription layers and the subsequent timestamp
        prediction layers.
    num_stacked_frames (`int`, *optional*, defaults to 7):
        Number of consecutive mel frames stacked by low-frame-rate feature extraction.
    num_timestamp_prediction_layers (`int`, *optional*, defaults to 20):
        Number of encoder layers (final ones) dedicated to timestamp prediction. The transcription output is taken
        after layer `num_hidden_layers - num_timestamp_prediction_layers - 1`.
    fsmn_kernel_size (`int`, *optional*, defaults to 11):
        Kernel size for the feedforward sequential memory network (FSMN) convolution.
    """

    model_type = "fun_asr_nano_encoder"

    num_mel_bins: int = 80
    hidden_size: int = 512
    num_attention_heads: int = 4
    intermediate_size: int = 2048
    num_hidden_layers: int = 70
    hidden_dropout: float = 0.1
    hidden_act: str = "relu"
    max_position_embeddings: int = 2049
    num_stacked_frames: int = 7
    num_timestamp_prediction_layers: int = 20
    fsmn_kernel_size: int = 11

    @property
    def input_size(self) -> int:
        return self.num_mel_bins * self.num_stacked_frames


@auto_docstring(checkpoint="FunAudioLLM/Fun-ASR-Nano-2512-hf")
@strict
class FunAsrNanoAdaptorConfig(PreTrainedConfig):
    hidden_size: int = 1024
    num_attention_heads: int = 8
    intermediate_size: int = 256
    num_hidden_layers: int = 2
    hidden_dropout: float = 0.0
    hidden_act: str = "relu"


@auto_docstring(checkpoint="FunAudioLLM/Fun-ASR-Nano-2512-hf")
@strict
class FunAsrNanoConfig(PreTrainedConfig):
    r"""
    adaptor_config (`dict` or `FunAsrNanoAdaptorConfig`, *optional*):
        Configuration for the bidirectional audio adaptor.
    """

    model_type = "fun_asr_nano"
    sub_configs = {
        "adaptor_config": FunAsrNanoAdaptorConfig,
        "audio_config": AutoConfig,
        "text_config": AutoConfig,
    }

    adaptor_config: dict | PreTrainedConfig | None = None
    audio_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    audio_token_id: int = 151646
    projector_hidden_act: str = "relu"
    projector_hidden_size: int = 2048
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if isinstance(self.audio_config, dict):
            self.audio_config["model_type"] = self.audio_config.get("model_type", "fun_asr_nano_encoder")
            self.audio_config = CONFIG_MAPPING[self.audio_config["model_type"]](**self.audio_config)
        elif self.audio_config is None:
            self.audio_config = CONFIG_MAPPING["fun_asr_nano_encoder"]()

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "qwen3")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["qwen3"](
                hidden_size=1024,
                intermediate_size=3072,
                num_hidden_layers=28,
                num_attention_heads=16,
                num_key_value_heads=8,
                max_position_embeddings=40960,
                tie_word_embeddings=True,
            )

        if isinstance(self.adaptor_config, dict):
            self.adaptor_config = FunAsrNanoAdaptorConfig(**self.adaptor_config)
        elif self.adaptor_config is None:
            self.adaptor_config = FunAsrNanoAdaptorConfig(
                hidden_size=self.text_config.hidden_size,
                intermediate_size=self.text_config.hidden_size // 4,
            )

        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.adaptor_config.hidden_size != self.text_config.hidden_size:
            raise ValueError(
                f"`adaptor_config.hidden_size` ({self.adaptor_config.hidden_size}) must match "
                f"`text_config.hidden_size` ({self.text_config.hidden_size})."
            )
        if self.adaptor_config.intermediate_size != self.text_config.hidden_size // 4:
            raise ValueError(
                f"`adaptor_config.intermediate_size` ({self.adaptor_config.intermediate_size}) must equal "
                f"`text_config.hidden_size // 4` ({self.text_config.hidden_size // 4})."
            )


__all__ = ["FunAsrNanoAdaptorConfig", "FunAsrNanoEncoderConfig", "FunAsrNanoConfig"]
