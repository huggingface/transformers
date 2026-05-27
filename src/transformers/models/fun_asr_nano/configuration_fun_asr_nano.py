# Copyright 2025 Alibaba DAMO Academy and the HuggingFace Inc. team. All rights reserved.
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
"""Fun-ASR-Nano model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING


@auto_docstring(checkpoint="FunAudioLLM/Fun-ASR-Nano-2512-hf")
@strict
class FunAsrNanoEncoderConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import FunAsrNanoEncoderConfig, FunAsrNanoEncoder

    >>> configuration = FunAsrNanoEncoderConfig()
    >>> model = FunAsrNanoEncoder(configuration)
    >>> configuration = model.config
    ```
    """

    model_type = "fun_asr_nano_encoder"

    input_size: int = 560
    output_size: int = 512
    attention_heads: int = 4
    linear_units: int = 2048
    num_blocks: int = 50
    tp_blocks: int = 20
    dropout_rate: float = 0.1
    positional_dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.0
    kernel_size: int = 11
    sanm_shift: int = 0
    initializer_range: float = 0.02


@strict
class FunAsrNanoAdaptorConfig(PreTrainedConfig):
    r"""
    Configuration for the Fun-ASR-Nano audio adaptor.
    """

    downsample_rate: int = 1
    encoder_dim: int = 512
    llm_dim: int = 1024
    ffn_dim: int = 2048
    num_layers: int = 2
    attention_heads: int = 8
    dropout_rate: float = 0.0
    use_low_frame_rate: bool = True


@strict
class FunAsrNanoCtcConfig(PreTrainedConfig):
    r"""
    Configuration for the Fun-ASR-Nano CTC decoder.
    """

    vocab_size: int = 60515
    encoder_dim: int = 512
    decoder_dim: int = 512
    ffn_dim: int = 2048
    num_layers: int = 5
    downsample_rate: int = 1
    blank_id: int = 60514
    dropout_rate: float = 0.0


@auto_docstring(checkpoint="FunAudioLLM/Fun-ASR-Nano-2512-hf")
@strict
class FunAsrNanoConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import FunAsrNanoConfig, FunAsrNanoForConditionalGeneration

    >>> configuration = FunAsrNanoConfig()
    >>> model = FunAsrNanoForConditionalGeneration(configuration)
    >>> configuration = model.config
    ```
    """

    model_type = "fun_asr_nano"
    sub_configs = {"text_config": "auto", "audio_encoder_config": FunAsrNanoEncoderConfig}

    audio_encoder_config: dict | FunAsrNanoEncoderConfig | None = None
    adaptor_config: dict | FunAsrNanoAdaptorConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    ctc_config: dict | FunAsrNanoCtcConfig | None = None
    audio_token_index: int = 151646
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if isinstance(self.audio_encoder_config, dict):
            self.audio_encoder_config = FunAsrNanoEncoderConfig(**self.audio_encoder_config)
        elif self.audio_encoder_config is None:
            self.audio_encoder_config = FunAsrNanoEncoderConfig()

        if isinstance(self.adaptor_config, dict):
            self.adaptor_config = FunAsrNanoAdaptorConfig(**self.adaptor_config)
        elif self.adaptor_config is None:
            self.adaptor_config = FunAsrNanoAdaptorConfig()

        if isinstance(self.text_config, dict):
            text_config_model_type = self.text_config.get("model_type", "qwen3")
            self.text_config = CONFIG_MAPPING[text_config_model_type](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["qwen3"]()

        if isinstance(self.ctc_config, dict):
            self.ctc_config = FunAsrNanoCtcConfig(**self.ctc_config)
        elif self.ctc_config is None:
            self.ctc_config = FunAsrNanoCtcConfig()

        super().__post_init__(**kwargs)


__all__ = [
    "FunAsrNanoConfig",
    "FunAsrNanoEncoderConfig",
    "FunAsrNanoAdaptorConfig",
    "FunAsrNanoCtcConfig",
]
