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
from ..auto import CONFIG_MAPPING, AutoConfig


@auto_docstring(checkpoint="FunAudioLLM/Fun-ASR-Nano-2512-hf")
@strict
class FunAsrNanoEncoderConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FunAsrNanoEncoder`]. It is used to instantiate a
    Fun-ASR-Nano audio encoder (a SenseVoice SAN-M encoder) according to the specified arguments, defining the model
    architecture. Like [`ParakeetEncoderConfig`], this is a standalone encoder configuration since the encoder is a
    standalone model registered in the auto mappings.

    input_size (`int`, *optional*, defaults to 560):
        Input feature dimension (after LFR: 80 mel bins * 7 frames = 560).
    output_size (`int`, *optional*, defaults to 512):
        Hidden size of the encoder layers.
    attention_heads (`int`, *optional*, defaults to 4):
        Number of attention heads in each SANM layer.
    linear_units (`int`, *optional*, defaults to 2048):
        Dimension of the feedforward layer.
    num_blocks (`int`, *optional*, defaults to 50):
        Number of main encoder blocks.
    tp_blocks (`int`, *optional*, defaults to 20):
        Number of timestamp prediction encoder blocks.
    dropout_rate (`float`, *optional*, defaults to 0.1):
        Dropout rate.
    attention_dropout_rate (`float`, *optional*, defaults to 0.0):
        Attention dropout rate.
    kernel_size (`int`, *optional*, defaults to 11):
        Kernel size for the FSMN convolution.
    sanm_shift (`int`, *optional*, defaults to 0):
        Shift for asymmetric padding in FSMN.
    initializer_range (`float`, *optional*, defaults to 0.02):
        Standard deviation for weight initialization.

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
    attention_dropout_rate: float = 0.0
    kernel_size: int = 11
    sanm_shift: int = 0
    initializer_range: float = 0.02


@auto_docstring(checkpoint="FunAudioLLM/Fun-ASR-Nano-2512-hf")
@strict
class FunAsrNanoConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FunAsrNanoForConditionalGeneration`]. It is used
    to instantiate a Fun-ASR-Nano model according to the specified arguments, defining the model architecture.

    The adaptor (audio projector) is *not* a standalone model, so following the [`VoxtralConfig`] pattern its
    parameters live directly on this config rather than in a nested sub-config.

    audio_encoder_config (`dict` or `FunAsrNanoEncoderConfig`, *optional*):
        Configuration for the audio encoder.
    text_config (`dict` or `PreTrainedConfig`, *optional*):
        Configuration for the language model (Qwen3).
    audio_token_index (`int`, *optional*, defaults to 151646):
        Token ID used as placeholder for audio features.
    adaptor_downsample_rate (`int`, *optional*, defaults to 1):
        Downsampling factor applied to the encoder sequence before projecting to the language model.
    adaptor_ffn_dim (`int`, *optional*, defaults to 2048):
        Hidden size of the adaptor feed-forward projection.
    adaptor_num_layers (`int`, *optional*, defaults to 2):
        Number of adaptor transformer layers.
    adaptor_attention_heads (`int`, *optional*, defaults to 8):
        Number of attention heads in the adaptor transformer layers.
    adaptor_dropout_rate (`float`, *optional*, defaults to 0.0):
        Dropout probability used in the adaptor.
    initializer_range (`float`, *optional*, defaults to 0.02):
        Standard deviation for weight initialization.

    Example:

    ```python
    >>> from transformers import FunAsrNanoConfig, FunAsrNanoForConditionalGeneration

    >>> configuration = FunAsrNanoConfig()
    >>> model = FunAsrNanoForConditionalGeneration(configuration)
    >>> configuration = model.config
    ```
    """

    model_type = "fun_asr_nano"
    attribute_map = {"audio_token_id": "audio_token_index"}
    sub_configs = {
        "text_config": AutoConfig,
        "audio_encoder_config": FunAsrNanoEncoderConfig,
    }

    audio_encoder_config: dict | FunAsrNanoEncoderConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    audio_token_index: int = 151646
    adaptor_downsample_rate: int = 1
    adaptor_ffn_dim: int = 2048
    adaptor_num_layers: int = 2
    adaptor_attention_heads: int = 8
    adaptor_dropout_rate: float = 0.0
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if isinstance(self.audio_encoder_config, dict):
            self.audio_encoder_config["model_type"] = self.audio_encoder_config.get(
                "model_type", "fun_asr_nano_encoder"
            )
            self.audio_encoder_config = FunAsrNanoEncoderConfig(**self.audio_encoder_config)
        elif self.audio_encoder_config is None:
            self.audio_encoder_config = FunAsrNanoEncoderConfig()

        if isinstance(self.text_config, dict):
            text_config_model_type = self.text_config.get("model_type", "qwen3")
            self.text_config = CONFIG_MAPPING[text_config_model_type](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["qwen3"]()

        super().__post_init__(**kwargs)


__all__ = [
    "FunAsrNanoConfig",
    "FunAsrNanoEncoderConfig",
]
