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
        Positional encoding dropout rate.
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
class FunAsrNanoAdaptorConfig(PreTrainedConfig):
    r"""
    Configuration for the Fun-ASR-Nano audio adaptor.

    downsample_rate (`int`, *optional*, defaults to 1):
        Downsampling factor applied to the encoder sequence before projecting to the language model.
    encoder_dim (`int`, *optional*, defaults to 512):
        Hidden size of the audio encoder output.
    llm_dim (`int`, *optional*, defaults to 1024):
        Hidden size of the language model input embeddings.
    ffn_dim (`int`, *optional*, defaults to 2048):
        Hidden size of the adaptor feed-forward projection.
    num_layers (`int`, *optional*, defaults to 2):
        Number of adaptor transformer layers.
    attention_heads (`int`, *optional*, defaults to 8):
        Number of attention heads in adaptor transformer layers.
    dropout_rate (`float`, *optional*, defaults to 0.0):
        Dropout probability used in the adaptor.
    use_low_frame_rate (`bool`, *optional*, defaults to `True`):
        Whether the adaptor expects low-frame-rate audio features.
    """

    model_type = "fun_asr_nano_adaptor"

    downsample_rate: int = 1
    encoder_dim: int = 512
    llm_dim: int = 1024
    ffn_dim: int = 2048
    num_layers: int = 2
    attention_heads: int = 8
    dropout_rate: float = 0.0
    use_low_frame_rate: bool = True


@auto_docstring(checkpoint="FunAudioLLM/Fun-ASR-Nano-2512-hf")
@strict
class FunAsrNanoCtcConfig(PreTrainedConfig):
    r"""
    Configuration for the Fun-ASR-Nano CTC decoder.

    vocab_size (`int`, *optional*, defaults to 60515):
        Size of the CTC decoder vocabulary.
    encoder_dim (`int`, *optional*, defaults to 512):
        Hidden size of the audio encoder output.
    decoder_dim (`int`, *optional*, defaults to 512):
        Hidden size of the CTC decoder.
    ffn_dim (`int`, *optional*, defaults to 2048):
        Hidden size of the CTC decoder feed-forward projection.
    num_layers (`int`, *optional*, defaults to 5):
        Number of CTC decoder transformer layers.
    downsample_rate (`int`, *optional*, defaults to 1):
        Downsampling factor applied before the CTC decoder projection.
    blank_id (`int`, *optional*, defaults to 60514):
        Token ID used as the CTC blank label.
    dropout_rate (`float`, *optional*, defaults to 0.0):
        Dropout probability used in the CTC decoder.
    """

    model_type = "fun_asr_nano_ctc"

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
    audio_encoder_config (`dict` or `FunAsrNanoEncoderConfig`, *optional*):
        Configuration for the audio encoder.
    adaptor_config (`dict` or `FunAsrNanoAdaptorConfig`, *optional*):
        Configuration for the audio adaptor.
    text_config (`dict` or `PreTrainedConfig`, *optional*):
        Configuration for the language model (Qwen3).
    ctc_config (`dict` or `FunAsrNanoCtcConfig`, *optional*):
        Configuration for the CTC decoder.
    audio_token_index (`int`, *optional*, defaults to 151646):
        Token ID used as placeholder for audio features.
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
    sub_configs = {
        "text_config": AutoConfig,
        "audio_encoder_config": FunAsrNanoEncoderConfig,
        "adaptor_config": FunAsrNanoAdaptorConfig,
        "ctc_config": FunAsrNanoCtcConfig,
    }

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
