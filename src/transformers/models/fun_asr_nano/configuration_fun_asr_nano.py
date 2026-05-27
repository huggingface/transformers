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
    Configuration class for the Fun-ASR-Nano audio encoder (SenseVoiceEncoderSmall / SANM architecture).

    Args:
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
            Number of additional timestamp prediction encoder blocks.
        dropout_rate (`float`, *optional*, defaults to 0.1):
            Dropout rate.
        positional_dropout_rate (`float`, *optional*, defaults to 0.1):
            Positional encoding dropout rate.
        attention_dropout_rate (`float`, *optional*, defaults to 0.0):
            Attention dropout rate.
        kernel_size (`int`, *optional*, defaults to 11):
            Kernel size for the FSMN (Feedforward Sequential Memory Network) convolution.
        sanm_shift (`int`, *optional*, defaults to 0):
            Shift for asymmetric padding in FSMN convolution.
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

    def __init__(
        self,
        input_size=560,
        output_size=512,
        attention_heads=4,
        linear_units=2048,
        num_blocks=50,
        tp_blocks=20,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.0,
        kernel_size=11,
        sanm_shift=0,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.output_size = output_size
        self.attention_heads = attention_heads
        self.linear_units = linear_units
        self.num_blocks = num_blocks
        self.tp_blocks = tp_blocks
        self.dropout_rate = dropout_rate
        self.positional_dropout_rate = positional_dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.kernel_size = kernel_size
        self.sanm_shift = sanm_shift
        self.initializer_range = initializer_range


@strict
class FunAsrNanoAdaptorConfig(PreTrainedConfig):
    r"""
    Configuration class for the Fun-ASR-Nano audio adaptor (Transformer-based projector).

    Args:
        downsample_rate (`int`, *optional*, defaults to 1):
            Frame downsampling rate.
        encoder_dim (`int`, *optional*, defaults to 512):
            Input dimension from the audio encoder.
        llm_dim (`int`, *optional*, defaults to 1024):
            Output dimension matching the LLM hidden size.
        ffn_dim (`int`, *optional*, defaults to 2048):
            Feedforward network intermediate dimension.
        num_layers (`int`, *optional*, defaults to 2):
            Number of transformer layers in the adaptor.
        attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads.
        dropout_rate (`float`, *optional*, defaults to 0.0):
            Dropout rate.
        use_low_frame_rate (`bool`, *optional*, defaults to `True`):
            Whether to use low frame rate mode (conv stride-based downsampling).

    Example:

    ```python
    >>> from transformers import FunAsrNanoAdaptorConfig

    >>> configuration = FunAsrNanoAdaptorConfig()
    ```
    """

    def __init__(
        self,
        downsample_rate=1,
        encoder_dim=512,
        llm_dim=1024,
        ffn_dim=2048,
        num_layers=2,
        attention_heads=8,
        dropout_rate=0.0,
        use_low_frame_rate=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.downsample_rate = downsample_rate
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
        self.ffn_dim = ffn_dim
        self.num_layers = num_layers
        self.attention_heads = attention_heads
        self.dropout_rate = dropout_rate
        self.use_low_frame_rate = use_low_frame_rate


@strict
class FunAsrNanoCtcConfig(PreTrainedConfig):
    r"""
    Configuration class for the Fun-ASR-Nano CTC decoder (used for timestamp prediction).

    Args:
        vocab_size (`int`, *optional*, defaults to 60515):
            CTC vocabulary size.
        encoder_dim (`int`, *optional*, defaults to 512):
            Input dimension from the audio encoder.
        decoder_dim (`int`, *optional*, defaults to 512):
            Hidden dimension of the CTC decoder.
        ffn_dim (`int`, *optional*, defaults to 2048):
            Feedforward network intermediate dimension.
        num_layers (`int`, *optional*, defaults to 5):
            Number of transformer layers in the CTC decoder.
        downsample_rate (`int`, *optional*, defaults to 1):
            Frame downsampling rate.
        blank_id (`int`, *optional*, defaults to 60514):
            Blank token ID for CTC.
        dropout_rate (`float`, *optional*, defaults to 0.0):
            Dropout rate.

    Example:

    ```python
    >>> from transformers import FunAsrNanoCtcConfig

    >>> configuration = FunAsrNanoCtcConfig()
    ```
    """

    def __init__(
        self,
        vocab_size=60515,
        encoder_dim=512,
        decoder_dim=512,
        ffn_dim=2048,
        num_layers=5,
        downsample_rate=1,
        blank_id=60514,
        dropout_rate=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.ffn_dim = ffn_dim
        self.num_layers = num_layers
        self.downsample_rate = downsample_rate
        self.blank_id = blank_id
        self.dropout_rate = dropout_rate


@auto_docstring(checkpoint="FunAudioLLM/Fun-ASR-Nano-2512-hf")
@strict
class FunAsrNanoConfig(PreTrainedConfig):
    r"""
    Configuration class for the Fun-ASR-Nano model.

    Fun-ASR-Nano is an end-to-end speech recognition model consisting of:
    - A SANM-based audio encoder (SenseVoiceEncoderSmall)
    - A Transformer-based audio adaptor
    - A Qwen3-0.6B language model
    - An optional CTC decoder for character-level timestamps

    Args:
        audio_encoder_config (`dict` or `FunAsrNanoEncoderConfig`, *optional*):
            Configuration for the audio encoder.
        adaptor_config (`dict` or `FunAsrNanoAdaptorConfig`, *optional*):
            Configuration for the audio adaptor.
        text_config (`dict` or `PreTrainedConfig`, *optional*):
            Configuration for the language model (Qwen3).
        ctc_config (`dict` or `FunAsrNanoCtcConfig`, *optional*):
            Configuration for the CTC decoder.
        audio_token_index (`int`, *optional*, defaults to 151646):
            Token ID used as placeholder for audio features in the input sequence.
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
    sub_configs = {"text_config": "auto", "audio_encoder_config": FunAsrNanoEncoderConfig}

    def __init__(
        self,
        audio_encoder_config=None,
        adaptor_config=None,
        text_config=None,
        ctc_config=None,
        audio_token_index=151646,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(audio_encoder_config, dict):
            audio_encoder_config = FunAsrNanoEncoderConfig(**audio_encoder_config)
        elif audio_encoder_config is None:
            audio_encoder_config = FunAsrNanoEncoderConfig()
        self.audio_encoder_config = audio_encoder_config

        if isinstance(adaptor_config, dict):
            adaptor_config = FunAsrNanoAdaptorConfig(**adaptor_config)
        elif adaptor_config is None:
            adaptor_config = FunAsrNanoAdaptorConfig()
        self.adaptor_config = adaptor_config

        if isinstance(text_config, dict):
            text_config_model_type = text_config.get("model_type", "qwen3")
            text_config = CONFIG_MAPPING[text_config_model_type](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["qwen3"]()
        self.text_config = text_config

        if isinstance(ctc_config, dict):
            ctc_config = FunAsrNanoCtcConfig(**ctc_config)
        elif ctc_config is None:
            ctc_config = FunAsrNanoCtcConfig()
        self.ctc_config = ctc_config

        self.audio_token_index = audio_token_index
        self.initializer_range = initializer_range


__all__ = [
    "FunAsrNanoConfig",
    "FunAsrNanoEncoderConfig",
    "FunAsrNanoAdaptorConfig",
    "FunAsrNanoCtcConfig",
]
