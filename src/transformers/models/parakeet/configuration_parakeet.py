# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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


@auto_docstring(checkpoint="nvidia/parakeet-ctc-1.1b")
@strict
class ParakeetEncoderConfig(PreTrainedConfig):
    r"""
    convolution_bias (`bool`, *optional*, defaults to `True`):
        Whether to use bias in convolutions of the conformer's convolution module.
    conv_kernel_size (`int`, *optional*, defaults to 9):
        The kernel size of the convolution layers in the Conformer block.
    subsampling_factor (`int`, *optional*, defaults to 8):
        The factor by which the input sequence is subsampled.
    subsampling_conv_channels (`int`, *optional*, defaults to 256):
        The number of channels in the subsampling convolution layers.
    num_mel_bins (`int`, *optional*, defaults to 80):
        Number of mel features.
    subsampling_conv_kernel_size (`int`, *optional*, defaults to 3):
        The kernel size of the subsampling convolution layers.
    subsampling_conv_stride (`int`, *optional*, defaults to 2):
        The stride of the subsampling convolution layers.
    dropout_positions (`float`, *optional*, defaults to 0.0):
        The dropout ratio for the positions in the input sequence.
    scale_input (`bool`, *optional*, defaults to `True`):
        Whether to scale the input embeddings.

    Example:
    ```python
    >>> from transformers import ParakeetEncoderModel, ParakeetEncoderConfig

    >>> # Initializing a `ParakeetEncoder` configuration
    >>> configuration = ParakeetEncoderConfig()

    >>> # Initializing a model from the configuration
    >>> model = ParakeetEncoderModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "parakeet_encoder"
    keys_to_ignore_at_inference = ["past_key_values"]

    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 8
    intermediate_size: int = 4096
    hidden_act: str = "silu"
    attention_bias: bool = True
    convolution_bias: bool = True
    conv_kernel_size: int = 9
    subsampling_factor: int = 8
    subsampling_conv_channels: int = 256
    num_mel_bins: int = 80
    subsampling_conv_kernel_size: int = 3
    subsampling_conv_stride: int = 2
    dropout: float | int = 0.1
    dropout_positions: float | int = 0.0
    layerdrop: float | int = 0.1
    activation_dropout: float | int = 0.1
    attention_dropout: float | int = 0.1
    max_position_embeddings: int = 5000
    scale_input: bool = True
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        self.num_key_value_heads = self.num_attention_heads
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="nvidia/parakeet-ctc-1.1b")
@strict
class ParakeetCTCConfig(PreTrainedConfig):
    r"""
    ctc_loss_reduction (`str`, *optional*, defaults to `"mean"`):
        Specifies the reduction to apply to the output of `torch.nn.CTCLoss`. Only relevant when training an
        instance of [`ParakeetForCTC`].
    ctc_zero_infinity (`bool`, *optional*, defaults to `True`):
        Whether to zero infinite losses and the associated gradients of `torch.nn.CTCLoss`. Infinite losses mainly
        occur when the inputs are too short to be aligned to the targets. Only relevant when training an instance
        of [`ParakeetForCTC`].
    encoder_config (`Union[dict, ParakeetEncoderConfig]`, *optional*):
        The config object or dictionary of the encoder.

    Example:

    ```python
    >>> from transformers import ParakeetForCTC, ParakeetCTCConfig
    >>> # Initializing a Parakeet configuration
    >>> configuration = ParakeetCTCConfig()
    >>> # Initializing a model from the configuration
    >>> model = ParakeetForCTC(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "parakeet_ctc"
    sub_configs = {"encoder_config": ParakeetEncoderConfig}

    vocab_size: int = 1025
    ctc_loss_reduction: str = "mean"
    ctc_zero_infinity: bool = True
    encoder_config: dict | PreTrainedConfig | None = None
    pad_token_id: int | None = 1024

    def __post_init__(self, **kwargs):
        if isinstance(self.encoder_config, dict):
            self.encoder_config = ParakeetEncoderConfig(**self.encoder_config)
        elif self.encoder_config is None:
            self.encoder_config = ParakeetEncoderConfig()
        self.initializer_range = self.encoder_config.initializer_range
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="nvidia/parakeet-tdt-0.6b-v3")
@strict
class ParakeetTDTConfig(PreTrainedConfig):
    r"""
    decoder_hidden_size (`int`, *optional*, defaults to 640):
        Hidden size of the LSTM prediction network and joint network.
    num_decoder_layers (`int`, *optional*, defaults to 2):
        Number of LSTM layers in the prediction network.
    max_symbols_per_step (`int`, *optional*, defaults to 10):
        Maximum number of symbols to emit per encoder time step during greedy decoding.
    durations (`list[int]`, *optional*, defaults to `[0, 1, 2, 3, 4]`):
        Token duration values that can be predicted. Each value represents how many frames a token or blank
        emission spans.
    encoder_config (`Union[dict, ParakeetEncoderConfig]`, *optional*):
        The config object or dictionary of the encoder.
    blank_token_id (`int`, *optional*, defaults to 8192):
        Blank token id. Different from `pad_token_id` for TDT.

    Example:
    ```python
    >>> from transformers import ParakeetForTDT, ParakeetTDTConfig

    >>> # Initializing a Parakeet TDT configuration
    >>> configuration = ParakeetTDTConfig()

    >>> # Initializing a model from the configuration
    >>> model = ParakeetForTDT(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "parakeet_tdt"
    sub_configs = {"encoder_config": ParakeetEncoderConfig}

    vocab_size: int = 8193
    decoder_hidden_size: int = 640
    num_decoder_layers: int = 2
    hidden_act: str = "relu"
    max_symbols_per_step: int = 10
    durations: list[int] | tuple[int, ...] = (0, 1, 2, 3, 4)
    encoder_config: dict | PreTrainedConfig | None = None
    pad_token_id: int = 2
    blank_token_id: int = 8192
    is_encoder_decoder: bool = True

    def __post_init__(self, **kwargs):
        if isinstance(self.encoder_config, dict):
            self.encoder_config = ParakeetEncoderConfig(**self.encoder_config)
        elif self.encoder_config is None:
            self.encoder_config = ParakeetEncoderConfig()
        self.initializer_range = self.encoder_config.initializer_range
        super().__post_init__(**kwargs)


__all__ = ["ParakeetCTCConfig", "ParakeetEncoderConfig", "ParakeetTDTConfig"]
