# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""FastSpeech2Conformer model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="espnet/fastspeech2_conformer")
@strict
class FastSpeech2ConformerConfig(PreTrainedConfig):
    r"""
    encoder_num_attention_heads (`int`, *optional*, defaults to 2):
        The number of attention heads in the encoder.
    encoder_linear_units (`int`, *optional*, defaults to 1536):
        The number of units in the linear layer of the encoder.
    decoder_linear_units (`int`, *optional*, defaults to 1536):
        The number of units in the linear layer of the decoder.
    speech_decoder_postnet_layers (`int`, *optional*, defaults to 5):
        The number of layers in the post-net of the speech decoder.
    speech_decoder_postnet_units (`int`, *optional*, defaults to 256):
        The number of units in the post-net layers of the speech decoder.
    speech_decoder_postnet_kernel (`int`, *optional*, defaults to 5):
        The kernel size in the post-net of the speech decoder.
    positionwise_conv_kernel_size (`int`, *optional*, defaults to 3):
        The size of the convolution kernel used in the position-wise layer.
    encoder_normalize_before (`bool`, *optional*, defaults to `False`):
        Specifies whether to normalize before encoder layers.
    decoder_normalize_before (`bool`, *optional*, defaults to `False`):
        Specifies whether to normalize before decoder layers.
    encoder_concat_after (`bool`, *optional*, defaults to `False`):
        Specifies whether to concatenate after encoder layers.
    decoder_concat_after (`bool`, *optional*, defaults to `False`):
        Specifies whether to concatenate after decoder layers.
    reduction_factor (`int`, *optional*, defaults to 1):
        The factor by which the speech frame rate is reduced.
    speaking_speed (`float`, *optional*, defaults to 1.0):
        The speed of the speech produced.
    use_macaron_style_in_conformer (`bool`, *optional*, defaults to `True`):
        Specifies whether to use macaron style in the conformer.
    use_cnn_in_conformer (`bool`, *optional*, defaults to `True`):
        Specifies whether to use convolutional neural networks in the conformer.
    encoder_kernel_size (`int`, *optional*, defaults to 7):
        The kernel size used in the encoder.
    decoder_kernel_size (`int`, *optional*, defaults to 31):
        The kernel size used in the decoder.
    duration_predictor_layers (`int`, *optional*, defaults to 2):
        The number of layers in the duration predictor.
    duration_predictor_channels (`int`, *optional*, defaults to 256):
        The number of channels in the duration predictor.
    duration_predictor_kernel_size (`int`, *optional*, defaults to 3):
        The kernel size used in the duration predictor.
    energy_predictor_layers (`int`, *optional*, defaults to 2):
        The number of layers in the energy predictor.
    energy_predictor_channels (`int`, *optional*, defaults to 256):
        The number of channels in the energy predictor.
    energy_predictor_kernel_size (`int`, *optional*, defaults to 3):
        The kernel size used in the energy predictor.
    energy_predictor_dropout (`float`, *optional*, defaults to 0.5):
        The dropout rate in the energy predictor.
    energy_embed_kernel_size (`int`, *optional*, defaults to 1):
        The kernel size used in the energy embed layer.
    energy_embed_dropout (`float`, *optional*, defaults to 0.0):
        The dropout rate in the energy embed layer.
    stop_gradient_from_energy_predictor (`bool`, *optional*, defaults to `False`):
        Specifies whether to stop gradients from the energy predictor.
    pitch_predictor_layers (`int`, *optional*, defaults to 5):
        The number of layers in the pitch predictor.
    pitch_predictor_channels (`int`, *optional*, defaults to 256):
        The number of channels in the pitch predictor.
    pitch_predictor_kernel_size (`int`, *optional*, defaults to 5):
        The kernel size used in the pitch predictor.
    pitch_predictor_dropout (`float`, *optional*, defaults to 0.5):
        The dropout rate in the pitch predictor.
    pitch_embed_kernel_size (`int`, *optional*, defaults to 1):
        The kernel size used in the pitch embed layer.
    pitch_embed_dropout (`float`, *optional*, defaults to 0.0):
        The dropout rate in the pitch embed layer.
    stop_gradient_from_pitch_predictor (`bool`, *optional*, defaults to `True`):
        Specifies whether to stop gradients from the pitch predictor.
    encoder_dropout_rate (`float`, *optional*, defaults to 0.2):
        The dropout rate in the encoder.
    encoder_positional_dropout_rate (`float`, *optional*, defaults to 0.2):
        The positional dropout rate in the encoder.
    encoder_attention_dropout_rate (`float`, *optional*, defaults to 0.2):
        The attention dropout rate in the encoder.
    decoder_dropout_rate (`float`, *optional*, defaults to 0.2):
        The dropout rate in the decoder.
    decoder_positional_dropout_rate (`float`, *optional*, defaults to 0.2):
        The positional dropout rate in the decoder.
    decoder_attention_dropout_rate (`float`, *optional*, defaults to 0.2):
        The attention dropout rate in the decoder.
    duration_predictor_dropout_rate (`float`, *optional*, defaults to 0.2):
        The dropout rate in the duration predictor.
    speech_decoder_postnet_dropout (`float`, *optional*, defaults to 0.5):
        The dropout rate in the speech decoder postnet.
    max_source_positions (`int`, *optional*, defaults to 5000):
        if `"relative"` position embeddings are used, defines the maximum source input positions.
    use_masking (`bool`, *optional*, defaults to `True`):
        Specifies whether to use masking in the model.
    use_weighted_masking (`bool`, *optional*, defaults to `False`):
        Specifies whether to use weighted masking in the model.
    num_speakers (`int`, *optional*):
        Number of speakers. If set to > 1, assume that the speaker ids will be provided as the input and use
        speaker id embedding layer.
    num_languages (`int`, *optional*):
        Number of languages. If set to > 1, assume that the language ids will be provided as the input and use the
        language id embedding layer.
    speaker_embed_dim (`int`, *optional*):
        Speaker embedding dimension. If set to > 0, assume that speaker_embedding will be provided as the input.
    convolution_bias (`bool`, *optional*, defaults to `True`):
        Specifies whether to use bias in convolutions of the conformer's convolution module.

    Example:

    ```python
    >>> from transformers import FastSpeech2ConformerModel, FastSpeech2ConformerConfig

    >>> # Initializing a FastSpeech2Conformer style configuration
    >>> configuration = FastSpeech2ConformerConfig()

    >>> # Initializing a model from the FastSpeech2Conformer style configuration
    >>> model = FastSpeech2ConformerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "fastspeech2_conformer"
    base_config_key = "model_config"
    attribute_map = {"num_hidden_layers": "encoder_layers", "num_attention_heads": "encoder_num_attention_heads"}

    hidden_size: int = 384
    vocab_size: int = 78
    num_mel_bins: int = 80
    encoder_num_attention_heads: int = 2
    encoder_layers: int = 4
    encoder_linear_units: int = 1536
    decoder_layers: int = 4
    decoder_num_attention_heads: int = 2
    decoder_linear_units: int = 1536
    speech_decoder_postnet_layers: int = 5
    speech_decoder_postnet_units: int = 256
    speech_decoder_postnet_kernel: int = 5
    positionwise_conv_kernel_size: int = 3
    encoder_normalize_before: bool = False
    decoder_normalize_before: bool = False
    encoder_concat_after: bool = False
    decoder_concat_after: bool = False
    reduction_factor: int = 1
    speaking_speed: float = 1.0
    use_macaron_style_in_conformer: bool = True
    use_cnn_in_conformer: bool = True
    encoder_kernel_size: int = 7
    decoder_kernel_size: int = 31
    duration_predictor_layers: int = 2
    duration_predictor_channels: int = 256
    duration_predictor_kernel_size: int = 3
    energy_predictor_layers: int = 2
    energy_predictor_channels: int = 256
    energy_predictor_kernel_size: int = 3
    energy_predictor_dropout: float | int = 0.5
    energy_embed_kernel_size: int = 1
    energy_embed_dropout: float | int = 0.0
    stop_gradient_from_energy_predictor: bool = False
    pitch_predictor_layers: int = 5
    pitch_predictor_channels: int = 256
    pitch_predictor_kernel_size: int = 5
    pitch_predictor_dropout: float | int = 0.5
    pitch_embed_kernel_size: int = 1
    pitch_embed_dropout: float | int = 0.0
    stop_gradient_from_pitch_predictor: bool = True
    encoder_dropout_rate: float = 0.2
    encoder_positional_dropout_rate: float = 0.2
    encoder_attention_dropout_rate: float = 0.2
    decoder_dropout_rate: float = 0.2
    decoder_positional_dropout_rate: float = 0.2
    decoder_attention_dropout_rate: float = 0.2
    duration_predictor_dropout_rate: float = 0.2
    speech_decoder_postnet_dropout: float | int = 0.5
    max_source_positions: int = 5000
    use_masking: bool = True
    use_weighted_masking: bool = False
    num_speakers: int | None = None
    num_languages: int | None = None
    speaker_embed_dim: int | None = None
    is_encoder_decoder: bool = True
    convolution_bias: bool = True

    def __post_init__(self, **kwargs):
        self.encoder_config = {
            "num_attention_heads": self.encoder_num_attention_heads,
            "layers": self.encoder_layers,
            "kernel_size": self.encoder_kernel_size,
            "attention_dropout_rate": self.encoder_attention_dropout_rate,
            "dropout_rate": self.encoder_dropout_rate,
            "positional_dropout_rate": self.encoder_positional_dropout_rate,
            "linear_units": self.encoder_linear_units,
            "normalize_before": self.encoder_normalize_before,
            "concat_after": self.encoder_concat_after,
        }
        self.decoder_config = {
            "num_attention_heads": self.decoder_num_attention_heads,
            "layers": self.decoder_layers,
            "kernel_size": self.decoder_kernel_size,
            "attention_dropout_rate": self.decoder_attention_dropout_rate,
            "dropout_rate": self.decoder_dropout_rate,
            "positional_dropout_rate": self.decoder_positional_dropout_rate,
            "linear_units": self.decoder_linear_units,
            "normalize_before": self.decoder_normalize_before,
            "concat_after": self.decoder_concat_after,
        }
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.positionwise_conv_kernel_size % 2 == 0:
            raise ValueError(
                f"positionwise_conv_kernel_size must be odd, but got {self.self.positionwise_conv_kernel_size} instead."
            )
        if self.encoder_kernel_size % 2 == 0:
            raise ValueError(f"encoder_kernel_size must be odd, but got {self.encoder_kernel_size} instead.")
        if self.decoder_kernel_size % 2 == 0:
            raise ValueError(f"decoder_kernel_size must be odd, but got {self.decoder_kernel_size} instead.")
        if self.duration_predictor_kernel_size % 2 == 0:
            raise ValueError(
                f"duration_predictor_kernel_size must be odd, but got {self.duration_predictor_kernel_size} instead."
            )
        if self.energy_predictor_kernel_size % 2 == 0:
            raise ValueError(
                f"energy_predictor_kernel_size must be odd, but got {self.energy_predictor_kernel_size} instead."
            )
        if self.energy_embed_kernel_size % 2 == 0:
            raise ValueError(f"energy_embed_kernel_size must be odd, but got {self.energy_embed_kernel_size} instead.")
        if self.pitch_predictor_kernel_size % 2 == 0:
            raise ValueError(
                f"pitch_predictor_kernel_size must be odd, but got {self.pitch_predictor_kernel_size} instead."
            )
        if self.pitch_embed_kernel_size % 2 == 0:
            raise ValueError(f"pitch_embed_kernel_size must be odd, but got {self.pitch_embed_kernel_size} instead.")
        if self.hidden_size % self.encoder_num_attention_heads != 0:
            raise ValueError("The hidden_size must be evenly divisible by encoder_num_attention_heads.")
        if self.hidden_size % self.decoder_num_attention_heads != 0:
            raise ValueError("The hidden_size must be evenly divisible by decoder_num_attention_heads.")
        if self.use_masking and self.use_weighted_masking:
            raise ValueError("Either use_masking or use_weighted_masking can be True, but not both.")


@auto_docstring(checkpoint="espnet/fastspeech2_conformer")
@strict
class FastSpeech2ConformerHifiGanConfig(PreTrainedConfig):
    r"""
    model_in_dim (`int`, *optional*, defaults to 80):
        The number of frequency bins in the input log-mel spectrogram.
    upsample_initial_channel (`int`, *optional*, defaults to 512):
        The number of input channels into the upsampling network.
    upsample_rates (`tuple[int]` or `list[int]`, *optional*, defaults to `[8, 8, 2, 2]`):
        A tuple of integers defining the stride of each 1D convolutional layer in the upsampling network. The
        length of *upsample_rates* defines the number of convolutional layers and has to match the length of
        *upsample_kernel_sizes*.
    upsample_kernel_sizes (`tuple[int]` or `list[int]`, *optional*, defaults to `[16, 16, 4, 4]`):
        A tuple of integers defining the kernel size of each 1D convolutional layer in the upsampling network. The
        length of *upsample_kernel_sizes* defines the number of convolutional layers and has to match the length of
        *upsample_rates*.
    resblock_kernel_sizes (`tuple[int]` or `list[int]`, *optional*, defaults to `[3, 7, 11]`):
        A tuple of integers defining the kernel sizes of the 1D convolutional layers in the multi-receptive field
        fusion (MRF) module.
    resblock_dilation_sizes (`tuple[tuple[int]]` or `list[list[int]]`, *optional*, defaults to `[[1, 3, 5], [1, 3, 5], [1, 3, 5]]`):
        A nested tuple of integers defining the dilation rates of the dilated 1D convolutional layers in the
        multi-receptive field fusion (MRF) module.
    leaky_relu_slope (`float`, *optional*, defaults to 0.1):
        The angle of the negative slope used by the leaky ReLU activation.
    normalize_before (`bool`, *optional*, defaults to `True`):
        Whether or not to normalize the spectrogram before vocoding using the vocoder's learned mean and variance.

    Example:

    ```python
    >>> from transformers import FastSpeech2ConformerHifiGan, FastSpeech2ConformerHifiGanConfig

    >>> # Initializing a FastSpeech2ConformerHifiGan configuration
    >>> configuration = FastSpeech2ConformerHifiGanConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = FastSpeech2ConformerHifiGan(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "hifigan"
    base_config_key = "vocoder_config"

    model_in_dim: int = 80
    upsample_initial_channel: int = 512
    upsample_rates: list[int] | tuple[int, ...] = (8, 8, 2, 2)
    upsample_kernel_sizes: list[int] | tuple[int, ...] = (16, 16, 4, 4)
    resblock_kernel_sizes: list[int] | tuple[int, ...] = (3, 7, 11)
    resblock_dilation_sizes: list | tuple | None = None
    initializer_range: float = 0.01
    leaky_relu_slope: float = 0.1
    normalize_before: bool = True

    def __post_init__(self, **kwargs):
        if self.resblock_dilation_sizes is None:
            self.resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="espnet/fastspeech2_conformer")
@strict
class FastSpeech2ConformerWithHifiGanConfig(PreTrainedConfig):
    r"""
    model_config ([`FastSpeech2ConformerConfig | dict`], *optional*):
        Configuration of the text-to-speech model.
    vocoder_config ([`FastSpeech2ConformerHiFiGanConfig | dict`], *optional*):
        Configuration of the vocoder model.

    Example:

    ```python
    >>> from transformers import (
    ...     FastSpeech2ConformerConfig,
    ...     FastSpeech2ConformerHifiGanConfig,
    ...     FastSpeech2ConformerWithHifiGanConfig,
    ...     FastSpeech2ConformerWithHifiGan,
    ... )

    >>> # Initializing FastSpeech2ConformerWithHifiGan sub-modules configurations.
    >>> model_config = FastSpeech2ConformerConfig()
    >>> vocoder_config = FastSpeech2ConformerHifiGanConfig()

    >>> # Initializing a FastSpeech2ConformerWithHifiGan module style configuration
    >>> configuration = FastSpeech2ConformerWithHifiGanConfig(model_config.to_dict(), vocoder_config.to_dict())

    >>> # Initializing a model (with random weights)
    >>> model = FastSpeech2ConformerWithHifiGan(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "fastspeech2_conformer_with_hifigan"
    sub_configs = {"model_config": FastSpeech2ConformerConfig, "vocoder_config": FastSpeech2ConformerHifiGanConfig}

    model_config: dict | PreTrainedConfig | None = None
    vocoder_config: dict | PreTrainedConfig | None = None

    def __post_init__(self, **kwargs):
        if self.model_config is None:
            self.model_config = FastSpeech2ConformerConfig()
            logger.info("model_config is None. initializing the model with default values.")
        elif isinstance(self.model_config, dict):
            self.model_config = FastSpeech2ConformerConfig(**self.model_config)

        if self.vocoder_config is None:
            self.vocoder_config = FastSpeech2ConformerHifiGanConfig()
            logger.info("vocoder_config is None. initializing the coarse model with default values.")
        elif isinstance(self.vocoder_config, dict):
            self.vocoder_config = FastSpeech2ConformerHifiGanConfig(**self.vocoder_config)

        super().__post_init__(**kwargs)


__all__ = ["FastSpeech2ConformerConfig", "FastSpeech2ConformerHifiGanConfig", "FastSpeech2ConformerWithHifiGanConfig"]
