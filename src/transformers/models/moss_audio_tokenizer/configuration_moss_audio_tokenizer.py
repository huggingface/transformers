# Copyright 2026 OpenMOSS and the HuggingFace Inc. team. All rights reserved.
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
"""MossAudioTokenizer model configuration."""

from typing import Any, ClassVar

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


def _list_or_default(value: list[Any] | tuple[Any, ...] | None, default: tuple[Any, ...]) -> list[Any]:
    return list(default if value is None else value)


def _stage_list_or_default(
    value: list[Any] | tuple[Any, ...] | None, default: tuple[Any, ...], num_stages: int, field_name: str
) -> list[Any]:
    if value is None:
        if num_stages == len(default):
            return list(default)
        if num_stages == 0:
            return []
        raise ValueError(f"`{field_name}` must be specified when defining {num_stages} transformer stages.")

    value = list(value)
    if len(value) != num_stages:
        raise ValueError(f"`{field_name}` must have length {num_stages}, got {len(value)}.")
    return value


def _expand_stage_value(value: Any, default: Any, num_stages: int, field_name: str) -> list:
    value = default if value is None else value
    if isinstance(value, (list, tuple)):
        value = list(value)
        if len(value) != num_stages:
            raise ValueError(f"`{field_name}` must have length {num_stages}, got {len(value)}.")
        return value
    return [value for _ in range(num_stages)]


@auto_docstring(checkpoint="OpenMOSS-Team/MOSS-Audio-Tokenizer")
@strict
class MossAudioTokenizerBackboneConfig(PreTrainedConfig):
    r"""
    patch_sizes (`List[int]`, *optional*):
        Patch sizes used by each downsampling or upsampling patch module.
    transformer_first (`bool`, *optional*, defaults to `False`):
        Whether transformer stages are ordered before patch stages.
    input_dimensions (`List[int]`, *optional*):
        Input dimensions for transformer stages.
    output_dimensions (`List[int]`, *optional*):
        Output dimensions for transformer stages.
    d_models (`List[int]`, *optional*):
        Hidden dimensions for transformer stages.
    num_heads (`List[int]`, *optional*):
        Number of attention heads for transformer stages.
    num_layers (`List[int]`, *optional*):
        Number of layers for transformer stages.
    dim_feedforward (`List[int]`, *optional*):
        Feed-forward dimensions for transformer stages.
    causal (`bool` or `List[bool]`, *optional*, defaults to `True`):
        Whether each transformer stage uses causal attention.
    norm (`str` or `List[str]`, *optional*, defaults to `"layer_norm"`):
        Normalization type for transformer stages.
    positional_embedding (`str` or `List[str]`, *optional*, defaults to `"rope"`):
        Positional embedding type for transformer stages.
    max_period (`int` or `List[int]`, *optional*, defaults to 10000):
        Max period used by sinusoidal or rotary positional embeddings.
    hidden_act (`str` or `List[str]`, *optional*, defaults to `"gelu"`):
        Activation function used by non-gated feed-forward layers.
    gating (`str` or `List[str]`, *optional*, defaults to `"none"`):
        Gated feed-forward activation. Use `"none"` to disable gated feed-forward layers.
    layer_scale (`float` or `List[float]`, *optional*, defaults to 0.01):
        Layer scale initialization for transformer stages.
    conv_layout (`bool` or `List[bool]`, *optional*, defaults to `True`):
        Whether transformer inputs are stored in convolutional layout.
    """

    model_type = "moss-audio-tokenizer-backbone"

    default_patch_sizes: ClassVar[tuple[int, ...]] = ()
    default_transformer_first: ClassVar[bool] = False
    default_input_dimensions: ClassVar[tuple[int, ...]] = ()
    default_output_dimensions: ClassVar[tuple[int, ...]] = ()
    default_d_models: ClassVar[tuple[int, ...]] = ()
    default_num_heads: ClassVar[tuple[int, ...]] = ()
    default_num_layers: ClassVar[tuple[int, ...]] = ()
    default_dim_feedforward: ClassVar[tuple[int, ...]] = ()
    default_causal: ClassVar[bool] = True
    default_norm: ClassVar[str] = "layer_norm"
    default_positional_embedding: ClassVar[str] = "rope"
    default_max_period: ClassVar[int] = 10000
    default_hidden_act: ClassVar[str] = "gelu"
    default_gating: ClassVar[str] = "none"
    default_layer_scale: ClassVar[float | None] = 0.01
    default_conv_layout: ClassVar[bool] = True

    patch_sizes: list[int] | tuple[int, ...] | None = None
    transformer_first: bool | None = None
    input_dimensions: list[int] | tuple[int, ...] | None = None
    output_dimensions: list[int] | tuple[int, ...] | None = None
    d_models: list[int] | tuple[int, ...] | None = None
    num_heads: list[int] | tuple[int, ...] | None = None
    num_layers: list[int] | tuple[int, ...] | None = None
    dim_feedforward: list[int] | tuple[int, ...] | None = None
    causal: bool | list[bool] | tuple[bool, ...] | None = None
    norm: str | list[str] | tuple[str, ...] | None = None
    positional_embedding: str | list[str] | tuple[str, ...] | None = None
    max_period: int | float | list[int | float] | tuple[int | float, ...] | None = None
    hidden_act: str | list[str] | tuple[str, ...] | None = None
    gating: str | list[str] | tuple[str, ...] | None = None
    layer_scale: float | list[float | None] | tuple[float | None, ...] | None = None
    conv_layout: bool | list[bool] | tuple[bool, ...] | None = None

    def __post_init__(self, **kwargs):
        self.patch_sizes = _list_or_default(self.patch_sizes, self.default_patch_sizes)
        self.transformer_first = (
            self.default_transformer_first if self.transformer_first is None else self.transformer_first
        )
        self.input_dimensions = _list_or_default(self.input_dimensions, self.default_input_dimensions)

        num_transformer_stages = len(self.input_dimensions)
        self.output_dimensions = _stage_list_or_default(
            self.output_dimensions, self.default_output_dimensions, num_transformer_stages, "output_dimensions"
        )
        self.d_models = _stage_list_or_default(
            self.d_models, self.default_d_models, num_transformer_stages, "d_models"
        )
        self.num_heads = _stage_list_or_default(
            self.num_heads, self.default_num_heads, num_transformer_stages, "num_heads"
        )
        self.num_layers = _stage_list_or_default(
            self.num_layers, self.default_num_layers, num_transformer_stages, "num_layers"
        )
        self.dim_feedforward = _stage_list_or_default(
            self.dim_feedforward, self.default_dim_feedforward, num_transformer_stages, "dim_feedforward"
        )
        self.causal = _expand_stage_value(self.causal, self.default_causal, num_transformer_stages, "causal")
        self.norm = _expand_stage_value(self.norm, self.default_norm, num_transformer_stages, "norm")
        self.positional_embedding = _expand_stage_value(
            self.positional_embedding,
            self.default_positional_embedding,
            num_transformer_stages,
            "positional_embedding",
        )
        self.max_period = _expand_stage_value(
            self.max_period, self.default_max_period, num_transformer_stages, "max_period"
        )
        self.hidden_act = _expand_stage_value(
            self.hidden_act, self.default_hidden_act, num_transformer_stages, "hidden_act"
        )
        self.gating = _expand_stage_value(self.gating, self.default_gating, num_transformer_stages, "gating")
        self.layer_scale = _expand_stage_value(
            self.layer_scale, self.default_layer_scale, num_transformer_stages, "layer_scale"
        )
        self.conv_layout = _expand_stage_value(
            self.conv_layout, self.default_conv_layout, num_transformer_stages, "conv_layout"
        )

        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="OpenMOSS-Team/MOSS-Audio-Tokenizer")
@strict
class MossAudioTokenizerEncoderConfig(MossAudioTokenizerBackboneConfig):
    r"""
    This is the configuration class for the MossAudioTokenizer encoder stack.

    patch_sizes (`List[int]`, *optional*):
        Patch sizes used by each downsampling patch module.
    transformer_first (`bool`, *optional*, defaults to `False`):
        Whether transformer stages are ordered before patch stages.
    input_dimensions (`List[int]`, *optional*):
        Input dimensions for transformer stages.
    output_dimensions (`List[int]`, *optional*):
        Output dimensions for transformer stages.
    d_models (`List[int]`, *optional*):
        Hidden dimensions for transformer stages.
    num_heads (`List[int]`, *optional*):
        Number of attention heads for transformer stages.
    num_layers (`List[int]`, *optional*):
        Number of layers for transformer stages.
    dim_feedforward (`List[int]`, *optional*):
        Feed-forward dimensions for transformer stages.
    causal (`bool` or `List[bool]`, *optional*, defaults to `True`):
        Whether each transformer stage uses causal attention.
    norm (`str` or `List[str]`, *optional*, defaults to `"layer_norm"`):
        Normalization type for transformer stages.
    positional_embedding (`str` or `List[str]`, *optional*, defaults to `"rope"`):
        Positional embedding type for transformer stages.
    max_period (`int` or `List[int]`, *optional*, defaults to 10000):
        Max period used by sinusoidal or rotary positional embeddings.
    hidden_act (`str` or `List[str]`, *optional*, defaults to `"gelu"`):
        Activation function used by non-gated feed-forward layers.
    gating (`str` or `List[str]`, *optional*, defaults to `"none"`):
        Gated feed-forward activation. Use `"none"` to disable gated feed-forward layers.
    layer_scale (`float` or `List[float]`, *optional*, defaults to 0.01):
        Layer scale initialization for transformer stages.
    conv_layout (`bool` or `List[bool]`, *optional*, defaults to `True`):
        Whether transformer inputs are stored in convolutional layout.
    """

    model_type = "moss-audio-tokenizer-encoder"
    base_config_key = "encoder_config"

    default_patch_sizes = (240, 2, 2, 2)
    default_transformer_first = False
    default_input_dimensions = (240, 768, 768, 1280)
    default_output_dimensions = (384, 384, 640, 768)
    default_d_models = (768, 768, 768, 1280)
    default_num_heads = (12, 12, 12, 20)
    default_num_layers = (12, 12, 12, 32)
    default_dim_feedforward = (3072, 3072, 3072, 5120)


@auto_docstring(checkpoint="OpenMOSS-Team/MOSS-Audio-Tokenizer")
@strict
class MossAudioTokenizerDecoderConfig(MossAudioTokenizerBackboneConfig):
    r"""
    This is the configuration class for the MossAudioTokenizer decoder stack.

    patch_sizes (`List[int]`, *optional*):
        Patch sizes used by each upsampling patch module.
    transformer_first (`bool`, *optional*, defaults to `True`):
        Whether transformer stages are ordered before patch stages.
    input_dimensions (`List[int]`, *optional*):
        Input dimensions for transformer stages.
    output_dimensions (`List[int]`, *optional*):
        Output dimensions for transformer stages.
    d_models (`List[int]`, *optional*):
        Hidden dimensions for transformer stages.
    num_heads (`List[int]`, *optional*):
        Number of attention heads for transformer stages.
    num_layers (`List[int]`, *optional*):
        Number of layers for transformer stages.
    dim_feedforward (`List[int]`, *optional*):
        Feed-forward dimensions for transformer stages.
    causal (`bool` or `List[bool]`, *optional*, defaults to `True`):
        Whether each transformer stage uses causal attention.
    norm (`str` or `List[str]`, *optional*, defaults to `"layer_norm"`):
        Normalization type for transformer stages.
    positional_embedding (`str` or `List[str]`, *optional*, defaults to `"rope"`):
        Positional embedding type for transformer stages.
    max_period (`int` or `List[int]`, *optional*, defaults to 10000):
        Max period used by sinusoidal or rotary positional embeddings.
    hidden_act (`str` or `List[str]`, *optional*, defaults to `"gelu"`):
        Activation function used by non-gated feed-forward layers.
    gating (`str` or `List[str]`, *optional*, defaults to `"none"`):
        Gated feed-forward activation. Use `"none"` to disable gated feed-forward layers.
    layer_scale (`float` or `List[float]`, *optional*, defaults to 0.01):
        Layer scale initialization for transformer stages.
    conv_layout (`bool` or `List[bool]`, *optional*, defaults to `True`):
        Whether transformer inputs are stored in convolutional layout.
    """

    model_type = "moss-audio-tokenizer-decoder"
    base_config_key = "decoder_config"

    default_patch_sizes = (2, 2, 2, 2, 240)
    default_transformer_first = True
    default_input_dimensions = (768, 640, 384, 384, 384)
    default_output_dimensions = (1280, 768, 768, 768, 240)
    default_d_models = (1280, 768, 768, 768, 768)
    default_num_heads = (20, 12, 12, 12, 12)
    default_num_layers = (32, 12, 12, 12, 12)
    default_dim_feedforward = (5120, 3072, 3072, 3072, 3072)


@auto_docstring(checkpoint="OpenMOSS-Team/MOSS-Audio-Tokenizer")
@strict
class MossAudioTokenizerQuantizerConfig(PreTrainedConfig):
    r"""
    input_dim (`int`, *optional*, defaults to 768):
        Input hidden size of the quantizer projection.
    rvq_dim (`int`, *optional*, defaults to 512):
        Hidden size used inside the residual quantizer.
    output_dim (`int`, *optional*, defaults to 768):
        Output hidden size of the quantizer projection.
    num_quantizers (`int`, *optional*, defaults to 32):
        Number of residual quantizers.
    codebook_size (`int`, *optional*, defaults to 1024):
        Number of entries in each codebook.
    codebook_dim (`int`, *optional*, defaults to 8):
        Dimension of each codebook entry.
    """

    model_type = "moss-audio-tokenizer-quantizer"
    base_config_key = "quantizer_config"

    input_dim: int = 768
    rvq_dim: int = 512
    output_dim: int = 768
    num_quantizers: int = 32
    codebook_size: int = 1024
    codebook_dim: int = 8


@auto_docstring(checkpoint="OpenMOSS-Team/MOSS-Audio-Tokenizer")
@strict
class MossAudioTokenizerConfig(PreTrainedConfig):
    r"""
    sampling_rate (`int`, *optional*, defaults to 24000):
        The sampling rate at which the audio waveform should be digitalized expressed in hertz (Hz).
    downsample_rate (`int`, *optional*, defaults to 1920):
        Total downsampling rate from waveform samples to audio code frames.
    causal_transformer_context_duration (`float`, *optional*, defaults to 10.0):
        Context duration in seconds for causal transformer stages.
    encoder_config (`dict` or `MossAudioTokenizerEncoderConfig`, *optional*):
        Encoder stack configuration.
    decoder_config (`dict` or `MossAudioTokenizerDecoderConfig`, *optional*):
        Decoder stack configuration.
    quantizer_config (`dict` or `MossAudioTokenizerQuantizerConfig`, *optional*):
        Residual LFQ quantizer configuration.

    Example:

    ```python
    >>> from transformers import MossAudioTokenizerModel, MossAudioTokenizerConfig

    >>> configuration = MossAudioTokenizerConfig()
    >>> model = MossAudioTokenizerModel(configuration)
    >>> configuration = model.config
    ```
    """

    model_type = "moss-audio-tokenizer"
    sub_configs = {
        "encoder_config": MossAudioTokenizerEncoderConfig,
        "decoder_config": MossAudioTokenizerDecoderConfig,
        "quantizer_config": MossAudioTokenizerQuantizerConfig,
    }

    sampling_rate: int = 24000
    downsample_rate: int = 1920
    causal_transformer_context_duration: int | float = 10.0
    encoder_config: dict[str, Any] | MossAudioTokenizerEncoderConfig | None = None
    decoder_config: dict[str, Any] | MossAudioTokenizerDecoderConfig | None = None
    quantizer_config: dict[str, Any] | MossAudioTokenizerQuantizerConfig | None = None

    def __post_init__(self, **kwargs):
        if isinstance(self.encoder_config, dict):
            self.encoder_config = MossAudioTokenizerEncoderConfig(**self.encoder_config)
        elif self.encoder_config is None:
            self.encoder_config = MossAudioTokenizerEncoderConfig()

        if isinstance(self.decoder_config, dict):
            self.decoder_config = MossAudioTokenizerDecoderConfig(**self.decoder_config)
        elif self.decoder_config is None:
            self.decoder_config = MossAudioTokenizerDecoderConfig()

        if isinstance(self.quantizer_config, dict):
            self.quantizer_config = MossAudioTokenizerQuantizerConfig(**self.quantizer_config)
        elif self.quantizer_config is None:
            self.quantizer_config = MossAudioTokenizerQuantizerConfig()

        super().__post_init__(**kwargs)

    @property
    def num_quantizers(self) -> int:
        return self.quantizer_config.num_quantizers

    @property
    def codebook_size(self) -> int:
        return self.quantizer_config.codebook_size

    @property
    def frame_rate(self) -> float:
        return self.sampling_rate / self.downsample_rate


__all__ = [
    "MossAudioTokenizerBackboneConfig",
    "MossAudioTokenizerConfig",
    "MossAudioTokenizerDecoderConfig",
    "MossAudioTokenizerEncoderConfig",
    "MossAudioTokenizerQuantizerConfig",
]
