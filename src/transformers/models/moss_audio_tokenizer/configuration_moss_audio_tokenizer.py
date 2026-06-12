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
"""MossAudioTokenizer model configuration"""

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


def _transformer_config_from_module_kwargs(module_kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "input_dimension": module_kwargs["input_dimension"],
        "output_dimension": module_kwargs["output_dimension"],
        "d_model": module_kwargs["d_model"],
        "num_heads": module_kwargs["num_heads"],
        "num_layers": module_kwargs["num_layers"],
        "dim_feedforward": module_kwargs["dim_feedforward"],
        "causal": module_kwargs.get("causal", True),
        "norm": module_kwargs.get("norm", "layer_norm"),
        "positional_embedding": module_kwargs.get("positional_embedding", "rope"),
        "max_period": module_kwargs.get("max_period", 10000),
        "gating": module_kwargs.get("gating", "none"),
        "layer_scale": module_kwargs.get("layer_scale", 0.01),
        "conv_layout": module_kwargs.get("conv_layout", True),
    }


class MossAudioTokenizerBackboneConfig:
    """
    Shared configuration for the encoder and decoder stack.
    """

    patch_sizes: list[int]
    transformer_first: bool
    input_dimensions: list[int]
    output_dimensions: list[int]
    d_models: list[int]
    num_heads: list[int]
    num_layers: list[int]
    dim_feedforward: list[int]
    causal: list[bool]
    norm: list[str]
    positional_embedding: list[str]
    max_period: list[int | float]
    gating: list[str]
    layer_scale: list[float | None]
    conv_layout: list[bool]

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
    default_gating: ClassVar[str] = "none"
    default_layer_scale: ClassVar[float | None] = 0.01
    default_conv_layout: ClassVar[bool] = True

    def __init__(
        self,
        patch_sizes: list[int] | tuple[int, ...] | None = None,
        transformer_first: bool | None = None,
        input_dimensions: list[int] | tuple[int, ...] | None = None,
        output_dimensions: list[int] | tuple[int, ...] | None = None,
        d_models: list[int] | tuple[int, ...] | None = None,
        num_heads: list[int] | tuple[int, ...] | None = None,
        num_layers: list[int] | tuple[int, ...] | None = None,
        dim_feedforward: list[int] | tuple[int, ...] | None = None,
        causal: bool | list[bool] | tuple[bool, ...] | None = None,
        norm: str | list[str] | tuple[str, ...] | None = None,
        positional_embedding: str | list[str] | tuple[str, ...] | None = None,
        max_period: int | float | list[int | float] | tuple[int | float, ...] | None = None,
        gating: str | list[str] | tuple[str, ...] | None = None,
        layer_scale: float | list[float | None] | tuple[float | None, ...] | None = None,
        conv_layout: bool | list[bool] | tuple[bool, ...] | None = None,
        **kwargs,
    ):
        kwargs.pop("model_type", None)

        self.patch_sizes = _list_or_default(patch_sizes, self.default_patch_sizes)
        self.transformer_first = self.default_transformer_first if transformer_first is None else transformer_first
        self.input_dimensions = _list_or_default(input_dimensions, self.default_input_dimensions)

        num_transformer_stages = len(self.input_dimensions)
        self.output_dimensions = _stage_list_or_default(
            output_dimensions, self.default_output_dimensions, num_transformer_stages, "output_dimensions"
        )
        self.d_models = _stage_list_or_default(d_models, self.default_d_models, num_transformer_stages, "d_models")
        self.num_heads = _stage_list_or_default(num_heads, self.default_num_heads, num_transformer_stages, "num_heads")
        self.num_layers = _stage_list_or_default(
            num_layers, self.default_num_layers, num_transformer_stages, "num_layers"
        )
        self.dim_feedforward = _stage_list_or_default(
            dim_feedforward, self.default_dim_feedforward, num_transformer_stages, "dim_feedforward"
        )
        self.causal = _expand_stage_value(causal, self.default_causal, num_transformer_stages, "causal")
        self.norm = _expand_stage_value(norm, self.default_norm, num_transformer_stages, "norm")
        self.positional_embedding = _expand_stage_value(
            positional_embedding, self.default_positional_embedding, num_transformer_stages, "positional_embedding"
        )
        self.max_period = _expand_stage_value(
            max_period, self.default_max_period, num_transformer_stages, "max_period"
        )
        self.gating = _expand_stage_value(gating, self.default_gating, num_transformer_stages, "gating")
        self.layer_scale = _expand_stage_value(
            layer_scale, self.default_layer_scale, num_transformer_stages, "layer_scale"
        )
        self.conv_layout = _expand_stage_value(
            conv_layout, self.default_conv_layout, num_transformer_stages, "conv_layout"
        )

        super().__init__(**kwargs)

    @classmethod
    def from_legacy_kwargs(cls, module_kwargs: list[dict[str, Any]]) -> "MossAudioTokenizerBackboneConfig":
        patch_sizes = []
        transformers = []
        transformer_first = False

        for index, module_config in enumerate(module_kwargs):
            module_config = dict(module_config)
            module_type = module_config.pop("module_type")
            if index == 0:
                transformer_first = module_type == "Transformer"

            if module_type == "PatchedPretransform":
                patch_sizes.append(module_config["patch_size"])
            elif module_type == "Transformer":
                transformers.append(_transformer_config_from_module_kwargs(module_config))
            else:
                raise ValueError(f"Unsupported MossAudioTokenizer module type: {module_type}")

        return cls(
            patch_sizes=patch_sizes,
            transformer_first=transformer_first,
            input_dimensions=[config["input_dimension"] for config in transformers],
            output_dimensions=[config["output_dimension"] for config in transformers],
            d_models=[config["d_model"] for config in transformers],
            num_heads=[config["num_heads"] for config in transformers],
            num_layers=[config["num_layers"] for config in transformers],
            dim_feedforward=[config["dim_feedforward"] for config in transformers],
            causal=[config["causal"] for config in transformers],
            norm=[config["norm"] for config in transformers],
            positional_embedding=[config["positional_embedding"] for config in transformers],
            max_period=[config["max_period"] for config in transformers],
            gating=[config["gating"] for config in transformers],
            layer_scale=[config["layer_scale"] for config in transformers],
            conv_layout=[config["conv_layout"] for config in transformers],
        )

    def _transformer_module_kwargs(self, index: int) -> dict[str, Any]:
        return {
            "module_type": "Transformer",
            "input_dimension": self.input_dimensions[index],
            "output_dimension": self.output_dimensions[index],
            "d_model": self.d_models[index],
            "num_heads": self.num_heads[index],
            "num_layers": self.num_layers[index],
            "dim_feedforward": self.dim_feedforward[index],
            "causal": self.causal[index],
            "norm": self.norm[index],
            "positional_embedding": self.positional_embedding[index],
            "max_period": self.max_period[index],
            "gating": self.gating[index],
            "layer_scale": self.layer_scale[index],
            "conv_layout": self.conv_layout[index],
        }

    def _patch_module_kwargs(self, index: int) -> dict[str, Any]:
        return {"module_type": "PatchedPretransform", "patch_size": self.patch_sizes[index]}

    def to_module_configs(self) -> list[dict[str, Any]]:
        module_configs = []
        num_patch_stages = len(self.patch_sizes)
        num_transformer_stages = len(self.input_dimensions)

        for index in range(max(num_patch_stages, num_transformer_stages)):
            if self.transformer_first:
                if index < num_transformer_stages:
                    module_configs.append(self._transformer_module_kwargs(index))
                if index < num_patch_stages:
                    module_configs.append(self._patch_module_kwargs(index))
            else:
                if index < num_patch_stages:
                    module_configs.append(self._patch_module_kwargs(index))
                if index < num_transformer_stages:
                    module_configs.append(self._transformer_module_kwargs(index))

        return module_configs


@auto_docstring(checkpoint="OpenMOSS-Team/MOSS-Audio-Tokenizer")
@strict
class MossAudioTokenizerEncoderConfig(MossAudioTokenizerBackboneConfig, PreTrainedConfig):
    r"""
    This is the configuration class for the MossAudioTokenizer encoder stack.
    """

    model_type = "moss-audio-tokenizer-encoder"

    default_patch_sizes = (240, 2, 2, 2)
    default_transformer_first = False
    default_input_dimensions = (240, 768, 768, 1280)
    default_output_dimensions = (384, 384, 640, 768)
    default_d_models = (768, 768, 768, 1280)
    default_num_heads = (12, 12, 12, 20)
    default_num_layers = (12, 12, 12, 32)
    default_dim_feedforward = (3072, 3072, 3072, 5120)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@auto_docstring(checkpoint="OpenMOSS-Team/MOSS-Audio-Tokenizer")
@strict
class MossAudioTokenizerDecoderConfig(MossAudioTokenizerBackboneConfig, PreTrainedConfig):
    r"""
    This is the configuration class for the MossAudioTokenizer decoder stack.
    """

    model_type = "moss-audio-tokenizer-decoder"

    default_patch_sizes = (2, 2, 2, 2, 240)
    default_transformer_first = True
    default_input_dimensions = (768, 640, 384, 384, 384)
    default_output_dimensions = (1280, 768, 768, 768, 240)
    default_d_models = (1280, 768, 768, 768, 768)
    default_num_heads = (20, 12, 12, 12, 12)
    default_num_layers = (32, 12, 12, 12, 12)
    default_dim_feedforward = (5120, 3072, 3072, 3072, 3072)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@auto_docstring(checkpoint="OpenMOSS-Team/MOSS-Audio-Tokenizer")
@strict
class MossAudioTokenizerQuantizerConfig(PreTrainedConfig):
    r"""
    This is the configuration class for the MossAudioTokenizer residual quantizer.

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
    quantizer_type (`str`, *optional*, defaults to `"rlfq"`):
        Quantizer implementation to instantiate.
    """

    model_type = "moss-audio-tokenizer-quantizer"

    input_dim: int
    rvq_dim: int
    output_dim: int
    num_quantizers: int
    codebook_size: int
    codebook_dim: int
    quantizer_type: str

    def __init__(
        self,
        input_dim: int = 768,
        rvq_dim: int = 512,
        output_dim: int = 768,
        num_quantizers: int = 32,
        codebook_size: int = 1024,
        codebook_dim: int = 8,
        quantizer_type: str = "rlfq",
        **kwargs,
    ):
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
        quantizer_type (`str`, *optional*, defaults to `"rlfq"`):
            Quantizer implementation to instantiate.
        """
        kwargs.pop("model_type", None)

        self.input_dim = input_dim
        self.rvq_dim = rvq_dim
        self.output_dim = output_dim
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer_type = quantizer_type

        super().__init__(**kwargs)

    def to_kwargs(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "rvq_dim": self.rvq_dim,
            "output_dim": self.output_dim,
            "num_quantizers": self.num_quantizers,
            "codebook_size": self.codebook_size,
            "codebook_dim": self.codebook_dim,
            "quantizer_type": self.quantizer_type,
        }


@auto_docstring(checkpoint="OpenMOSS-Team/MOSS-Audio-Tokenizer")
@strict
class MossAudioTokenizerConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MossAudioTokenizerModel`]. It is used to
    instantiate a MossAudioTokenizer model according to the specified arguments, defining the model architecture.

    Instantiating a configuration with the defaults will yield a similar configuration to that of the
    [VoiceAgentGroup/moss_audio_tokenizer](https://huggingface.co/VoiceAgentGroup/moss_audio_tokenizer) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    version (`str`, *optional*):
        Version string accepted for checkpoint compatibility.
    sampling_rate (`int`, *optional*, defaults to 24000):
        The sampling rate at which the audio waveform should be digitalized expressed in hertz (Hz).
    downsample_rate (`int`, *optional*, defaults to 1920):
        Total downsampling rate from waveform to tokens.
    causal_transformer_context_duration (`float`, *optional*, defaults to 10.0):
        Context duration in seconds for causal transformer.
    encoder_config (`dict` or `MossAudioTokenizerEncoderConfig`, *optional*):
        Encoder stack configuration.
    decoder_config (`dict` or `MossAudioTokenizerDecoderConfig`, *optional*):
        Decoder stack configuration.
    quantizer_config (`dict` or `MossAudioTokenizerQuantizerConfig`, *optional*):
        Residual quantizer configuration.
    quantizer_type (`str`, *optional*, defaults to `"rlfq"`):
        Quantizer implementation to instantiate when `quantizer_config` is not specified.

    Example:

    ```python
    >>> from transformers import MossAudioTokenizerModel, MossAudioTokenizerConfig

    >>> # Initializing a MossAudioTokenizer style configuration
    >>> configuration = MossAudioTokenizerConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = MossAudioTokenizerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "moss-audio-tokenizer"
    sub_configs = {
        "encoder_config": MossAudioTokenizerEncoderConfig,
        "decoder_config": MossAudioTokenizerDecoderConfig,
        "quantizer_config": MossAudioTokenizerQuantizerConfig,
    }

    # Backward-compatible alias used by some checkpoints.
    attribute_map = {"sample_rate": "sampling_rate"}

    sampling_rate: int
    downsample_rate: int
    causal_transformer_context_duration: int | float
    encoder_config: MossAudioTokenizerEncoderConfig
    decoder_config: MossAudioTokenizerDecoderConfig
    quantizer_config: MossAudioTokenizerQuantizerConfig
    quantizer_type: str

    def __init__(
        self,
        version: str | None = None,
        sampling_rate: int = 24000,
        downsample_rate: int = 1920,
        causal_transformer_context_duration: int | float = 10.0,
        encoder_config: dict[str, Any] | MossAudioTokenizerEncoderConfig | None = None,
        decoder_config: dict[str, Any] | MossAudioTokenizerDecoderConfig | None = None,
        quantizer_config: dict[str, Any] | MossAudioTokenizerQuantizerConfig | None = None,
        quantizer_type: str = "rlfq",
        **kwargs,
    ):
        r"""
        version (`str`, *optional*):
            Version string accepted for checkpoint compatibility.
        sampling_rate (`int`, *optional*, defaults to 24000):
            The sampling rate at which the audio waveform should be digitalized expressed in hertz (Hz).
        downsample_rate (`int`, *optional*, defaults to 1920):
            Total downsampling rate from waveform to tokens.
        causal_transformer_context_duration (`float`, *optional*, defaults to 10.0):
            Context duration in seconds for causal transformer.
        encoder_config (`dict` or `MossAudioTokenizerEncoderConfig`, *optional*):
            Encoder stack configuration.
        decoder_config (`dict` or `MossAudioTokenizerDecoderConfig`, *optional*):
            Decoder stack configuration.
        quantizer_config (`dict` or `MossAudioTokenizerQuantizerConfig`, *optional*):
            Residual quantizer configuration.
        quantizer_type (`str`, *optional*, defaults to `"rlfq"`):
            Quantizer implementation to instantiate when `quantizer_config` is not specified.
        """
        kwargs.pop("model_type", None)

        # Accepted for compatibility with original checkpoints but not used by the Transformers implementation.
        kwargs.pop("code_dim", None)
        kwargs.pop("reversed_decoder_kwargs", None)
        encoder_kwargs = kwargs.pop("encoder_kwargs", None)
        decoder_kwargs = kwargs.pop("decoder_kwargs", None)
        quantizer_kwargs = kwargs.pop("quantizer_kwargs", None)

        self.version = version
        self.sampling_rate = sampling_rate
        self.downsample_rate = downsample_rate
        self.causal_transformer_context_duration = causal_transformer_context_duration

        if encoder_config is None:
            self.encoder_config = (
                MossAudioTokenizerEncoderConfig.from_legacy_kwargs(encoder_kwargs)
                if encoder_kwargs is not None
                else MossAudioTokenizerEncoderConfig()
            )
        elif isinstance(encoder_config, dict):
            self.encoder_config = MossAudioTokenizerEncoderConfig(**encoder_config)
        else:
            self.encoder_config = encoder_config

        if decoder_config is None:
            self.decoder_config = (
                MossAudioTokenizerDecoderConfig.from_legacy_kwargs(decoder_kwargs)
                if decoder_kwargs is not None
                else MossAudioTokenizerDecoderConfig()
            )
        elif isinstance(decoder_config, dict):
            self.decoder_config = MossAudioTokenizerDecoderConfig(**decoder_config)
        else:
            self.decoder_config = decoder_config

        if quantizer_config is None:
            if quantizer_kwargs is not None:
                quantizer_kwargs = dict(quantizer_kwargs)
                quantizer_kwargs["quantizer_type"] = quantizer_kwargs.get("quantizer_type", quantizer_type)
                self.quantizer_config = MossAudioTokenizerQuantizerConfig(**quantizer_kwargs)
            else:
                self.quantizer_config = MossAudioTokenizerQuantizerConfig(quantizer_type=quantizer_type)
        elif isinstance(quantizer_config, dict):
            self.quantizer_config = MossAudioTokenizerQuantizerConfig(**quantizer_config)
        else:
            self.quantizer_config = quantizer_config

        self.quantizer_type = self.quantizer_config.quantizer_type

        super().__init__(**kwargs)

    @property
    def encoder_kwargs(self) -> list[dict[str, Any]]:
        """Backward-compatible encoder module kwargs."""
        return self.encoder_config.to_module_configs()

    @encoder_kwargs.setter
    def encoder_kwargs(self, value: list[dict[str, Any]]):
        self.encoder_config = MossAudioTokenizerEncoderConfig.from_legacy_kwargs(value)

    @property
    def decoder_kwargs(self) -> list[dict[str, Any]]:
        """Backward-compatible decoder module kwargs."""
        return self.decoder_config.to_module_configs()

    @decoder_kwargs.setter
    def decoder_kwargs(self, value: list[dict[str, Any]]):
        self.decoder_config = MossAudioTokenizerDecoderConfig.from_legacy_kwargs(value)

    @property
    def quantizer_kwargs(self) -> dict[str, Any]:
        """Backward-compatible quantizer kwargs."""
        return self.quantizer_config.to_kwargs()

    @quantizer_kwargs.setter
    def quantizer_kwargs(self, value: dict[str, Any]):
        value = dict(value)
        value["quantizer_type"] = value.get("quantizer_type", self.quantizer_type)
        self.quantizer_config = MossAudioTokenizerQuantizerConfig(**value)
        self.quantizer_type = self.quantizer_config.quantizer_type

    @property
    def num_quantizers(self) -> int:
        """Return the number of quantizers."""
        return self.quantizer_config.num_quantizers

    @property
    def codebook_size(self) -> int:
        """Return the codebook size."""
        return self.quantizer_config.codebook_size

    @property
    def frame_rate(self) -> float:
        """Return the frame rate (tokens per second)."""
        return self.sampling_rate / self.downsample_rate


__all__ = [
    "MossAudioTokenizerConfig",
    "MossAudioTokenizerDecoderConfig",
    "MossAudioTokenizerEncoderConfig",
    "MossAudioTokenizerQuantizerConfig",
]
