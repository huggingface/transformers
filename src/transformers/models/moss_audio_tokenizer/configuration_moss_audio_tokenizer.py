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

import math
from dataclasses import field
from typing import Any

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="OpenMOSS-Team/MOSS-Audio-Tokenizer")
@strict
class MossAudioTokenizerQuantizerConfig(PreTrainedConfig):
    r"""
    input_hidden_size (`int`, *optional*, defaults to 768):
        Input hidden size of the quantizer projection.
    output_hidden_size (`int`, *optional*, defaults to 768):
        Output hidden size of the quantizer projection.
    """

    input_hidden_size: int = 768
    hidden_size: int = 512
    output_hidden_size: int = 768
    n_codebooks: int = 32
    codebook_size: int = 1024
    codebook_dim: int = 8


@auto_docstring(checkpoint="OpenMOSS-Team/MOSS-Audio-Tokenizer")
@strict
class MossAudioTokenizerTransformerConfig(PreTrainedConfig):
    r"""
    input_hidden_size (`int`, *optional*, defaults to 768):
        Input hidden size of the transformer stage projection.
    output_hidden_size (`int`, *optional*, defaults to 768):
        Output hidden size of the transformer stage projection.
    sliding_window (`int`, *optional*, defaults to 125):
        Sliding-window attention size, in number of frames.
    layerscale_value (`float`, *optional*, defaults to 0.01):
        Initial value of the layer scale parameters.
    """

    input_hidden_size: int = 768
    output_hidden_size: int = 768
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_attention_heads: int = 12
    num_key_value_heads: int | None = None
    num_hidden_layers: int = 12
    head_dim: int | None = None
    max_position_embeddings: int = 2048
    sliding_window: int = 125
    layerscale_value: float = 0.01
    rope_parameters: RopeParameters | dict | None = None
    use_cache: bool = True
    attention_bias: bool = False
    attention_dropout: float = 0.0
    dropout: float = 0.0

    def __post_init__(self, **kwargs):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                f"heads ({self.num_attention_heads})."
            )
        if self.sliding_window <= 0:
            raise ValueError(f"`sliding_window` must be positive, got {self.sliding_window}.")

    @property
    def d_model(self):
        return self.hidden_size

    @property
    def encoder_attention_heads(self):
        return self.num_attention_heads

    @property
    def encoder_ffn_dim(self):
        return self.intermediate_size


@auto_docstring(checkpoint="OpenMOSS-Team/MOSS-Audio-Tokenizer")
@strict
class MossAudioTokenizerConfig(PreTrainedConfig):
    r"""
    sliding_window_duration (`float`, *optional*, defaults to 10.0):
        Sliding-window attention duration in seconds.
    downsampling_ratios (`List[int]`, *optional*, defaults to `[240, 2, 2, 2]`):
        Downsampling ratios for each encoder stage.
    input_hidden_sizes (`List[int]`, *optional*):
        Input hidden sizes for transformer stages.
    output_hidden_sizes (`List[int]`, *optional*):
        Output hidden sizes for transformer stages.
    hidden_sizes (`List[int]`, *optional*):
        Hidden sizes for transformer stages.
    intermediate_sizes (`List[int]`, *optional*):
        Feed-forward dimensions for transformer stages.
    quantizer_config (`dict` or `MossAudioTokenizerQuantizerConfig`, *optional*):
        Residual LFQ quantizer configuration.
    """

    model_type = "moss_audio_tokenizer"
    sub_configs = {"quantizer_config": MossAudioTokenizerQuantizerConfig}

    sampling_rate: int = 24000
    sliding_window_duration: int | float = 10.0
    downsampling_ratios: list[int] | tuple[int, ...] = (240, 2, 2, 2)
    input_hidden_sizes: list[int] | tuple[int, ...] = (240, 768, 768, 1280)
    output_hidden_sizes: list[int] | tuple[int, ...] = (384, 384, 640, 768)
    hidden_sizes: list[int] | tuple[int, ...] = (768, 768, 768, 1280)
    num_attention_heads: list[int] | tuple[int, ...] = (12, 12, 12, 20)
    max_position_embeddings: int = 2048
    num_hidden_layers: list[int] | tuple[int, ...] = (12, 12, 12, 32)
    intermediate_sizes: list[int] | tuple[int, ...] = (3072, 3072, 3072, 5120)
    layer_scale_init_value: float = 0.01
    rope_parameters: RopeParameters | dict | None = None
    use_cache: bool = True
    quantizer_config: dict[str, Any] | MossAudioTokenizerQuantizerConfig = field(
        default_factory=MossAudioTokenizerQuantizerConfig
    )

    def __post_init__(self, **kwargs):
        if isinstance(self.quantizer_config, dict):
            self.quantizer_config = MossAudioTokenizerQuantizerConfig(**self.quantizer_config)

        super().__post_init__(**kwargs)
        self.validate_architecture()

    def validate_architecture(self):
        num_transformer_stages = len(self.input_hidden_sizes)
        stage_attributes = (
            "output_hidden_sizes",
            "hidden_sizes",
            "num_attention_heads",
            "num_hidden_layers",
            "intermediate_sizes",
            "downsampling_ratios",
        )
        for attribute in stage_attributes:
            length = len(getattr(self, attribute))
            if length != num_transformer_stages:
                raise ValueError(
                    f"`{attribute}` must have the same length as `input_hidden_sizes`, got "
                    f"{length} and {num_transformer_stages}."
                )

        if self.sliding_window_duration <= 0:
            raise ValueError(f"`sliding_window_duration` must be positive, got {self.sliding_window_duration}.")

        for stage_index, ratio in enumerate(self.downsampling_ratios):
            if ratio <= 0:
                raise ValueError(f"`downsampling_ratios[{stage_index}]` must be positive, got {ratio}.")

        for stage_index, (hidden_size, num_attention_heads) in enumerate(
            zip(self.hidden_sizes, self.num_attention_heads)
        ):
            if hidden_size % num_attention_heads != 0:
                raise ValueError(
                    f"`hidden_sizes[{stage_index}]` must be divisible by "
                    f"`num_attention_heads[{stage_index}]`, got {hidden_size} and {num_attention_heads}."
                )

    def _build_transformer_config(self, stage_index: int, frame_rate: float) -> MossAudioTokenizerTransformerConfig:
        return MossAudioTokenizerTransformerConfig(
            input_hidden_size=self.input_hidden_sizes[stage_index],
            output_hidden_size=self.output_hidden_sizes[stage_index],
            hidden_size=self.hidden_sizes[stage_index],
            intermediate_size=self.intermediate_sizes[stage_index],
            num_attention_heads=self.num_attention_heads[stage_index],
            num_hidden_layers=self.num_hidden_layers[stage_index],
            max_position_embeddings=self.max_position_embeddings,
            sliding_window=int(frame_rate * self.sliding_window_duration),
            layerscale_value=self.layer_scale_init_value,
            rope_parameters=self.rope_parameters,
            use_cache=self.use_cache,
        )

    @property
    def hop_length(self):
        return int(math.prod(self.downsampling_ratios))

    @property
    def encoder_config(self):
        return MossAudioTokenizerEncoderConfig(**self.to_dict())

    @property
    def decoder_config(self):
        config_dict = self.to_dict()
        config_dict["input_hidden_sizes"] = list(reversed(config_dict["output_hidden_sizes"]))
        config_dict["output_hidden_sizes"] = list(reversed(self.input_hidden_sizes))
        config_dict["hidden_sizes"] = list(reversed(config_dict["hidden_sizes"]))
        config_dict["num_attention_heads"] = list(reversed(config_dict["num_attention_heads"]))
        config_dict["num_hidden_layers"] = list(reversed(config_dict["num_hidden_layers"]))
        config_dict["intermediate_sizes"] = list(reversed(config_dict["intermediate_sizes"]))
        return MossAudioTokenizerDecoderConfig(**config_dict)


@auto_docstring(checkpoint="OpenMOSS-Team/MOSS-Audio-Tokenizer")
@strict
class MossAudioTokenizerEncoderConfig(MossAudioTokenizerConfig):
    r"""
    sliding_window_duration (`float`, *optional*, defaults to 10.0):
        Sliding-window attention duration in seconds.
    downsampling_ratios (`List[int]`, *optional*, defaults to `[240, 2, 2, 2]`):
        Downsampling ratios for each encoder stage.
    input_hidden_sizes (`List[int]`, *optional*):
        Input hidden sizes for transformer stages.
    output_hidden_sizes (`List[int]`, *optional*):
        Output hidden sizes for transformer stages.
    hidden_sizes (`List[int]`, *optional*):
        Hidden sizes for transformer stages.
    intermediate_sizes (`List[int]`, *optional*):
        Feed-forward dimensions for transformer stages.
    quantizer_config (`dict` or `MossAudioTokenizerQuantizerConfig`, *optional*):
        Residual LFQ quantizer configuration.
    """

    model_type = "moss_audio_tokenizer_encoder"
    base_config_key = "encoder_config"

    @property
    def encoder_config(self):
        return None

    @property
    def decoder_config(self):
        return None

    @property
    def transformer_configs(self) -> list[MossAudioTokenizerTransformerConfig]:
        configs = []
        current_frame_rate = float(self.sampling_rate)
        for stage_index, ratio in enumerate(self.downsampling_ratios):
            current_frame_rate /= ratio
            configs.append(self._build_transformer_config(stage_index, current_frame_rate))
        return configs


@auto_docstring(checkpoint="OpenMOSS-Team/MOSS-Audio-Tokenizer")
@strict
class MossAudioTokenizerDecoderConfig(MossAudioTokenizerConfig):
    r"""
    sliding_window_duration (`float`, *optional*, defaults to 10.0):
        Sliding-window attention duration in seconds.
    downsampling_ratios (`List[int]`, *optional*, defaults to `[240, 2, 2, 2]`):
        Downsampling ratios for each encoder stage.
    input_hidden_sizes (`List[int]`, *optional*):
        Input hidden sizes for transformer stages.
    output_hidden_sizes (`List[int]`, *optional*):
        Output hidden sizes for transformer stages.
    hidden_sizes (`List[int]`, *optional*):
        Hidden sizes for transformer stages.
    intermediate_sizes (`List[int]`, *optional*):
        Feed-forward dimensions for transformer stages.
    quantizer_config (`dict` or `MossAudioTokenizerQuantizerConfig`, *optional*):
        Residual LFQ quantizer configuration.
    """

    model_type = "moss_audio_tokenizer_decoder"
    base_config_key = "decoder_config"

    @property
    def encoder_config(self):
        return None

    @property
    def decoder_config(self):
        return None

    @property
    def upsampling_ratios(self):
        return list(reversed(self.downsampling_ratios))

    @property
    def transformer_configs(self) -> list[MossAudioTokenizerTransformerConfig]:
        configs = []
        current_frame_rate = float(self.sampling_rate) / self.hop_length
        for stage_index, ratio in enumerate(self.upsampling_ratios):
            configs.append(self._build_transformer_config(stage_index, current_frame_rate))
            current_frame_rate *= ratio
        return configs


__all__ = [
    "MossAudioTokenizerConfig",
    "MossAudioTokenizerDecoderConfig",
    "MossAudioTokenizerEncoderConfig",
    "MossAudioTokenizerQuantizerConfig",
    "MossAudioTokenizerTransformerConfig",
]
