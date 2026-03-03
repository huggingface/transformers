# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Hiera model configuration"""

from dataclasses import dataclass

from huggingface_hub.dataclasses import strict

from ...backbone_utils import BackboneConfigMixin
from ...configuration_utils import PreTrainedConfig


@strict(accept_kwargs=True)
@dataclass(repr=False)
class HieraConfig(BackboneConfigMixin, PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`HieraModel`]. It is used to instantiate a Hiera
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Hiera
    [facebook/hiera-base-224](https://huggingface.co/facebook/hiera-base-224) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        embed_dim (`int`, *optional*, defaults to 96):
            Dimensionality of patch embedding.
        image_size (`list(int)`, *optional*, defaults to `[224, 224]`):
            The size (resolution) of input in the format (height, width) for images
            and (frames, height, width) for videos.
        patch_size (`list(int)`, *optional*, defaults to `[7, 7]`):
            The size (resolution) of each patch.
        patch_stride (`list(int)`, *optional*, defaults to `[4, 4]`):
            The stride of the patch.
        patch_padding (`list(int)`, *optional*, defaults to `[3, 3]`):
            The padding of the patch.
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            The ratio of mlp hidden dim to embedding dim.
        depths (`list(int)`, *optional*, defaults to `[2, 3, 16, 3]`):
            Depth of each layer in the Transformer encoder.
        num_heads (`list(int)`, *optional*, defaults to `[1, 2, 4, 8]`):
            Number of attention heads in each layer of the Transformer encoder.
        embed_dim_multiplier (`float`, *optional*, defaults to 2.0):
            The multiplier to the dimensionality of patch embedding in each layer of the Transformer encoder.
        num_query_pool (`int`, *optional*, defaults to 3):
            The number of query pool stages.
        query_stride (`list(int)`, *optional*, defaults to `[2, 2]`):
            The stride of the query pool.
        masked_unit_size (`list(int)`, *optional*, defaults to `[8, 8]`):
            The size of the masked unit.
        masked_unit_attention (`list(bool)`, *optional*, defaults to `[True, True, False, False]`):
            Whether to use masked unit attention in each layer of the Transformer encoder.
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            The drop path rate.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder. If string, `"gelu"`, `"relu"`,
            `"selu"` and `"gelu_new"` are supported.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices and
            the zero_initializer for initializing all bias vectors.
        layer_norm_init (`float`, *optional*, defaults to 1.0):
            The initial weight value for layer normalization layers.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        decoder_hidden_size (`int`, *optional*):
            Dimensionality of decoder embeddings for MAE pretraining.
        decoder_depth (`int`, *optional*):
            Depth of the decoder for MAE pretraining.
        decoder_num_heads (`int`, *optional*):
            Number of attention heads in each layer of the decoder for MAE pretraining.
        normalize_pixel_loss (`bool`, *optional*, defaults to `True`):
            Whether to normalize the pixel loss by the number of pixels.
        mask_ratio (`float`, *optional*, defaults to 0.6):
            The ratio of masked tokens in the input.
        out_features (`list[str]`, *optional*):
            If used as backbone, list of features to output. Can be any of `"stem"`, `"stage1"`, `"stage2"`, etc.
            (depending on how many stages the model has). If unset and `out_indices` is set, will default to the
            corresponding stages. If unset and `out_indices` is unset, will default to the last stage. Must be in the
            same order as defined in the `stage_names` attribute.
        out_indices (`list[int]`, *optional*):
            If used as backbone, list of indices of features to output. Can be any of 0, 1, 2, etc. (depending on how
            many stages the model has). If unset and `out_features` is set, will default to the corresponding stages.
            If unset and `out_features` is unset, will default to the last stage. Must be in the
            same order as defined in the `stage_names` attribute.

    Example:

    ```python
    >>> from transformers import HieraConfig, HieraModel

    >>> # Initializing a Hiera hiera-base-patch16-224 style configuration
    >>> configuration = HieraConfig()

    >>> # Initializing a model (with random weights) from the hiera-base-patch16-224 style configuration
    >>> model = HieraModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "hiera"

    attribute_map = {"num_hidden_layers": "num_layers"}

    embed_dim: int = 96
    image_size: list[int] | tuple[int, ...] = (224, 224)
    patch_size: list[int] | tuple[int, ...] = (7, 7)
    patch_stride: list[int] | tuple[int, ...] = (4, 4)
    patch_padding: list[int] | tuple[int, ...] = (3, 3)
    mlp_ratio: float = 4.0
    depths: list[int] | tuple[int, ...] = (2, 3, 16, 3)
    num_heads: list[int] | tuple[int, ...] = (1, 2, 4, 8)
    embed_dim_multiplier: float = 2.0
    num_query_pool: int = 3
    query_stride: list[int] | tuple[int, ...] = (2, 2)
    masked_unit_size: list[int] | tuple[int, ...] = (8, 8)
    masked_unit_attention: list[bool] | tuple[bool, ...] = (True, True, False, False)
    drop_path_rate: float = 0.0
    num_channels: int = 3
    hidden_act: str = "gelu"
    initializer_range: float = 0.02
    layer_norm_init: float = 1.0
    layer_norm_eps: float = 1e-6
    decoder_hidden_size: int | None = None
    decoder_depth: int | None = None
    decoder_num_heads: int | None = None
    normalize_pixel_loss: bool | None = True
    mask_ratio: float = 0.6
    _out_features: list[str] | None = None
    _out_indices: list[int] | None = None

    def __post_init__(self, **kwargs):
        # we set the hidden_size attribute in order to make Hiera work with VisionEncoderDecoderModel
        # this indicates the channel dimension after the last stage of the model
        self.hidden_size = int(self.embed_dim * self.embed_dim_multiplier ** (len(self.depths) - 1))
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(self.depths) + 1)]
        self.set_output_features_output_indices(
            out_indices=kwargs.pop("out_indices", None), out_features=kwargs.pop("out_features", None)
        )
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.masked_unit_size[0] % self.query_stride[0] ** (len(self.depths) - 1) != 0:
            raise ValueError(
                f"masked_unit_size[0] ({self.masked_unit_size[0]}) must be divisible by query_stride[0] ({self.query_stride[0]}) "
                f"raised to the power of the number of layers ({len(self.depths) - 1})"
            )

        if self.num_query_pool >= len(self.depths):
            raise ValueError(
                f"num_query_pool ({self.num_query_pool}) must be less than the number of layers ({len(self.depths)})"
            )


__all__ = ["HieraConfig"]
