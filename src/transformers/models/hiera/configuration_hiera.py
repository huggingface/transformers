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

from ...backbone_utils import BackboneConfigMixin
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="facebook/hiera-base-224")
class HieraConfig(BackboneConfigMixin, PreTrainedConfig):
    r"""
    layer_norm_init (`float`, *optional*, defaults to 1.0):
        The initial weight value for layer normalization layers.
    patch_stride (`list(int)`, *optional*, defaults to `[4, 4]`):
        The stride of the patch.
    patch_padding (`list(int)`, *optional*, defaults to `[3, 3]`):
        The padding of the patch.
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
    decoder_depth (`int`, *optional*):
        Depth of the decoder for MAE pretraining.
    normalize_pixel_loss (`bool`, *optional*, defaults to `True`):
        Whether to normalize the pixel loss by the number of pixels.
    mask_ratio (`float`, *optional*, defaults to 0.6):
        The ratio of masked tokens in the input.

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

    def __init__(
        self,
        embed_dim=96,
        image_size=[224, 224],
        patch_size=[7, 7],
        patch_stride=[4, 4],
        patch_padding=[3, 3],
        mlp_ratio=4.0,
        depths=[2, 3, 16, 3],
        num_heads=[1, 2, 4, 8],
        embed_dim_multiplier=2.0,
        num_query_pool=3,
        query_stride=[2, 2],
        masked_unit_size=[8, 8],
        masked_unit_attention=[True, True, False, False],
        drop_path_rate=0.0,
        num_channels=3,
        hidden_act="gelu",
        initializer_range=0.02,
        layer_norm_init=1.0,
        layer_norm_eps=1e-6,
        decoder_hidden_size=None,
        decoder_depth=None,
        decoder_num_heads=None,
        normalize_pixel_loss=True,
        mask_ratio=0.6,
        out_features=None,
        out_indices=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if masked_unit_size[0] % query_stride[0] ** (len(depths) - 1) != 0:
            raise ValueError(
                f"masked_unit_size[0] ({masked_unit_size[0]}) must be divisible by query_stride[0] ({query_stride[0]}) "
                f"raised to the power of the number of layers ({len(depths) - 1})"
            )

        if num_query_pool >= len(depths):
            raise ValueError(
                f"num_query_pool ({num_query_pool}) must be less than the number of layers ({len(depths)})"
            )

        self.embed_dim = embed_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.patch_padding = patch_padding
        self.mlp_ratio = mlp_ratio
        self.depths = depths
        self.num_heads = num_heads
        self.num_layers = len(depths)
        self.embed_dim_multiplier = embed_dim_multiplier
        self.num_query_pool = num_query_pool
        self.query_stride = query_stride
        self.masked_unit_size = masked_unit_size
        self.masked_unit_attention = masked_unit_attention
        self.drop_path_rate = drop_path_rate
        self.num_channels = num_channels
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.layer_norm_init = layer_norm_init
        self.layer_norm_eps = layer_norm_eps
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_depth = decoder_depth
        self.decoder_num_heads = decoder_num_heads
        self.normalize_pixel_loss = normalize_pixel_loss
        self.mask_ratio = mask_ratio
        # we set the hidden_size attribute in order to make Hiera work with VisionEncoderDecoderModel
        # this indicates the channel dimension after the last stage of the model
        self.hidden_size = int(embed_dim * embed_dim_multiplier ** (len(depths) - 1))
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(depths) + 1)]
        self.set_output_features_output_indices(out_indices=out_indices, out_features=out_features)


__all__ = ["HieraConfig"]
