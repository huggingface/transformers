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
"""ZoeDepth model configuration"""

from dataclasses import dataclass
from typing import Literal

from huggingface_hub.dataclasses import strict

from ...backbone_utils import consolidate_backbone_kwargs_to_config
from ...configuration_utils import PreTrainedConfig
from ..auto.configuration_auto import AutoConfig


ZOEDEPTH_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "Intel/zoedepth-nyu": "https://huggingface.co/Intel/zoedepth-nyu/resolve/main/config.json",
}


@strict(accept_kwargs=True)
@dataclass(repr=False)
class ZoeDepthConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ZoeDepthForDepthEstimation`]. It is used to instantiate an ZoeDepth
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the ZoeDepth
    [Intel/zoedepth-nyu](https://huggingface.co/Intel/zoedepth-nyu) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        backbone_config (`Union[dict, "PreTrainedConfig"]`, *optional*, defaults to `BeitConfig()`):
            The configuration of the backbone model.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        batch_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the batch normalization layers.
        readout_type (`str`, *optional*, defaults to `"project"`):
            The readout type to use when processing the readout token (CLS token) of the intermediate hidden states of
            the ViT backbone. Can be one of [`"ignore"`, `"add"`, `"project"`].

            - "ignore" simply ignores the CLS token.
            - "add" passes the information from the CLS token to all other tokens by adding the representations.
            - "project" passes information to the other tokens by concatenating the readout to all other tokens before
              projecting the
            representation to the original feature dimension D using a linear layer followed by a GELU non-linearity.
        reassemble_factors (`list[int]`, *optional*, defaults to `[4, 2, 1, 0.5]`):
            The up/downsampling factors of the reassemble layers.
        neck_hidden_sizes (`list[str]`, *optional*, defaults to `[96, 192, 384, 768]`):
            The hidden sizes to project to for the feature maps of the backbone.
        fusion_hidden_size (`int`, *optional*, defaults to 256):
            The number of channels before fusion.
        head_in_index (`int`, *optional*, defaults to -1):
            The index of the features to use in the heads.
        use_batch_norm_in_fusion_residual (`bool`, *optional*, defaults to `False`):
            Whether to use batch normalization in the pre-activate residual units of the fusion blocks.
        use_bias_in_fusion_residual (`bool`, *optional*, defaults to `True`):
            Whether to use bias in the pre-activate residual units of the fusion blocks.
        num_relative_features (`int`, *optional*, defaults to 32):
            The number of features to use in the relative depth estimation head.
        add_projection (`bool`, *optional*, defaults to `False`):
            Whether to add a projection layer before the depth estimation head.
        bottleneck_features (`int`, *optional*, defaults to 256):
            The number of features in the bottleneck layer.
        num_attractors (`list[int], *optional*, defaults to `[16, 8, 4, 1]`):
            The number of attractors to use in each stage.
        bin_embedding_dim (`int`, *optional*, defaults to 128):
            The dimension of the bin embeddings.
        attractor_alpha (`int`, *optional*, defaults to 1000):
            The alpha value to use in the attractor.
        attractor_gamma (`int`, *optional*, defaults to 2):
            The gamma value to use in the attractor.
        attractor_kind (`str`, *optional*, defaults to `"mean"`):
            The kind of attractor to use. Can be one of [`"mean"`, `"sum"`].
        min_temp (`float`, *optional*, defaults to 0.0212):
            The minimum temperature value to consider.
        max_temp (`float`, *optional*, defaults to 50.0):
            The maximum temperature value to consider.
        bin_centers_type (`str`, *optional*, defaults to `"softplus"`):
            Activation type used for bin centers. Can be "normed" or "softplus". For "normed" bin centers, linear normalization trick
            is applied. This results in bounded bin centers. For "softplus", softplus activation is used and thus are unbounded.
        bin_configurations (`list[dict]`, *optional*, defaults to `[{'n_bins': 64, 'min_depth': 0.001, 'max_depth': 10.0}]`):
            Configuration for each of the bin heads.
            Each configuration should consist of the following keys:
            - name (`str`): The name of the bin head - only required in case of multiple bin configurations.
            - `n_bins` (`int`): The number of bins to use.
            - `min_depth` (`float`): The minimum depth value to consider.
            - `max_depth` (`float`): The maximum depth value to consider.
            In case only a single configuration is passed, the model will use a single head with the specified configuration.
            In case multiple configurations are passed, the model will use multiple heads with the specified configurations.
        num_patch_transformer_layers (`int`, *optional*):
            The number of transformer layers to use in the patch transformer. Only used in case of multiple bin configurations.
        patch_transformer_hidden_size (`int`, *optional*):
            The hidden size to use in the patch transformer. Only used in case of multiple bin configurations.
        patch_transformer_intermediate_size (`int`, *optional*):
            The intermediate size to use in the patch transformer. Only used in case of multiple bin configurations.
        patch_transformer_num_attention_heads (`int`, *optional*):
            The number of attention heads to use in the patch transformer. Only used in case of multiple bin configurations.

    Example:

    ```python
    >>> from transformers import ZoeDepthConfig, ZoeDepthForDepthEstimation

    >>> # Initializing a ZoeDepth zoedepth-large style configuration
    >>> configuration = ZoeDepthConfig()

    >>> # Initializing a model from the zoedepth-large style configuration
    >>> model = ZoeDepthForDepthEstimation(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "zoedepth"
    sub_configs = {"backbone_config": AutoConfig}

    backbone_config: dict | PreTrainedConfig | None = None
    hidden_act: str = "gelu"
    initializer_range: float = 0.02
    batch_norm_eps: float = 1e-05
    readout_type: Literal["ignore", "add", "project"] = "project"
    reassemble_factors: list[int | float] | tuple[int | float, ...] = (4, 2, 1, 0.5)
    neck_hidden_sizes: list[int] | tuple[int, ...] = (96, 192, 384, 768)
    fusion_hidden_size: int = 256
    head_in_index: int = -1
    use_batch_norm_in_fusion_residual: bool = False
    use_bias_in_fusion_residual: bool | None = None
    num_relative_features: int = 32
    add_projection: bool = False
    bottleneck_features: int = 256
    num_attractors: list[int] | tuple[int, ...] = (16, 8, 4, 1)
    bin_embedding_dim: int = 128
    attractor_alpha: int = 1000
    attractor_gamma: int = 2
    attractor_kind: Literal["mean", "sum"] = "mean"
    min_temp: float = 0.0212
    max_temp: float = 50.0
    bin_centers_type: str = "softplus"
    bin_configurations: list[dict] | None = None
    num_patch_transformer_layers: int | None = None
    patch_transformer_hidden_size: int | None = None
    patch_transformer_intermediate_size: int | None = None
    patch_transformer_num_attention_heads: int | None = None

    def __post_init__(self, **kwargs):
        self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=self.backbone_config,
            default_config_type="beit",
            default_config_kwargs={
                "image_size": 384,
                "num_hidden_layers": 24,
                "hidden_size": 1024,
                "intermediate_size": 4096,
                "num_attention_heads": 16,
                "use_relative_position_bias": True,
                "reshape_hidden_states": False,
                "out_features": ["stage6", "stage12", "stage18", "stage24"],
            },
            **kwargs,
        )
        self.bin_configurations = self.bin_configurations or [{"n_bins": 64, "min_depth": 0.001, "max_depth": 10.0}]

        super().__post_init__(**kwargs)


__all__ = ["ZOEDEPTH_PRETRAINED_CONFIG_ARCHIVE_MAP", "ZoeDepthConfig"]
