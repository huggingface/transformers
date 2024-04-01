# coding=utf-8
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
""" ZoeDepth model configuration"""

import copy

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto.configuration_auto import CONFIG_MAPPING


logger = logging.get_logger(__name__)

ZOEDEPTH_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "Intel/zoedepth-base": "https://huggingface.co/Intel/zoedepth-base/resolve/main/config.json",
}


class ZoeDepthConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ZoeDepthForDepthEstimation`]. It is used to instantiate an ZoeDepth
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the ZoeDepth
    [Intel/zoedepth-base](https://huggingface.co/Intel/zoedepth-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        image_size (`int`, *optional*, defaults to 384):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        readout_type (`str`, *optional*, defaults to `"project"`):
            The readout type to use when processing the readout token (CLS token) of the intermediate hidden states of
            the ViT backbone. Can be one of [`"ignore"`, `"add"`, `"project"`].

            - "ignore" simply ignores the CLS token.
            - "add" passes the information from the CLS token to all other tokens by adding the representations.
            - "project" passes information to the other tokens by concatenating the readout to all other tokens before
              projecting the
            representation to the original feature dimension D using a linear layer followed by a GELU non-linearity.
        reassemble_factors (`List[int]`, *optional*, defaults to `[4, 2, 1, 0.5]`):
            The up/downsampling factors of the reassemble layers.
        neck_hidden_sizes (`List[str]`, *optional*, defaults to `[96, 192, 384, 768]`):
            The hidden sizes to project to for the feature maps of the backbone.
        fusion_hidden_size (`int`, *optional*, defaults to 256):
            The number of channels before fusion.
        head_in_index (`int`, *optional*, defaults to -1):
            The index of the features to use in the heads.
        use_batch_norm_in_fusion_residual (`bool`, *optional*, defaults to `False`):
            Whether to use batch normalization in the pre-activate residual units of the fusion blocks.
        use_bias_in_fusion_residual (`bool`, *optional*, defaults to `True`):
            Whether to use bias in the pre-activate residual units of the fusion blocks.
        add_projection (`bool`, *optional*, defaults to `False`):
            Whether to add a projection layer before the depth estimation head.
        neck_ignore_stages (`List[int]`, *optional*, defaults to `[0, 1]`):
            Used only for the `hybrid` embedding type. The stages of the readout layers to ignore.
        backbone_config (`Union[Dict[str, Any], PretrainedConfig]`, *optional*):
            The configuration of the backbone model. Only used in case `is_hybrid` is `True` or in case you want to
            leverage the [`AutoBackbone`] API.
        backbone (`str`, *optional*):
            Name of backbone to use when `backbone_config` is `None`. If `use_pretrained_backbone` is `True`, this
            will load the corresponding pretrained weights from the timm or transformers library. If `use_pretrained_backbone`
            is `False`, this loads the backbone's config and uses that to initialize the backbone with random weights.
        use_pretrained_backbone (`bool`, *optional*, defaults to `False`):
            Whether to use pretrained weights for the backbone.
        bottleneck_features (`int`, *optional*, defaults to 256):
            The number of features in the bottleneck layer.
        n_bins (`int`, *optional*, defaults to 64):
            The number of bins to use in the metric depth estimation head.
        min_depth (`float`, *optional*, defaults to 0.001):
            The minimum depth value to consider.
        max_depth (`float`, *optional*, defaults to 10):
            The maximum depth value to consider.
        num_attractors (`List[int], *optional*, defaults to `[16, 8, 4, 1]`):
            The number of attractors to use in each stage.
        bin_embedding_dim (`int`, *optional*, defaults to 128):
            The dimension of the bin embeddings.
        num_out_features (`List[int]`, *optional*, defaults to `[256, 256, 256, 256]`):
            The number of output features for each stage.
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

    def __init__(
        self,
        hidden_act="gelu",
        initializer_range=0.02,
        image_size=384,
        patch_size=16,
        readout_type="project",
        reassemble_factors=[4, 2, 1, 0.5],
        neck_hidden_sizes=[96, 192, 384, 768],
        fusion_hidden_size=256,
        head_in_index=-1,
        use_batch_norm_in_fusion_residual=False,
        use_bias_in_fusion_residual=None,
        add_projection=False,
        neck_ignore_stages=[0, 1],
        backbone_config=None,
        backbone=None,
        use_pretrained_backbone=False,
        bottleneck_features=256,
        n_bins=64,
        min_depth=0.001,
        max_depth=10,
        num_attractors=[16, 8, 4, 1],
        bin_embedding_dim=128,
        num_out_features=[256, 256, 256, 256],
        attractor_alpha=1000,
        attractor_gamma=2,
        attractor_kind="mean",
        min_temp=0.0212,
        max_temp=50.0,
        bin_centers_type="softplus",
        **kwargs,
    ):
        super().__init__(**kwargs)

        if use_pretrained_backbone:
            raise ValueError("Pretrained backbones are not supported yet.")

        if backbone_config is not None and backbone is not None:
            raise ValueError("You can't specify both `backbone` and `backbone_config`.")

        use_autobackbone = False
        if backbone_config is not None:
            use_autobackbone = True

            if isinstance(backbone_config, dict):
                backbone_model_type = backbone_config.get("model_type")
                config_class = CONFIG_MAPPING[backbone_model_type]
                backbone_config = config_class.from_dict(backbone_config)

            self.backbone_config = backbone_config
            self.neck_ignore_stages = []

        else:
            self.backbone_config = backbone_config
            self.neck_ignore_stages = []

        self.backbone = backbone
        self.hidden_act = hidden_act
        self.use_pretrained_backbone = use_pretrained_backbone
        self.image_size = None if use_autobackbone else image_size
        self.patch_size = None if use_autobackbone else patch_size

        if readout_type not in ["ignore", "add", "project"]:
            raise ValueError("Readout_type must be one of ['ignore', 'add', 'project']")
        self.initializer_range = initializer_range
        self.readout_type = readout_type
        self.reassemble_factors = reassemble_factors
        self.neck_hidden_sizes = neck_hidden_sizes
        self.fusion_hidden_size = fusion_hidden_size
        self.head_in_index = head_in_index
        self.use_batch_norm_in_fusion_residual = use_batch_norm_in_fusion_residual
        self.use_bias_in_fusion_residual = use_bias_in_fusion_residual
        self.add_projection = add_projection

        self.bottleneck_features = bottleneck_features
        self.n_bins = n_bins
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.num_attractors = num_attractors
        self.bin_embedding_dim = bin_embedding_dim
        self.num_out_features = num_out_features
        self.attractor_alpha = attractor_alpha
        self.attractor_gamma = attractor_gamma
        self.attractor_kind = attractor_kind
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.bin_centers_type = bin_centers_type

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`]. Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)

        if output["backbone_config"] is not None:
            output["backbone_config"] = self.backbone_config.to_dict()

        output["model_type"] = self.__class__.model_type
        return output
