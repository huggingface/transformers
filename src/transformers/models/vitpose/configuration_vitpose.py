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
"""ViTPose model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto.configuration_auto import CONFIG_MAPPING


logger = logging.get_logger(__name__)


class ViTPoseConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ViTPoseForPoseEstimation`]. It is used to instantiate a
    ViTPose model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the ViTPose
    [google/vitpose-base-patch16-224](https://huggingface.co/google/vitpose-base-patch16-224) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        backbone_config (`PretrainedConfig` or `dict`, *optional*, defaults to `VitPoseBackboneConfig()`):
            The configuration of the backbone model.
        backbone (`str`, *optional*):
            Name of backbone to use when `backbone_config` is `None`. If `use_pretrained_backbone` is `True`, this
            will load the corresponding pretrained weights from the timm or transformers library. If `use_pretrained_backbone`
            is `False`, this loads the backbone's config and uses that to initialize the backbone with random weights.
        use_pretrained_backbone (`bool`, *optional*, defaults to `False`):
            Whether to use pretrained weights for the backbone.
        use_timm_backbone (`bool`, *optional*, defaults to `False`):
            Whether to load `backbone` from the timm library. If `False`, the backbone is loaded from the transformers
            library.
        backbone_kwargs (`dict`, *optional*):
            Keyword arguments to be passed to AutoBackbone when loading from a checkpoint
            e.g. `{'out_indices': (0, 1, 2, 3)}`. Cannot be specified if `backbone_config` is set.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        scale_factor (`int`, *optional*, defaults to 4):
            Factor to upscale the feature maps coming from the ViT backbone.
        use_simple_decoder (`bool`, *optional*, defaults to `True`):
            Whether to use a simple decoder to decode the feature maps from the backbone into heatmaps.

    Example:

    ```python
    >>> from transformers import ViTPoseConfig, ViTPoseForPoseEstimation

    >>> # Initializing a ViTPose configuration
    >>> configuration = ViTPoseConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = ViTPoseForPoseEstimation(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "vitpose"

    def __init__(
        self,
        backbone_config: PretrainedConfig = None,
        backbone: str = None,
        use_pretrained_backbone: bool = False,
        use_timm_backbone: bool = False,
        backbone_kwargs: dict = None,
        initializer_range: float = 0.02,
        scale_factor: int = 4,
        use_simple_decoder: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if use_pretrained_backbone:
            raise ValueError("Pretrained backbones are not supported yet.")

        if backbone_config is not None and backbone is not None:
            raise ValueError("You can't specify both `backbone` and `backbone_config`.")

        if backbone_config is None and backbone is None:
            logger.info("`backbone_config` is `None`. Initializing the config with the default `VitPose` backbone.")
            backbone_config = CONFIG_MAPPING["vitpose_backbone"](out_features=["stage4"])
        elif isinstance(backbone_config, dict):
            backbone_model_type = backbone_config.get("model_type")
            config_class = CONFIG_MAPPING[backbone_model_type]
            backbone_config = config_class.from_dict(backbone_config)

        if backbone_kwargs is not None and backbone_kwargs and backbone_config is not None:
            raise ValueError("You can't specify both `backbone_kwargs` and `backbone_config`.")

        self.backbone_config = backbone_config
        self.backbone = backbone
        self.use_pretrained_backbone = use_pretrained_backbone
        self.use_timm_backbone = use_timm_backbone
        self.backbone_kwargs = backbone_kwargs

        self.initializer_range = initializer_range
        self.scale_factor = scale_factor
        self.use_simple_decoder = use_simple_decoder
