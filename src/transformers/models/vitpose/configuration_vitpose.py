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
"""VitPose model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...modeling_backbone_utils import consolidate_backbone_kwargs_to_config
from ...utils import logging
from ..auto.configuration_auto import AutoConfig


logger = logging.get_logger(__name__)


class VitPoseConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`VitPoseForPoseEstimation`]. It is used to instantiate a
    VitPose model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the VitPose
    [usyd-community/vitpose-base-simple](https://huggingface.co/usyd-community/vitpose-base-simple) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        backbone_config (`Union[dict, "PreTrainedConfig"]`, *optional*, defaults to `VitPoseBackboneConfig()`):
            The configuration of the backbone model. Currently, only `backbone_config` with `vitpose_backbone` as `model_type` is supported.
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
            Whether to use a `VitPoseSimpleDecoder` to decode the feature maps from the backbone into heatmaps. Otherwise it uses `VitPoseClassicDecoder`.


    Example:

    ```python
    >>> from transformers import VitPoseConfig, VitPoseForPoseEstimation

    >>> # Initializing a VitPose configuration
    >>> configuration = VitPoseConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = VitPoseForPoseEstimation(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "vitpose"
    sub_configs = {"backbone_config": AutoConfig}

    def __init__(
        self,
        backbone_config: PreTrainedConfig | None = None,
        initializer_range: float = 0.02,
        scale_factor: int = 4,
        use_simple_decoder: bool = True,
        **kwargs,
    ):
        backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=backbone_config,
            default_config_type="vitpose_backbone",
            default_config_kwargs={"out_indices": [4]},
            **kwargs,
        )

        self.backbone_config = backbone_config
        self.initializer_range = initializer_range
        self.scale_factor = scale_factor
        self.use_simple_decoder = use_simple_decoder

        super().__init__(**kwargs)


__all__ = ["VitPoseConfig"]
