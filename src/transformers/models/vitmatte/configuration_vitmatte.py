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
"""VitMatte model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...modeling_backbone_utils import consolidate_backbone_kwargs_to_config
from ...utils import logging
from ..auto.configuration_auto import AutoConfig


logger = logging.get_logger(__name__)


class VitMatteConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of [`VitMatteForImageMatting`]. It is used to
    instantiate a ViTMatte model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the ViTMatte
    [hustvl/vitmatte-small-composition-1k](https://huggingface.co/hustvl/vitmatte-small-composition-1k) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        backbone_config (`Union[dict, "PreTrainedConfig"]`, *optional*, defaults to `VitDetConfig()`):
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
        hidden_size (`int`, *optional*, defaults to 384):
            The number of input channels of the decoder.
        batch_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the batch norm layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        convstream_hidden_sizes (`list[int]`, *optional*, defaults to `[48, 96, 192]`):
            The output channels of the ConvStream module.
        fusion_hidden_sizes (`list[int]`, *optional*, defaults to `[256, 128, 64, 32]`):
            The output channels of the Fusion blocks.

    Example:

    ```python
    >>> from transformers import VitMatteConfig, VitMatteForImageMatting

    >>> # Initializing a ViTMatte hustvl/vitmatte-small-composition-1k style configuration
    >>> configuration = VitMatteConfig()

    >>> # Initializing a model (with random weights) from the hustvl/vitmatte-small-composition-1k style configuration
    >>> model = VitMatteForImageMatting(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "vitmatte"
    sub_configs = {"backbone_config": AutoConfig}

    def __init__(
        self,
        backbone_config: PreTrainedConfig | None = None,
        backbone=None,
        hidden_size: int = 384,
        batch_norm_eps: float = 1e-5,
        initializer_range: float = 0.02,
        convstream_hidden_sizes: list[int] = [48, 96, 192],
        fusion_hidden_sizes: list[int] = [256, 128, 64, 32],
        **kwargs,
    ):
        backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=backbone_config,
            backbone=backbone,
            default_config_type="vitdet",
            default_config_kwargs={
                "out_features": ["stage4"],
            },
            **kwargs,
        )

        self.backbone_config = backbone_config
        self.batch_norm_eps = batch_norm_eps
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.convstream_hidden_sizes = convstream_hidden_sizes
        self.fusion_hidden_sizes = fusion_hidden_sizes

        super().__init__(**kwargs)


__all__ = ["VitMatteConfig"]
