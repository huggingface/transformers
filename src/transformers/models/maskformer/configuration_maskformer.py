# coding=utf-8
# Copyright 2022 Facebook AI Research and The HuggingFace Inc. team. All rights reserved.
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
""" MaskFormer model configuration"""
from __future__ import annotations

from typing import Dict, Optional

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto.configuration_auto import AutoConfig
from ..detr import DetrConfig
from ..swin import SwinConfig


MASKFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = [
    "facebook/maskformer-swin-base-ade-640",
    # See all MaskFormer models at https://huggingface.co/models?filter=maskformer
]

logger = logging.get_logger(__name__)


class MaskFormerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MaskFormer`]. It is used to instantiate a
    MaskFormer model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the "maskformer-swin-base-ade-640"
    architecture trained on ade20k-150

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        dataset_metadata (DatasetMetadata, optional): [description]. Defaults to None.
        mask_feature_size (Optional[int], optional):
            The masks' features size, this value will also be used to specify the Feature Pyramid Network features
            size. Defaults to 256.
        no_object_weight (Optional[float], optional): Weight to apply to the null class . Defaults to 0.1.
        use_auxilary_loss (Optional[bool], optional): If `true` [`MaskFormerOutput`] will contain. Defaults to False.
        backbone_config (Optional[Dict], optional): [description]. Defaults to None.
        detr_config (Optional[Dict], optional): [description]. Defaults to None.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        init_xavier_std (`float`, *optional*, defaults to 1):
            The scaling factor used for the Xavier initialization gain in the HM Attention map module.
        dice_weight (Optional[float], optional): [description]. Defaults to 1.0.
        cross_entropy_weight (Optional[float], optional): [description]. Defaults to 1.0.
        mask_weight (Optional[float], optional): [description]. Defaults to 20.0.

    Raises:
        ValueError: Raised if the backbone model type selected is not in `MaskFormerConfig.backbones_supported`

    Examples:

    ```python
    >>> from transformers import MaskFormerModel, MaskFormerConfig

    >>> # Initializing a maskFormer facebook/maskformer-swin-base-ade-640 configuration
    >>> configuration = MaskFormerConfig()

    >>> # Initializing a model from the facebook/maskformer-swin-base-ade-640 style configuration
    >>> model = MaskFormerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

    """
    model_type = "maskformer"

    attribute_map = {"hidden_size": "d_model"}

    backbones_supported = ["swin"]

    def __init__(
        self,
        fpn_feature_size: Optional[int] = 256,
        mask_feature_size: Optional[int] = 256,
        no_object_weight: Optional[float] = 0.1,
        use_auxilary_loss: Optional[bool] = False,
        backbone_config: Optional[Dict] = None,
        detr_config: Optional[Dict] = None,
        init_std: float = 0.02,
        init_xavier_std: float = 1.0,
        dice_weight: Optional[float] = 1.0,
        cross_entropy_weight: Optional[float] = 1.0,
        mask_weight: Optional[float] = 20.0,
        num_labels: Optional[int] = 150,
        **kwargs,
    ):
        if backbone_config is None:
            # fall back to https://huggingface.co/microsoft/swin-base-patch4-window12-384-in22k
            backbone = SwinConfig(
                image_size=384,
                in_channels=3,
                patch_size=4,
                embed_dim=128,
                depths=[2, 2, 18, 2],
                num_heads=[4, 8, 16, 32],
                window_size=12,
                drop_path_rate=0.3,
            )
        else:
            backbone_model_type = backbone_config.pop("model_type")
            if backbone_model_type not in self.backbones_supported:
                raise ValueError(
                    f"Backbone {backbone_model_type} not supported, please use one of {','.join(self.backbones_supported)}"
                )
            backbone = AutoConfig.for_model(backbone_model_type, **backbone_config)

        if detr_config is None:
            transformer_decoder = DetrConfig()

        else:
            transformer_decoder = DetrConfig(**detr_config)

        self.backbone = backbone

        self.transformer_decoder = transformer_decoder

        self.fpn_feature_size = fpn_feature_size
        self.mask_feature_size = mask_feature_size
        self.no_object_weight = no_object_weight
        self.use_auxilary_loss = use_auxilary_loss
        # initializer
        self.init_std = init_std
        self.init_xavier_std = init_xavier_std
        # Hungarian matcher && loss
        self.cross_entropy_weight = cross_entropy_weight
        self.dice_weight = dice_weight
        self.mask_weight = mask_weight

        super().__init__(num_labels=num_labels, **kwargs)

    @classmethod
    def from_backbone_and_detr_configs(
        cls, backbone_config: PretrainedConfig, detr_config: DetrConfig, **kwargs
    ) -> MaskFormerConfig:
        """Instantiate a [`MaskFormerConfig`] (or a derived class) from a pre-trained backbone model configuration and DETR model
configuration.

        Args:
            backbone_config (PretrainedConfig): The backbone configuration
            detr_config (DetrConfig): The transformer decoder configuration to use

        Returns:
            [`MaskFormerConfig`]: An instance of a configuration object
        """
        return cls(
            backbone_config=backbone_config.to_dict(),
            detr_config=detr_config.to_dict(),
            **kwargs,
        )
