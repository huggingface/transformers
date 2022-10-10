# coding=utf-8
# Copyright 2022 Meta Platforms, Inc.and The HuggingFace Inc. team. All rights reserved.
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
""" Mask2Former model configuration"""
import copy
from typing import Dict, List, Optional

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto.configuration_auto import AutoConfig
from ..detr import DetrConfig
from ..swin import SwinConfig
from ..deformable_detr import DeformableDetrConfig

MASK2FORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "shivi/mask2former-instance-swin-small-coco": "https://huggingface.co/shivi/mask2former-instance-swin-small-coco/resolve/main/config.json",
}


logger = logging.get_logger(__name__)


class Mask2FormerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Mask2FormerModel`]. It is used to instantiate a
    Mask2Former model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Mask2Former
    [shivi/mask2former-instance-swin-small-coco](https://huggingface.co/shivi/mask2former-instance-swin-small-coco) architecture trained
    on [ADE20k-150](https://huggingface.co/datasets/scene_parse_150).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Currently, Mask2Former only supports the [Swin Transformer](swin) as backbone.

    Args:
        mask_feature_size (`int`, *optional*, defaults to 256):
            The masks' features size, this value will also be used to specify the Feature Pyramid Network features'
            size.
        no_object_weight (`float`, *optional*, defaults to 0.1):
            Weight to apply to the null (no object) class.
        use_auxiliary_loss(`bool`, *optional*, defaults to `False`):
            If `True` [`Mask2FormerForInstanceSegmentationOutput`] will contain the auxiliary losses computed using the
            logits from each decoder's stage.
        backbone_config (`Dict`, *optional*):
            The configuration passed to the backbone, if unset, the configuration corresponding to
            `swin-base-patch4-window12-384` will be used.
        decoder_config (`Dict`, *optional*):
            The configuration passed to the transformer decoder model, if unset the base config for `detr-resnet-50`
            will be used.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        init_xavier_std (`float`, *optional*, defaults to 1):
            The scaling factor used for the Xavier initialization gain in the HM Attention map module.
        dice_weight (`float`, *optional*, defaults to 1.0):
            The weight for the dice loss.
        cross_entropy_weight (`float`, *optional*, defaults to 1.0):
            The weight for the cross entropy loss.
        mask_weight (`float`, *optional*, defaults to 20.0):
            The weight for the mask loss.
        output_auxiliary_logits (`bool`, *optional*):
            Should the model output its `auxiliary_logits` or not.
        train_num_points (`int`, *optional*, defaults to 12544):
            Number of points used for sampling during loss calculation.
        importance_sample_ratio (`float`, *optional*, defaults to 0.75):
            Ratio of points that are sampled via importance sampling.
        oversample_ratio (`float`, *optional*, defaults to 3.0):
            Oversampling parameter used for calculating no. of sampled points
        pixel_decoder_config (`Dict`, *optional*):
            The configuration passed to the pixel decoder module, if unset, the configuration corresponding to
            `deformable-detr` will be used.
        common_stride (`int`, *optional*, defaults to 4):
            parameter used for determining number of FPN levels used as part of pixel decoder
        feature_strides (`List`, *optional*, defaults to [4, 8, 16, 32]):
            feature strides corresponding to features generated from backbone network
        mask2former_num_feature_levels (`int`, *optional*, defaults to 3):
            Number of feature levels for Mask2former model
    Raises:
        `ValueError`:
            Raised if the backbone model type selected is not in `["swin"]` or the decoder model type selected is not
            in `["detr"]` or the pixel decoder type selected is not in `["deformable_detr"]`

    Examples:

    ```python
    >>> from transformers import Mask2FormerConfig, Mask2FormerModel

    >>> # Initializing a Mask2Former shivi/mask2former-instance-swin-small-coco configuration
    >>> configuration = Mask2FormerConfig()

    >>> # Initializing a model from the shivi/mask2former-instance-swin-small-coco style configuration
    >>> model = Mask2FormerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

    """
    model_type = "maskformer" #or mask2former?
    attribute_map = {"hidden_size": "mask_feature_size"}
    backbones_supported = ["swin"]
    decoders_supported = ["detr"]
    pixel_decoders_supported = ["deformable_detr"]

    def __init__(
        self,
        feature_size: int = 256,
        mask_feature_size: int = 256,
        no_object_weight: float = 0.1,
        use_auxiliary_loss: bool = False,
        backbone_config: Optional[Dict] = None,
        decoder_config: Optional[Dict] = None,
        init_std: float = 0.02,
        init_xavier_std: float = 1.0,
        dice_weight: float = 1.0,
        cross_entropy_weight: float = 1.0,
        mask_weight: float = 20.0,
        output_auxiliary_logits: Optional[bool] = None,
        train_num_points: Optional[int] = 12544,
        importance_sample_ratio: Optional[float] = 0.75,
        oversample_ratio: Optional[float] = 3.0,
        pixel_decoder_config: Optional[Dict] = None,
        common_stride: Optional[int] = 4,
        feature_strides: Optional[List[int]] = [4, 8, 16, 32],
        mask2former_num_feature_levels: Optional[int] = 3,
        **kwargs,
    ):
        if backbone_config is None:
            # fall back to https://huggingface.co/microsoft/swin-base-patch4-window12-384-in22k
            backbone_config = SwinConfig(
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
                    f"Backbone {backbone_model_type} not supported, please use one of"
                    f" {','.join(self.backbones_supported)}"
                )
            backbone_config = AutoConfig.for_model(backbone_model_type, **backbone_config)

        if decoder_config is None:
            # fall back to https://huggingface.co/facebook/detr-resnet-50
            decoder_config = DetrConfig()
        else:
            decoder_type = decoder_config.pop("model_type")
            if decoder_type not in self.decoders_supported:
                raise ValueError(
                    f"Transformer Decoder {decoder_type} not supported, please use one of"
                    f" {','.join(self.decoders_supported)}"
                )
            decoder_config = AutoConfig.for_model(decoder_type, **decoder_config)

        if pixel_decoder_config is None:
            # fall back to https://huggingface.co/sensetime/deformable-detr
            pixel_decoder_config = DeformableDetrConfig()
        else:
            pixel_decoder_type = pixel_decoder_config.pop("model_type")
            if pixel_decoder_type not in self.pixel_decoders_supported:
                raise ValueError(
                    f"Pixel Decoder {pixel_decoder_type} not supported, please use one of"
                    f" {'.'.join(self.pixel_decoders_supported)}"
                )
            pixel_decoder_config = AutoConfig.for_model(pixel_decoder_type, **pixel_decoder_config)

        self.backbone_config = backbone_config
        self.decoder_config = decoder_config
        # main feature dimension for the model
        self.feature_size = feature_size
        self.mask_feature_size = mask_feature_size
        # initializer
        self.init_std = init_std
        self.init_xavier_std = init_xavier_std
        # Hungarian matcher && loss
        self.cross_entropy_weight = cross_entropy_weight
        self.dice_weight = dice_weight
        self.mask_weight = mask_weight
        self.use_auxiliary_loss = use_auxiliary_loss
        self.no_object_weight = no_object_weight
        self.output_auxiliary_logits = output_auxiliary_logits
        self.train_num_points = train_num_points
        self.importance_sample_ratio = importance_sample_ratio
        self.oversample_ratio = oversample_ratio
        ##Pixel Decoder Config
        self.pixel_decoder_config = pixel_decoder_config
        self.pixel_decoder_config.feature_strides = feature_strides
        self.pixel_decoder_config.common_stride = common_stride
        self.pixel_decoder_config.mask2former_num_feature_levels = mask2former_num_feature_levels

        self.num_attention_heads = self.decoder_config.encoder_attention_heads
        self.num_hidden_layers = self.decoder_config.num_hidden_layers
        super().__init__(**kwargs)

    @classmethod
    def from_backbone_decoder_pixel_decoder_configs(
        cls, backbone_config: PretrainedConfig, decoder_config: PretrainedConfig, pixel_decoder_config: PretrainedConfig,  **kwargs
    ):
        """Instantiate a [`Mask2FormerConfig`] (or a derived class) from a pre-trained backbone model configuration, DETR model
        configuration and Deformable Detr model configuration.

            Args:
                backbone_config ([`PretrainedConfig`]):
                    The backbone configuration.
                decoder_config ([`PretrainedConfig`]):
                    The transformer decoder configuration to use.
                pixel_decoder_config ([`PretrainedConfig`]):
                    The pixel decoder configuration to use.

            Returns:
                [`Mask2FormerConfig`]: An instance of a configuration object
        """
        return cls(
            backbone_config=backbone_config.to_dict(),
            decoder_config=decoder_config.to_dict(),
            pixel_decoder_config=pixel_decoder_config.to_dict(),
            **kwargs,
        )

    def to_dict(self) -> Dict[str, any]:
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["backbone_config"] = self.backbone_config.to_dict()
        output["decoder_config"] = self.decoder_config.to_dict()
        output["pixel_decoder_config"] = self.pixel_decoder_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
