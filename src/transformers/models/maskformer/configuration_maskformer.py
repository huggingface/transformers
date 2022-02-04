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

from typing import Dict, List, Optional, Tuple, TypedDict

from ..detr import DetrConfig

from ...configuration_utils import PretrainedConfig
from ...utils import logging


MASKFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = [
    "facebook/maskformer-swin-base-ade-640",
    # See all MaskFormer models at https://huggingface.co/models?filter=maskformer
]

logger = logging.get_logger(__name__)


class ClassSpec(TypedDict):
    is_thing: bool
    label: str
    color: Tuple[int, int, int]


class DatasetMetadata(TypedDict):
    classes: List[ClassSpec]


class MaskFormerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MaskFormer`]. It is used to instantiate a
    MaskFormer model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the "maskformer-swin-base-ade-640"
    architecture trained on ade20k-150

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """
    model_type = "maskformer"

    attribute_map = {"hidden_size": "d_model"}

    backbones_supported = ["swin"]

    def __init__(
        self,
        dataset_metadata: DatasetMetadata = None,
        fpn_feature_size: Optional[int] = 256,
        mask_feature_size: Optional[int] = 256,
        no_object_weight: Optional[float] = 0.1,
        use_auxilary_loss: Optional[bool] = False,

        backbone_config: Optional[Dict] = None,
        # TODO better name?
        transformer_decoder_config: Optional[Dict] = None,
        dice_weight: Optional[float] = 1.0,
        cross_entropy_weight: Optional[float] = 1.0,
        mask_weight: Optional[float] = 20.0,
        mask_classification: Optional[bool] = True,
        **kwargs,
    ):
        from ..auto.configuration_auto import AutoConfig

        if backbone_config is None:
            backbone_model_type = "swin"
            backbone = AutoConfig.from_pretrained("microsoft/swin-base-patch4-window12-384")
        else:
            backbone_model_type = backbone_config.pop("model_type")
            backbone = AutoConfig.for_model(backbone_model_type, **backbone_config)

        if transformer_decoder_config is None:
            # NOTE we have to force detr -> fix it
            transformer_decoder_model_type = "detr"
            transformer_decoder = DetrConfig()

        else:
            # NOTE we have to force detr -> fix it
            transformer_decoder_model_type = transformer_decoder_config.pop("model_type")
            transformer_decoder = DetrConfig(**transformer_decoder_config)

        self.backbone = backbone

        self.transformer_decoder = transformer_decoder

        self.dataset_metadata = dataset_metadata

        self.fpn_feature_size = fpn_feature_size
        self.mask_feature_size = mask_feature_size
        self.no_object_weight = no_object_weight
        self.use_auxilary_loss = use_auxilary_loss

        # Hungarian matcher && loss
        self.cross_entropy_weight = cross_entropy_weight
        self.dice_weight = dice_weight
        self.mask_weight = mask_weight

        self.mask_classification = mask_classification

        super().__init__(**kwargs)

    @classmethod
    def from_backbone_and_transformer_decoder_configs(
        cls, backbone_config: PretrainedConfig, transformer_decode_config: DetrConfig, **kwargs
    ) -> PretrainedConfig:
        r"""
        Instantiate a [`EncoderDecoderConfig`] (or a derived class) from a pre-trained encoder model configuration and
        decoder model configuration.
        Returns:
            [`EncoderDecoderConfig`]: An instance of a configuration object
        """
        return cls(backbone_config=backbone_config.to_dict(), generator=transformer_decode_config.to_dict(), **kwargs)
