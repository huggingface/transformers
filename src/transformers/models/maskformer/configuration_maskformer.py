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

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TypedDict

from ...configuration_utils import PretrainedConfig
from ...utils import logging


MASKFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    # "facebook/detr-resnet-50": "https://huggingface.co/facebook/detr-resnet-50/resolve/main/config.json",
    # See all MaskFormer models at https://huggingface.co/models?filter=detr
}

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
    configuration with the defaults will yield a similar configuration to that of the "askformer-swin-base-640"
    architecture trained on ade20k-150

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """
    model_type = "maskformer"

    attribute_map = {"hidden_size": "d_model"}

    def __init__(
        self,
        dataset_metadata: DatasetMetadata = None,
        fpn_feature_size: Optional[int] = 256,
        mask_feature_size: Optional[int] = 256,
        no_object_weight: Optional[float] = 0.1,
        use_auxilary_loss: Optional[bool] = False,
        num_queries: Optional[int] = 100,
        swin_image_size: Optional[int] = 384,
        swin_in_channels: Optional[int] = 3,
        swin_patch_size: Optional[int] = 4,
        swin_embed_dim: Optional[int] = 128,
        swin_depths: Optional[List[int]] = [2, 2, 18, 2],
        swin_num_heads: Optional[List[int]] = [4, 8, 16, 32],
        swin_window_size: Optional[int] = 12,
        swin_drop_path_rate: Optional[float] = 0.3,
        dice_weight: Optional[float] = 1.0,
        cross_entropy_weight: Optional[float] = 1.0,
        mask_weight: Optional[float] = 20.0,
        mask_classification: Optional[bool] = True,
        detr_max_position_embeddings: Optional[int] = 1024,
        detr_encoder_layers: Optional[int] = 6,
        detr_encoder_ffn_dim: Optional[int] = 2048,
        detr_encoder_attention_heads: Optional[int] = 8,
        detr_decoder_layers: Optional[int] = 6,
        detr_decoder_ffn_dim: Optional[int] = 2048,
        detr_decoder_attention_heads: Optional[int] = 8,
        detr_encoder_layerdrop: Optional[int] = 0.0,
        detr_decoder_layerdrop: Optional[int] = 0.0,
        detr_d_model: Optional[int] = 256,
        detr_dropout: Optional[int] = 0.1,
        detr_attention_dropout: Optional[int] = 0.0,
        detr_activation_dropout: Optional[int] = 0.0,
        detr_init_std: Optional[int] = 0.02,
        detr_init_xavier_std: Optional[int] = 1.0,
        detr_scale_embedding: Optional[int] = False,
        detr_auxiliary_loss: Optional[int] = False,
        detr_dilation: Optional[int] = False,
        **kwargs,
    ):
        self.dataset_metadata = dataset_metadata

        self.fpn_feature_size = fpn_feature_size
        self.mask_feature_size = mask_feature_size
        self.num_queries = num_queries
        self.no_object_weight = no_object_weight
        self.use_auxilary_loss = use_auxilary_loss
        # swin backbone parameters
        self.swin_image_size = swin_image_size
        self.swin_in_channels = swin_in_channels
        self.swin_patch_size = swin_patch_size
        self.swin_embed_dim = swin_embed_dim
        self.swin_depths = swin_depths
        self.swin_num_heads = swin_num_heads
        self.swin_window_size = swin_window_size
        self.swin_drop_path_rate = swin_drop_path_rate
        # Hungarian matcher && loss
        self.cross_entropy_weight = cross_entropy_weight
        self.dice_weight = dice_weight
        self.mask_weight = mask_weight

        self.mask_classification = mask_classification
        # DETR parameters
        self.detr_max_position_embeddings = detr_max_position_embeddings
        self.detr_d_model = detr_d_model
        self.detr_encoder_ffn_dim = detr_encoder_ffn_dim
        self.detr_encoder_layers = detr_encoder_layers
        self.detr_encoder_attention_heads = detr_encoder_attention_heads
        self.detr_decoder_ffn_dim = detr_decoder_ffn_dim
        self.detr_decoder_layers = detr_decoder_layers
        self.detr_decoder_attention_heads = detr_decoder_attention_heads
        self.detr_dropout = detr_dropout
        self.detr_attention_dropout = detr_attention_dropout
        self.detr_activation_dropout = detr_activation_dropout
        self.detr_init_std = detr_init_std
        self.detr_init_xavier_std = detr_init_xavier_std
        self.detr_encoder_layerdrop = detr_encoder_layerdrop
        self.detr_decoder_layerdrop = detr_decoder_layerdrop
        self.detr_num_hidden_layers = detr_encoder_layers
        self.detr_scale_embedding = detr_scale_embedding  # scale factor will be sqrt(d_model) if True
        self.detr_auxiliary_loss = detr_auxiliary_loss
        self.detr_dilation = detr_dilation

        super().__init__(**kwargs)

    @property
    def hidden_size(self) -> int:
        return self.detr_d_model

    @property
    def d_model(self) -> int:
        return self.detr_d_model
