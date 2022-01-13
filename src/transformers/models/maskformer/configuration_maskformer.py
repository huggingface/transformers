# coding=utf-8
# Copyright 2021 Facebook AI Research and The HuggingFace Inc. team. All rights reserved.
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

from typing import List, Optional

from ...configuration_utils import PretrainedConfig
from ...utils import logging


MASKFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    # "facebook/detr-resnet-50": "https://huggingface.co/facebook/detr-resnet-50/resolve/main/config.json",
    # See all MaskFormer models at https://huggingface.co/models?filter=detr
}

logger = logging.get_logger(__name__)


class MaskFormerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MaskFormer`]. It is used to instantiate a MaskFormer
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the <TODO!!> "maskformer_panoptic_swin_base_IN21k_384_bs64_554k" architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """
    model_type = "mask_former"

    attribute_map = {"hidden_size": "d_model"}

    def __init__(
        self,
        fpn_feature_size: Optional[int] = 256,
        mask_feature_size: Optional[int] = 256,
        num_classes: Optional[int] = 133,
        no_object_weight: Optional[float] = 0.1,
        num_queries: Optional[int] = 100,
        swin_pretrain_img_size: Optional[int] = 384,
        swin_in_channels: Optional[int] = 3,
        swin_patch_size: Optional[int] = 4,
        swin_embed_dim: Optional[int] = 128,
        swin_depths: Optional[List[int]] = None,
        swin_num_heads: Optional[List[int]] = None,
        swin_window_size: Optional[int] = 12,
        swin_drop_path_rate: Optional[float] = 0.3,
        dice_weight: Optional[float] = 1.0,
        ce_weight: Optional[float] = 1.0,
        mask_weight: Optional[float] = 20.0,
        mask_classification: Optional[bool] = True,
        max_position_embeddings: Optional[int] = 1024,
        encoder_layers: Optional[int] = 6,
        encoder_ffn_dim: Optional[int] = 2048,
        encoder_attention_heads: Optional[int] = 8,
        decoder_layers: Optional[int] = 6,
        decoder_ffn_dim: Optional[int] = 2048,
        decoder_attention_heads: Optional[int] = 8,
        encoder_layerdrop: Optional[int] = 0.0,
        decoder_layerdrop: Optional[int] = 0.0,
        d_model: Optional[int] = 256,
        dropout: Optional[int] = 0.1,
        attention_dropout: Optional[int] = 0.0,
        activation_dropout: Optional[int] = 0.0,
        init_std: Optional[int] = 0.02,
        init_xavier_std: Optional[int] = 1.0,
        scale_embedding: Optional[int] = False,
        auxiliary_loss: Optional[int] = False,
        dilation: Optional[int] = False,
        **kwargs
    ):

        self.fpn_feature_size = fpn_feature_size
        self.mask_feature_size = mask_feature_size
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.no_object_weight = no_object_weight
        # swin backbone parameters
        self.swin_pretrain_img_size = swin_pretrain_img_size
        self.swin_in_channels = swin_in_channels
        self.swin_patch_size = swin_patch_size
        self.swin_embed_dim = swin_embed_dim
        self.swin_depths = swin_depths or [2, 2, 18, 2]
        self.swin_num_heads = swin_num_heads or [4, 8, 16, 32]
        self.swin_window_size = swin_window_size
        self.swin_drop_path_rate = swin_drop_path_rate
        # Hungarian matcher && loss
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.mask_weight = mask_weight

        self.mask_classification = mask_classification
        # DETR parameters
        self.detr_max_position_embeddings = max_position_embeddings
        self.detr_d_model = d_model
        self.detr_encoder_ffn_dim = encoder_ffn_dim
        self.detr_encoder_layers = encoder_layers
        self.detr_encoder_attention_heads = encoder_attention_heads
        self.detr_decoder_ffn_dim = decoder_ffn_dim
        self.detr_decoder_layers = decoder_layers
        self.detr_decoder_attention_heads = decoder_attention_heads
        self.detr_dropout = dropout
        self.detr_attention_dropout = attention_dropout
        self.detr_activation_dropout = activation_dropout
        self.detr_init_std = init_std
        self.detr_init_xavier_std = init_xavier_std
        self.detr_encoder_layerdrop = encoder_layerdrop
        self.detr_decoder_layerdrop = decoder_layerdrop
        self.detr_num_hidden_layers = encoder_layers
        self.detr_scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        self.detr_auxiliary_loss = auxiliary_loss
        self.detr_dilation = dilation

        super().__init__(is_encoder_decoder=True, **kwargs)

    @property
    def hidden_size(self) -> int:
        return self.detr_d_model

    @property
    def d_model(self) -> int:
        # DETR needs `.d_model`
        return self.detr_d_model
