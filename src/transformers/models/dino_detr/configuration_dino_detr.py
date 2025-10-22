# coding=utf-8
# Copyright 2022 SenseTime and The HuggingFace Inc. team. All rights reserved.
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
"""Dino DETR model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ...utils.backbone_utils import verify_backbone_config_arguments
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


class DinoDetrConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DinoDetrModel`]. It is used to instantiate
    a Dino DETR model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Dino DETR
    [SenseTime/deformable-detr](https://huggingface.co/SenseTime/deformable-detr) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
            is_encoder_decoder (`<fill_type>`, *optional*, defaults to `True`): <fill_docstring>
            d_model (`<fill_type>`, *optional*, defaults to 256): <fill_docstring>
            disable_custom_kernels (`<fill_type>`, *optional*, defaults to `False`): <fill_docstring>
            use_timm_backbone (`<fill_type>`, *optional*, defaults to `True`): <fill_docstring>
            num_channels (`<fill_type>`, *optional*, defaults to 3): <fill_docstring>
            use_pretrained_backbone (`<fill_type>`, *optional*, defaults to `True`): <fill_docstring>
            init_std (`<fill_type>`, *optional*, defaults to 0.02): <fill_docstring>
            backbone (`<fill_type>`, *optional*, defaults to `"resnet50"`): <fill_docstring>
            num_feature_levels (`<fill_type>`, *optional*, defaults to 4): <fill_docstring>
            num_heads (`<fill_type>`, *optional*, defaults to 8): <fill_docstring>
            decoder_n_points (`<fill_type>`, *optional*, defaults to 4): <fill_docstring>
            dilation (`<fill_type>`, *optional*, defaults to `False`): <fill_docstring>
            position_embedding_type (`<fill_type>`, *optional*, defaults to `"SineHW"`): <fill_docstring>
            encoder_n_points (`<fill_type>`, *optional*, defaults to 4): <fill_docstring>
            dropout (`<fill_type>`, *optional*, defaults to 0.0): <fill_docstring>
            activation_function (`<fill_type>`, *optional*, defaults to `"relu"`): <fill_docstring>
            activation_dropout (`<fill_type>`, *optional*, defaults to 0.0): <fill_docstring>
            encoder_ffn_dim (`<fill_type>`, *optional*, defaults to 2048): <fill_docstring>
            d_ffn (`<fill_type>`, *optional*, defaults to 2048): <fill_docstring>
            activation (`<fill_type>`, *optional*, defaults to `"relu"`): <fill_docstring>
            num_queries (`<fill_type>`, *optional*, defaults to 900): <fill_docstring>
            query_dim (`<fill_type>`, *optional*, defaults to 4): <fill_docstring>
            num_encoder_layers (`<fill_type>`, *optional*, defaults to 6): <fill_docstring>
            num_decoder_layers (`<fill_type>`, *optional*, defaults to 6): <fill_docstring>
            embed_init_tgt (`<fill_type>`, *optional*, defaults to `True`): <fill_docstring>
            num_classes (`<fill_type>`, *optional*, defaults to 91): <fill_docstring>
            use_dn (`<fill_type>`, *optional*, defaults to `True`): <fill_docstring>
            dn_num_classes (`<fill_type>`, *optional*, defaults to 91): <fill_docstring>
            dn_number (`<fill_type>`, *optional*, defaults to 100): <fill_docstring>
            dn_box_noise_scale (`<fill_type>`, *optional*, defaults to 0.4): <fill_docstring>
            dn_label_noise_ratio (`<fill_type>`, *optional*, defaults to 0.5): <fill_docstring>
            auxiliary_loss (`<fill_type>`, *optional*, defaults to `True`): <fill_docstring>
            use_masks (`<fill_type>`, *optional*, defaults to `True`): <fill_docstring>
            class_cost (`<fill_type>`, *optional*, defaults to 2.0): <fill_docstring>
            bbox_cost (`<fill_type>`, *optional*, defaults to 5.0): <fill_docstring>
            giou_cost (`<fill_type>`, *optional*, defaults to 2.0): <fill_docstring>
            mask_loss_coefficient (`<fill_type>`, *optional*, defaults to 1.0): <fill_docstring>
            dice_loss_coefficient (`<fill_type>`, *optional*, defaults to 1.0): <fill_docstring>
            cls_loss_coefficient (`<fill_type>`, *optional*, defaults to 1.0): <fill_docstring>
            bbox_loss_coefficient (`<fill_type>`, *optional*, defaults to 5.0): <fill_docstring>
            giou_loss_coefficient (`<fill_type>`, *optional*, defaults to 2.0): <fill_docstring>
            focal_alpha (`<fill_type>`, *optional*, defaults to 0.25): <fill_docstring>
            dec_pred_class_embed_share (`<fill_type>`, *optional*, defaults to `True`): <fill_docstring>
            dec_pred_bbox_embed_share (`<fill_type>`, *optional*, defaults to `True`): <fill_docstring>
            enc_layer_share (`<fill_type>`, *optional*, defaults to `False`): <fill_docstring>
            dec_layer_share (`<fill_type>`, *optional*, defaults to `False`): <fill_docstring>
            pe_temperature_H (`<fill_type>`, *optional*, defaults to 20): <fill_docstring>
            pe_temperature_W (`<fill_type>`, *optional*, defaults to 20): <fill_docstring>
            backbone_config (`<fill_type>`, *optional*): <fill_docstring>
            backbone_kwargs (`<fill_type>`, *optional*, defaults to `{'out_indices': [2, 3, 4]}`): <fill_docstring>

    Examples:

    ```python
    >>> from transformers import DinoDetrConfig, DinoDetrModel

    >>> # Initializing a Dino DETR SenseTime/deformable-detr style configuration
    >>> configuration = DinoDetrConfig()

    >>> # Initializing a model (with random weights) from the SenseTime/deformable-detr style configuration
    >>> model = DinoDetrModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "deformable_detr"
    sub_configs = {"backbone_config": AutoConfig}
    attribute_map = {
        "encoder_attention_heads": "num_heads",
    }

    def __init__(
        self,
        is_encoder_decoder=True,
        d_model=256,
        disable_custom_kernels=False,
        use_timm_backbone=True,
        num_channels=3,
        use_pretrained_backbone=True,
        init_std=0.02,
        backbone="resnet50",
        num_feature_levels=4,
        num_heads=8,
        decoder_n_points=4,
        dilation=False,
        position_embedding_type="SineHW",
        encoder_n_points=4,
        dropout=0.0,
        activation_function="relu",
        activation_dropout=0.0,
        encoder_ffn_dim=2048,
        d_ffn=2048,
        activation="relu",
        num_queries=900,
        query_dim=4,
        num_encoder_layers=6,
        num_decoder_layers=6,
        embed_init_tgt=True,
        num_classes=91,
        use_dn=True,
        dn_num_classes=91,
        dn_number=100,
        dn_box_noise_scale=0.4,
        dn_label_noise_ratio=0.5,
        auxiliary_loss=True,
        use_masks=True,
        class_cost=2.0,
        bbox_cost=5.0,
        giou_cost=2.0,
        mask_loss_coefficient=1.0,
        dice_loss_coefficient=1.0,
        cls_loss_coefficient=1.0,
        bbox_loss_coefficient=5.0,
        giou_loss_coefficient=2.0,
        focal_alpha=0.25,
        dec_pred_class_embed_share=True,
        dec_pred_bbox_embed_share=True,
        enc_layer_share=False,
        dec_layer_share=False,
        pe_temperature_H=20,
        pe_temperature_W=20,
        backbone_config=None,
        backbone_kwargs={"out_indices": [2, 3, 4]},
        **kwargs,
    ):
        if backbone_config is None:
            logger.info("`backbone_config` is `None`. Initializing the config with the default `ResNet50` backbone.")
        elif isinstance(backbone_config, dict):
            backbone_model_type = backbone_config.pop("model_type")
            config_class = CONFIG_MAPPING[backbone_model_type]
            backbone_config = config_class.from_dict(backbone_config)

        verify_backbone_config_arguments(
            use_timm_backbone=use_timm_backbone,
            use_pretrained_backbone=use_pretrained_backbone,
            backbone=backbone,
            backbone_config=backbone_config,
            backbone_kwargs=backbone_kwargs,
        )

        self.d_model = d_model
        self.num_feature_levels = num_feature_levels
        self.num_heads = num_heads
        self.decoder_n_points = decoder_n_points
        self.disable_custom_kernels = disable_custom_kernels
        self.use_timm_backbone = use_timm_backbone
        self.num_channels = num_channels
        self.dilation = dilation
        self.backbone = backbone
        self.backbone_config = backbone_config
        self.backbone_kwargs = backbone_kwargs
        self.use_pretrained_backbone = use_pretrained_backbone
        self.position_embedding_type = position_embedding_type
        self.encoder_n_points = encoder_n_points
        self.dropout = dropout
        self.activation_function = activation_function
        self.activation_dropout = activation_dropout
        self.encoder_ffn_dim = encoder_ffn_dim
        self.d_ffn = d_ffn
        self.activation = activation
        self.num_queries = num_queries
        self.query_dim = query_dim
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.embed_init_tgt = embed_init_tgt
        self.init_std = init_std
        self.num_classes = num_classes
        self.dn_num_classes = dn_num_classes
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.auxiliary_loss = auxiliary_loss
        self.dec_pred_class_embed_share = dec_pred_class_embed_share
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        self.use_dn = use_dn
        self.use_masks = use_masks
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        self.mask_loss_coefficient = mask_loss_coefficient
        self.dice_loss_coefficient = dice_loss_coefficient
        self.cls_loss_coefficient = cls_loss_coefficient
        self.bbox_loss_coefficient = bbox_loss_coefficient
        self.giou_loss_coefficient = giou_loss_coefficient
        self.focal_alpha = focal_alpha
        self.enc_layer_share = enc_layer_share
        self.dec_layer_share = dec_layer_share
        self.pe_temperature_H = pe_temperature_H
        self.pe_temperature_W = pe_temperature_W
        super().__init__(is_encoder_decoder=is_encoder_decoder, **kwargs)

    @property
    def encoder_attention_heads(self) -> int:
        return self.num_heads


__all__ = ["DinoDetrConfig"]
