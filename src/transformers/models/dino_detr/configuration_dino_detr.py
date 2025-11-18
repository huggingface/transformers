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
    a Dino DETR model according to the specified arguments, defining the model architecture.Instantiating a configuration
    with the defaults will yield a similar configuration to that of the DINO-DETR
    [kostaspitas/dino_detr](https://huggingface.co/kostaspitas/dino_detr) architecture.


    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
            is_encoder_decoder (`bool`, *optional*, defaults to `True`): Whether the model is an encoder-decoder architecture.
            d_model (`int`, *optional*, defaults to 256): Dimension of the layers.
            disable_custom_kernels (`bool`, *optional*, defaults to `False`):
                Disable the use of custom CUDA and CPU kernels. This option is necessary for the ONNX export, as custom
                kernels are not supported by PyTorch ONNX export.
            use_timm_backbone (`bool`, *optional*, defaults to `True`):
                Whether or not to use the `timm` library for the backbone. If set to `False`, will use the [`AutoBackbone`]
                API.
            num_channels (`int`, *optional*, defaults to 3): The number of input channels.
            use_pretrained_backbone (`bool`, *optional*, defaults to `True`): Whether to use pretrained weights for the backbone.
            init_std (`float`, *optional*, defaults to 0.02):
                The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            backbone (`str`, *optional*, defaults to `"resnet50"`):
                Name of backbone to use when `backbone_config` is `None`. If `use_pretrained_backbone` is `True`, this
                will load the corresponding pretrained weights from the timm or transformers library. If `use_pretrained_backbone`
                is `False`, this loads the backbone's config and uses that to initialize the backbone with random weights.
            num_feature_levels (`int`, *optional*, defaults to 4): The number of input feature levels.
            num_heads (`int`, *optional*, defaults to 8): The number of heads in all the attention layers.
            decoder_n_points (`int`, *optional*, defaults to 4):
                The number of sampled keys in each feature level for each attention head in the decoder.
            dilation (`bool`, *optional*, defaults to `False`):
                Whether to replace stride with dilation in the last convolutional block (DC5). Only supported when
                `use_timm_backbone` = `True`.
            encoder_n_points (`int`, *optional*, defaults to 4):
                The number of sampled keys in each feature level for each attention head in the encoder.
            dropout (`float`, *optional*, defaults to 0.0):
                The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
            activation_function (`str` or `function`, *optional*, defaults to `"relu"`):
                The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
                `"relu"`, `"silu"` and `"gelu_new"` are supported.
            activation_dropout (`float`, *optional*, defaults to 0.0): The dropout ratio for activations inside the fully connected layer.
            encoder_ffn_dim (`int`, *optional*, defaults to 2048):
                Dimension of the "intermediate" (often named feed-forward) layer in decoder.
            d_ffn (`int`, *optional*, defaults to 2048): The hidden dimensions of the fully connected layers.
            activation (`str`, *optional*, defaults to `"relu"`): Could be `"relu"`, `"gelu"`, `"glu"`, `"prelu"`, `"selu"`.
            num_queries (`int`, *optional*, defaults to 900):
                Number of object queries, i.e. detection slots. This is the maximal number of objects
                [`DinoDetrModel`] can detect in a single image.
            query_dim (`int`, *optional*, defaults to 4): The dimension of the object query embeddings.
            num_encoder_layers (`int`, *optional*, defaults to 6): Number of encoder layers.
            num_decoder_layers (`int`, *optional*, defaults to 6): Number of decoder layers.
            embed_init_tgt (`bool`, *optional*, defaults to `True`): Whether to initialize the target embeddings.
            num_classes (`int`, *optional*, defaults to 91): The number of object classes the model can predict.
            use_dn (`bool`, *optional*, defaults to `True`): Whether to use denoising training.
            dn_num_classes (`int`, *optional*, defaults to 91): The size of the label book for denoising training.
            dn_number (`int`, *optional*, defaults to 100): The number of denoising queries.
            dn_box_noise_scale (`float`, *optional*, defaults to 0.4): The scale of noise added to bounding boxes during denoising training.
            dn_label_noise_ratio (`float`, *optional*, defaults to 0.5): The ratio of noise added to labels during denoising training.
            auxiliary_loss (`bool`, *optional*, defaults to `True`):
                Whether auxiliary decoding losses (loss at each decoder layer) are to be used.
            use_masks (`bool`, *optional*, defaults to `True`): Whether to use masks in the model.
            class_cost (`float`, *optional*, defaults to 2.0):
                Relative weight of the classification error in the Hungarian matching cost.
            bbox_cost (`float`, *optional*, defaults to 5.0):
                Relative weight of the L1 error of the bounding box coordinates in the Hungarian matching cost.
            giou_cost (`float`, *optional*, defaults to 2.0):
                Relative weight of the generalized IoU loss of the bounding box in the Hungarian matching cost.
            mask_loss_coefficient (`float`, *optional*, defaults to 1.0):
                Relative weight of the Focal loss in the panoptic segmentation loss.
            dice_loss_coefficient (`float`, *optional*, defaults to 1.0):
                Relative weight of the DICE/F-1 loss in the panoptic segmentation loss.
            cls_loss_coefficient (`float`, *optional*, defaults to 1.0): The weight of the classification loss.
            bbox_loss_coefficient (`float`, *optional*, defaults to 5.0):
                Relative weight of the L1 bounding box loss in the object detection loss.
            giou_loss_coefficient (`float`, *optional*, defaults to 2.0):
                Relative weight of the generalized IoU loss in the object detection loss.
            focal_alpha (`float`, *optional*, defaults to 0.25):
                Alpha parameter in the focal loss.
            pe_temperatureH (`float`, *optional*, defaults to 20): The temperature for positional encoding along the height dimension.
            pe_temperatureW (`float`, *optional*, defaults to 20): The temperature for positional encoding along the width dimension.
            backbone_config (`PretrainedConfig` or `dict`, *optional*):
                The configuration of the backbone model. Only used in case `use_timm_backbone` is set to `False` in which
                case it will default to `ResNetConfig()`.
            backbone_kwargs (`dict`, *optional*, defaults to `{'out_indices': [2, 3, 4]}`):
                Keyword arguments to be passed to AutoBackbone when loading from a checkpoint
                e.g. `{'out_indices': (0, 1, 2, 3)}`. Cannot be specified if `backbone_config` is set.
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
        self.pe_temperature_H = pe_temperature_H
        self.pe_temperature_W = pe_temperature_W
        super().__init__(is_encoder_decoder=is_encoder_decoder, **kwargs)

    @property
    def encoder_attention_heads(self) -> int:
        return self.num_heads


__all__ = ["DinoDetrConfig"]
