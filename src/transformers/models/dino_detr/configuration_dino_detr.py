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
        d_model (`int`, *optional*, defaults to 256):
            Dimension of the layers.
        disable_custom_kernels (`bool`, *optional*, defaults to `False`):
            Disable the use of custom CUDA and CPU kernels. This option is necessary for the ONNX export, as custom
            kernels are not supported by PyTorch ONNX export.
        use_timm_backbone (`bool`, *optional*, defaults to `True`):
            Whether or not to use the `timm` library for the backbone. If set to `False`, will use the [`AutoBackbone`]
            API.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        use_pretrained_backbone (`bool`, *optional*, defaults to `True`):
            Whether to use pretrained weights for the backbone.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        key_aware_type (`str`, *optional*): Can take values `"mean"` or `"proj_mean"` selects the way memory is added to queries
            in forward_ca of the DinoDetrDecoderLayer.
        enc_layer_dropout_prob (`float`, *optional*): The dropout probability for all fully connected layers in the encoder.
        dec_layer_dropout_prob (`float`, *optional*): The dropout probability for all fully connected layers in the decoder.
        rm_self_attn_layers (`int`, *optional*): The number of decoder layers from which to remove self attention.
        dec_detach (`bool`, *optional*): Whether to detach the new reference points in the decoder.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        backbone (`str`, *optional*, defaults to `"resnet50"`):
            Name of backbone to use when `backbone_config` is `None`. If `use_pretrained_backbone` is `True`, this
            will load the corresponding pretrained weights from the timm or transformers library. If `use_pretrained_backbone`
            is `False`, this loads the backbone's config and uses that to initialize the backbone with random weights.
        num_feature_levels (`int`, *optional*, defaults to 4):
            The number of input feature levels.
        num_heads (`int`, *optional*, defaults to 8): The number of heads in all the attention layers.
        decoder_n_points (`int`, *optional*, defaults to 4):
            The number of sampled keys in each feature level for each attention head in the decoder.
        dilation (`bool`, *optional*, defaults to `False`):
            Whether to replace stride with dilation in the last convolutional block (DC5). Only supported when
            `use_timm_backbone` = `True`.
        position_embedding_type (`str`, *optional*, defaults to `"SineHW"`):
            Type of position embeddings to be used on top of the image features. One of `"sine"` or `"learned"`.
        encoder_n_points (`int`, *optional*, defaults to 4):
            The number of sampled keys in each feature level for each attention head in the encoder.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        activation_function (`str` or `function`, *optional*, defaults to `"relu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        encoder_ffn_dim (`int`, *optional*, defaults to 2048):
            Dimension of the "intermediate" (often named feed-forward) layer in decoder.
        module_seq (`List[str]`, *optional*, defaults to `['sa', 'ca', 'ffn']`): The sequence of modules in the decoder.
        d_ffn (`int`, *optional*, defaults to 2048): The hidden dimensions of the fully connected layers.
        activation (`str`, *optional*, defaults to `"relu"`): Could be `"relu"`, `"gelu"`, `"glu"`, `"prelu"`, `"selu"`.
        decoder_sa_type (`str`, *optional*, defaults to `"sa"`): The type of decoder self attention can take values
        num_queries (`int`, *optional*, defaults to 900):
            Number of object queries, i.e. detection slots. This is the maximal number of objects
            [`DinoDetrModel`] can detect in a single image. In case `two_stage` is set to `True`, we use
            `two_stage_num_proposals` instead.
        two_stage_type (`str`, *optional*, defaults to `"standard"`): The type of two-stage mechanism to use.
        query_dim (`int`, *optional*, defaults to 4): The dimension of the object query embeddings.
        use_detached_boxes_dec_out (`bool`, *optional*, defaults to `False`): Whether to use detached boxes in the decoder output.
        dec_layer_number (`int`, *optional*): The number of layers in the decoder.
        decoder_layer_noise (`bool`, *optional*, defaults to `False`): Whether to add noise to the decoder layers.
        dln_xy_noise (`float`, *optional*, defaults to 0.2): The noise level of the DinoDetrRandomBoxPerturber along the x and y axes.
        dln_hw_noise (`float`, *optional*, defaults to 0.2): The noise level of the DinoDetrRandomBoxPerturber along the w and h axes.
        num_encoder_layers (`int`, *optional*, defaults to 6): Number of encoder layers.
        num_decoder_layers (`int`, *optional*, defaults to 6): Number of decoder layers.
        two_stage_keep_all_tokens (`bool`, *optional*, defaults to `False`): Whether to keep all tokens in the two-stage process.
        random_refpoints_xy (`bool`, *optional*, defaults to `False`): Whether to initialize reference points randomly along the x and y axes.
        normalize_before (`bool`, *optional*, defaults to `False`): Whether to apply layer normalization before other operations.
        num_patterns (`int`, *optional*, defaults to 0): The number of patterns to use in the decoder.
        embed_init_tgt (`bool`, *optional*, defaults to `True`): Whether to initialize the target embeddings.
        two_stage_pat_embed (`int`, *optional*, defaults to 0): The number of pattern embeddings in the two-stage process.
        two_stage_add_query_num (`int`, *optional*, defaults to 0): The number of additional queries in the two-stage process.
        two_stage_learn_wh (`bool`, *optional*, defaults to `False`): Whether to learn the width and height in the two-stage process.
        num_classes (`int`, *optional*, defaults to 91): The number of object classes the model can predict.
        dn_labelbook_size (`int`, *optional*, defaults to 91): The size of the label book for denoising training.
        dn_number (`int`, *optional*, defaults to 100): The number of denoising queries.
        dn_box_noise_scale (`float`, *optional*, defaults to 0.4): The scale of noise added to bounding boxes during denoising training.
        dn_label_noise_ratio (`float`, *optional*, defaults to 0.5): The ratio of noise added to labels during denoising training.
        auxiliary_loss (`bool`, *optional*, defaults to `True`):
            Whether auxiliary decoding losses (loss at each decoder layer) are to be used.
        dec_pred_class_embed_share (`bool`, *optional*, defaults to `True`): Whether to share class embeddings across decoder layers.
        dec_pred_bbox_embed_share (`bool`, *optional*, defaults to `True`): Whether to share bounding box embeddings across decoder layers.
        two_stage_bbox_embed_share (`bool`, *optional*, defaults to `False`): Whether to share bounding box embeddings in the two-stage process.
        two_stage_class_embed_share (`bool`, *optional*, defaults to `False`): Whether to share class embeddings in the two-stage process.
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
        interm_loss_coef (`float`, *optional*, defaults to 1.0): The weight of the intermediate loss.
        no_interm_box_loss (`bool`, *optional*, defaults to `False`): Whether to disable intermediate bounding box loss.
        use_dn (`bool`, *optional*, defaults to `True`): Whether to use denoising training.
        use_masks (`bool`, *optional*, defaults to `True`): Whether to use masks in the model.
        focal_alpha (`float`, *optional*, defaults to 0.25):
            Alpha parameter in the focal loss.
        enc_layer_share (`bool`, *optional*, defaults to `False`): Whether to share encoder layers.
        dec_layer_share (`bool`, *optional*, defaults to `False`): Whether to share decoder layers.
        backbone_config (`PretrainedConfig` or `dict`, *optional*):
            The configuration of the backbone model. Only used in case `use_timm_backbone` is set to `False` in which
            case it will default to `ResNetConfig()`.
        backbone_kwargs (`dict`, *optional*, defaults to `{'out_indices': [2, 3, 4]}`):
            Keyword arguments to be passed to AutoBackbone when loading from a checkpoint
            e.g. `{'out_indices': (0, 1, 2, 3)}`. Cannot be specified if `backbone_config` is set.
        is_encoder_decoder (`bool`, *optional*, defaults to `True`): Whether the model is an encoder-decoder architecture.
        pe_temperatureH (`float`, *optional*, defaults to 20): The temperature for positional encoding along the height dimension.
        pe_temperatureW (`float`, *optional*, defaults to 20): The temperature for positional encoding along the width dimension.

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
        d_model=256,
        disable_custom_kernels=False,
        use_timm_backbone=True,
        num_channels=3,
        use_pretrained_backbone=True,
        activation_dropout=0.0,
        key_aware_type=None,
        enc_layer_dropout_prob=None,
        dec_layer_dropout_prob=None,
        rm_self_attn_layers=None,
        dec_detach=None,
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
        encoder_ffn_dim=2048,
        module_seq=["sa", "ca", "ffn"],
        d_ffn=2048,
        activation="relu",
        decoder_sa_type="sa",
        num_queries=900,
        two_stage_type="standard",
        query_dim=4,
        use_detached_boxes_dec_out=False,
        dec_layer_number=None,
        decoder_layer_noise=False,
        dln_xy_noise=0.2,
        dln_hw_noise=0.2,
        num_encoder_layers=6,
        num_decoder_layers=6,
        two_stage_keep_all_tokens=False,
        random_refpoints_xy=False,
        normalize_before=False,
        num_patterns=0,
        embed_init_tgt=True,
        two_stage_pat_embed=0,
        two_stage_add_query_num=0,
        two_stage_learn_wh=False,
        num_classes=91,
        dn_labelbook_size=91,
        dn_number=100,
        dn_box_noise_scale=0.4,
        dn_label_noise_ratio=0.5,
        auxiliary_loss=True,
        dec_pred_class_embed_share=True,
        dec_pred_bbox_embed_share=True,
        two_stage_bbox_embed_share=False,
        two_stage_class_embed_share=False,
        class_cost=2.0,
        bbox_cost=5.0,
        giou_cost=2.0,
        mask_loss_coefficient=1.0,
        dice_loss_coefficient=1.0,
        cls_loss_coefficient=1.0,
        bbox_loss_coefficient=5.0,
        giou_loss_coefficient=2.0,
        interm_loss_coef=1.0,
        no_interm_box_loss=False,
        use_dn=True,
        use_masks=True,
        focal_alpha=0.25,
        enc_layer_share=False,
        dec_layer_share=False,
        backbone_config=None,
        backbone_kwargs={"out_indices": [2, 3, 4]},
        is_encoder_decoder=True,
        pe_temperatureH=20,
        pe_temperatureW=20,
        **kwargs,
    ):
        if use_timm_backbone:
            if backbone_kwargs is None:
                backbone_kwargs = {}
            if dilation:
                backbone_kwargs["output_stride"] = 16
            if "out_indices" not in backbone_kwargs:
                backbone_kwargs["out_indices"] = (
                    list(range(num_feature_levels + 1))[-3:] if num_feature_levels > 1 else [num_feature_levels]
                )
            backbone_kwargs["in_chans"] = num_channels

        elif not use_timm_backbone and backbone in (None, "resnet50"):
            if backbone_config is None:
                logger.info("`backbone_config` is `None`. Initializing the config with the default `ResNet` backbone.")
                backbone_config = CONFIG_MAPPING["resnet"](out_features=["stage4"])
            elif isinstance(backbone_config, dict):
                backbone_model_type = backbone_config.get("model_type")
                config_class = CONFIG_MAPPING[backbone_model_type]
                backbone_config = config_class.from_dict(backbone_config)
        if not isinstance(num_patterns, int):
            Warning(f"num_patterns should be int but {type(num_patterns)}")
            num_patterns = 0

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
        self.num_feature_levels = num_feature_levels
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

        self.module_seq = module_seq
        self.num_heads = num_heads
        self.decoder_n_points = decoder_n_points
        self.dropout = dropout
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.activation = activation
        self.key_aware_type = key_aware_type
        self.decoder_sa_type = decoder_sa_type

        self.num_queries = num_queries
        self.d_model = d_model
        self.enc_layer_dropout_prob = enc_layer_dropout_prob
        self.two_stage_type = two_stage_type

        self.query_dim = query_dim
        self.num_feature_levels = num_feature_levels
        self.use_detached_boxes_dec_out = use_detached_boxes_dec_out
        self.query_dim = query_dim
        self.d_model = d_model
        self.dec_layer_number = dec_layer_number
        self.dec_layer_dropout_prob = dec_layer_dropout_prob

        self.decoder_layer_noise = decoder_layer_noise
        self.dln_xy_noise = dln_xy_noise
        self.dln_hw_noise = dln_hw_noise
        self.num_feature_levels = num_feature_levels
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.two_stage_keep_all_tokens = two_stage_keep_all_tokens
        self.num_queries = num_queries
        self.random_refpoints_xy = random_refpoints_xy
        self.use_detached_boxes_dec_out = use_detached_boxes_dec_out
        self.query_dim = query_dim
        self.decoder_sa_type = decoder_sa_type
        self.d_model = d_model
        self.normalize_before = normalize_before
        self.num_patterns = num_patterns
        self.embed_init_tgt = embed_init_tgt
        self.two_stage_type = two_stage_type
        self.two_stage_pat_embed = two_stage_pat_embed
        self.two_stage_add_query_num = two_stage_add_query_num
        self.two_stage_learn_wh = two_stage_learn_wh
        self.rm_self_attn_layers = rm_self_attn_layers
        self.dec_detach = dec_detach

        self.init_std = init_std

        self.num_queries = num_queries
        self.num_classes = num_classes
        self.num_feature_levels = num_feature_levels
        self.dn_labelbook_size = dn_labelbook_size
        self.query_dim = query_dim
        self.random_refpoints_xy = random_refpoints_xy
        self.num_patterns = num_patterns
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size
        self.two_stage_type = two_stage_type
        self.auxiliary_loss = auxiliary_loss
        self.dec_pred_class_embed_share = dec_pred_class_embed_share
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        self.num_classes = num_classes
        self.two_stage_add_query_num = two_stage_add_query_num
        self.two_stage_bbox_embed_share = two_stage_bbox_embed_share
        self.two_stage_class_embed_share = two_stage_class_embed_share
        self.decoder_sa_type = decoder_sa_type

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
        self.interm_loss_coef = interm_loss_coef
        self.no_interm_box_loss = no_interm_box_loss
        self.focal_alpha = focal_alpha
        self.disable_custom_kernels = disable_custom_kernels

        self.enc_layer_share = enc_layer_share
        self.dec_layer_share = dec_layer_share

        self.pe_temperatureH = pe_temperatureH
        self.pe_temperatureW = pe_temperatureW
        super().__init__(is_encoder_decoder=is_encoder_decoder, **kwargs)

    @property
    def encoder_attention_heads(self) -> int:
        return self.num_heads


__all__ = ["DinoDetrConfig"]
