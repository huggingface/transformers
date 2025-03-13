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
from ..auto import CONFIG_MAPPING


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
        use_timm_backbone (`bool`, *optional*, defaults to `True`):
            Whether or not to use the `timm` library for the backbone. If set to `False`, will use the [`AutoBackbone`]
            API.
        backbone_config (`PretrainedConfig` or `dict`, *optional*):
            The configuration of the backbone model. Only used in case `use_timm_backbone` is set to `False` in which
            case it will default to `ResNetConfig()`.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        num_queries (`int`, *optional*, defaults to 300):
            Number of object queries, i.e. detection slots. This is the maximal number of objects
            [`DinoDetrModel`] can detect in a single image. In case `two_stage` is set to `True`, we use
            `two_stage_num_proposals` instead.
        d_model (`int`, *optional*, defaults to 256):
            Dimension of the layers.
        encoder_layers (`int`, *optional*, defaults to 6):
            Number of encoder layers.
        decoder_layers (`int`, *optional*, defaults to 6):
            Number of decoder layers.
        encoder_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer decoder.
        decoder_ffn_dim (`int`, *optional*, defaults to 1024):
            Dimension of the "intermediate" (often named feed-forward) layer in decoder.
        encoder_ffn_dim (`int`, *optional*, defaults to 1024):
            Dimension of the "intermediate" (often named feed-forward) layer in decoder.
        activation_function (`str` or `function`, *optional*, defaults to `"relu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        init_xavier_std (`float`, *optional*, defaults to 1):
            The scaling factor used for the Xavier initialization gain in the HM Attention map module.
        encoder_layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the encoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        auxiliary_loss (`bool`, *optional*, defaults to `False`):
            Whether auxiliary decoding losses (loss at each decoder layer) are to be used.
        position_embedding_type (`str`, *optional*, defaults to `"sine"`):
            Type of position embeddings to be used on top of the image features. One of `"sine"` or `"learned"`.
        backbone (`str`, *optional*, defaults to `"resnet50"`):
            Name of backbone to use when `backbone_config` is `None`. If `use_pretrained_backbone` is `True`, this
            will load the corresponding pretrained weights from the timm or transformers library. If `use_pretrained_backbone`
            is `False`, this loads the backbone's config and uses that to initialize the backbone with random weights.
        use_pretrained_backbone (`bool`, *optional*, defaults to `True`):
            Whether to use pretrained weights for the backbone.
        backbone_kwargs (`dict`, *optional*):
            Keyword arguments to be passed to AutoBackbone when loading from a checkpoint
            e.g. `{'out_indices': (0, 1, 2, 3)}`. Cannot be specified if `backbone_config` is set.
        dilation (`bool`, *optional*, defaults to `False`):
            Whether to replace stride with dilation in the last convolutional block (DC5). Only supported when
            `use_timm_backbone` = `True`.
        class_cost (`float`, *optional*, defaults to 1):
            Relative weight of the classification error in the Hungarian matching cost.
        bbox_cost (`float`, *optional*, defaults to 5):
            Relative weight of the L1 error of the bounding box coordinates in the Hungarian matching cost.
        giou_cost (`float`, *optional*, defaults to 2):
            Relative weight of the generalized IoU loss of the bounding box in the Hungarian matching cost.
        mask_loss_coefficient (`float`, *optional*, defaults to 1):
            Relative weight of the Focal loss in the panoptic segmentation loss.
        dice_loss_coefficient (`float`, *optional*, defaults to 1):
            Relative weight of the DICE/F-1 loss in the panoptic segmentation loss.
        bbox_loss_coefficient (`float`, *optional*, defaults to 5):
            Relative weight of the L1 bounding box loss in the object detection loss.
        giou_loss_coefficient (`float`, *optional*, defaults to 2):
            Relative weight of the generalized IoU loss in the object detection loss.
        eos_coefficient (`float`, *optional*, defaults to 0.1):
            Relative classification weight of the 'no-object' class in the object detection loss.
        num_feature_levels (`int`, *optional*, defaults to 4):
            The number of input feature levels.
        encoder_n_points (`int`, *optional*, defaults to 4):
            The number of sampled keys in each feature level for each attention head in the encoder.
        decoder_n_points (`int`, *optional*, defaults to 4):
            The number of sampled keys in each feature level for each attention head in the decoder.
        two_stage (`bool`, *optional*, defaults to `False`):
            Whether to apply a two-stage deformable DETR, where the region proposals are also generated by a variant of
            Dino DETR, which are further fed into the decoder for iterative bounding box refinement.
        two_stage_num_proposals (`int`, *optional*, defaults to 300):
            The number of region proposals to be generated, in case `two_stage` is set to `True`.
        with_box_refine (`bool`, *optional*, defaults to `False`):
            Whether to apply iterative bounding box refinement, where each decoder layer refines the bounding boxes
            based on the predictions from the previous layer.
        focal_alpha (`float`, *optional*, defaults to 0.25):
            Alpha parameter in the focal loss.
        disable_custom_kernels (`bool`, *optional*, defaults to `False`):
            Disable the use of custom CUDA and CPU kernels. This option is necessary for the ONNX export, as custom
            kernels are not supported by PyTorch ONNX export.

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
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
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
        deformable_encoder=True,
        enc_layer_dropout_prob=None,
        return_intermediate=True,
        deformable_decoder=True,
        rm_dec_query_scale=True,
        modulate_hw_attn=True,
        dec_layer_dropout_prob=None,
        layer_share_type=None,
        learnable_tgt_init=True,
        rm_self_attn_layers=None,
        rm_detach=None,
        init_std=0.02,
        iter_update=True,
        backbone="resnet50",  # From here and down things are pretty clear
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
        num_unicoder_layers=0,
        num_decoder_layers=6,
        two_stage_keep_all_tokens=False,
        random_refpoints_xy=False,
        use_deformable_box_attn=False,
        box_attn_type="roi_align",
        normalize_before=False,
        num_patterns=0,
        embed_init_tgt=True,
        two_stage_pat_embed=0,
        two_stage_add_query_num=0,
        two_stage_learn_wh=False,
        num_classes=91,
        dn_labelbook_size=91,
        fix_refpoints_hw=-1,
        dn_number=100,
        dn_box_noise_scale=0.4,
        dn_label_noise_ratio=0.5,
        aux_loss=True,
        dec_pred_class_embed_share=True,
        dec_pred_bbox_embed_share=True,
        two_stage_bbox_embed_share=False,
        two_stage_class_embed_share=False,
        class_cost=2.0,
        bbox_cost=5.0,
        giou_cost=2.0,
        mask_loss_coefficient=1.0,
        dice_loss_coefficient=1.0,
        bbox_loss_coefficient=5.0,
        giou_loss_coefficient=2.0,
        interm_loss_coef=1.0,
        no_interm_box_loss=False,
        use_dn=True,
        match_unstable_error=True,
        focal_alpha=0.25,
        enc_layer_share=None,
        dec_layer_share=None,
        backbone_config=None,
        backbone_kwargs=None,
        is_encoder_decoder=True,
        pe_temperatureH=20,
        pe_temperatureW=20,
        **kwargs,
    ):
        # We default to values which were previously hard-coded in the model. This enables configurability of the config
        # while keeping the default behavior the same.
        if use_timm_backbone and backbone_kwargs is None:
            backbone_kwargs = {}
            if dilation:
                backbone_kwargs["output_stride"] = 16
            backbone_kwargs["out_indices"] = [2, 3, 4] if num_feature_levels > 1 else [4]
            backbone_kwargs["in_chans"] = num_channels
        # Backwards compatibility
        elif not use_timm_backbone and backbone in (None, "resnet50"):
            if backbone_config is None:
                logger.info("`backbone_config` is `None`. Initializing the config with the default `ResNet` backbone.")
                backbone_config = CONFIG_MAPPING["resnet"](out_features=["stage4"])
            elif isinstance(backbone_config, dict):
                backbone_model_type = backbone_config.get("model_type")
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
        self.num_feature_levels = num_feature_levels
        self.num_channels = num_channels
        self.dilation = dilation
        self.backbone = backbone
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
        self.deformable_encoder = deformable_encoder
        self.d_model = d_model
        self.enc_layer_dropout_prob = enc_layer_dropout_prob
        self.two_stage_type = two_stage_type

        self.return_intermediate = return_intermediate
        self.query_dim = query_dim
        self.num_feature_levels = num_feature_levels
        self.use_detached_boxes_dec_out = use_detached_boxes_dec_out
        self.query_dim = query_dim
        self.d_model = d_model
        self.deformable_decoder = deformable_decoder
        self.rm_dec_query_scale = rm_dec_query_scale
        self.modulate_hw_attn = modulate_hw_attn
        self.deformable_decoder = deformable_decoder
        self.modulate_hw_attn = modulate_hw_attn
        self.dec_layer_number = dec_layer_number
        self.dec_layer_dropout_prob = dec_layer_dropout_prob

        self.decoder_layer_noise = decoder_layer_noise
        self.dln_xy_noise = dln_xy_noise
        self.dln_hw_noise = dln_hw_noise
        self.num_feature_levels = num_feature_levels
        self.num_encoder_layers = num_encoder_layers
        self.num_unicoder_layers = num_unicoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.deformable_encoder = deformable_encoder
        self.deformable_decoder = deformable_decoder
        self.two_stage_keep_all_tokens = two_stage_keep_all_tokens
        self.num_queries = num_queries
        self.random_refpoints_xy = random_refpoints_xy
        self.use_detached_boxes_dec_out = use_detached_boxes_dec_out
        self.query_dim = query_dim
        self.use_deformable_box_attn = use_deformable_box_attn
        self.layer_share_type = layer_share_type
        self.decoder_sa_type = decoder_sa_type
        self.d_model = d_model
        self.normalize_before = normalize_before
        self.num_patterns = num_patterns
        self.learnable_tgt_init = learnable_tgt_init
        self.embed_init_tgt = embed_init_tgt
        self.two_stage_type = two_stage_type
        self.two_stage_pat_embed = two_stage_pat_embed
        self.two_stage_add_query_num = two_stage_add_query_num
        self.two_stage_learn_wh = two_stage_learn_wh
        self.rm_self_attn_layers = rm_self_attn_layers
        self.rm_detach = rm_detach

        self.init_std = init_std

        self.num_queries = num_queries
        self.num_classes = num_classes
        self.num_feature_levels = num_feature_levels
        self.dn_labelbook_size = dn_labelbook_size
        self.query_dim = query_dim
        self.random_refpoints_xy = random_refpoints_xy
        self.fix_refpoints_hw = fix_refpoints_hw
        self.num_patterns = num_patterns
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size
        self.two_stage_type = two_stage_type
        self.aux_loss = aux_loss
        self.iter_update = iter_update
        self.dec_pred_class_embed_share = dec_pred_class_embed_share
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        self.num_classes = num_classes
        self.two_stage_add_query_num = two_stage_add_query_num
        self.two_stage_bbox_embed_share = two_stage_bbox_embed_share
        self.two_stage_class_embed_share = two_stage_class_embed_share
        self.decoder_sa_type = decoder_sa_type

        self.use_dn = use_dn
        self.match_unstable_error = match_unstable_error
        # Hungarian matcher
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        # Loss coefficients
        self.mask_loss_coefficient = mask_loss_coefficient
        self.dice_loss_coefficient = dice_loss_coefficient
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
    def num_attention_heads(self) -> int:
        return self.encoder_attention_heads

    @property
    def hidden_size(self) -> int:
        return self.d_model


__all__ = ["DinoDetrConfig"]
