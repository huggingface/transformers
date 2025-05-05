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
        key_aware_type (`<fill_type>`, *optional*): <fill_docstring>
        enc_layer_dropout_prob (`<fill_type>`, *optional*): <fill_docstring>
        dec_layer_dropout_prob (`<fill_type>`, *optional*): <fill_docstring>
        learnable_tgt_init (`<fill_type>`, *optional*, defaults to `True`): <fill_docstring>
        rm_self_attn_layers (`<fill_type>`, *optional*): <fill_docstring>
        rm_detach (`<fill_type>`, *optional*): <fill_docstring>
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        backbone (`str`, *optional*, defaults to `"resnet50"`):
            Name of backbone to use when `backbone_config` is `None`. If `use_pretrained_backbone` is `True`, this
            will load the corresponding pretrained weights from the timm or transformers library. If `use_pretrained_backbone`
            is `False`, this loads the backbone's config and uses that to initialize the backbone with random weights.
        num_feature_levels (`int`, *optional*, defaults to 4):
            The number of input feature levels.
        num_heads (`<fill_type>`, *optional*, defaults to 8): <fill_docstring>
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
        module_seq (`<fill_type>`, *optional*, defaults to `['sa', 'ca', 'ffn']`): <fill_docstring>
        d_ffn (`<fill_type>`, *optional*, defaults to 2048): <fill_docstring>
        activation (`<fill_type>`, *optional*, defaults to `"relu"`): <fill_docstring>
        decoder_sa_type (`<fill_type>`, *optional*, defaults to `"sa"`): <fill_docstring>
        num_queries (`int`, *optional*, defaults to 900):
            Number of object queries, i.e. detection slots. This is the maximal number of objects
            [`DinoDetrModel`] can detect in a single image. In case `two_stage` is set to `True`, we use
            `two_stage_num_proposals` instead.
        two_stage_type (`<fill_type>`, *optional*, defaults to `"standard"`): <fill_docstring>
        query_dim (`<fill_type>`, *optional*, defaults to 4): <fill_docstring>
        use_detached_boxes_dec_out (`<fill_type>`, *optional*, defaults to `False`): <fill_docstring>
        dec_layer_number (`<fill_type>`, *optional*): <fill_docstring>
        decoder_layer_noise (`<fill_type>`, *optional*, defaults to `False`): <fill_docstring>
        dln_xy_noise (`<fill_type>`, *optional*, defaults to 0.2): <fill_docstring>
        dln_hw_noise (`<fill_type>`, *optional*, defaults to 0.2): <fill_docstring>
        num_encoder_layers (`<fill_type>`, *optional*, defaults to 6): <fill_docstring>
        num_unicoder_layers (`<fill_type>`, *optional*, defaults to 0): <fill_docstring>
        num_decoder_layers (`<fill_type>`, *optional*, defaults to 6): <fill_docstring>
        two_stage_keep_all_tokens (`<fill_type>`, *optional*, defaults to `False`): <fill_docstring>
        random_refpoints_xy (`<fill_type>`, *optional*, defaults to `False`): <fill_docstring>
        normalize_before (`<fill_type>`, *optional*, defaults to `False`): <fill_docstring>
        num_patterns (`<fill_type>`, *optional*, defaults to 0): <fill_docstring>
        embed_init_tgt (`<fill_type>`, *optional*, defaults to `True`): <fill_docstring>
        two_stage_pat_embed (`<fill_type>`, *optional*, defaults to 0): <fill_docstring>
        two_stage_add_query_num (`<fill_type>`, *optional*, defaults to 0): <fill_docstring>
        two_stage_learn_wh (`<fill_type>`, *optional*, defaults to `False`): <fill_docstring>
        num_classes (`<fill_type>`, *optional*, defaults to 91): <fill_docstring>
        dn_labelbook_size (`<fill_type>`, *optional*, defaults to 91): <fill_docstring>
        fix_refpoints_hw (`<fill_type>`, *optional*, defaults to -1): <fill_docstring>
        dn_number (`<fill_type>`, *optional*, defaults to 100): <fill_docstring>
        dn_box_noise_scale (`<fill_type>`, *optional*, defaults to 0.4): <fill_docstring>
        dn_label_noise_ratio (`<fill_type>`, *optional*, defaults to 0.5): <fill_docstring>
        auxiliary_loss (`bool`, *optional*, defaults to `True`):
            Whether auxiliary decoding losses (loss at each decoder layer) are to be used.
        dec_pred_class_embed_share (`<fill_type>`, *optional*, defaults to `True`): <fill_docstring>
        dec_pred_bbox_embed_share (`<fill_type>`, *optional*, defaults to `True`): <fill_docstring>
        two_stage_bbox_embed_share (`<fill_type>`, *optional*, defaults to `False`): <fill_docstring>
        two_stage_class_embed_share (`<fill_type>`, *optional*, defaults to `False`): <fill_docstring>
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
        cls_loss_coefficient (`<fill_type>`, *optional*, defaults to 1.0): <fill_docstring>
        bbox_loss_coefficient (`float`, *optional*, defaults to 5.0):
            Relative weight of the L1 bounding box loss in the object detection loss.
        giou_loss_coefficient (`float`, *optional*, defaults to 2.0):
            Relative weight of the generalized IoU loss in the object detection loss.
        interm_loss_coef (`<fill_type>`, *optional*, defaults to 1.0): <fill_docstring>
        no_interm_box_loss (`<fill_type>`, *optional*, defaults to `False`): <fill_docstring>
        use_dn (`<fill_type>`, *optional*, defaults to `True`): <fill_docstring>
        use_masks (`<fill_type>`, *optional*, defaults to `True`): <fill_docstring>
        focal_alpha (`float`, *optional*, defaults to 0.25):
            Alpha parameter in the focal loss.
        enc_layer_share (`<fill_type>`, *optional*, defaults to `False`): <fill_docstring>
        dec_layer_share (`<fill_type>`, *optional*, defaults to `False`): <fill_docstring>
        backbone_config (`PretrainedConfig` or `dict`, *optional*):
            The configuration of the backbone model. Only used in case `use_timm_backbone` is set to `False` in which
            case it will default to `ResNetConfig()`.
        backbone_kwargs (`dict`, *optional*, defaults to `{'out_indices': [2, 3, 4]}`):
            Keyword arguments to be passed to AutoBackbone when loading from a checkpoint
            e.g. `{'out_indices': (0, 1, 2, 3)}`. Cannot be specified if `backbone_config` is set.
        is_encoder_decoder (`<fill_type>`, *optional*, defaults to `True`): <fill_docstring>
        pe_temperatureH (`<fill_type>`, *optional*, defaults to 20): <fill_docstring>
        pe_temperatureW (`<fill_type>`, *optional*, defaults to 20): <fill_docstring>

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
        learnable_tgt_init=True,
        rm_self_attn_layers=None,
        rm_detach=None,
        init_std=0.02,
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
        # We default to values which were previously hard-coded in the model. This enables configurability of the config
        # while keeping the default behavior the same.
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

        # Backwards compatibility
        elif not use_timm_backbone and backbone in (None, "resnet50"):
            if backbone_config is None:
                logger.info("`backbone_config` is `None`. Initializing the config with the default `ResNet` backbone.")
                backbone_config = CONFIG_MAPPING["resnet"](out_features=["stage4"])
            elif isinstance(backbone_config, dict):
                backbone_model_type = backbone_config.get("model_type")
                config_class = CONFIG_MAPPING[backbone_model_type]
                backbone_config = config_class.from_dict(backbone_config)
        if not isinstance(num_patterns, int):
            Warning("num_patterns should be int but {}".format(type(num_patterns)))
            num_patterns = 0

        verify_backbone_config_arguments(
            use_timm_backbone=use_timm_backbone,
            use_pretrained_backbone=use_pretrained_backbone,
            backbone=backbone,
            backbone_config=backbone_config,
            backbone_kwargs=backbone_kwargs,
        )

        # Verify arguments
        assert sorted(module_seq) == ["ca", "ffn", "sa"]
        assert decoder_sa_type in ["sa", "ca_label", "ca_content"]
        if enc_layer_dropout_prob is not None:
            assert isinstance(enc_layer_dropout_prob, list)
            assert len(enc_layer_dropout_prob) == num_encoder_layers
            for i in enc_layer_dropout_prob:
                assert 0.0 <= i <= 1.0
        if dec_layer_number is not None:
            assert isinstance(dec_layer_number, list)
            assert len(dec_layer_number) == num_decoder_layers
        if dec_layer_dropout_prob is not None:
            assert isinstance(dec_layer_dropout_prob, list)
            assert len(dec_layer_dropout_prob) == num_decoder_layers
            for i in dec_layer_dropout_prob:
                assert 0.0 <= i <= 1.0

        assert decoder_sa_type in ["sa", "ca_label", "ca_content"]
        assert learnable_tgt_init, "learnable_tgt_init should be True"
        assert two_stage_type in [
            "no",
            "standard",
        ], "Unknown param {} of two_stage_type".format(two_stage_type)
        if dec_layer_number is not None:
            if two_stage_type != "no" or num_patterns == 0:
                assert (
                    dec_layer_number[0] == num_queries
                ), f"dec_layer_number[0]({dec_layer_number[0]}) != num_queries({num_queries})"
            else:
                assert (
                    dec_layer_number[0] == num_queries * num_patterns
                ), f"dec_layer_number[0]({dec_layer_number[0]}) != num_queries({num_queries}) * num_patterns({num_patterns})"
        if rm_detach:
            assert isinstance(rm_detach, list)
            assert any(i in ["enc_ref", "enc_tgt", "dec"] for i in rm_detach)

        if num_feature_levels <= 1:
            assert two_stage_type == "no", "two_stage_type should be no if num_feature_levels=1"
        assert two_stage_type in [
            "no",
            "standard",
        ], "unknown param {} of two_stage_type".format(two_stage_type)
        if two_stage_type != "no":
            if two_stage_bbox_embed_share:
                assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
            if two_stage_class_embed_share:
                assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
        assert decoder_sa_type in ["sa", "ca_label", "ca_content"]
        if fix_refpoints_hw > 0:
            assert random_refpoints_xy
        elif int(fix_refpoints_hw) == -2:
            assert random_refpoints_xy
        assert query_dim in [2, 4], "Query_dim should be 2/4 but {}".format(query_dim)
        if use_timm_backbone and backbone_kwargs is not None:
            assert backbone_kwargs["in_chans"] == num_channels

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
        self.num_unicoder_layers = num_unicoder_layers
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
        # Hungarian matcher
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        # Loss coefficients
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
