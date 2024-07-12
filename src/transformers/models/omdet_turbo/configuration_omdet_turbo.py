# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""OmDet-Turbo model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ...utils.backbone_utils import verify_backbone_config_arguments
from ..auto import CONFIG_MAPPING


logger = logging.get_logger(__name__)


class OmDetTurboConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`OmDetTurboModel`]. It is used to instantiate a
    OmDet-Turbo model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the OmDet-Turbo
    [omlab/omdet-turbo-tiny](https://huggingface.co/omlab/omdet-turbo-tiny) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        TODO: Add arguments

    Examples:

    ```python
    >>> from transformers import OmDetTurboConfig, OmDetTurboModel

    >>> # Initializing a OmDet-Turbo omlab/omdet-turbo-tiny style configuration
    >>> configuration = OmDetTurboConfig()

    >>> # Initializing a model (with random weights) from the omlab/omdet-turbo-tiny style configuration
    >>> model = OmDetTurboModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "omdet-turbo"
    attribute_map = {
        "encoder_hidden_dim": "d_model",
        "encoder_feat_strides": "feat_strides",
        "num_attention_heads": "encoder_attention_heads",
    }

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        vision_backbone=None,
        use_timm_backbone=False,
        backbone_kwargs=None,
        encoder_hidden_dim=256,
        decoder_hidden_dim=256,
        backbone_feat_channels=(256, 256, 256),
        num_feature_levels=3,
        disable_custom_kernels=False,
        text_projection_in_features=512,
        text_projection_out_features=512,
        num_queries=900,
        size_divisibility=32,
        layer_norm_eps=1e-5,
        batch_norm_eps=1e-5,
        activation_function="silu",
        encoder_activation="gelu",
        encoder_activation_function="relu",
        hidden_expansion=1,
        dropout=0.0,
        activation_dropout=0.0,
        encoder_in_channels=[192, 384, 768],
        encoder_feat_strides=[8, 16, 32],
        encode_proj_layers=[2],
        encoder_attention_heads=8,
        normalize_before=False,
        eval_size=None,
        encoder_layers=1,
        positional_encoding_temperature=10000,
        encoder_ffn_dim=2048,
        decoder_num_heads=8,
        decoder_num_layers=6,
        label_dim=512,
        cls_type="cosine",
        decoder_activation="relu",
        decoder_dim_feedforward=2048,
        decoder_num_points=4,
        decoder_dropout=0.0,
        decoder_eval_idx=-1,
        fuse_type="merged_attn",
        is_encoder_decoder=True,
        **kwargs,
    ):
        if vision_config is None and vision_backbone is None:
            logger.info("`backbone_config` is `None`. Initializing the config with the default `Swin` backbone.")
            backbone_config = CONFIG_MAPPING["swin"](
                window_size=7,
                image_size=224,
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                out_indices=[2, 3, 4],
            )
        elif isinstance(vision_config, dict):
            backbone_model_type = backbone_config.pop("model_type")
            config_class = CONFIG_MAPPING[backbone_model_type]
            vision_config = config_class.from_dict(backbone_config)

        verify_backbone_config_arguments(
            use_timm_backbone=use_timm_backbone,
            use_pretrained_backbone=False,
            backbone=vision_backbone,
            backbone_config=vision_config,
            backbone_kwargs=backbone_kwargs,
        )

        self.text_config = text_config
        self.vision_config = vision_config
        self.vision_backbone = vision_backbone
        self.use_timm_backbone = use_timm_backbone
        self.backbone_kwargs = backbone_kwargs
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.backbone_feat_channels = backbone_feat_channels
        self.num_feature_levels = num_feature_levels
        self.disable_custom_kernels = disable_custom_kernels
        self.text_projection_in_features = text_projection_in_features
        self.text_projection_out_features = text_projection_out_features
        self.num_queries = num_queries
        self.size_divisibility = size_divisibility
        self.layer_norm_eps = layer_norm_eps
        self.batch_norm_eps = batch_norm_eps
        self.activation_function = activation_function
        self.encoder_activation = encoder_activation
        self.encoder_activation_function = encoder_activation_function
        self.hidden_expansion = hidden_expansion
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.encoder_in_channels = encoder_in_channels
        self.encoder_feat_strides = encoder_feat_strides
        self.encode_proj_layers = encode_proj_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.normalize_before = normalize_before
        self.eval_size = eval_size
        self.encoder_layers = encoder_layers
        self.positional_encoding_temperature = positional_encoding_temperature
        self.encoder_ffn_dim = encoder_ffn_dim
        self.decoder_num_heads = decoder_num_heads
        self.decoder_num_layers = decoder_num_layers
        self.label_dim = label_dim
        self.cls_type = cls_type
        self.decoder_activation = decoder_activation
        self.decoder_dim_feedforward = decoder_dim_feedforward
        self.decoder_num_points = decoder_num_points
        self.decoder_dropout = decoder_dropout
        self.decoder_eval_idx = decoder_eval_idx
        self.fuse_type = fuse_type

        super().__init__(is_encoder_decoder=is_encoder_decoder, **kwargs)
