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
import os
from typing import Dict, Optional, Union

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING
from ..swin import SwinConfig


MASK2FORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/mask2former-instance-swin-small-coco": (
        "https://huggingface.co/facebook/mask2former-instance-swin-small-coco/blob/main/config.json"
    )
    # See all Mask2Former models at https://huggingface.co/models?filter=mask2former
}


logger = logging.get_logger(__name__)


class Mask2FormerDecoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of pixel decoder and transformer decoder modules of
    [`Mask2FormerModel`]. It is used to instantiate an Mask2Former model according to the specified arguments, defining
    the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that
    of the Mask2Former
    [facebook/mask2former-instance-swin-small-coco](https://huggingface.co/facebook/mask2former-instance-swin-small-coco)
    architecture. Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs.
    Read the documentation from [`PretrainedConfig`] for more information.

    Args:
        feature_size (`int`, *optional*, defaults to 256):
            The features (channels) of the resulting feature maps.
        mask_feature_size (`int`, *optional*, defaults to 256):
            The masks' features size, this value will also be used to specify the Feature Pyramid Network features'
            size.
        hidden_dim (`int`, *optional*, defaults to 256):
            Dimensionality of the encoder layers
        encoder_feedforward_dim (`int`, *optional*, defaults to 1024):
            Dimension of feedforward network for deformable detr encoder used as part of pixel decoder
        norm (`str`, *optional*, defaults to `"GN"`):
            Normalization for all convolution layers
        encoder_layers (`int`, *optional*, defaults to 6):
            Number of layers in the deformable detr encoder used as part of pixel decoder.
        decoder_layers (`int`, *optional*, defaults to 10):
            Number of layers in the Transformer decoder used by Mask2Former
        num_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder
        dim_feedforward (`int`, *optional*, defaults to 2048):
            Feature dimension in feedforward network for transformer decoder
        pre_norm (`boolean``, *optional*, defaults to False):
            Whether to use pre-LayerNorm or not in layers of transformer decoder
        enforce_input_proj (`boolean``, *optional*, defaults to False):
            add input project 1x1 conv even if input channels and hidden dim is identical in transformer decoder
        common_stride (`int`, *optional*, defaults to 4):
            parameter used for determining number of FPN levels used as part of pixel decoder

    Example:
    ```python
    >>> from transformers import Mask2FormerDecoderConfig, Mask2FormerConfig, Mask2FormerModel

    >>> # Initializing a Mask2FormerDecoderConfig with facebook/mask2former-instance-swin-small-coco style configuration
    >>> decoder_config = Mask2FormerDecoderConfig()
    >>> configuration = Mask2FormerConfig(decoder_config=decoder_config.to_dict())
    >>> # Initializing a Mask2FormerModel (with random weights) from the facebook/mask2former-instance-swin-small-coco style configuration
    >>> model = Mask2FormerModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "mask2Former_decoder"

    def __init__(
        self,
        feature_size=256,
        mask_feature_size=256,
        hidden_dim=256,
        encoder_feedforward_dim=1024,
        norm="GN",
        encoder_layers=6,
        decoder_layers=10,
        num_heads=8,
        dropout=0.1,
        dim_feedforward=2048,
        pre_norm=False,
        enforce_input_proj=False,
        common_stride=4,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.feature_size = feature_size
        self.mask_feature_size = mask_feature_size
        self.hidden_dim = hidden_dim
        self.encoder_feedforward_dim = encoder_feedforward_dim
        self.norm = norm
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.dim_feedforward = dim_feedforward
        self.pre_norm = pre_norm
        self.enforce_input_proj = enforce_input_proj
        self.common_stride = common_stride

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the decoder config dict if we are loading from Mask2FormerConfig
        if config_dict.get("model_type") == "mask2former":
            config_dict = config_dict["decoder_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class Mask2FormerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Mask2FormerModel`]. It is used to instantiate a
    Mask2Former model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Mask2Former
    [facebook/mask2former-instance-swin-small-coco](facebook/mask2former-instance-swin-small-coco) architecture trained
    on [ADE20k-150](https://huggingface.co/datasets/scene_parse_150).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Currently, Mask2Former only supports the [Swin Transformer](swin) as backbone.

    Args:
        backbone_config (`PretrainedConfig` or `dict`, *optional*, defaults to `SwinConfig()`):
            The configuration of the backbone model. if unset, the configuration corresponding to
            `swin-base-patch4-window12-384` will be used.
        decoder_config (`dict`, *optional*):
            The configuration passed to the pixel decoder and transformer decoder models. Includes the number of
            layers, hidden state dimensions, normalization settings, etc.
        ignore_value (`int`, *optional*, defaults to 255):
            Category id to be ignored during training.
        num_queries (`int`, *optional*, defaults to 150):
            Number of queries
        no_object_weight (`int`, *optional*, defaults to 0.1):
            Weight to apply to the null (no object) class.
        class_weight (`int`, *optional*, defaults to 2.0):
            The weight for the cross entropy loss.
        mask_weight (`int`, *optional*, defaults to 5.0):
            The weight for the mask loss.
        dice_weight (`int`, *optional*, defaults to 5.0):
            The weight for the dice loss.
        train_num_points (`str` or `function`, *optional*, defaults to 12544):
            Number of points used for sampling during loss calculation.
        oversample_ratio (`float`, *optional*, defaults to 3.0):
            Oversampling parameter used for calculating no. of sampled points
        importance_sample_ratio (`float`, *optional*, defaults to 0.75):
            Ratio of points that are sampled via importance sampling.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        init_xavier_std (`float``, *optional*, defaults to 1.0):
            The scaling factor used for the Xavier initialization gain in the HM Attention map module.
        layer_norm_eps (`float``, *optional*, defaults to 1e-05):
            description
        is_train (`boolean``, *optional*, defaults to False):
            description
        use_auxiliary_loss (`boolean``, *optional*, defaults to True):
            If `True` [`Mask2FormerForUniversalSegmentationOutput`] will contain the auxiliary losses computed using
            the logits from each decoder's stage.
        feature_strides (`List[int]`, *optional*, defaults to [4, 8, 16, 32]):
            Feature strides corresponding to features generated from backbone network
        deep_supervision (`boolean``, *optional*, defaults to True):
            description

    Raises:
        `ValueError`:
            Raised if the backbone model type selected is not in `["swin"]`.

    Examples:

    ```python
    >>> from transformers import Mask2FormerConfig, Mask2FormerModel

    >>> # Initializing a Mask2Former facebook/mask2former-instance-swin-small-coco configuration
    >>> configuration = Mask2FormerConfig()

    >>> # Initializing a model (with random weights) from the facebook/mask2former-instance-swin-small-coco style configuration
    >>> model = Mask2FormerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

    """
    model_type = "mask2former"
    backbones_supported = ["swin"]
    attribute_map = {"hidden_size": "mask_feature_size"}

    def __init__(
        self,
        backbone_config: Optional[Dict] = None,
        decoder_config: Optional[Dict] = None,
        ignore_value=255,
        num_queries=150,
        no_object_weight=0.1,
        class_weight=2.0,
        mask_weight=5.0,
        dice_weight=5.0,
        train_num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        init_std=0.02,
        init_xavier_std=1.0,
        layer_norm_eps=1e-05,
        is_train=False,
        use_auxiliary_loss=True,
        feature_strides=[4, 8, 16, 32],
        deep_supervision=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if backbone_config is None:
            logger.info("`backbone_config` is `None`. Initializing the config with the default `Swin` backbone.")
            backbone_config = SwinConfig(
                image_size=224,
                in_channels=3,
                patch_size=4,
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                drop_path_rate=0.3,
                out_features=["stage1", "stage2", "stage3", "stage4"],
            )
        else:
            backbone_model_type = (
                backbone_config.pop("model_type") if isinstance(backbone_config, dict) else backbone_config.model_type
            )
            if backbone_model_type not in self.backbones_supported:
                raise ValueError(
                    f"Backbone {backbone_model_type} not supported, please use one of"
                    f" {','.join(self.backbones_supported)}"
                )
            if isinstance(backbone_config, dict):
                config_class = CONFIG_MAPPING[backbone_model_type]
                backbone_config = config_class.from_dict(backbone_config)

        decoder_config_dict = kwargs.pop("decoder_config_dict", None)

        if decoder_config_dict is not None:
            decoder_config = decoder_config_dict

        if decoder_config is None:
            decoder_config = {}
            logger.info("decoder_config is None. Initializing the Mask2FormerDecoderConfig with default values.")

        self.backbone_config = backbone_config
        self.decoder_config = Mask2FormerDecoderConfig(**decoder_config)

        self.ignore_value = ignore_value
        self.num_queries = num_queries
        self.no_object_weight = no_object_weight
        self.class_weight = class_weight
        self.mask_weight = mask_weight
        self.dice_weight = dice_weight
        self.train_num_points = train_num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.init_std = init_std
        self.init_xavier_std = init_xavier_std
        self.layer_norm_eps = layer_norm_eps
        self.is_train = is_train
        self.use_auxiliary_loss = use_auxiliary_loss
        self.feature_strides = feature_strides
        self.deep_supervision = deep_supervision

        self.hidden_size = self.decoder_config.hidden_dim
        self.num_attention_heads = self.decoder_config.num_heads
        self.num_hidden_layers = self.decoder_config.decoder_layers

    @classmethod
    def from_backbone_decoder_configs(
        cls, backbone_config: PretrainedConfig, decoder_config: Mask2FormerDecoderConfig, **kwargs
    ):
        """
        Instantiate a [`Mask2FormerConfig`] (or a derived class) from a pre-trained backbone model configuration and
        Mask2FormerDecoder configuration.

        Args:
            backbone_config ([`PretrainedConfig`]):
                The backbone configuration to be used for Mask2Former model.
            decoder_config ([`PretrainedConfig`]):
                The pixel decoder & transformer decoder configuration to use for Mask2Former model.

        Returns:
            [`Mask2FormerConfig`]: An instance of a configuration object
        """
        return cls(
            backbone_config=backbone_config.to_dict(),
            decoder_config=decoder_config.to_dict(),
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
        output["model_type"] = self.__class__.model_type
        return output
