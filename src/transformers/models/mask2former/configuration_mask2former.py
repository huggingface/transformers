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
from typing import Dict, Optional

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING
from ..swin import SwinConfig


MASK2FORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "shivi/mask2former-instance-swin-small-coco": (
        "https://huggingface.co/shivi/mask2former-instance-swin-small-coco/resolve/main/config.json"
    ),
}


logger = logging.get_logger(__name__)


class Mask2FormerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Mask2FormerModel`]. It is used to instantiate a
    Mask2Former model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Mask2Former
    [shivi/mask2former-instance-swin-small-coco](https://huggingface.co/shivi/mask2former-instance-swin-small-coco)
    architecture trained on [ADE20k-150](https://huggingface.co/datasets/scene_parse_150).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Currently, Mask2Former only supports the [Swin Transformer](swin) as backbone.

    Args:
        general_config (`Dict`, *optional*):
            General configuration settings for model initialization and loss terms computations.
        backbone_config (`Dict`, *optional*, defaults to `swin-tiny-patch4-window7-224`):
            The configuration passed to the backbone, if unset, the configuration corresponding to
            `swin-base-patch4-window12-384` will be used.
        decoder_config (`Dict`, *optional*):
            The configuration passed to the pixel decoder and transformer decoder models. Includes the number of
            layers, hidden state dimensions, normalization settings, etc.

    Raises:
        `ValueError`:
            Raised if the backbone model type selected is not in `["swin"]`.

    Examples:

    ```python
    >>> from transformers import Mask2FormerConfig, Mask2FormerModel

    >>> # Initializing a Mask2Former shivi/mask2former-instance-swin-small-coco configuration
    >>> configuration = Mask2FormerConfig()

    >>> # Initializing a model (with random weights) from the shivi/mask2former-instance-swin-small-coco style configuration
    >>> model = Mask2FormerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

    """
    model_type = "mask2former"
    backbones_supported = ["swin"]

    def __init__(
        self,
        general_config: Optional[Dict] = None,
        backbone_config: Optional[Dict] = None,
        decoder_config: Optional[Dict] = None,
        **kwargs,
    ):
        cfgs = self._setup_cfg(general_config, backbone_config, decoder_config)

        general_config, backbone_config, decoder_config = cfgs

        self.general_config = general_config
        self.backbone_config = backbone_config
        self.decoder_config = decoder_config

        self.hidden_size = self.decoder_config["hidden_dim"]
        self.num_attention_heads = self.decoder_config["num_heads"]
        self.num_hidden_layers = self.decoder_config["decoder_layers"]
        self.init_std = self.general_config["init_std"]
        self.init_xavier_std = self.general_config["init_xavier_std"]

        super().__init__(**kwargs)

    def _setup_cfg(
        self,
        general_config: Optional[Dict] = None,
        backbone_config: Optional[SwinConfig] = None,
        decoder_config: Optional[Dict] = None,
    ) -> Dict[str, any]:

        if general_config is None:
            general_config = {}
            general_config["ignore_value"] = 255
            general_config["num_classes"] = 150
            general_config["num_queries"] = 150
            general_config["no_object_weight"] = 0.1
            general_config["class_weight"] = 2.0
            general_config["mask_weight"] = 5.0
            general_config["dice_weight"] = 5.0
            general_config["train_num_points"] = 12544
            general_config["oversample_ratio"] = 3.0
            general_config["importance_sample_ratio"] = 0.75
            general_config["init_std"] = 0.02
            general_config["init_xavier_std"] = 1.0
            general_config["layer_norm_eps"] = 1e-05
            general_config["is_train"] = False
            general_config["use_auxiliary_loss"] = False
            general_config["output_auxiliary_logits"] = False
            general_config["feature_strides"] = ([4, 8, 16, 32],)
            general_config["deep_supervision"] = True

        if backbone_config is None:
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

        if decoder_config is None:
            decoder_config = {}
            decoder_config["feature_size"] = 256
            decoder_config["mask_feature_size"] = 256
            decoder_config["hidden_dim"] = 256
            decoder_config["encoder_feedforward_dim"] = 1024
            decoder_config["norm"] = "GN"
            decoder_config["encoder_layers"] = 6
            decoder_config["decoder_layers"] = 10
            decoder_config["num_heads"] = 8
            decoder_config["dropout"] = 0.1
            decoder_config["dim_feedforward"] = 2048
            decoder_config["pre_norm"] = False
            decoder_config["enforce_input_proj"] = False
            decoder_config["common_stride"] = 4

        return general_config, backbone_config, decoder_config

    def to_dict(self) -> Dict[str, any]:
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`]. Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["backbone_config"] = self.backbone_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
