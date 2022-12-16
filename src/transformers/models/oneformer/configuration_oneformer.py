# coding=utf-8
# Copyright 2022 SHI-Labs Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""OneFormer model configuration"""
import copy
from typing import Dict, Optional

from transformers import SwinConfig

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING


logger = logging.get_logger(__name__)

ONEFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "shi-labs/oneformer_ade20k_swin_tiny": (
        "https://huggingface.co/shi-labs/oneformer_ade20k_swin_tiny/blob/main/config.json"
    ),
    # See all OneFormer models at https://huggingface.co/models?filter=oneformer
}


class OneFormerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`OneFormerModel`]. It is used to instantiate a
    OneFormer model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the OneFormer
    [shi-labs/oneformer_ade20k_swin_tiny](https://huggingface.co/shi-labs/oneformer_ade20k_swin_tiny) architecture
    trained on [ADE20k-150](https://huggingface.co/datasets/scene_parse_150).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Currently, OneFormer supports the [Swin Transformer](swin) and [Dilated Neighborhood Attention Transformer](dinat)
    as backbones.

    Args:
        general_config (`dict`, *optional*)
            Dictionary containing general configuration like backbone_type, loss weights, number of classes, etc.
        backbone_config (`PretrainedConfig`, *optional*, defaults to `SwinConfig`)
            The configuration of the backbone model.
        text_encoder_config (`dict`, *optional*, defaults to a dictionary with the following keys)
            Dictionary containing configuration for the text-mapper module and task encoder like sequence length,
            number of linear layers in MLP, etc.
        decoder_config (`dict`, *optional*, defaults to a dictionary with the following keys)
            Dictionary containing configuration for the text-mapper module and task encoder like sequence length,
            number of linear layers in MLP, etc.

    Raises:
        `ValueError`:
            Raised if the backbone model type selected is not in `["swin", "dinat"]`
    Examples:
    ```python
    >>> from transformers import OneFormerConfig, OneFormerModel

    >>> # Initializing a OneFormer shi-labs/oneformer_ade20k_swin_tiny configuration
    >>> configuration = OneFormerConfig()
    >>> # Initializing a model (with random weights) from the shi-labs/oneformer_ade20k_swin_tiny style configuration
    >>> model = OneFormerModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """
    model_type = "oneformer"
    attribute_map = {"hidden_size": "hidden_dim"}
    backbones_supported = ["swin", "dinat"]

    def __init__(
        self,
        backbone_config: Optional[Dict] = None,
        ignore_value: Optional[int] = 255,
        num_labels: Optional[int] = 150,
        num_queries: Optional[int] = 150,
        no_object_weight: Optional[int] = 0.1,
        deep_supervision: Optional[bool] = True,
        class_weight: Optional[float] = 2.0,
        mask_weight: Optional[float] = 5.0,
        dice_weight: Optional[float] = 5.0,
        contrastive_weight: Optional[float] = 0.5,
        contrastive_temperature: Optional[float] = 0.07,
        train_num_points: Optional[int] = 12544,
        oversample_ratio: Optional[float] = 3.0,
        importance_sample_ratio: Optional[float] = 0.75,
        init_std: Optional[float] = 0.02,
        init_xavier_std: Optional[float] = 1.0,
        layer_norm_eps: Optional[float] = 1e-05,
        training: Optional[bool] = False,
        use_auxiliary_loss: Optional[bool] = True,
        output_auxiliary_logits: Optional[bool] = True,
        strides: Optional[list] = [4, 8, 16, 32],
        task_seq_len: Optional[int] = 77,
        max_seq_len: Optional[int] = 77,
        text_encoder_width: Optional[int] = 256,
        text_encoder_context_length: Optional[int] = 77,
        text_encoder_num_layers: Optional[int] = 6,
        text_encoder_vocab_size: Optional[int] = 49408,
        text_encoder_proj_layers: Optional[int] = 2,
        text_encoder_n_ctx: Optional[int] = 16,
        conv_dim: Optional[int] = 256,
        mask_dim: Optional[int] = 256,
        hidden_dim: Optional[int] = 256,
        encoder_feedforward_dim: Optional[int] = 1024,
        norm: Optional[str] = "GN",
        encoder_layers: Optional[int] = 6,
        decoder_layers: Optional[int] = 10,
        use_task_norm: Optional[bool] = True,
        num_attention_heads: Optional[int] = 8,
        dropout: Optional[float] = 0.1,
        dim_feedforward: Optional[int] = 2048,
        pre_norm: Optional[bool] = False,
        enforce_input_proj: Optional[bool] = False,
        query_dec_layers: Optional[int] = 2,
        common_stride: Optional[int] = 4,
        **kwargs,
    ):
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
                use_absolute_embeddings=False,
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

        self.backbone_config = backbone_config

        self.ignore_value = ignore_value
        self.num_labels = num_labels
        self.num_queries = num_queries
        self.no_object_weight = no_object_weight
        self.deep_supervision = deep_supervision
        self.class_weight = class_weight
        self.mask_weight = mask_weight
        self.dice_weight = dice_weight
        self.contrastive_weight = contrastive_weight
        self.contrastive_temperature = contrastive_temperature
        self.train_num_points = train_num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.init_std = init_std
        self.init_xavier_std = init_xavier_std
        self.layer_norm_eps = layer_norm_eps
        self.training = training
        self.use_auxiliary_loss = use_auxiliary_loss
        self.output_auxiliary_logits = output_auxiliary_logits
        self.strides = strides
        self.task_seq_len = task_seq_len
        self.max_seq_len = max_seq_len
        self.text_encoder_width = text_encoder_width
        self.text_encoder_context_length = text_encoder_context_length
        self.text_encoder_num_layers = text_encoder_num_layers
        self.text_encoder_vocab_size = text_encoder_vocab_size
        self.text_encoder_proj_layers = text_encoder_proj_layers
        self.text_encoder_n_ctx = text_encoder_n_ctx
        self.conv_dim = conv_dim
        self.mask_dim = mask_dim
        self.hidden_dim = hidden_dim
        self.encoder_feedforward_dim = encoder_feedforward_dim
        self.norm = norm
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.use_task_norm = use_task_norm
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.dim_feedforward = dim_feedforward
        self.pre_norm = pre_norm
        self.enforce_input_proj = enforce_input_proj
        self.query_dec_layers = query_dec_layers
        self.common_stride = common_stride
        self.num_hidden_layers = decoder_layers

        super().__init__(**kwargs)

    def to_dict(self) -> Dict[str, any]:
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`]. Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["backbone_config"] = self.backbone_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
