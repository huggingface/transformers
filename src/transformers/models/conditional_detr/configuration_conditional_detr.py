# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""Conditional DETR model configuration"""

from huggingface_hub.dataclasses import strict

from ...backbone_utils import consolidate_backbone_kwargs_to_config
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import AutoConfig


@auto_docstring(checkpoint="microsoft/conditional-detr-resnet-50")
@strict
class ConditionalDetrConfig(PreTrainedConfig):
    r"""
    num_queries (`int`, *optional*, defaults to 100):
        Number of object queries, i.e. detection slots. This is the maximal number of objects
        [`ConditionalDetrModel`] can detect in a single image. For COCO, we recommend 100 queries.
    auxiliary_loss (`bool`, *optional*, defaults to `False`):
        Whether auxiliary decoding losses (loss at each decoder layer) are to be used.
    position_embedding_type (`str`, *optional*, defaults to `"sine"`):
        Type of position embeddings to be used on top of the image features. One of `"sine"` or `"learned"`.
    dilation (`bool`, *optional*, defaults to `False`):
        Whether to replace stride with dilation in the last convolutional block (DC5). Only supported when
        `use_timm_backbone` = `True`.

    Examples:

    ```python
    >>> from transformers import ConditionalDetrConfig, ConditionalDetrModel

    >>> # Initializing a Conditional DETR microsoft/conditional-detr-resnet-50 style configuration
    >>> configuration = ConditionalDetrConfig()

    >>> # Initializing a model (with random weights) from the microsoft/conditional-detr-resnet-50 style configuration
    >>> model = ConditionalDetrModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "conditional_detr"
    sub_configs = {"backbone_config": AutoConfig}
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
        "num_hidden_layers": "encoder_layers",
    }

    backbone_config: dict | PreTrainedConfig | None = None
    num_channels: int = 3
    num_queries: int = 300
    encoder_layers: int = 6
    encoder_ffn_dim: int = 2048
    encoder_attention_heads: int = 8
    decoder_layers: int = 6
    decoder_ffn_dim: int = 2048
    decoder_attention_heads: int = 8
    encoder_layerdrop: float | int = 0.0
    decoder_layerdrop: float | int = 0.0
    is_encoder_decoder: bool = True
    activation_function: str = "relu"
    d_model: int = 256
    dropout: float | int = 0.1
    attention_dropout: float | int = 0.0
    activation_dropout: float | int = 0.0
    init_std: float = 0.02
    init_xavier_std: float = 1.0
    auxiliary_loss: bool = False
    position_embedding_type: str = "sine"
    dilation: bool = False
    class_cost: int = 2
    bbox_cost: int = 5
    giou_cost: int = 2
    mask_loss_coefficient: int = 1
    dice_loss_coefficient: int = 1
    cls_loss_coefficient: int = 2
    bbox_loss_coefficient: int = 5
    giou_loss_coefficient: int = 2
    focal_alpha: float = 0.25

    def __post_init__(self, **kwargs):
        # Init timm backbone with hardcoded values for BC
        backbone_kwargs = kwargs.get("backbone_kwargs", {})
        timm_default_kwargs = {
            "num_channels": backbone_kwargs.get("num_channels", self.num_channels),
            "features_only": True,
            "use_pretrained_backbone": False,
            "out_indices": backbone_kwargs.get("out_indices", [1, 2, 3, 4]),
        }
        if self.dilation:
            timm_default_kwargs["output_stride"] = backbone_kwargs.get("output_stride", 16)

        self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=self.backbone_config,
            default_backbone="resnet50",
            default_config_type="resnet",
            default_config_kwargs={"out_features": ["stage4"]},
            timm_default_kwargs=timm_default_kwargs,
            **kwargs,
        )

        super().__post_init__(**kwargs)


__all__ = ["ConditionalDetrConfig"]
