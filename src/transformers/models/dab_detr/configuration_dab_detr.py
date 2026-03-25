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
"""DAB-DETR model configuration"""

from huggingface_hub.dataclasses import strict

from ...backbone_utils import consolidate_backbone_kwargs_to_config
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import AutoConfig


@auto_docstring(checkpoint="IDEA-Research/dab-detr-resnet-50")
@strict
class DabDetrConfig(PreTrainedConfig):
    r"""
    num_queries (`int`, *optional*, defaults to 300):
        Number of object queries, i.e. detection slots. This is the maximal number of objects
        [`DabDetrModel`] can detect in a single image. For COCO, we recommend 100 queries.
    dilation (`bool`, *optional*, defaults to `False`):
        Whether to replace stride with dilation in the last convolutional block (DC5). Only supported when `use_timm_backbone` = `True`.
    temperature_height (`int`, *optional*, defaults to 20):
        Temperature parameter to tune the flatness of positional attention (HEIGHT)
    temperature_width (`int`, *optional*, defaults to 20):
        Temperature parameter to tune the flatness of positional attention (WIDTH)
    query_dim (`int`, *optional*, defaults to 4):
        Query dimension parameter represents the size of the output vector.
    random_refpoints_xy (`bool`, *optional*, defaults to `False`):
        Whether to fix the x and y coordinates of the anchor boxes with random initialization.
    keep_query_pos (`bool`, *optional*, defaults to `False`):
        Whether to concatenate the projected positional embedding from the object query into the original query (key) in every decoder layer.
    num_patterns (`int`, *optional*, defaults to 0):
        Number of pattern embeddings.
    normalize_before (`bool`, *optional*, defaults to `False`):
        Whether we use a normalization layer in the Encoder or not.
    sine_position_embedding_scale (`float`, *optional*, defaults to 'None'):
        Scaling factor applied to the normalized positional encodings.
    initializer_bias_prior_prob (`float`, *optional*):
        The prior probability used by the bias initializer to initialize biases for `enc_score_head` and `class_embed`.
        If `None`, `prior_prob` computed as `prior_prob = 1 / (num_labels + 1)` while initializing model weights.

    Examples:

    ```python
    >>> from transformers import DabDetrConfig, DabDetrModel

    >>> # Initializing a DAB-DETR IDEA-Research/dab-detr-resnet-50 style configuration
    >>> configuration = DabDetrConfig()

    >>> # Initializing a model (with random weights) from the IDEA-Research/dab-detr-resnet-50 style configuration
    >>> model = DabDetrModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "dab-detr"
    sub_configs = {"backbone_config": AutoConfig}
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_attention_heads": "encoder_attention_heads",
        "num_hidden_layers": "encoder_layers",
    }

    backbone_config: dict | PreTrainedConfig | None = None
    num_queries: int = 300
    encoder_layers: int = 6
    encoder_ffn_dim: int = 2048
    encoder_attention_heads: int = 8
    decoder_layers: int = 6
    decoder_ffn_dim: int = 2048
    decoder_attention_heads: int = 8
    is_encoder_decoder: int = True
    activation_function: str = "prelu"
    hidden_size: int = 256
    dropout: float | int = 0.1
    attention_dropout: float | int = 0.0
    activation_dropout: float | int = 0.0
    init_std: float = 0.02
    init_xavier_std: float = 1.0
    auxiliary_loss: bool = False
    dilation: bool = False
    class_cost: int = 2
    bbox_cost: int = 5
    giou_cost: int = 2
    cls_loss_coefficient: int = 2
    bbox_loss_coefficient: int = 5
    giou_loss_coefficient: int = 2
    focal_alpha: float = 0.25
    temperature_height: int = 20
    temperature_width: int = 20
    query_dim: int = 4
    random_refpoints_xy: bool = False
    keep_query_pos: bool = False
    num_patterns: int = 0
    normalize_before: bool = False
    sine_position_embedding_scale: float | None = None
    initializer_bias_prior_prob: float | None = None
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        # Init timm backbone with hardcoded values for BC
        timm_default_kwargs = {
            "num_channels": 3,
            "features_only": True,
            "use_pretrained_backbone": False,
            "out_indices": [1, 2, 3, 4],
        }
        if self.dilation:
            timm_default_kwargs["output_stride"] = 16

        self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=self.backbone_config,
            default_backbone="resnet50",
            default_config_type="resnet50",
            default_config_kwargs={"out_features": ["stage4"]},
            timm_default_kwargs=timm_default_kwargs,
            **kwargs,
        )

        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.query_dim != 4:
            raise ValueError("The query dimensions has to be 4.")


__all__ = ["DabDetrConfig"]
