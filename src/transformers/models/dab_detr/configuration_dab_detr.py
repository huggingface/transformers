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

from ...backbone_utils import consolidate_backbone_kwargs_to_config
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging
from ..auto import AutoConfig


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="IDEA-Research/dab-detr-resnet-50")
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
    }

    def __init__(
        self,
        backbone_config=None,
        num_queries=300,
        encoder_layers=6,
        encoder_ffn_dim=2048,
        encoder_attention_heads=8,
        decoder_layers=6,
        decoder_ffn_dim=2048,
        decoder_attention_heads=8,
        is_encoder_decoder=True,
        activation_function="prelu",
        hidden_size=256,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        init_std=0.02,
        init_xavier_std=1.0,
        auxiliary_loss=False,
        dilation=False,
        class_cost=2,
        bbox_cost=5,
        giou_cost=2,
        cls_loss_coefficient=2,
        bbox_loss_coefficient=5,
        giou_loss_coefficient=2,
        focal_alpha=0.25,
        temperature_height=20,
        temperature_width=20,
        query_dim=4,
        random_refpoints_xy=False,
        keep_query_pos=False,
        num_patterns=0,
        normalize_before=False,
        sine_position_embedding_scale=None,
        initializer_bias_prior_prob=None,
        tie_word_embeddings=True,
        **kwargs,
    ):
        if query_dim != 4:
            raise ValueError("The query dimensions has to be 4.")

        # Init timm backbone with hardcoded values for BC
        timm_default_kwargs = {
            "num_channels": 3,
            "features_only": True,
            "use_pretrained_backbone": False,
            "out_indices": [1, 2, 3, 4],
        }
        if dilation:
            timm_default_kwargs["output_stride"] = 16

        backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=backbone_config,
            default_backbone="resnet50",
            default_config_type="resnet50",
            default_config_kwargs={"out_features": ["stage4"]},
            timm_default_kwargs=timm_default_kwargs,
            **kwargs,
        )

        self.backbone_config = backbone_config
        self.num_queries = num_queries
        self.hidden_size = hidden_size
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.init_xavier_std = init_xavier_std
        self.num_hidden_layers = encoder_layers
        self.auxiliary_loss = auxiliary_loss
        # Hungarian matcher
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        # Loss coefficients
        self.cls_loss_coefficient = cls_loss_coefficient
        self.bbox_loss_coefficient = bbox_loss_coefficient
        self.giou_loss_coefficient = giou_loss_coefficient
        self.focal_alpha = focal_alpha
        self.query_dim = query_dim
        self.random_refpoints_xy = random_refpoints_xy
        self.keep_query_pos = keep_query_pos
        self.num_patterns = num_patterns
        self.normalize_before = normalize_before
        self.temperature_width = temperature_width
        self.temperature_height = temperature_height
        self.sine_position_embedding_scale = sine_position_embedding_scale
        self.initializer_bias_prior_prob = initializer_bias_prior_prob
        self.tie_word_embeddings = tie_word_embeddings

        super().__init__(is_encoder_decoder=is_encoder_decoder, **kwargs)


__all__ = ["DabDetrConfig"]
