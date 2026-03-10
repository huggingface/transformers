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
"""Mask2Former model configuration"""

from ...backbone_utils import consolidate_backbone_kwargs_to_config
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging
from ..auto import AutoConfig


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="facebook/mask2former-swin-small-coco-instance")
class Mask2FormerConfig(PreTrainedConfig):
    r"""
    feature_size (`int`, *optional*, defaults to 256):
        The features (channels) of the resulting feature maps.
    mask_feature_size (`int`, *optional*, defaults to 256):
        The masks' features size, this value will also be used to specify the Feature Pyramid Network features'
        size.
    encoder_feedforward_dim (`int`, *optional*, defaults to 1024):
        Dimension of feedforward network for deformable detr encoder used as part of pixel decoder.
    dim_feedforward (`int`, *optional*, defaults to 2048):
        Feature dimension in feedforward network for transformer decoder.
    pre_norm (`bool`, *optional*, defaults to `False`):
        Whether to use pre-LayerNorm or not for transformer decoder.
    enforce_input_projection (`bool`, *optional*, defaults to `False`):
        Whether to add an input projection 1x1 convolution even if the input channels and hidden dim are identical
        in the Transformer decoder.
    common_stride (`int`, *optional*, defaults to 4):
        Parameter used for determining number of FPN levels used as part of pixel decoder.
    ignore_value (`int`, *optional*, defaults to 255):
        Category id to be ignored during training.
    num_queries (`int`, *optional*, defaults to 100):
        Number of queries for the decoder.
    train_num_points (`str` or `function`, *optional*, defaults to 12544):
        Number of points used for sampling during loss calculation.
    oversample_ratio (`float`, *optional*, defaults to 3.0):
        Oversampling parameter used for calculating no. of sampled points
    importance_sample_ratio (`float`, *optional*, defaults to 0.75):
        Ratio of points that are sampled via importance sampling.
    feature_strides (`list[int]`, *optional*, defaults to `[4, 8, 16, 32]`):
        Feature strides corresponding to features generated from backbone network.
    output_auxiliary_logits (`bool`, *optional*):
        Should the model output its `auxiliary_logits` or not.

    Examples:

    ```python
    >>> from transformers import Mask2FormerConfig, Mask2FormerModel

    >>> # Initializing a Mask2Former facebook/mask2former-swin-small-coco-instance configuration
    >>> configuration = Mask2FormerConfig()

    >>> # Initializing a model (with random weights) from the facebook/mask2former-swin-small-coco-instance style configuration
    >>> model = Mask2FormerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

    """

    model_type = "mask2former"
    sub_configs = {"backbone_config": AutoConfig}
    backbones_supported = ["swin"]
    attribute_map = {"hidden_size": "hidden_dim"}

    def __init__(
        self,
        backbone_config: dict | PreTrainedConfig | None = None,
        feature_size: int = 256,
        mask_feature_size: int = 256,
        hidden_dim: int = 256,
        encoder_feedforward_dim: int = 1024,
        activation_function: str = "relu",
        encoder_layers: int = 6,
        decoder_layers: int = 10,
        num_attention_heads: int = 8,
        dropout: float = 0.0,
        dim_feedforward: int = 2048,
        pre_norm: bool = False,
        enforce_input_projection: bool = False,
        common_stride: int = 4,
        ignore_value: int = 255,
        num_queries: int = 100,
        no_object_weight: float = 0.1,
        class_weight: float = 2.0,
        mask_weight: float = 5.0,
        dice_weight: float = 5.0,
        train_num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        init_std: float = 0.02,
        init_xavier_std: float = 1.0,
        use_auxiliary_loss: bool = True,
        feature_strides: list[int] = [4, 8, 16, 32],
        output_auxiliary_logits: bool | None = None,
        **kwargs,
    ):
        backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=backbone_config,
            default_config_type="swin",
            default_config_kwargs={
                "depths": [2, 2, 18, 2],
                "drop_path_rate": 0.3,
                "out_features": ["stage1", "stage2", "stage3", "stage4"],
            },
            **kwargs,
        )

        # verify that the backbone is supported
        if backbone_config.model_type not in self.backbones_supported:
            logger.warning_once(
                f"Backbone {backbone_config.model_type} is not a supported model and may not be compatible with Mask2Former. "
                f"Supported model types: {','.join(self.backbones_supported)}"
            )

        self.backbone_config = backbone_config
        self.feature_size = feature_size
        self.mask_feature_size = mask_feature_size
        self.hidden_dim = hidden_dim
        self.encoder_feedforward_dim = encoder_feedforward_dim
        self.activation_function = activation_function
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.dim_feedforward = dim_feedforward
        self.pre_norm = pre_norm
        self.enforce_input_projection = enforce_input_projection
        self.common_stride = common_stride
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
        self.use_auxiliary_loss = use_auxiliary_loss
        self.feature_strides = feature_strides
        self.output_auxiliary_logits = output_auxiliary_logits
        self.num_hidden_layers = decoder_layers

        super().__init__(**kwargs)


__all__ = ["Mask2FormerConfig"]
