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

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        encoder_hidden_dim=256,
        decoder_hidden_dim=256,
        backbone_feat_channels=(256, 256, 256),
        disable_custom_kernels=False,
        text_projection_in_features=512,
        text_projection_out_features=512,
        num_queries=900,
        size_divisibility=32,
        batch_norm_eps=1e-5,
        activation_function="silu",
        encoder_activation="gelu",
        hidden_expansion=1,
        encoder_in_channels=[192, 384, 768],
        encoder_feat_strides=[8, 16, 32],
        use_encoder_idx=[2],
        num_encoder_layers=1,
        pe_temperature=10000,
        encoder_dim_feedforward=2048,
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
    ):
        super().__init__()

        self.text_config = text_config
        self.vision_config = vision_config
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.backbone_feat_channels = backbone_feat_channels
        self.disable_custom_kernels = disable_custom_kernels
        self.text_projection_in_features = text_projection_in_features
        self.text_projection_out_features = text_projection_out_features
        self.num_queries = num_queries
        self.size_divisibility = size_divisibility
        self.batch_norm_eps = batch_norm_eps
        self.activation_function = activation_function
        self.encoder_activation = encoder_activation
        self.hidden_expansion = hidden_expansion
        self.encoder_in_channels = encoder_in_channels
        self.encoder_feat_strides = encoder_feat_strides
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.encoder_dim_feedforward = encoder_dim_feedforward
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
