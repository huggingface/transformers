# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""VitDet model configuration"""

from ...backbone_utils import BackboneConfigMixin
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="google/vitdet-base-patch16-224")
class VitDetConfig(BackboneConfigMixin, PreTrainedConfig):
    r"""
    pretrain_image_size (`int`, *optional*, defaults to 224):
        The size (resolution) of each image during pretraining.
    window_block_indices (`list[int]`, *optional*, defaults to `[]`):
        List of indices of blocks that should have window attention instead of regular global self-attention.
    residual_block_indices (`list[int]`, *optional*, defaults to `[]`):
        List of indices of blocks that should have an extra residual block after the MLP.
    use_relative_position_embeddings (`bool`, *optional*, defaults to `False`):
        Whether to add relative position embeddings to the attention maps.
    window_size (`int`, *optional*, defaults to 0):
        The size of the attention window.
    Example:

    ```python
    >>> from transformers import VitDetConfig, VitDetModel

    >>> # Initializing a VitDet configuration
    >>> configuration = VitDetConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = VitDetModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "vitdet"

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        mlp_ratio=4,
        hidden_act="gelu",
        dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-6,
        image_size=224,
        pretrain_image_size=224,
        patch_size=16,
        num_channels=3,
        qkv_bias=True,
        drop_path_rate=0.0,
        window_block_indices=[],
        residual_block_indices=[],
        use_absolute_position_embeddings=True,
        use_relative_position_embeddings=False,
        window_size=0,
        out_features=None,
        out_indices=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.mlp_ratio = mlp_ratio
        self.hidden_act = hidden_act
        self.dropout_prob = dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.pretrain_image_size = pretrain_image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.drop_path_rate = drop_path_rate
        self.window_block_indices = window_block_indices
        self.residual_block_indices = residual_block_indices
        self.use_absolute_position_embeddings = use_absolute_position_embeddings
        self.use_relative_position_embeddings = use_relative_position_embeddings
        self.window_size = window_size

        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, self.num_hidden_layers + 1)]
        self.set_output_features_output_indices(out_indices=out_indices, out_features=out_features)


__all__ = ["VitDetConfig"]
