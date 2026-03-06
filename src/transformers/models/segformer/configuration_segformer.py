# Copyright 2021 NVIDIA and The HuggingFace Inc. team. All rights reserved.
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
"""SegFormer model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="ByteDance-Seed/SeedOss-36B")
class SegformerConfig(PreTrainedConfig):
    r"""
    num_encoder_blocks (`int`, *optional*, defaults to 4):
        The number of encoder blocks (i.e. stages in the Mix Transformer encoder).
    depths (`list[int]`, *optional*, defaults to `[2, 2, 2, 2]`):
        The number of layers in each encoder block.
    sr_ratios (`list[int]`, *optional*, defaults to `[8, 4, 2, 1]`):
        Sequence reduction ratios in each encoder block.
    patch_sizes (`list[int]`, *optional*, defaults to `[7, 3, 3, 3]`):
        Patch size before each encoder block.
    strides (`list[int]`, *optional*, defaults to `[4, 2, 2, 2]`):
        Stride before each encoder block.
    mlp_ratios (`list[int]`, *optional*, defaults to `[4, 4, 4, 4]`):
        Ratio of the size of the hidden layer compared to the size of the input layer of the Mix FFNs in the
        encoder blocks.

    Example:

    ```python
    >>> from transformers import SegformerModel, SegformerConfig

    >>> # Initializing a SegFormer nvidia/segformer-b0-finetuned-ade-512-512 style configuration
    >>> configuration = SegformerConfig()

    >>> # Initializing a model from the nvidia/segformer-b0-finetuned-ade-512-512 style configuration
    >>> model = SegformerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "segformer"

    def __init__(
        self,
        num_channels=3,
        num_encoder_blocks=4,
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        hidden_sizes=[32, 64, 160, 256],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        num_attention_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        classifier_dropout_prob=0.1,
        initializer_range=0.02,
        drop_path_rate=0.1,
        layer_norm_eps=1e-6,
        decoder_hidden_size=256,
        semantic_loss_ignore_index=255,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_channels = num_channels
        self.num_encoder_blocks = num_encoder_blocks
        self.depths = depths
        self.sr_ratios = sr_ratios
        self.hidden_sizes = hidden_sizes
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.mlp_ratios = mlp_ratios
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.classifier_dropout_prob = classifier_dropout_prob
        self.initializer_range = initializer_range
        self.drop_path_rate = drop_path_rate
        self.layer_norm_eps = layer_norm_eps
        self.decoder_hidden_size = decoder_hidden_size
        self.reshape_last_stage = kwargs.get("reshape_last_stage", True)
        self.semantic_loss_ignore_index = semantic_loss_ignore_index


__all__ = ["SegformerConfig"]
