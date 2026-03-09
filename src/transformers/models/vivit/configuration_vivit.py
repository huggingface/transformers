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
"""ViViT model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="google/vivit-b-16x2-kinetics400")
class VivitConfig(PreTrainedConfig):
    r"""
    num_frames (`int`, *optional*, defaults to 32):
        The number of frames in each video.
    tubelet_size (`list[int]`, *optional*, defaults to `[2, 16, 16]`):
        The size (resolution) of each tubelet.

    Example:

    ```python
    >>> from transformers import VivitConfig, VivitModel

    >>> # Initializing a ViViT google/vivit-b-16x2-kinetics400 style configuration
    >>> configuration = VivitConfig()

    >>> # Initializing a model (with random weights) from the google/vivit-b-16x2-kinetics400 style configuration
    >>> model = VivitModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "vivit"

    def __init__(
        self,
        image_size=224,
        num_frames=32,
        tubelet_size=[2, 16, 16],
        num_channels=3,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu_fast",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-06,
        qkv_bias=True,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

        self.image_size = image_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias

        super().__init__(**kwargs)


__all__ = ["VivitConfig"]
