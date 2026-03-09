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
"""TimeSformer model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="facebook/timesformer-base-finetuned-k600")
class TimesformerConfig(PreTrainedConfig):
    r"""
    num_frames (`int`, *optional*, defaults to 8):
        The number of frames in each video.
    attention_type (`str`, *optional*, defaults to `"divided_space_time"`):
        The attention type to use. Must be one of `"divided_space_time"`, `"space_only"`, `"joint_space_time"`.

    Example:

    ```python
    >>> from transformers import TimesformerConfig, TimesformerModel

    >>> # Initializing a TimeSformer timesformer-base style configuration
    >>> configuration = TimesformerConfig()

    >>> # Initializing a model from the configuration
    >>> model = TimesformerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "timesformer"

    def __init__(
        self,
        image_size=224,
        patch_size=16,
        num_channels=3,
        num_frames=8,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-6,
        qkv_bias=True,
        attention_type="divided_space_time",
        drop_path_rate=0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_frames = num_frames

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.qkv_bias = qkv_bias

        self.attention_type = attention_type
        self.drop_path_rate = drop_path_rate


__all__ = ["TimesformerConfig"]
