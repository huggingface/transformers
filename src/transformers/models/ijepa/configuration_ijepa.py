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
"""I-JEPA model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="facebook/ijepa_vith14_1k")
class IJepaConfig(PreTrainedConfig):
    r"""
    pooler_output_size (`int`, *optional*):
       Dimensionality of the pooler layer. If None, defaults to `hidden_size`.
    pooler_act (`str`, *optional*, defaults to `"tanh"`):
       The activation function to be used by the pooler.

    Example:

    ```python
    >>> from transformers import IJepaConfig, IJepaModel

    >>> # Initializing a IJEPA ijepa-base-patch16-224 style configuration
    >>> configuration = IJepaConfig()

    >>> # Initializing a model (with random weights) from the ijepa-base-patch16-224 style configuration
    >>> model = IJepaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "ijepa"

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        image_size=224,
        patch_size=16,
        num_channels=3,
        qkv_bias=True,
        pooler_output_size=None,
        pooler_act="tanh",
        **kwargs,
    ):
        super().__init__(**kwargs)

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
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.pooler_output_size = pooler_output_size if pooler_output_size else hidden_size
        self.pooler_act = pooler_act


__all__ = ["IJepaConfig"]
