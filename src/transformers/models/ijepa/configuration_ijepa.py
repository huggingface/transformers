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

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="facebook/ijepa_vith14_1k")
@strict
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

    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    image_size: int | list[int] | tuple[int, int] = 224
    patch_size: int | list[int] | tuple[int, int] = 16
    num_channels: int = 3
    qkv_bias: bool = True
    pooler_output_size: int | None = None
    pooler_act: str = "tanh"

    def __post_init__(self, **kwargs):
        self.pooler_output_size = self.pooler_output_size if self.pooler_output_size else self.hidden_size
        super().__post_init__(**kwargs)


__all__ = ["IJepaConfig"]
