# Copyright 2024 HuggingFace Inc. team. All rights reserved.
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
"""Pixtral model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="pixtral-hf/pixtral-9b")
class PixtralVisionConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import PixtralVisionModel, PixtralVisionConfig

    >>> # Initializing a Pixtral-12B style configuration
    >>> config = PixtralVisionConfig()

    >>> # Initializing a model (with randomly initialized weights) from the configuration
    >>> model = PixtralVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "pixtral"

    def __init__(
        self,
        hidden_size: int | None = 1024,
        intermediate_size: int | None = 4096,
        num_hidden_layers: int | None = 24,
        num_attention_heads: int | None = 16,
        num_channels: int | None = 3,
        image_size: int | None = 1024,
        patch_size: int | None = 16,
        hidden_act: str | None = "gelu",
        attention_dropout: float | None = 0.0,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        initializer_range: float | None = 0.02,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.head_dim = hidden_size // num_attention_heads
        self.initializer_range = initializer_range
        self.rope_parameters = rope_parameters

        super().__init__(**kwargs)


__all__ = ["PixtralVisionConfig"]
