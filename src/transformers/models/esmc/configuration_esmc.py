# Copyright 2026 Biohub. All rights reserved.
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
"""ESMC model configuration."""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig  # type: ignore[import]
from ...utils import auto_docstring


@auto_docstring(checkpoint="biohub/ESMC-600M")
@strict
class ESMCConfig(PreTrainedConfig):
    r"""
    d_model (`int`, *optional*, defaults to 2560):
        Dimensionality of the encoder layers and the pooler layer.
    n_heads (`int`, *optional*, defaults to 40):
        Number of attention heads for each attention layer in the Transformer encoder.
    n_layers (`int`, *optional*, defaults to 80):
        Number of hidden layers in the Transformer encoder.
    mask_token_id (`int`, *optional*, defaults to 32):
        Index of the mask token in the vocabulary (``"<mask>"``), used for masked language modelling.
    classifier_dropout (`float`, *optional*, defaults to 0.1):
        Dropout ratio for the classification head.
    rope_theta (`float`, *optional*, defaults to 10000.0):
        The base period of the rotary position embeddings (RoPE).

    Examples:

    ```python
    >>> from transformers import ESMCConfig, ESMCModel

    >>> # Initializing an ESMC biohub/ESMC-600M style configuration
    >>> configuration = ESMCConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = ESMCModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "esmc"

    vocab_size: int | None = 64
    d_model: int | None = 2560
    n_heads: int | None = 40
    n_layers: int | None = 80
    pad_token_id: int | None = 1
    mask_token_id: int | None = 32
    initializer_range: float | None = 0.02
    classifier_dropout: float | None = 0.1
    rope_theta: float | None = 10000.0
    tie_word_embeddings: bool | None = False


__all__ = ["ESMCConfig"]
