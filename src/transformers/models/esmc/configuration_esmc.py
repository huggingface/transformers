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

from ...configuration_utils import PretrainedConfig  # type: ignore[import]


class ESMCConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`ESMCModel`]. It is used to
    instantiate an ESMC model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model
    outputs. Read the documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 64):
            Vocabulary size of the ESMC model. Defines the number of different amino acid tokens that
            can be represented by the ``input_ids`` passed to [`ESMCModel`].
        d_model (`int`, *optional*, defaults to 2560):
            Dimensionality of the encoder layers and the pooler layer.
        n_heads (`int`, *optional*, defaults to 40):
            Number of attention heads for each attention layer in the Transformer encoder.
        n_layers (`int`, *optional*, defaults to 80):
            Number of hidden layers in the Transformer encoder.
        pad_token_id (`int`, *optional*, defaults to 1):
            Index of the padding token in the vocabulary (``"<pad>"``).
        mask_token_id (`int`, *optional*, defaults to 32):
            Index of the mask token in the vocabulary (``"<mask>"``), used for masked language modelling.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated normal initialiser for weight matrix initialisation.
        classifier_dropout (`float`, *optional*, defaults to 0.1):
            Dropout ratio for the classification head.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the rotary position embeddings (RoPE).

    Examples:

    ```python
    >>> from transformers import ESMCConfig, ESMCModel

    >>> # Initializing an ESMC EvolutionaryScale/esmc-600m-2024-12 style configuration
    >>> configuration = ESMCConfig()

    >>> # Initializing a model (with random weights) from the EvolutionaryScale/esmc-600m-2024-12 style configuration
    >>> model = ESMCModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "esmc"

    def __init__(
        self,
        vocab_size: int = 64,
        d_model: int = 2560,
        n_heads: int = 40,
        n_layers: int = 80,
        pad_token_id: int = 1,
        mask_token_id: int = 32,
        initializer_range: float = 0.02,
        classifier_dropout: float = 0.1,
        rope_theta: float = 10000.0,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id, mask_token_id=mask_token_id, **kwargs
        )

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.initializer_range = initializer_range
        self.classifier_dropout = classifier_dropout
        self.rope_theta = rope_theta
        self.tie_word_embeddings = False


__all__ = ["ESMCConfig"]
