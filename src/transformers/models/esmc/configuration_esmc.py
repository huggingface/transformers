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

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="biohub/ESMC-6B")
@strict
class ESMCConfig(PreTrainedConfig):
    r"""
    hidden_size (`int`, *optional*, defaults to 2560):
        Dimensionality of the encoder layers and the pooler layer.
    num_attention_heads (`int`, *optional*, defaults to 40):
        Number of attention heads for each attention layer in the Transformer encoder.
    num_hidden_layers (`int`, *optional*, defaults to 80):
        Number of hidden layers in the Transformer encoder.
    mask_token_id (`int`, *optional*, defaults to 32):
        Index of the mask token in the vocabulary (``"<mask>"``), used for masked language modelling.
    classifier_dropout (`float`, *optional*, defaults to 0.1):
        Dropout ratio for the classification head.
    max_position_embeddings (`int`, *optional*, defaults to 2048):
        Nominal maximum sequence length. RoPE imposes no hard limit, so this only sizes
        the rotary cache for dynamic RoPE variants (unused by the default RoPE).
    rope_parameters (`RopeParameters` or `dict`, *optional*):
        Dictionary configuring the rotary position embeddings (RoPE). When omitted, the
        default RoPE is used with `rope_theta` = `default_theta` (10000.0). See
        [`~modeling_rope_utils.RopeParameters`] for the accepted keys.
    expansion_ratio (`float`, *optional*, defaults to `8/3`):
        Hidden-dim expansion ratio for the SwiGLU feed-forward network. When
        `intermediate_size` is not given it is derived from this as
        `expansion_ratio * hidden_size` rounded up to a multiple of 256.
    intermediate_size (`int`, *optional*):
        Dimensionality of the SwiGLU feed-forward layer. Defaults to the value
        derived from `expansion_ratio` (see above).
    hidden_act (`str`, *optional*, defaults to `"silu"`):
        The non-linear activation function in the feed-forward network.
    qk_layernorm (`bool`, *optional*, defaults to `True`):
        Whether to apply LayerNorm to queries and keys before computing attention.
    scale_residue (`bool`, *optional*, defaults to `True`):
        Whether to apply ESM3 residual scaling (`1 / sqrt(num_hidden_layers / 36)`
        per block) to stabilise deep networks.

    Examples:

    ```python
    >>> from transformers import ESMCConfig, ESMCModel

    >>> # Initializing an ESMC biohub/ESMC-6B style configuration
    >>> configuration = ESMCConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = ESMCModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "esmc"
    default_theta = 10000.0
    attribute_map = {
        "d_model": "hidden_size",
        "n_heads": "num_attention_heads",
        "n_layers": "num_hidden_layers",
    }

    vocab_size: int | None = 64
    hidden_size: int | None = 2560
    num_attention_heads: int | None = 40
    num_hidden_layers: int | None = 80
    pad_token_id: int | None = 1
    mask_token_id: int | None = 32
    initializer_range: float | None = 0.02
    classifier_dropout: float | None = 0.1
    max_position_embeddings: int | None = 2048
    rope_parameters: RopeParameters | dict | None = None
    expansion_ratio: float | None = 8 / 3
    intermediate_size: int | None = None
    hidden_act: str | None = "silu"
    mlp_bias: bool | None = False
    qk_layernorm: bool | None = True
    scale_residue: bool | None = True
    tie_word_embeddings: bool | None = False

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        if self.intermediate_size is None:
            self.intermediate_size = int(((self.expansion_ratio * self.hidden_size) + 255) // 256 * 256)


__all__ = ["ESMCConfig"]
