# Copyright 2025 The BitNet Team and The HuggingFace Inc. team. All rights reserved.
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
"""BitNet model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="microsoft/bitnet-b1.58-2B-4T")
@strict
class BitNetConfig(PreTrainedConfig):
    r"""
    ```python
    >>> from transformers import BitNetModel, BitNetConfig

    >>> # Initializing a BitNet style configuration
    >>> configuration = BitNetConfig()

    >>> # Initializing a model from the BitNet style configuration
    >>> model = BitNetModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "bitnet"
    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = 500000.0

    vocab_size: int = 128256
    hidden_size: int = 2560
    intermediate_size: int = 6912
    num_hidden_layers: int = 30
    num_attention_heads: int = 20
    num_key_value_heads: int | None = 5
    hidden_act: str = "relu2"
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = 128000
    eos_token_id: int | list[int] | None = 128001
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    attention_dropout: float | int | None = 0.0
    rope_parameters: RopeParameters | dict | None = None

    def __post_init__(self, **kwargs):
        # for backward compatibility
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        super().__post_init__(**kwargs)


__all__ = ["BitNetConfig"]
