# Copyright 2022 The OpenBMB Team and The HuggingFace Inc. team. All rights reserved.
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
"""CPMAnt model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="openbmb/cpm-ant-10b")
@strict
class CpmAntConfig(PreTrainedConfig):
    r"""
    position_bias_num_buckets (`int`, *optional*, defaults to 512):
        The number of position_bias buckets.
    position_bias_max_distance (`int`, *optional*, defaults to 2048):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    prompt_types (`int`, *optional*, defaults to 32):
        The type of prompt.
    prompt_length (`int`, *optional*, defaults to 32):
        The length of prompt.
    segment_types (`int`, *optional*, defaults to 32):
        The type of segment.

    Example:

    ```python
    >>> from transformers import CpmAntModel, CpmAntConfig

    >>> # Initializing a CPMAnt cpm-ant-10b style configuration
    >>> configuration = CpmAntConfig()

    >>> # Initializing a model from the cpm-ant-10b style configuration
    >>> model = CpmAntModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "cpmant"

    vocab_size: int = 30720
    hidden_size: int = 4096
    num_attention_heads: int = 32
    dim_head: int = 128
    dim_ff: int = 10240
    num_hidden_layers: int = 48
    dropout_p: float = 0.0
    position_bias_num_buckets: int = 512
    position_bias_max_distance: int = 2048
    eps: float = 1e-6
    init_std: float = 1.0
    prompt_types: int = 32
    prompt_length: int = 32
    segment_types: int = 32
    use_cache: bool = True
    tie_word_embeddings: bool = True


__all__ = ["CpmAntConfig"]
