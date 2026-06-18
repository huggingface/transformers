# Copyright 2024 weak-kajuma and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on Llama implementations in this library and Microsoft's
# Differential Transformer implementations.

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
"""DiffLlama model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="kajuma/DiffLlama-0.3B-handcut")
@strict
class DiffLlamaConfig(PreTrainedConfig):
    r"""
    lambda_std_dev (`float`, *optional*, defaults to 0.1):
        The standard deviation for initialization of parameter lambda in attention layer.

    ```python
    >>> from transformers import DiffLlamaModel, DiffLlamaConfig

    >>> # Initializing a DiffLlama diffllama-7b style configuration
    >>> configuration = DiffLlamaConfig()

    >>> # Initializing a model from the diffllama-7b style configuration
    >>> model = DiffLlamaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "diffllama"
    keys_to_ignore_at_inference = ["past_key_values"]

    vocab_size: int = 32000
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 16
    num_attention_heads: int = 32
    num_key_value_heads: int | None = None
    hidden_act: str = "silu"
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int | None = 0.0
    lambda_std_dev: float | None = 0.1
    head_dim: int | None = None

    def __post_init__(self, **kwargs):
        # for backward compatibility
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        self.head_dim = self.head_dim if self.head_dim is not None else self.hidden_size // self.num_attention_heads
        super().__post_init__(**kwargs)


__all__ = ["DiffLlamaConfig"]
