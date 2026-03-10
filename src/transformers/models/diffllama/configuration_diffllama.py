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

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="kajuma/DiffLlama-0.3B-handcut")
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
    ```"""

    model_type = "diffllama"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int | None = 32000,
        hidden_size: int | None = 2048,
        intermediate_size: int | None = 8192,
        num_hidden_layers: int | None = 16,
        num_attention_heads: int | None = 32,
        num_key_value_heads: int | None = None,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 2048,
        initializer_range: float | None = 0.02,
        rms_norm_eps: int | None = 1e-5,
        use_cache: bool | None = True,
        pad_token_id: int | None = None,
        bos_token_id: int | None = 1,
        eos_token_id: int | None = 2,
        tie_word_embeddings: bool | None = False,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        attention_bias: bool | None = False,
        attention_dropout: float | None = 0.0,
        lambda_std_dev: float | None = 0.1,
        head_dim: int | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.lambda_std_dev = lambda_std_dev
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        self.rope_parameters = rope_parameters

        self.tie_word_embeddings = tie_word_embeddings
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        super().__init__(**kwargs)


__all__ = ["DiffLlamaConfig"]
