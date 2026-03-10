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

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="microsoft/bitnet-b1.58-2B-4T")
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
    ```"""

    model_type = "bitnet"
    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = 500000.0

    def __init__(
        self,
        vocab_size: int | None = 128256,
        hidden_size: int | None = 2560,
        intermediate_size: int | None = 6912,
        num_hidden_layers: int | None = 30,
        num_attention_heads: int | None = 20,
        num_key_value_heads: int | None = 5,
        hidden_act: str | None = "relu2",
        max_position_embeddings: int | None = 2048,
        initializer_range: float | None = 0.02,
        rms_norm_eps: int | None = 1e-5,
        use_cache: bool | None = True,
        pad_token_id: int | None = None,
        bos_token_id: int | None = 128000,
        eos_token_id: int | None = 128001,
        tie_word_embeddings: bool | None = False,
        attention_bias: bool | None = False,
        attention_dropout: str | None = 0.0,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
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
        self.rope_parameters = rope_parameters

        self.tie_word_embeddings = tie_word_embeddings
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        super().__init__(**kwargs)


__all__ = ["BitNetConfig"]
