# Copyright 2024 Stability AI and The HuggingFace Inc. team. All rights reserved.
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
"""StableLM model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="stabilityai/stablelm-3b-4e1t")
class StableLmConfig(PreTrainedConfig):
    r"""
    use_parallel_residual (`bool`, *optional*, defaults to `False`):
        Whether to use a "parallel" formulation in each Transformer layer, which can provide a slight training
        speedup at large scales.
    hidden_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio after applying the MLP to the hidden states.

    Example:

    ```python
    >>> from transformers import StableLmModel, StableLmConfig

    >>> # Initializing a StableLM stablelm-3b style configuration
    >>> configuration = StableLmConfig()
    ```"""

    model_type = "stablelm"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int | None = 50304,
        intermediate_size: int | None = 6912,
        hidden_size: int | None = 2560,
        num_hidden_layers: int | None = 32,
        num_attention_heads: int | None = 32,
        num_key_value_heads: int | None = 32,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 4096,
        initializer_range: float | None = 0.02,
        layer_norm_eps: float | None = 1.0e-5,
        use_cache: bool | None = True,
        tie_word_embeddings: bool | None = False,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        use_qkv_bias: bool | None = False,
        qk_layernorm: bool | None = False,
        use_parallel_residual: bool | None = False,
        hidden_dropout: float | None = 0.0,
        attention_dropout: float | None = 0.0,
        bos_token_id: int | None = 0,
        eos_token_id: int | None = 0,
        pad_token_id: int | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act

        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.use_qkv_bias = use_qkv_bias
        self.qk_layernorm = qk_layernorm
        self.use_parallel_residual = use_parallel_residual
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.rope_parameters = rope_parameters
        kwargs.setdefault("partial_rotary_factor", 0.25)  # assign default for BC

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(**kwargs)


__all__ = ["StableLmConfig"]
