# Copyright 2023 Adept AI and the HuggingFace Inc. team. All rights reserved.
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
"""Persimmon model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="adept/persimmon-8b-base")
class PersimmonConfig(PreTrainedConfig):
    r""".
    qk_layernorm (`bool`, *optional*, default to `True`):
        Whether or not to normalize the Queries and Keys after projecting the hidden states

    Example:

    ```python
    >>> from transformers import PersimmonModel, PersimmonConfig

    >>> # Initializing a Persimmon persimmon-7b style configuration
    >>> configuration = PersimmonConfig()
    ```"""

    model_type = "persimmon"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int | None = 262144,
        hidden_size: int | None = 4096,
        intermediate_size: int | None = 16384,
        num_hidden_layers: int | None = 36,
        num_attention_heads: int | None = 64,
        hidden_act: str | None = "relu2",
        max_position_embeddings: int | None = 16384,
        initializer_range: float | None = 0.02,
        layer_norm_eps: int | None = 1e-5,
        use_cache: bool | None = True,
        tie_word_embeddings: bool | None = False,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        qk_layernorm: bool | None = True,
        hidden_dropout: float | None = 0.0,
        attention_dropout: float | None = 0.0,
        pad_token_id: int | None = None,
        bos_token_id: int | None = 1,
        eos_token_id: int | None = 2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.qk_layernorm = qk_layernorm
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.rope_parameters = rope_parameters
        kwargs.setdefault("partial_rotary_factor", 0.5)  # assign default for BC

        self.tie_word_embeddings = tie_word_embeddings
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        super().__init__(**kwargs)


__all__ = ["PersimmonConfig"]
