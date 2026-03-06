# Copyright 2024 JetMoe AI and the HuggingFace Inc. team. All rights reserved.
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
"""JetMoe model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="jetmoe/jetmoe-8b")
class JetMoeConfig(PreTrainedConfig):
    r"""
    kv_channels (`int`, *optional*, defaults to 128):
        Defines the number of channels for the key and value tensors.
    num_local_experts (`int`, *optional*, defaults to 8):
        Defines the number of experts in the MoE and MoA.

    ```python
    >>> from transformers import JetMoeModel, JetMoeConfig

    >>> # Initializing a JetMoe 4B style configuration
    >>> configuration = JetMoeConfig()

    >>> # Initializing a model from the JetMoe 4B style configuration
    >>> model = JetMoeModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "jetmoe"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"head_dim": "kv_channels"}

    def __init__(
        self,
        vocab_size: int | None = 32000,
        hidden_size: int | None = 2048,
        num_hidden_layers: int | None = 12,
        num_key_value_heads: int | None = 16,
        kv_channels: int | None = 128,
        intermediate_size: int | None = 5632,
        max_position_embeddings: int | None = 4096,
        activation_function: str | None = "silu",
        num_local_experts: int | None = 8,
        num_experts_per_tok: int | None = 2,
        output_router_logits: bool | None = False,
        aux_loss_coef: float | None = 0.01,
        use_cache: bool | None = True,
        bos_token_id: int | None = 1,
        eos_token_id: int | None = 2,
        pad_token_id: int | None = None,
        tie_word_embeddings: bool | None = True,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        rms_norm_eps: int | None = 1e-6,
        initializer_range: float | None = 0.01,
        attention_dropout: float | None = 0.0,
        **kwargs,
    ):
        if num_experts_per_tok > num_local_experts:
            raise ValueError("`num_experts_per_tok` must be less than or equal to `num_local_experts`")
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_key_value_heads * num_experts_per_tok
        self.num_key_value_heads = num_key_value_heads
        self.kv_channels = kv_channels
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.activation_function = activation_function
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.output_router_logits = output_router_logits
        self.aux_loss_coef = aux_loss_coef
        self.use_cache = use_cache
        self.initializer_range = initializer_range
        self.attention_dropout = attention_dropout

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.rms_norm_eps = rms_norm_eps
        self.rope_parameters = rope_parameters
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(**kwargs)


__all__ = ["JetMoeConfig"]
