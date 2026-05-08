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

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="jetmoe/jetmoe-8b")
@strict
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
    ```
    """

    model_type = "jetmoe"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"head_dim": "kv_channels"}

    vocab_size: int = 32000
    hidden_size: int = 2048
    num_hidden_layers: int = 12
    num_key_value_heads: int = 16
    kv_channels: int = 128
    intermediate_size: int = 5632
    max_position_embeddings: int = 4096
    activation_function: str = "silu"
    num_local_experts: int = 8
    num_experts_per_tok: int = 2
    output_router_logits: bool = False
    aux_loss_coef: float = 0.01
    use_cache: bool = True
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    pad_token_id: int | None = None
    tie_word_embeddings: bool = True
    rope_parameters: RopeParameters | dict | None = None
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.01
    attention_dropout: float | int = 0.0

    def __post_init__(self, **kwargs):
        self.num_attention_heads = self.num_key_value_heads * self.num_experts_per_tok
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.num_experts_per_tok > self.num_local_experts:
            raise ValueError("`num_experts_per_tok` must be less than or equal to `num_local_experts`")


__all__ = ["JetMoeConfig"]
