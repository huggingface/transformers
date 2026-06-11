# Copyright 2024 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""GraniteMoeShared model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="ibm-granite/granite-speech-3.2-8b")
@strict
class GraniteMoeSharedConfig(PreTrainedConfig):
    r"""
    embedding_multiplier (`float`, *optional*, defaults to 1.0):
        embedding multiplier
    logits_scaling (`float`, *optional*, defaults to 1.0):
        divisor for output logits
    residual_multiplier (`float`, *optional*, defaults to 1.0):
        residual multiplier
    attention_multiplier (`float`, *optional*, defaults to 1.0):
        attention multiplier
    shared_intermediate_size (`int`, *optional*, defaults to 1024):
        intermediate size for shared experts.

    ```python
    >>> from transformers import GraniteMoeSharedModel, GraniteMoeSharedConfig

    >>> # Initializing a GraniteMoeShared granitemoe-3b style configuration
    >>> configuration = GraniteMoeSharedConfig()

    >>> # Initializing a model from the granitemoe-7b style configuration
    >>> model = GraniteMoeSharedModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "granitemoeshared"
    keys_to_ignore_at_inference = ["past_key_values"]

    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int | None = None
    hidden_act: str = "silu"
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int | None = 0.0
    embedding_multiplier: float | int | None = 1.0
    logits_scaling: float | int | None = 1.0
    residual_multiplier: float | int | None = 1.0
    attention_multiplier: float | int | None = 1.0
    num_local_experts: int | None = 8
    num_experts_per_tok: int | None = 2
    output_router_logits: bool | None = False
    router_aux_loss_coef: float | None = 0.001
    shared_intermediate_size: int = 0

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        super().__post_init__(**kwargs)


__all__ = ["GraniteMoeSharedConfig"]
