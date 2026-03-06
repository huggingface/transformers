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
"""GraniteMoe model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="ibm-granite/granite-speech-3.2-8b")
class GraniteMoeConfig(PreTrainedConfig):
    r"""
        embedding_multiplier (`float`, *optional*, defaults to 1.0): embedding multiplier
        logits_scaling (`float`, *optional*, defaults to 1.0): divisor for output logits
        residual_multiplier (`float`, *optional*, defaults to 1.0): residual multiplier
        attention_multiplier (`float`, *optional*, defaults to 1.0): attention multiplier

    ```python
    >>> from transformers import GraniteMoeModel, GraniteMoeConfig

    >>> # Initializing a GraniteMoe granitemoe-3b style configuration
    >>> configuration = GraniteMoeConfig()

    >>> # Initializing a model from the granitemoe-7b style configuration
    >>> model = GraniteMoeModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "granitemoe"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int | None = 32000,
        hidden_size: int | None = 4096,
        intermediate_size: int | None = 11008,
        num_hidden_layers: int | None = 32,
        num_attention_heads: int | None = 32,
        num_key_value_heads: int | None = None,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 2048,
        initializer_range: float | None = 0.02,
        rms_norm_eps: int | None = 1e-6,
        use_cache: bool | None = True,
        pad_token_id: int | None = None,
        bos_token_id: int | None = 1,
        eos_token_id: int | None = 2,
        tie_word_embeddings: bool | None = False,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        attention_bias: bool | None = False,
        attention_dropout: float | None = 0.0,
        embedding_multiplier: float | None = 1.0,
        logits_scaling: float | None = 1.0,
        residual_multiplier: float | None = 1.0,
        attention_multiplier: float | None = 1.0,
        num_local_experts: int | None = 8,
        num_experts_per_tok: int | None = 2,
        output_router_logits: bool | None = False,
        router_aux_loss_coef: float | None = 0.001,
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

        self.embedding_multiplier = embedding_multiplier
        self.logits_scaling = logits_scaling
        self.residual_multiplier = residual_multiplier
        self.attention_multiplier = attention_multiplier

        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef

        self.tie_word_embeddings = tie_word_embeddings
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.rope_parameters = rope_parameters

        super().__init__(**kwargs)


__all__ = ["GraniteMoeConfig"]
