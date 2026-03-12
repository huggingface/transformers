# Copyright 2023 Mixtral AI and the HuggingFace Inc. team. All rights reserved.
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
"""Mixtral model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="mistralai/Mixtral-8x7B-v0.1")
class MixtralConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import MixtralModel, MixtralConfig

    >>> # Initializing a Mixtral 7B style configuration
    >>> configuration = MixtralConfig()

    >>> # Initializing a model from the Mixtral 7B style configuration
    >>> model = MixtralModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "mixtral"
    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = 1000000.0
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    attribute_map = {
        "num_experts": "num_local_experts",
    }

    def __init__(
        self,
        vocab_size: int | None = 32000,
        hidden_size: int | None = 4096,
        intermediate_size: int | None = 14336,
        num_hidden_layers: int | None = 32,
        num_attention_heads: int | None = 32,
        num_key_value_heads: int | None = 8,
        head_dim: int | None = None,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 4096 * 32,
        initializer_range: float | None = 0.02,
        rms_norm_eps: int | None = 1e-5,
        use_cache: bool | None = True,
        pad_token_id: int | None = None,
        bos_token_id: int | None = 1,
        eos_token_id: int | None = 2,
        tie_word_embeddings: bool | None = False,
        sliding_window: int | None = None,
        attention_dropout: float | None = 0.0,
        num_experts_per_tok: int | None = 2,
        num_local_experts: int | None = 8,
        output_router_logits: bool | None = False,
        router_aux_loss_coef: float | None = 0.001,
        router_jitter_noise: float | None = 0.0,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim

        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_jitter_noise = router_jitter_noise
        self.tie_word_embeddings = tie_word_embeddings
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.rope_parameters = rope_parameters

        super().__init__(**kwargs)


__all__ = ["MixtralConfig"]
