# Copyright 2025 Arcee AI and the HuggingFace Inc. team. All rights reserved.
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
"""AFMoE model configuration"""

from ...configuration_utils import PreTrainedConfig, layer_type_validation
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(
    custom_intro="""
    AFMoE is an Adaptive Feedforward MoE (Mixture of Experts) model with token-choice routing, shared experts, and a
    hybrid attention mechanism combining sliding window and full attention patterns.
    """,
    checkpoint="arcee-ai/Trinity-Mini",
)
class AfmoeConfig(PreTrainedConfig):
    r"""
    rope_theta (`float`, *optional*, defaults to 10000.0):
        The base period of the RoPE embeddings.
    global_attn_every_n_layers (`int`, *optional*, defaults to 4):
        The frequency of full attention layers. Every Nth layer will use full attention, while others use sliding
        window attention.
    mup_enabled (`bool`, *optional*, defaults to `False`):
        Whether to enable muP (Maximal Update Parametrization) input scaling. When enabled, input embeddings
        are scaled by `sqrt(hidden_size)`.

    Example:
    ```python
    >>> from transformers import AfmoeModel, AfmoeConfig

    >>> # Initializing an AFMoE configuration
    >>> configuration = AfmoeConfig()

    >>> # Initializing a model from the afmoe-small-sft-v1 style configuration
    >>> model = AfmoeModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "afmoe"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Default pipeline parallel plan for base model
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size: int | None = 200192,
        hidden_size: int | None = 2048,
        intermediate_size: int | None = 6144,
        moe_intermediate_size: int | None = 1408,
        num_hidden_layers: int | None = 32,
        num_dense_layers: int | None = 1,
        num_attention_heads: int | None = 16,
        num_key_value_heads: int | None = None,
        head_dim: int | None = 128,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 16384,
        initializer_range: float | None = 0.02,
        rms_norm_eps: float | None = 1e-5,
        use_cache: bool | None = True,
        tie_word_embeddings: bool | None = False,
        rope_theta: float | None = 10000.0,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        num_experts: int | None = 64,
        num_experts_per_tok: int | None = 6,
        num_shared_experts: int | None = 2,
        route_scale: float | None = 1.0,
        global_attn_every_n_layers: int | None = 4,
        sliding_window: int | None = 1024,
        layer_types: list | None = None,
        attention_dropout: float | None = 0.0,
        mup_enabled: bool | None = False,
        eos_token_id: bool | None = None,
        pad_token_id: bool | None = None,
        bos_token_id: bool | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_dense_layers = num_dense_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_parameters = rope_parameters

        # MoE specific
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.route_scale = route_scale
        self.attention_bias = False

        # Attention specific
        self.attention_dropout = attention_dropout
        self.global_attn_every_n_layers = global_attn_every_n_layers
        self.sliding_window = sliding_window
        self.mup_enabled = mup_enabled
        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % global_attn_every_n_layers) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types)

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.tie_word_embeddings = tie_word_embeddings

        super().__init__(**kwargs)


__all__ = ["AfmoeConfig"]
