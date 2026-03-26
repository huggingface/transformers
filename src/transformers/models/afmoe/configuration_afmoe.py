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

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@strict
@auto_docstring(
    custom_intro="""
    AFMoE is an Adaptive Feedforward MoE (Mixture of Experts) model with token-choice routing, shared experts, and a
    hybrid attention mechanism combining sliding window and full attention patterns.
    """,
    checkpoint="arcee-ai/Trinity-Mini",
)
class AfmoeConfig(PreTrainedConfig):
    r"""
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

    vocab_size: int = 200192
    hidden_size: int = 2048
    intermediate_size: int = 6144
    moe_intermediate_size: int = 1408
    num_hidden_layers: int = 32
    num_dense_layers: int | None = 1
    num_attention_heads: int = 16
    num_key_value_heads: int | None = None
    head_dim: int | None = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 16384
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    num_experts: int | None = 64
    num_experts_per_tok: int | None = 6
    num_shared_experts: int | None = 2
    route_scale: float | None = 1.0
    output_router_logits: bool = False
    global_attn_every_n_layers: int | None = 4
    sliding_window: int | None = 1024
    layer_types: list[str] | None = None
    attention_dropout: float | int | None = 0.0
    mup_enabled: bool | None = False
    eos_token_id: int | list[int] | None = None
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    attention_bias: bool = False

    def __post_init__(self, **kwargs):
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % self.global_attn_every_n_layers) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        super().__post_init__(**kwargs)


__all__ = ["AfmoeConfig"]
