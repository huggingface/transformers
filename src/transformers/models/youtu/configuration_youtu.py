# Copyright 2025 Tencent Youtu Lab and the HuggingFace Inc. team. All rights reserved.

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
"""Youtu-LLM model configuration"""

from ...modeling_rope_utils import RopeParameters
from ..deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config


class YoutuConfig(DeepseekV3Config):
    r"""
    This is the configuration class to store the configuration of a [`YoutuModel`]. It is used to instantiate an Youtu
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Youtu-LLM-2B.
    e.g. [tencent/Youtu-LLM-2B](https://huggingface.co/tencent/Youtu-LLM-2B)
    Configuration objects inherit from [`DeepseekV3Config`] and can be used to control the model outputs. Read the
    documentation from [`DeepseekV3Config`] for more information.
    Args:
            vocab_size (`int | None`, *optional*, defaults to 128256): <fill_docstring>
            hidden_size (`int | None`, *optional*, defaults to 2048): <fill_docstring>
            intermediate_size (`int | None`, *optional*, defaults to 6144): <fill_docstring>
            num_hidden_layers (`int | None`, *optional*, defaults to 32): <fill_docstring>
            num_attention_heads (`int | None`, *optional*, defaults to 16): <fill_docstring>
            num_key_value_heads (`int | None`, *optional*, defaults to 16): <fill_docstring>
            kv_lora_rank (`int | None`, *optional*, defaults to 512): <fill_docstring>
            q_lora_rank (`int | None`, *optional*, defaults to 1536): <fill_docstring>
            qk_rope_head_dim (`int | None`, *optional*, defaults to 64): <fill_docstring>
            v_head_dim (`int | None`, *optional*, defaults to 128): <fill_docstring>
            qk_nope_head_dim (`int | None`, *optional*, defaults to 128): <fill_docstring>
            hidden_act (`str | None`, *optional*, defaults to `"silu"`): <fill_docstring>
            max_position_embeddings (`int | None`, *optional*, defaults to 131072): <fill_docstring>
            initializer_range (`float | None`, *optional*): <fill_docstring>
            embedding_initializer_range (`float | None`, *optional*): <fill_docstring>
            rms_norm_eps (`int | None`, *optional*, defaults to 1e-06): <fill_docstring>
            use_cache (`bool | None`, *optional*, defaults to `True`): <fill_docstring>
            pad_token_id (`int | None`, *optional*): <fill_docstring>
            bos_token_id (`int | None`, *optional*, defaults to 128000): <fill_docstring>
            eos_token_id (`int | None`, *optional*, defaults to 128001): <fill_docstring>
            tie_word_embeddings (`bool | None`, *optional*, defaults to `True`): <fill_docstring>
            rope_parameters (`transformers.modeling_rope_utils.RopeParameters | dict[str, transformers.modeling_rope_utils.RopeParameters]`, *optional*): <fill_docstring>
            rope_interleave (`bool | None`, *optional*, defaults to `True`): <fill_docstring>
            attention_bias (`bool | None`, *optional*, defaults to `False`): <fill_docstring>
            attention_dropout (`float | None`, *optional*, defaults to 0.0): <fill_docstring>
    ```python
    >>> from transformers import YoutuModel, YoutuConfig
    >>> # Initializing a Youtu-LLM-2B style configuration
    >>> configuration = YoutuConfig()
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "youtu"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size: int | None = 128256,
        hidden_size: int | None = 2048,
        intermediate_size: int | None = 6144,
        num_hidden_layers: int | None = 32,
        num_attention_heads: int | None = 16,
        num_key_value_heads: int | None = 16,
        kv_lora_rank: int | None = 512,
        q_lora_rank: int | None = 1536,
        qk_rope_head_dim: int | None = 64,
        v_head_dim: int | None = 128,
        qk_nope_head_dim: int | None = 128,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 131072,
        initializer_range: float | None = None,
        embedding_initializer_range: float | None = None,
        rms_norm_eps: int | None = 1e-6,
        use_cache: bool | None = True,
        pad_token_id: int | None = None,
        bos_token_id: int | None = 128000,
        eos_token_id: int | None = 128001,
        tie_word_embeddings: bool | None = True,
        rope_parameters: RopeParameters | dict[str, RopeParameters] = None,
        rope_interleave: bool | None = True,
        attention_bias: bool | None = False,
        attention_dropout: float | None = 0.0,
        **kwargs,
    ):
        super().__init__()
        # remove unused attribute
        del self.n_shared_experts
        del self.n_routed_experts
        del self.routed_scaling_factor
        del self.n_group
        del self.topk_group
        del self.num_experts_per_tok
        del self.first_k_dense_replace
        del self.norm_topk_prob
        del self.pretraining_tp

        # if initializer_range is None, set it to 2.0 / (5.0 * self.hidden_size) ** 0.5
        self.initializer_range = (
            (2.0 / (5.0 * self.hidden_size)) ** 0.5 if initializer_range is None else initializer_range
        )
        # if embedding_initializer_range is None, set it to 2.0 * self.initializer_range
        self.embedding_initializer_range = (
            self.initializer_range * 2.0 if embedding_initializer_range is None else embedding_initializer_range
        )


__all__ = ["YoutuConfig"]
