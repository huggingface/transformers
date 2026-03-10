# Copyright 2026 Sarvam AI and the HuggingFace Inc. team. All rights reserved.
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
"""SarvamMLA model configuration"""

from ...utils import auto_docstring
from ..deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config


@auto_docstring(checkpoint="sarvamai/sarvam-105b")
class SarvamMLAConfig(DeepseekV3Config):
    r"""
    n_group (`int`, *optional*, defaults to 16):
        Number of groups for routed experts.
    rope_interleave (`bool`, *optional*, defaults to `True`):
        Whether to interleave the rotary position embeddings.
    first_k_dense_replace (`int`, *optional*, defaults to 1):
        Number of dense layers in shallow layers(embed->dense->moe->moe...->lm_head).
                                                        \--k dense layers--/

    Example:

    ```python
    >>> from transformers import SarvamMLAModel, SarvamMLAConfig

    >>> # Initializing a SarvamMLA style configuration
    >>> configuration = SarvamMLAConfig()

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "sarvam_mla"
    attribute_map = {
        "n_routed_experts": "num_experts",
        "n_shared_experts": "num_shared_experts",
        "num_local_experts": "num_experts",
    }

    def __init__(
        self,
        vocab_size=262144,
        hidden_size=4096,
        intermediate_size=16384,
        moe_intermediate_size=2048,
        num_hidden_layers=32,
        num_attention_heads=64,
        num_key_value_heads=None,
        num_shared_experts=1,
        num_experts=128,
        routed_scaling_factor=2.5,
        kv_lora_rank=512,
        q_lora_rank=None,
        qk_rope_head_dim=64,
        v_head_dim=128,
        qk_nope_head_dim=128,
        n_group=16,
        topk_group=2,
        num_experts_per_tok=8,
        first_k_dense_replace=1,
        norm_topk_prob=True,
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.006,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=None,
        eos_token_id=1,
        tie_word_embeddings=False,
        rope_parameters=None,
        rope_interleave=True,
        attention_bias=False,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts

        # head_dim in the Hub config.json is set to kv_lora_rank + qk_rope_head_dim
        # for vLLM MLA compatibility, but internally the model uses qk_rope_head_dim
        # for rotary embeddings. Remove it from kwargs to prevent overriding.
        kwargs.pop("head_dim", None)
        kwargs.pop("q_head_dim", None)

        # The Hub config uses "deepseek_yarn" as rope type; normalize to "yarn"
        # which is the standard type in ROPE_INIT_FUNCTIONS.
        if rope_parameters is not None and rope_parameters.get("type") == "deepseek_yarn":
            rope_parameters = dict(rope_parameters)
            rope_parameters["type"] = "yarn"
        rope_scaling = kwargs.pop("rope_scaling", None)
        if rope_scaling is not None:
            if rope_scaling.get("type") == "deepseek_yarn":
                rope_scaling = dict(rope_scaling)
                rope_scaling["type"] = "yarn"
            if rope_parameters is None:
                rope_parameters = rope_scaling

        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            moe_intermediate_size=moe_intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            n_shared_experts=num_shared_experts,
            n_routed_experts=num_experts,
            routed_scaling_factor=routed_scaling_factor,
            kv_lora_rank=kv_lora_rank,
            q_lora_rank=q_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            qk_nope_head_dim=qk_nope_head_dim,
            n_group=n_group,
            topk_group=topk_group,
            num_experts_per_tok=num_experts_per_tok,
            first_k_dense_replace=first_k_dense_replace,
            norm_topk_prob=norm_topk_prob,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            rope_parameters=rope_parameters,
            rope_interleave=rope_interleave,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            **kwargs,
        )


__all__ = ["SarvamMLAConfig"]
