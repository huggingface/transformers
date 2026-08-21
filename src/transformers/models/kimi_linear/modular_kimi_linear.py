# Copyright 2026 The HuggingFace Inc. team
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
from huggingface_hub.dataclasses import strict

from ...modeling_rope_utils import RopeParameters
from ...models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from ...utils import auto_docstring


@auto_docstring(checkpoint="moonshotai/Kimi-Linear-48B-A3B-Base")
@strict
class KimiLinearConfig(DeepseekV3Config):

    model_type: str = "kimi_linear"
    attribute_map = {
        "model_max_length": "max_position_embeddings",
        "moe_renormalize": "norm_topk_prob",
        "num_expert_group": "n_group",
        "num_experts": "n_routed_experts",
        "num_mtp_layers": "num_nextn_predict_layers",
    }

    vocab_size: int = 163840
    hidden_size: int = 2304
    intermediate_size: int = 9216
    moe_intermediate_size: int = 1024
    num_hidden_layers: int = 27
    num_attention_heads: int = 32
    num_key_value_heads: int | None = 32
    num_shared_experts: int = 1
    n_routed_experts: int = 256
    routed_scaling_factor: float = 2.446
    kv_lora_rank: int = 512
    q_lora_rank: int | None = None
    qk_rope_head_dim: int = 64
    v_head_dim: int | None = 128
    qk_nope_head_dim: int = 128
    n_group: int = 8
    topk_group: int | None = 1
    num_experts_per_tok: int | None = 8
    first_k_dense_replace: int | None = 1
    norm_topk_prob: bool = True
    hidden_act: str = "silu"
    max_position_embeddings: int = 1048576
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int | None = 163839
    bos_token_id: int | None = 163584
    eos_token_id: int | list[int] | None = 163586
    # pretraining_tp: int | None = 1
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    # rope_interleave: bool | None = True
    # attention_bias: bool = False
    # attention_dropout: float | int | None = 0.0
    layer_types: list[str] | None = None
    num_mtp_layers: int = 0

    head_dim: int = 72
    linear_key_head_dim: int = 128
    linear_num_key_heads: int = 32
    linear_conv_kernel_dim: int = 4

    def __post_init__(self, **kwargs):
        # Checkpoint stores linear attention attributes in a config sub-dict: if it's there, extract them
        linear_attn_config = kwargs.pop("linear_attn_config", {})
        self.linear_key_head_dim = linear_attn_config.get("head_dim", self.linear_key_head_dim)
        self.linear_num_key_heads = linear_attn_config.get("num_heads", self.linear_num_key_heads)
        self.linear_conv_kernel_dim = linear_attn_config.get("short_conv_kernel_size", self.linear_conv_kernel_dim)
        # Values head have the same config as key heads
        self.linear_value_head_dim = self.linear_key_head_dim
        self.linear_num_value_heads = self.linear_num_key_heads

        # For layer types, the precedence is: explcit `layer_types` > checkpoint config > default
        if self.layer_types is None:
            pass  # nothing to do here
        elif "full_attn_layers" in linear_attn_config and "kda_layers" in linear_attn_config:
            self.layer_types = [None] * self.num_hidden_layers
            for layer in linear_attn_config["full_attn_layers"]:
                self.layer_types[layer - 1] = "full_attention"  # for some reason, types are 1-indexed in the checkpoint
            for layer in linear_attn_config["kda_layers"]:
                self.layer_types[layer - 1] = "kda_attention"
            if None in self.layer_types:
                raise ValueError(
                    "Layer types are not fully specified. You can provide an explicit `layer_types` list to solve this."
                )
        else:
            self.layer_types = [
                "full_attention" if i and i % 4 == 0 else "kda_attention" for i in range(self.num_hidden_layers)
            ]

        # By default, only full attention layers use rotary embeddings
        if self.rope_parameters is None:
            rope_scaling = kwargs.pop("rope_scaling", None)
            rope_theta = kwargs.pop("rope_theta", 10000.0)
            full_attention_rope = {"rope_type": "?", "rope_scaling": rope_scaling, "rope_theta": rope_theta}
            mla_rope = {} if kwargs.pop("mla_use_nope", True) else dict(full_attention_rope.items())
            self.rope_parameters = {"full_attention": full_attention_rope, "kda_attention": mla_rope}
