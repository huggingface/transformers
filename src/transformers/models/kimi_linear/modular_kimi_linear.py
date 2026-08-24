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


from collections.abc import Callable

import torch
import torch.nn.functional as F
from einops import rearrange
from huggingface_hub.dataclasses import strict
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...integrations import use_kernel_func_from_hub_with_fallback
from ...masking_utils import create_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from ...models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3Attention,
    DeepseekV3Experts,
    DeepseekV3ForCausalLM,
    DeepseekV3MLP,
    DeepseekV3MoE,
    DeepseekV3TopkRouter,
)
from ...models.llama.modeling_llama import LlamaRMSNorm, eager_attention_forward  # used in modeling
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import OutputRecorder, check_model_inputs


@auto_docstring(checkpoint="moonshotai/Kimi-Linear-48B-A3B-Base")
@strict
class KimiLinearConfig(DeepseekV3Config):
    model_type = "kimi_linear"
    attribute_map = {
        "model_max_length": "max_position_embeddings",
        "moe_renormalize": "norm_topk_prob",
        "num_expert_group": "n_group",
        "num_experts": "n_routed_experts",
        "num_experts_per_token": "num_experts_per_tok",
        "num_mtp_layers": "num_nextn_predict_layers",
    }

    vocab_size: int = 163840
    hidden_size: int = 2304
    intermediate_size: int = 9216
    moe_intermediate_size: int = 1024
    num_hidden_layers: int = 27
    num_attention_heads: int = 32
    num_key_value_heads: int | None = 32
    routed_scaling_factor: float = 2.446
    q_lora_rank: int | None = None
    n_group: int = 1
    topk_group: int | None = 1
    first_k_dense_replace: int | None = 1
    norm_topk_prob: bool = True
    max_position_embeddings: int = 1048576
    rms_norm_eps: float = 1e-5
    pad_token_id: int | None = 163839
    bos_token_id: int | None = 163584
    eos_token_id: int | list[int] | None = 163586
    layer_types: list[str] | None = None
    num_mtp_layers: int = 0

    head_dim: int = 72
    linear_key_head_dim: int = 128
    linear_num_key_heads: int = 32
    linear_conv_kernel_dim: int = 4

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        # Checkpoint stores linear attention attributes in a config sub-dict: if it's there, extract them
        linear_attn_config = kwargs.pop("linear_attn_config", {})
        self.linear_key_head_dim = linear_attn_config.get("head_dim", self.linear_key_head_dim)
        self.linear_num_key_heads = linear_attn_config.get("num_heads", self.linear_num_key_heads)
        self.linear_conv_kernel_dim = linear_attn_config.get("short_conv_kernel_size", self.linear_conv_kernel_dim)
        # Values head have the same config as key heads
        self.linear_value_head_dim = self.linear_key_head_dim
        self.linear_num_value_heads = self.linear_num_key_heads

        # For layer types, the precedence is: explcit `layer_types` > checkpoint config > default
        if self.layer_types is not None:
            pass  # nothing to do here
        elif "full_attn_layers" in linear_attn_config and "kda_layers" in linear_attn_config:
            self.layer_types = [None] * self.num_hidden_layers
            for layer in linear_attn_config["full_attn_layers"]:
                self.layer_types[layer - 1] = "full_attention"  # types are 1-indexed in the checkpoint
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

        # Attention layers never use bias, this is kept to inherit from DSV3
        self.attention_bias: bool = False

class KimiLinearRMSNorm(LlamaRMSNorm):
    pass


class KimiLinearExperts(DeepseekV3Experts):
    pass


class KimiLinearMLP(DeepseekV3MLP):
    pass



class KimiLinearAttention(DeepseekV3Attention):
    """Multi-headed Latent Attention (MLA) from Deepseek V2 with NoPE, but the part of the keys where RoPE is applied is
    still shared."""

    def __init__(self, config: KimiLinearConfig, layer_idx: int):
        config.attention_bias = False
        super().__init__(config, layer_idx)
        self.scaling = self.qk_head_dim ** (-0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)

        if self.q_lora_rank is None:
            q_states = self.q_proj(hidden_states)
        else:
            q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        query_states = q_states.view(query_shape).transpose(1, 2)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        kv_nope, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_nope = self.kv_a_layernorm(kv_nope)
        # Both latents are viewed as single-head, 4D tensors so all cache layers handle them correctly
        kv_nope = kv_nope.view(batch_size, 1, seq_length, self.kv_lora_rank)
        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        # Cache read / write is performed while latent KV is still compressed
        if past_key_values is not None:
            kv_nope, k_rot = past_key_values.update(kv_nope, k_rot, self.layer_idx)

        key_states, value_states = self.expand_kv(kv_nope, k_rot)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights



class KimiLinearTopkRouter(DeepseekV3TopkRouter):
    pass


class KimiLinearSparseMoeBlock(DeepseekV3MoE):
    pass


