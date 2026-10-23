# Copyright 2026 Recursive and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Flint model."""

import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import maybe_autocast
from ..gpt_oss.modeling_gpt_oss import GptOssRMSNorm
from ..qwen2_moe.configuration_qwen2_moe import Qwen2MoeConfig
from ..qwen2_moe.modeling_qwen2_moe import Qwen2MoeModel
from ..qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
    Qwen3MoeMLP,
    Qwen3MoePreTrainedModel,
    Qwen3MoeSparseMoeBlock,
    Qwen3MoeTopKRouter,
    apply_rotary_pos_emb,
    repeat_kv,
)


@auto_docstring(checkpoint="RecursiveSI/flint")
@strict
class FlintConfig(Qwen2MoeConfig):
    r"""
    shared_expert_intermediate_size (`int`, *optional*, defaults to 1024):
        Intermediate size of the ungated shared expert that every sparse layer adds to its routed experts.
    mlp_only_layers (`list[int]`, *optional*, defaults to `[0, 1]`):
        Indices of the layers that use a dense MLP instead of the sparse mixture-of-experts block.

    Example:

    ```python
    >>> from transformers import FlintConfig, FlintModel

    >>> configuration = FlintConfig()
    >>> model = FlintModel(configuration)
    >>> configuration = model.config
    ```
    """

    model_type = "flint"
    # Attention is replicated: tensor parallelism cannot shard the per-head `sinks` along with the heads yet
    base_model_tp_plan = {
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.mlp.shared_expert.gate_proj": "colwise",
        "layers.*.mlp.shared_expert.up_proj": "colwise",
        "layers.*.mlp.shared_expert.down_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    vocab_size: int = 150144
    intermediate_size: int = 6144
    num_hidden_layers: int = 49
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    head_dim: int = 128
    attention_bias: bool = False
    max_position_embeddings: int = 262144
    sliding_window: int = 513
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 1024
    num_experts_per_tok: int = 6
    num_experts: int = 256
    norm_topk_prob: bool = True
    router_aux_loss_coef: float = 0.0005

    use_sliding_window = AttributeError()
    max_window_layers = AttributeError()
    qkv_bias = AttributeError()

    def __post_init__(self, **kwargs):
        # Full attention on the first two layers, every fourth layer and the last layer; sliding window elsewhere
        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if i < 2 or i % 4 == 1 or i == self.num_hidden_layers - 1 else "sliding_attention"
                for i in range(self.num_hidden_layers)
            ]
        if self.mlp_only_layers is None:
            self.mlp_only_layers = [0, 1]
        PreTrainedConfig.__post_init__(self, **kwargs)


class FlintRMSNorm(GptOssRMSNorm):
    pass


class FlintMLP(Qwen3MoeMLP):
    pass


class FlintTopKRouter(Qwen3MoeTopKRouter):
    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        # `device.type` is a proxy under FX tracing, which vLLM uses to fuse this router
        device_type = hidden_states.device.type if isinstance(hidden_states.device.type, str) else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
            # A no-op on the float32 weight; `from_config`, `.to()` and disk offload can leave it in half precision
            router_logits = F.linear(hidden_states.float(), self.weight.float())
        # Sigmoid scores without expert bias; the selected scores are renormalized
        routing_weights, selected_experts = torch.topk(router_logits.sigmoid(), self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        return router_logits, routing_weights, selected_experts


class FlintSparseMoeBlock(Qwen3MoeSparseMoeBlock):
    def __init__(self, config: FlintConfig):
        super().__init__(config)
        self.shared_expert = FlintMLP(config, intermediate_size=config.shared_expert_intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        _, routing_weights, selected_experts = self.gate(hidden_states)
        # Unlike Qwen2-MoE, the shared expert is always added in full, without a gate
        final_hidden_states = self.experts(hidden_states, selected_experts, routing_weights)
        final_hidden_states = final_hidden_states + self.shared_expert(hidden_states)
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    # QK-normed attention logits reach tens, so the scores and the softmax stay in float32, even under autocast
    with maybe_autocast(device_type=query.device.type, enabled=False):
        attn_weights = torch.matmul(query.float(), key_states.float().transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # Each head's learned sink is an extra softmax logit that attends to nothing
    sinks = module.sinks.reshape(1, -1, 1, 1).expand(*attn_weights.shape[:-1], 1)
    attn_weights = F.softmax(torch.cat([attn_weights, sinks], dim=-1), dim=-1)[..., :-1].to(value_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class FlintAttention(Qwen3MoeAttention):
    def __init__(self, config: FlintConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None
        self.sinks = nn.Parameter(torch.empty(config.num_attention_heads))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # Only sliding-window layers are rotary; full-attention layers use no positional encoding
        if self.sliding_window is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
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
            sliding_window=self.sliding_window,
            s_aux=self.sinks,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class FlintDecoderLayer(Qwen3MoeDecoderLayer):
    pass


class FlintPreTrainedModel(Qwen3MoePreTrainedModel):
    _supports_sdpa = False  # SDPA cannot apply the attention sinks
    _compatible_flash_implementations = ["kernels-community/vllm-flash-attn3", "flash_attention_4"]
    _keep_in_fp32_modules_strict = ["gate.weight", "sinks"]

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, FlintAttention):
            init.normal_(module.sinks, mean=0.0, std=self.config.initializer_range)


class FlintModel(Qwen2MoeModel):
    pass


class FlintForCausalLM(Qwen3MoeForCausalLM):
    def forward(self, **super_kwargs):
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, FlintForCausalLM

        >>> model = FlintForCausalLM.from_pretrained("RecursiveSI/flint", dtype="auto", device_map="auto")
        >>> tokenizer = AutoTokenizer.from_pretrained("RecursiveSI/flint")

        >>> messages = [{"role": "user", "content": "Who are you?"}]
        >>> inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        >>> outputs = model.generate(**inputs.to(model.device), max_new_tokens=32)
        >>> print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True))
        ```"""
        return super().forward(**super_kwargs)


__all__ = ["FlintConfig", "FlintForCausalLM", "FlintModel", "FlintPreTrainedModel"]
