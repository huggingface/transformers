# Copyright 2026 Translated SRL and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch LaraMini model."""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...integrations import use_kernelized_func
from ...masking_utils import create_causal_mask, create_recurrent_attention_mask, create_sliding_window_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ..flint.modeling_flint import FlintSparseMoeBlock
from ..glm5_next.modeling_glm5_next import Glm5NextTextRMSNormGated
from ..helium.modeling_helium import HeliumRMSNorm
from ..qwen3.modeling_qwen3 import Qwen3Attention, Qwen3RotaryEmbedding, apply_rotary_pos_emb, eager_attention_forward
from ..qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5GatedDeltaNet,
    causal_conv1d_fn,
    causal_conv1d_forward,
    causal_conv1d_update,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)
from ..qwen3_next.configuration_qwen3_next import Qwen3NextConfig
from ..qwen3_next.modeling_qwen3_next import (
    Qwen3NextExperts,
    Qwen3NextForCausalLM,
    Qwen3NextModel,
    Qwen3NextPreTrainedModel,
    Qwen3NextTopKRouter,
)


@auto_docstring(checkpoint="translated/Lara-4-Mini-Base")
@strict
class LaraMiniConfig(Qwen3NextConfig):
    r"""
    linear_conv_kernel_dim (`int`, *optional*, defaults to 4):
        Kernel size of the causal depthwise convolution in the gated delta net (`"linear_attention"`) layers.
    linear_key_head_dim (`int`, *optional*, defaults to 128):
        Dimension of each query and key head in the gated delta net layers.
    linear_value_head_dim (`int`, *optional*, defaults to 128):
        Dimension of each value head in the gated delta net layers.
    linear_num_key_heads (`int`, *optional*, defaults to 8):
        Number of query and key heads in the gated delta net layers.
    linear_num_value_heads (`int`, *optional*, defaults to 16):
        Number of value heads in the gated delta net layers. Each query and key head serves
        `linear_num_value_heads // linear_num_key_heads` consecutive value heads.

    ```python
    >>> from transformers import LaraMiniModel, LaraMiniConfig

    >>> # Initializing a LaraMini style configuration
    >>> configuration = LaraMiniConfig()

    >>> # Initializing a model from the LaraMini style configuration
    >>> model = LaraMiniModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "lara_mini"

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.shared_expert.gate_proj": "colwise",
        "layers.*.mlp.shared_expert.up_proj": "colwise",
        "layers.*.mlp.shared_expert.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
        # The `linear_attn.in_proj_*` stay replicated: they are split from the checkpoint's fused `in_proj` at load time,
        # which the `Split` conversion can't do on a sharded tensor
        "layers.*.linear_attn.out_proj": "colwise_gather_output",
    }

    hidden_size: int = 1024
    num_hidden_layers: int = 34
    num_key_value_heads: int = 4
    max_position_embeddings: int = 131072
    tie_word_embeddings: bool = True
    head_dim: int = 128
    sliding_window: int = 512
    linear_num_key_heads: int = 8
    linear_num_value_heads: int = 16
    num_experts_per_tok: int = 4
    num_experts: int = 128
    eos_token_id: int | list[int] | None = 151643

    intermediate_size = AttributeError()
    decoder_sparse_step = AttributeError()
    mlp_only_layers = AttributeError()

    def __post_init__(self, **kwargs):
        if self.layer_types is None:
            # The released pattern, with period 5
            pattern = [
                "linear_attention",
                "sliding_attention",
                "linear_attention",
                "full_attention",
                "sliding_attention",
            ]
            self.layer_types = [pattern[i % len(pattern)] for i in range(self.num_hidden_layers)]
        PreTrainedConfig.__post_init__(self, **kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("`num_attention_heads` must be a multiple of `num_key_value_heads`.")
        if self.linear_num_value_heads % self.linear_num_key_heads != 0:
            raise ValueError("`linear_num_value_heads` must be a multiple of `linear_num_key_heads`.")


class LaraMiniRMSNorm(HeliumRMSNorm):
    """RMSNorm that applies its weight in float32 and only then casts back to the input dtype."""


class LaraMiniRMSNormGated(Glm5NextTextRMSNormGated):
    def __init__(self, hidden_size, eps=1e-6, **kwargs) -> None:
        super().__init__(hidden_size, eps, **kwargs)
        self.activation = "silu"


class LaraMiniRotaryEmbedding(Qwen3RotaryEmbedding):
    pass


class LaraMiniAttention(Qwen3Attention):
    """Qwen3 attention that applies rotary embeddings only when it gets them: the `"full_attention"` layers get none."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        attention_mask: torch.Tensor | None,
        cache: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if cache is not None:
            key_states, value_states = cache.update(key_states, value_states, self.layer_idx, kwargs=kwargs)

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
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


@use_kernelized_func(
    [
        torch_recurrent_gated_delta_rule,
        torch_chunk_gated_delta_rule,
        causal_conv1d_forward,
        causal_conv1d_fn,
        causal_conv1d_update,
    ]
)
class LaraMiniGatedDeltaNet(Qwen3_5GatedDeltaNet):
    """Qwen3.5 gated delta net. The published checkpoints fuse its input projections into one `in_proj` (rows
    `[query | key | value | z | b | a]`) and name its gated norm `out_norm`: `conversion_mapping.py` splits and
    renames them when loading, and restores them when saving."""


class LaraMiniExperts(Qwen3NextExperts):
    pass


class LaraMiniTopKRouter(Qwen3NextTopKRouter):
    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        # Unlike Qwen3-Next, the router logits are computed in float32
        router_logits = F.linear(hidden_states.float(), self.weight.float())  # (seq_len, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)
        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)  # (seq_len, top_k)
        if self.norm_topk_prob:
            router_top_value = router_top_value / router_top_value.sum(dim=-1, keepdim=True)
        router_scores = router_top_value.to(hidden_states.dtype)
        return router_logits, router_scores, router_indices


class LaraMiniSparseMoeBlock(FlintSparseMoeBlock):
    pass


class LaraMiniDecoderLayer(Qwen3_5DecoderLayer):
    def __init__(self, config: LaraMiniConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        del self.hidden_size
        self.mlp = LaraMiniSparseMoeBlock(config)


@auto_docstring
class LaraMiniPreTrainedModel(Qwen3NextPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = AttributeError()

    @torch.no_grad()
    def _init_weights(self, module):
        # Not Qwen3-Next's: it zero-initializes its RMSNorms, which scale by `1 + weight`
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, LaraMiniGatedDeltaNet):
            init.ones_(module.dt_bias)
            init.copy_(
                module.A_log,
                torch.empty(module.num_v_heads, device=module.A_log.device).uniform_(1, 16).log_(),
            )
        elif isinstance(module, LaraMiniExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LaraMiniTopKRouter):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)


@auto_docstring
class LaraMiniModel(Qwen3NextModel):
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and cache is None:
            cache = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = cache.get_seq_length() if cache is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache": cache,
                "position_ids": position_ids,
            }
            mask_creation_functions = {
                "full_attention": create_causal_mask,
                "sliding_attention": create_sliding_window_causal_mask,
                "linear_attention": create_recurrent_attention_mask,
            }
            causal_mask_mapping = {
                layer_type: mask_creation_functions[layer_type](**mask_kwargs)
                for layer_type in set(self.config.layer_types)
            }

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            layer_type = self.config.layer_types[i]
            hidden_states = decoder_layer(
                hidden_states,
                # The global layers use no positional encoding
                position_embeddings=position_embeddings if layer_type == "sliding_attention" else None,
                attention_mask=causal_mask_mapping[layer_type],
                position_ids=position_ids,
                cache=cache,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            cache=cache,
        )


class LaraMiniForCausalLM(Qwen3NextForCausalLM):
    def forward(self, **super_kwargs) -> MoeCausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, LaraMiniForCausalLM

        >>> model = LaraMiniForCausalLM.from_pretrained("translated/Lara-4-Mini-Base")
        >>> tokenizer = AutoTokenizer.from_pretrained("translated/Lara-4-Mini-Base")

        >>> prompt = "La capitale d'Italia è"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_new_tokens=16)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        ```"""
        return super().forward(**super_kwargs)


__all__ = ["LaraMiniConfig", "LaraMiniForCausalLM", "LaraMiniModel", "LaraMiniPreTrainedModel"]
