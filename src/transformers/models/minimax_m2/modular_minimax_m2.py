# coding=utf-8
# Copyright 2025 the MiniMax AI Team and HuggingFace Team. All rights reserved.
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
from typing import Optional

import torch
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_rope_utils import RopeParameters, rope_config_validation, standardize_rope_params
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import OutputRecorder
from ..glm4_moe.modeling_glm4_moe import (
    Glm4MoeRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from ..mixtral.modeling_mixtral import (
    MixtralDecoderLayer,
    MixtralForCausalLM,
    MixtralForQuestionAnswering,
    MixtralForSequenceClassification,
    MixtralForTokenClassification,
    MixtralModel,
    MixtralRMSNorm,
    MixtralSparseMoeBlock,
)
from ..qwen3_moe.modeling_qwen3_moe import Qwen3MoeAttention


class MiniMaxM2Config(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MiniMaxM2Model`]. It is used to instantiate an
    MiniMaxM2 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the MiniMaxM2.

    [MiniMaxAI/MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-M2)

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.


    Args:
            vocab_size (`int`, *optional*, defaults to 32000):
                Vocabulary size of the MiniMaxM2 model. Defines the number of different tokens that can be represented by the
                `inputs_ids` passed when calling [`MiniMaxM2Model`]
            hidden_size (`int`, *optional*, defaults to 4096):
                Dimension of the hidden representations.
            intermediate_size (`int`, *optional*, defaults to 14336):
                Dimension of the MLP representations.
            num_hidden_layers (`int`, *optional*, defaults to 32):
                Number of hidden layers in the Transformer encoder.
            num_attention_heads (`int`, *optional*, defaults to 32):
                Number of attention heads for each attention layer in the Transformer encoder.
            num_key_value_heads (`int`, *optional*, defaults to 8):
                This is the number of key_value heads that should be used to implement Grouped Query Attention. If
                `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
                `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
                converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
                by meanpooling all the original heads within that group. For more details, check out [this
                paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `8`.
            head_dim (`int`, *optional*, defaults to `hidden_size // num_attention_heads`):
                The attention head dimension.
            hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
                The non-linear activation function (function or string) in the decoder.
            max_position_embeddings (`int`, *optional*, defaults to `4096*32`):
                The maximum sequence length that this model might ever be used with. MiniMaxM2's sliding window attention
                allows sequence of up to 4096*32 tokens.
            initializer_range (`float`, *optional*, defaults to 0.02):
                The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            rms_norm_eps (`float`, *optional*, defaults to 1e-05):
                The epsilon used by the rms normalization layers.
            use_cache (`bool`, *optional*, defaults to `True`):
                Whether or not the model should return the last key/values attentions (not used by all models). Only
                relevant if `config.is_decoder=True`.
            pad_token_id (`int`, *optional*):
                The id of the padding token.
            bos_token_id (`int`, *optional*, defaults to 1):
                The id of the "beginning-of-sequence" token.
            eos_token_id (`int`, *optional*, defaults to 2):
                The id of the "end-of-sequence" token.
            tie_word_embeddings (`bool`, *optional*, defaults to `False`):
                Whether the model's input and output word embeddings should be tied.
            sliding_window (`int`, *optional*):
                Sliding window attention window size. If not specified, will default to `4096`.
            attention_dropout (`float`, *optional*, defaults to 0.0):
                The dropout ratio for the attention probabilities.
            num_experts_per_tok (`int`, *optional*, defaults to 2):
                The number of experts to route per-token, can be also interpreted as the `top-k` routing
                parameter
            num_local_experts (`int`, *optional*, defaults to 8):
                Number of experts per Sparse MLP layer.
            output_router_logits (`bool`, *optional*, defaults to `False`):
                Whether or not the router logits should be returned by the model. Enabling this will also
                allow the model to output the auxiliary loss. See [here]() for more details
            router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
                The aux loss factor for the total loss.
            router_jitter_noise (`float`, *optional*, defaults to 0.0):
                Amount of noise to add to the router.
            rope_theta (`float`, *optional*, defaults to 10000.0):
                The base period of the RoPE embeddings.
            rope_parameters (`RopeParameters`, *optional*):
                Dictionary containing the configuration parameters for the RoPE embeddings. The dictionaty should contain
                a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
                with longer `max_position_embeddings`.

    ```python
    >>> from transformers import MiniMaxM2Model, MiniMaxM2Config

    >>> # Initializing a MiniMaxM2 style configuration
    >>> configuration = MiniMaxM2Config()

    >>> # Initializing a model from the MiniMaxM2 style configuration
    >>> model = MiniMaxM2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "minimax_m2"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.block_sparse_moe.gate": "colwise_rep",  # we need to replicate here to correctly route experts
        "layers.*.block_sparse_moe.experts.*.w1": "colwise",
        "layers.*.block_sparse_moe.experts.*.w2": "rowwise",
        "layers.*.block_sparse_moe.experts.*.w3": "colwise",
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
        vocab_size: Optional[int] = 32000,
        hidden_size: Optional[int] = 4096,
        intermediate_size: Optional[int] = 14336,
        num_hidden_layers: Optional[int] = 32,
        num_attention_heads: Optional[int] = 32,
        num_key_value_heads: Optional[int] = 8,
        head_dim: Optional[int] = None,
        hidden_act: Optional[str] = "silu",
        max_position_embeddings: Optional[int] = 4096 * 32,
        initializer_range: Optional[float] = 0.02,
        rms_norm_eps: Optional[int] = 1e-5,
        use_cache: Optional[bool] = True,
        pad_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = 1,
        eos_token_id: Optional[int] = 2,
        tie_word_embeddings: Optional[bool] = False,
        sliding_window: Optional[int] = None,
        attention_dropout: Optional[float] = 0.0,
        num_experts_per_tok: Optional[int] = 2,
        num_local_experts: Optional[int] = 8,
        output_router_logits: Optional[bool] = False,
        router_aux_loss_coef: Optional[float] = 0.001,
        router_jitter_noise: Optional[float] = 0.0,
        rope_theta: Optional[float] = 1000000.0,
        rope_parameters: Optional[RopeParameters | dict[RopeParameters]] = None,
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
        # Try to set `rope_scaling` if available, otherwise use `rope_parameters`
        rope_scaling = kwargs.pop("rope_scaling", None)
        self.rope_parameters = rope_scaling or rope_parameters

        # Validate the correctness of rotary position embeddings parameters
        standardize_rope_params(self, rope_theta=rope_theta)
        rope_config_validation(self)

        rotary_dim = kwargs.pop("rotary_dim", head_dim)

        if self.head_dim is not None:
            self.partial_rotary_factor = rotary_dim / self.head_dim
        else:
            self.partial_rotary_factor = 1.0

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class MiniMaxM2MLP(nn.Module):
    def __init__(self, config: MiniMaxM2Config):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


class MiniMaxM2Experts(nn.ModuleList):
    """
    ModuleList of experts.
    """

    def __init__(self, config: MiniMaxM2Config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        for _ in range(self.num_experts):
            self.append(MiniMaxM2MLP(config))

    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size * sequence_length, hidden_dim)
            selected_experts: (batch_size * sequence_length, top_k)
            routing_weights: (batch_size * sequence_length, top_k)
        Returns:
            (batch_size * sequence_length, hidden_dim)
        """
        final_hidden_states = torch.zeros_like(hidden_states)
        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)

        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            current_state = hidden_states[None, top_x].reshape(-1, hidden_states.shape[-1])
            current_hidden_states = self[expert_idx](current_state) * top_k_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        return final_hidden_states


class MiniMaxM2SparseMoeBlock(MixtralSparseMoeBlock):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.top_k = config.num_experts_per_tok
        self.jitter_noise = config.router_jitter_noise
        self.gate = nn.Linear(config.hidden_size, config.num_local_experts, bias=False)
        self.experts = MiniMaxM2Experts(config)
        self.register_buffer("e_score_correction_bias", torch.zeros(config.num_local_experts))

    def route_tokens_to_experts(self, router_logits):
        routing_weights = torch.nn.functional.sigmoid(router_logits.float())
        scores_for_choice = routing_weights + self.e_score_correction_bias
        _, top_k_index = torch.topk(scores_for_choice, self.top_k, dim=-1, sorted=False)
        top_k_weights = routing_weights.gather(1, top_k_index)
        top_k_weights /= top_k_weights.sum(dim=-1, keepdim=True)
        return top_k_index, top_k_weights.to(router_logits.dtype)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        router_logits = self.gate(hidden_states)
        top_k_index, top_k_weights = self.route_tokens_to_experts(router_logits)
        hidden_states = self.experts(hidden_states, top_k_index, top_k_weights.to(hidden_states.dtype))
        hidden_states = hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return hidden_states


class MiniMaxM2RMSNorm(MixtralRMSNorm):
    pass


class MiniMaxM2RotaryEmbedding(Glm4MoeRotaryEmbedding):
    pass


class MiniMaxM2Attention(Qwen3MoeAttention):
    def __init__(self, config: MiniMaxM2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        self.q_norm = MiniMaxM2RMSNorm(self.head_dim * config.num_attention_heads, eps=config.rms_norm_eps)
        self.k_norm = MiniMaxM2RMSNorm(self.head_dim * config.num_key_value_heads, eps=config.rms_norm_eps)

        del self.sliding_window

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # main diff from Llama
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        key_states = key_states.view(hidden_shape)
        query_states = query_states.view(hidden_shape)
        value_states = value_states.view(hidden_shape)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; position_ids needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

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

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class MiniMaxM2DecoderLayer(MixtralDecoderLayer):
    def __init__(self, config: MiniMaxM2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        del self.mlp
        self.block_sparse_moe = MiniMaxM2SparseMoeBlock(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.block_sparse_moe(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class MiniMaxM2PreTrainedModel(PreTrainedModel):
    config: MiniMaxM2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MiniMaxM2DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = False  # MoE models don't work with torch.compile (`torch.where(condition)` not supported)
    _supports_attention_backend = True
    _can_record_outputs = {
        "router_logits": OutputRecorder(nn.Linear, layer_name="block_sparse_moe.gate", index=0),
        "hidden_states": MiniMaxM2DecoderLayer,
        "attentions": MiniMaxM2Attention,
    }


class MiniMaxM2Model(MixtralModel):
    pass


class MiniMaxM2ForCausalLM(MixtralForCausalLM):
    pass


class MiniMaxM2ForSequenceClassification(MixtralForSequenceClassification):
    pass


class MiniMaxM2ForTokenClassification(MixtralForTokenClassification):
    pass


class MiniMaxM2ForQuestionAnswering(MixtralForQuestionAnswering):
    pass


__all__ = [
    "MiniMaxM2Config",
    "MiniMaxM2ForCausalLM",
    "MiniMaxM2ForQuestionAnswering",
    "MiniMaxM2Model",
    "MiniMaxM2PreTrainedModel",
    "MiniMaxM2ForSequenceClassification",
    "MiniMaxM2ForTokenClassification",
]
