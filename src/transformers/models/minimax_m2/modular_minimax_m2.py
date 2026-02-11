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


import torch
import torch.nn.functional as F
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...masking_utils import create_causal_mask
from ...modeling_outputs import MoeModelOutputWithPast
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..flex_olmo.modeling_flex_olmo import FlexOlmoAttention
from ..glm4_moe.modeling_glm4_moe import (
    Glm4MoeRotaryEmbedding,
    apply_rotary_pos_emb,  # noqa: F401
)
from ..mixtral.modeling_mixtral import (
    MixtralExperts,
    MixtralForCausalLM,
    MixtralModel,
    MixtralPreTrainedModel,
    MixtralRMSNorm,
    MixtralSparseMoeBlock,
    MixtralTopKRouter,
)


class MiniMaxM2Config(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MiniMaxM2Model`]. It is used to instantiate an
    MiniMaxM2 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the MiniMaxM2.

    [MiniMaxAI/MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-M2)

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.


    Args:
        vocab_size (`Optional`, *optional*, defaults to 200064):
            Vocabulary size of the MiniMaxM2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`MiniMaxM2Model`]
        hidden_size (`Optional`, *optional*, defaults to 3072):
            Dimension of the hidden representations.
        intermediate_size (`Optional`, *optional*, defaults to 1536):
            Dimension of the MLP representations.
        num_hidden_layers (`Optional`, *optional*, defaults to 62):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`Optional`, *optional*, defaults to 48):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`Optional`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `8`.
        head_dim (`Optional`, *optional*, defaults to 128):
            The attention head dimension.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`Optional`, *optional*, defaults to 196608):
            The maximum sequence length that this model might ever be used with. MiniMaxM2's sliding window attention
            allows sequence of up to 196608 tokens.
        initializer_range (`Optional`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`Optional`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`Optional`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`Optional`, *optional*):
            The id of the padding token.
        bos_token_id (`Optional`, *optional*, defaults to 200034):
            The id of the "beginning-of-sequence" token.
        eos_token_id (`Optional`, *optional*, defaults to 200020):
            The id of the "end-of-sequence" token.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        attention_dropout (`Optional`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        num_experts_per_tok (`Optional`, *optional*, defaults to 8):
            The number of experts to route per-token, can be also interpreted as the `top-k` routing
            parameter
        num_local_experts (`Optional`, *optional*, defaults to 256):
            Number of experts per Sparse MLP layer.
        output_router_logits (`Optional`, *optional*, defaults to `False`):
            Whether or not the router logits should be returned by the model. Enabling this will also
            allow the model to output the auxiliary loss. See [here]() for more details
        router_aux_loss_coef (`Optional`, *optional*, defaults to 0.001):
            The aux loss factor for the total loss.
        router_jitter_noise (`Optional`, *optional*, defaults to 0.0):
            Amount of noise to add to the router.
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
        "layers.*.self_attn.q_proj": "colwise_rep",
        "layers.*.self_attn.k_proj": "colwise_rep",
        "layers.*.self_attn.v_proj": "colwise_rep",
        "layers.*.self_attn.o_proj": "rowwise_rep",
        "layers.*.mlp.gate": "colwise_rep",  # we need to replicate here to correctly route experts
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
    default_theta = 5000000.0

    def __init__(
        self,
        vocab_size: int | None = 200064,
        hidden_size: int | None = 3072,
        intermediate_size: int | None = 1536,
        num_hidden_layers: int | None = 62,
        num_attention_heads: int | None = 48,
        num_key_value_heads: int | None = 8,
        head_dim: int | None = 128,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 196608,
        initializer_range: float | None = 0.02,
        rms_norm_eps: int | None = 1e-06,
        use_cache: bool | None = True,
        pad_token_id: int | None = None,
        bos_token_id: int | None = 200034,
        eos_token_id: int | None = 200020,
        tie_word_embeddings: bool | None = False,
        attention_dropout: float | None = 0.0,
        num_experts_per_tok: int | None = 8,
        num_local_experts: int | None = 256,
        output_router_logits: bool | None = False,
        router_aux_loss_coef: float | None = 0.001,
        router_jitter_noise: float | None = 0.0,
        rope_parameters: RopeParameters | dict[RopeParameters] | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim
        self.rope_parameters = rope_parameters

        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_jitter_noise = router_jitter_noise
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings

        super().__init__(**kwargs)


class MiniMaxM2TopKRouter(MixtralTopKRouter):
    def forward(self, hidden_states, e_score_correction_bias):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states.to(self.weight.dtype), self.weight)  # (seq_len, num_experts)
        # Main difference to other Moe, using Sigmoid activation instead of Softmax
        routing_weights = nn.functional.sigmoid(router_logits.float())
        scores_for_choice = routing_weights + e_score_correction_bias
        _, top_k_index = torch.topk(scores_for_choice, self.top_k, dim=-1, sorted=False)
        top_k_weights = routing_weights.gather(1, top_k_index)
        top_k_weights /= top_k_weights.sum(dim=-1, keepdim=True)
        router_scores = top_k_weights
        return router_logits, router_scores, top_k_index


class MiniMaxM2Experts(MixtralExperts):
    pass


class MiniMaxM2SparseMoeBlock(MixtralSparseMoeBlock):
    def __init__(self, config):
        super().__init__()
        self.register_buffer("e_score_correction_bias", torch.zeros(config.num_local_experts))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        _, top_k_weights, top_k_index = self.gate(hidden_states, self.e_score_correction_bias)
        hidden_states = self.experts(hidden_states, top_k_index, top_k_weights)
        hidden_states = hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return hidden_states


class MiniMaxM2RMSNorm(MixtralRMSNorm):
    pass


class MiniMaxM2RotaryEmbedding(Glm4MoeRotaryEmbedding):
    pass


class MiniMaxM2Attention(FlexOlmoAttention):
    def __init__(self, config: MiniMaxM2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)


class MiniMaxM2PreTrainedModel(MixtralPreTrainedModel):
    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        std = self.config.initializer_range
        if isinstance(module, MiniMaxM2Experts):
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, MiniMaxM2TopKRouter):
            init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, MiniMaxM2SparseMoeBlock):
            init.zeros_(module.e_score_correction_bias)


class MiniMaxM2Model(MixtralModel):
    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # No sliding window opposed to mixtral
        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return MoeModelOutputWithPast(  # only diff with Mistral is the output type, we need MoE
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class MiniMaxM2ForCausalLM(MixtralForCausalLM):
    pass


__all__ = [
    "MiniMaxM2Config",
    "MiniMaxM2ForCausalLM",
    "MiniMaxM2Model",  # noqa: F822
    "MiniMaxM2PreTrainedModel",  # noqa: F822
]
