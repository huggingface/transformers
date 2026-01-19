# Copyright 2025 MiniMaxAI and HuggingFace Inc. teams. All rights reserved.
#
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
"""PyTorch MiniMax model."""

import torch
import torch.nn.functional as F
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig, layer_type_validation
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import MoeModelOutputWithPast
from ...modeling_rope_utils import RopeParameters
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, logging
from ...utils.generic import OutputRecorder, check_model_inputs
from ..gemma2.modeling_gemma2 import Gemma2RotaryEmbedding
from ..mixtral.modeling_mixtral import (
    MixtralAttention,
    MixtralDecoderLayer,
    MixtralForCausalLM,
    MixtralForQuestionAnswering,
    MixtralForSequenceClassification,
    MixtralForTokenClassification,
    MixtralModel,
    MixtralPreTrainedModel,
    MixtralRMSNorm,
    MixtralSparseMoeBlock,
    MixtralTopKRouter,
)


logger = logging.get_logger(__name__)


class MiniMaxConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MiniMaxModel`]. It is used to instantiate an
    MiniMax model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the MiniMax.

    [MiniMaxAI/MiniMax-Text-01-hf](https://huggingface.co/MiniMaxAI/MiniMax-Text-01-hf)

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the MiniMax model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`MiniMaxModel`]
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
            The maximum sequence length that this model might ever be used with. MiniMax's sliding window attention
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
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
        layer_types (`list`, *optional*):
            Attention pattern for each layer.
        block_size (`int`, *optional*, defaults to 256):
            The length of each attention block, determining how queries, keys, and values
            are grouped and processed for intra- and inter-block attention.
        full_attn_alpha_factor (`float`, *optional*, defaults to 1):
            Weight for residual value in residual connection after normal attention.
        full_attn_beta_factor (`float`, *optional*, defaults to 1):
            Weight for hidden state value in residual connection after normal attention.
        linear_attn_alpha_factor (`float`, *optional*, defaults to 1):
            Weight for residual value in residual connection after lightning attention.
        linear_attn_beta_factor (`float`, *optional*, defaults to 1):
            Weight for hidden state value in residual connection after lightning attention.
        mlp_alpha_factor (`float`, *optional*, defaults to 1):
            Weight for residual value in residual connection after MLP.
        mlp_beta_factor (`float`, *optional*, defaults to 1):
            Weight for hidden state value in residual connection after MLP.

    ```python
    >>> from transformers import MiniMaxModel, MiniMaxConfig

    >>> # Initializing a MiniMax style configuration
    >>> configuration = MiniMaxConfig()

    >>> # Initializing a model from the MiniMax style configuration
    >>> model = MiniMaxModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "minimax"
    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = 1000000.0
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate": "colwise_rep",  # we need to replicate here to correctly route experts
        "layers.*.mlp.experts.gate_up_proj": "local_rowwise",
        "layers.*.mlp.experts.down_proj": "local_rowwise",
        "layers.*.mlp.experts": "gather",
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
        vocab_size: int | None = 32000,
        hidden_size: int | None = 4096,
        intermediate_size: int | None = 14336,
        num_hidden_layers: int | None = 32,
        num_attention_heads: int | None = 32,
        num_key_value_heads: int | None = 8,
        head_dim: int | None = None,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 4096 * 32,
        initializer_range: float | None = 0.02,
        rms_norm_eps: int | None = 1e-5,
        use_cache: bool | None = True,
        pad_token_id: int | None = None,
        bos_token_id: int | None = 1,
        eos_token_id: int | None = 2,
        tie_word_embeddings: bool | None = False,
        sliding_window: int | None = None,
        attention_dropout: float | None = 0.0,
        num_experts_per_tok: int | None = 2,
        num_local_experts: int | None = 8,
        output_router_logits: bool | None = False,
        router_aux_loss_coef: float | None = 0.001,
        router_jitter_noise: float | None = 0.0,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        layer_types: list[str] | None = None,
        block_size: int | None = 256,
        full_attn_alpha_factor: int | None = 1,
        full_attn_beta_factor: int | None = 1,
        linear_attn_alpha_factor: int | None = 1,
        linear_attn_beta_factor: int | None = 1,
        mlp_alpha_factor: int | None = 1,
        mlp_beta_factor: int | None = 1,
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
        self.tie_word_embeddings = tie_word_embeddings
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        self.layer_types = layer_types
        self.block_size = block_size
        self.full_attn_alpha_factor = full_attn_alpha_factor
        self.full_attn_beta_factor = full_attn_beta_factor
        self.linear_attn_alpha_factor = linear_attn_alpha_factor
        self.linear_attn_beta_factor = linear_attn_beta_factor
        self.mlp_alpha_factor = mlp_alpha_factor
        self.mlp_beta_factor = mlp_beta_factor

        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if bool((i + 1) % 2) else "linear_attention" for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types, self.num_hidden_layers)

        self.rope_parameters = rope_parameters
        super().__init__(**kwargs)


class MiniMaxRMSNorm(MixtralRMSNorm):
    pass


class MiniMaxCache(DynamicCache):
    def __init__(self):
        super().__init__()
        self.linear_cache: list[torch.Tensor] = []

    def set_linear_cache(self, layer_idx, linear_cache):
        # There may be skipped layers, fill them with empty lists
        for _ in range(len(self.linear_cache), layer_idx + 1):
            self.linear_cache.append([])
        self.linear_cache[layer_idx] = linear_cache

    def get_linear_cache(self, layer_idx: int):
        if layer_idx < len(self):
            return self.linear_cache[layer_idx]
        return None

    def __len__(self):
        return max(super().__len__(), len(self.linear_cache))

    def batch_repeat_interleave(self, repeats: int):
        for layer_idx in range(len(self)):
            if self.linear_cache[layer_idx] != []:
                self.linear_cache[layer_idx] = self.linear_cache[layer_idx].repeat_interleave(repeats, dim=0)
            else:
                self.layers[layer_idx].batch_repeat_interleave(repeats)

    def batch_select_indices(self, indices: torch.Tensor):
        for layer_idx in range(len(self)):
            if self.linear_cache[layer_idx] != []:
                self.linear_cache[layer_idx] = self.linear_cache[layer_idx][indices, ...]
            else:
                self.layers[layer_idx].batch_select_indices(indices)

    def crop(self, max_length: int):
        raise RuntimeError("MiniMaxCache doesnot support `crop` method")


class MiniMaxLightningAttention(nn.Module):
    def __init__(self, config: MiniMaxConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        self.num_attention_heads = config.num_attention_heads
        self.num_hidden_layers = config.num_hidden_layers
        self.block_size = config.block_size

        self.act_fn = ACT2FN[config.hidden_act]
        self.norm = MiniMaxRMSNorm(self.head_dim * self.num_attention_heads)
        self.qkv_proj = nn.Linear(config.hidden_size, self.num_attention_heads * self.head_dim * 3, bias=False)
        self.out_proj = nn.Linear(self.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.output_gate = nn.Linear(config.hidden_size, self.num_attention_heads * self.head_dim, bias=False)

        slope_rate = self.get_slope_rate()
        query_decay, key_decay, diagonal_decay = self.decay_factors(slope_rate)

        self.register_buffer("slope_rate", slope_rate)
        self.register_buffer("query_decay", query_decay)
        self.register_buffer("key_decay", key_decay)
        self.register_buffer("diagonal_decay", diagonal_decay)

    def get_slope_rate(self):
        base = 1 / (2 ** (8 / self.num_attention_heads))
        exponent = torch.arange(self.num_attention_heads) + 1
        factor = 1 - self.layer_idx / (self.num_hidden_layers - 1 + 1e-5) + 1e-5

        rate = base**exponent
        rate = rate * factor
        rate = rate[:, None, None]

        return rate

    def decay_factors(self, slope_rate):
        block_size_range = torch.arange(self.block_size) + 1

        query_decay = torch.exp(-slope_rate * block_size_range[:, None])
        key_decay = torch.exp(-slope_rate * (self.block_size - block_size_range[:, None]))

        diagonal_decay = block_size_range[:, None] - block_size_range[None, :]
        diagonal_decay = diagonal_decay[None, None, :, :]
        diagonal_decay = slope_rate * diagonal_decay
        diagonal_decay = torch.where(diagonal_decay >= 0, -diagonal_decay, float("-inf"))
        diagonal_decay = torch.exp(diagonal_decay)

        return query_decay, key_decay, diagonal_decay

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        num_blocks = (seq_len + self.block_size - 1) // self.block_size

        qkv_states = self.act_fn(self.qkv_proj(hidden_states))
        qkv_states = qkv_states.reshape(batch_size, seq_len, self.num_attention_heads, 3 * self.head_dim)

        query_states, key_states, value_states = torch.split(qkv_states, self.head_dim, dim=3)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # calculated (K.T @ V) and saved as cache
        attn_weights_inter = None
        if past_key_values is not None:
            attn_weights_inter = past_key_values.get_linear_cache(self.layer_idx)

        if attn_weights_inter is None:
            attn_weights_inter = torch.zeros(batch_size, self.num_attention_heads, self.head_dim, self.head_dim).to(
                value_states
            )

            # apply attention_mask
            if attention_mask is not None:
                attention_mask = attention_mask.to(dtype=torch.bool)  # Ensure it's a boolean tensor
                value_states = value_states.masked_fill(~attention_mask.unsqueeze(1).unsqueeze(-1), 0)

            attn_output = []
            for i in range(num_blocks):
                start_idx = i * self.block_size
                end_idx = min(start_idx + self.block_size, seq_len)
                current_block_size = end_idx - start_idx

                current_query_states = query_states[:, :, start_idx:end_idx]
                current_key_states = key_states[:, :, start_idx:end_idx]
                current_value_states = value_states[:, :, start_idx:end_idx]

                current_query_decay = self.query_decay[:, :current_block_size]
                current_key_decay = self.key_decay[:, -current_block_size:]
                current_diagonal_decay = self.diagonal_decay[:, :, :current_block_size, :current_block_size]
                block_decay = torch.exp(-self.slope_rate * current_block_size)

                # intra: ( Q @ K.T ) @ V -> QK * V
                attn_weights_intra = torch.matmul(current_query_states, current_key_states.transpose(-1, -2))
                attn_output_intra = torch.matmul(attn_weights_intra * current_diagonal_decay, current_value_states)

                # inter: Q @ ( K.T @ V ) -> Q * KV
                attn_output_inter = torch.matmul(current_query_states * current_query_decay, attn_weights_inter)

                # final attention output
                current_attn_output = attn_output_inter + attn_output_intra
                attn_output.append(current_attn_output)

                # calculate attn_weights_inter for next block or cache
                next_attn_weights_inter = torch.matmul(
                    (current_key_states * current_key_decay).transpose(-1, -2), current_value_states
                )
                attn_weights_inter = attn_weights_inter * block_decay + next_attn_weights_inter

        else:
            ratio = torch.exp(-self.slope_rate)
            attn_output = []
            for i in range(seq_len):
                current_query_states = query_states[:, :, i : i + 1]
                current_key_states = key_states[:, :, i : i + 1]
                current_value_states = value_states[:, :, i : i + 1]

                current_attn_weights_inter = torch.matmul(current_key_states.transpose(-1, -2), current_value_states)
                attn_weights_inter = ratio * attn_weights_inter + current_attn_weights_inter
                current_attn_output = torch.matmul(current_query_states, attn_weights_inter)

                attn_output.append(current_attn_output)

        # concatenate attention outputs over all blocks
        attn_output = torch.cat(attn_output, dim=-2)

        # final output projection
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, seq_len, self.num_attention_heads * self.head_dim)
        attn_output = self.norm(attn_output)
        attn_output = F.sigmoid(self.output_gate(hidden_states)) * attn_output
        attn_output = self.out_proj(attn_output)

        # update cache
        if past_key_values is not None:
            past_key_values.set_linear_cache(self.layer_idx, attn_weights_inter)

        return attn_output, attn_weights_inter


class MiniMaxRotaryEmbedding(Gemma2RotaryEmbedding):
    pass


class MiniMaxAttention(MixtralAttention):
    pass


class MiniMaxTopKRouter(MixtralTopKRouter):
    pass


class MiniMaxSparseMoeBlock(MixtralSparseMoeBlock):
    pass


class MiniMaxDecoderLayer(MixtralDecoderLayer, GradientCheckpointingLayer):
    def __init__(self, config: MiniMaxConfig, layer_idx: int):
        super().__init__(config, layer_idx)

        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.mlp_alpha_factor = config.mlp_alpha_factor
        self.mlp_beta_factor = config.mlp_beta_factor
        del self.mlp
        self.mlp = MiniMaxSparseMoeBlock(config)
        if self.layer_type == "linear_attention":
            self.self_attn = MiniMaxLightningAttention(config, layer_idx)
            self.attn_alpha_factor = config.linear_attn_alpha_factor
            self.attn_beta_factor = config.linear_attn_beta_factor
        else:
            self.self_attn = MiniMaxAttention(config, layer_idx)
            self.attn_alpha_factor = config.full_attn_alpha_factor
            self.attn_beta_factor = config.full_attn_beta_factor

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        hidden_states = self.input_layernorm(hidden_states)
        residual = hidden_states
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual * self.attn_alpha_factor + hidden_states * self.attn_beta_factor
        hidden_states = self.post_attention_layernorm(hidden_states)
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual * self.mlp_alpha_factor + hidden_states * self.mlp_beta_factor

        return hidden_states


class MiniMaxPreTrainedModel(MixtralPreTrainedModel):
    _can_compile_fullgraph = False  # uses a non-compilable custom cache class MiniMaxCache
    _can_record_outputs = {
        "router_logits": OutputRecorder(MiniMaxTopKRouter, layer_name="mlp.gate", index=0),
        "hidden_states": MiniMaxDecoderLayer,
        "attentions": [MiniMaxAttention, MiniMaxLightningAttention],
    }

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, MiniMaxLightningAttention):
            slope_rate = module.get_slope_rate()
            query_decay, key_decay, diagonal_decay = module.decay_factors(slope_rate)
            init.copy_(module.slope_rate, slope_rate)
            init.copy_(module.query_decay, query_decay)
            init.copy_(module.key_decay, key_decay)
            init.copy_(module.diagonal_decay, diagonal_decay)


class MiniMaxModel(MixtralModel):
    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: MiniMaxCache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = MiniMaxCache()
        elif use_cache and not isinstance(past_key_values, MiniMaxCache):
            raise ValueError(
                f"MiniMax uses cache of its own and is not compatible with `past_key_values` of type {type(past_key_values)}."
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        mask_function = create_causal_mask if self.config.sliding_window is None else create_sliding_window_causal_mask
        causal_mask = mask_function(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers:
            if decoder_layer.layer_type == "full_attention":
                input_attention_mask = causal_mask
            else:
                # lightning attention uses original attention_mask, and uses it only for the first step
                input_attention_mask = attention_mask

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=input_attention_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class MiniMaxForCausalLM(MixtralForCausalLM):
    def forward(self, **super_kwargs):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, MiniMaxForCausalLM

        >>> model = MiniMaxForCausalLM.from_pretrained("MiniMaxAI/MiniMax-Text-01-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("MiniMaxAI/MiniMax-Text-01-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        return super().forward(**super_kwargs)


class MiniMaxForSequenceClassification(MixtralForSequenceClassification):
    pass


class MiniMaxForTokenClassification(MixtralForTokenClassification):
    pass


class MiniMaxForQuestionAnswering(MixtralForQuestionAnswering):
    pass


__all__ = [
    "MiniMaxConfig",
    "MiniMaxPreTrainedModel",
    "MiniMaxModel",
    "MiniMaxForCausalLM",
    "MiniMaxForSequenceClassification",
    "MiniMaxForTokenClassification",
    "MiniMaxForQuestionAnswering",
]
