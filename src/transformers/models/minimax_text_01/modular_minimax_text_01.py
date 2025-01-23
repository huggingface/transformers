# coding=utf-8
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
"""PyTorch MiniMax-Text-01 model."""

# TODO: remove these
from icecream import ic
from pack_minimax import show_tensor

from typing import Callable, List, Optional, Tuple, Union

import math
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache
from ...configuration_utils import PretrainedConfig
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import (
    logging,
)
from ..mixtral.modeling_mixtral import (
    eager_attention_forward,
    MixtralRMSNorm,
    MixtralAttention,
    MixtralDecoderLayer,
    MixtralModel,
    MixtralForCausalLM,
    MixtralForSequenceClassification,
    MixtralForTokenClassification,
    MixtralForQuestionAnswering,
)


logger = logging.get_logger(__name__)


_CHECKPOINT_FOR_DOC = "mistralai/Mixtral-8x7B-v0.1"
_CONFIG_FOR_DOC = "MixtralConfig"


class MiniMaxText01Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MiniMaxText01Model`]. It is used to instantiate an
    MiniMaxText01 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the MiniMaxText01.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the MiniMaxText01 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`MiniMaxText01Model`]
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
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `8`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to `4096*32`):
            The maximum sequence length that this model might ever be used with. MiniMaxText01's sliding window attention
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
        rope_theta (`float`, *optional*, defaults to 1000000.0):
            The base period of the RoPE embeddings.
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
            Whether or not the router logits should be returned by the model. Enabeling this will also
            allow the model to output the auxiliary loss. See [here]() for more details
        router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
            The aux loss factor for the total loss.
        router_jitter_noise (`float`, *optional*, defaults to 0.0):
            Amount of noise to add to the router.

    ```python
    >>> from transformers import MiniMaxText01Model, MiniMaxText01Config

    >>> # Initializing a MiniMaxText01 style configuration
    >>> configuration = MiniMaxText01Config()

    >>> # Initializing a model from the MiniMaxText01 style configuration
    >>> model = MiniMaxText01Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "minimax_text_01"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=None,
        hidden_act="silu",
        max_position_embeddings=4096 * 32,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
        tie_word_embeddings=False,
        rope_theta=1e6,
        sliding_window=None,
        attention_dropout=0.0,
        num_experts_per_tok=2,
        num_local_experts=8,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        router_jitter_noise=0.0,
        attn_type_list = None,
        block_size = 256,
        residual_post_norm=False,
        layernorm_attention_alpha=1,
        layernorm_attention_beta=1,
        layernorm_lightning_attention_alpha=1,
        layernorm_lightning_attention_beta=1,
        layernorm_mlp_alpha=1,
        layernorm_mlp_beta=1,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads

        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_jitter_noise = router_jitter_noise

        # use softmax-attention after `interval` lightning-attentions
        interval = num_hidden_layers // 10
        self.attn_type_list = (
            [1 if i%interval==interval-1 else 0 for i in range(num_hidden_layers)]
            if attn_type_list is None else attn_type_list
        )

        self.block_size = block_size
        self.residual_post_norm = residual_post_norm
        self.layernorm_attention_alpha = layernorm_attention_alpha
        self.layernorm_attention_beta = layernorm_attention_beta
        self.layernorm_lightning_attention_alpha = layernorm_lightning_attention_alpha
        self.layernorm_lightning_attention_beta = layernorm_lightning_attention_beta
        self.layernorm_mlp_alpha = layernorm_mlp_alpha
        self.layernorm_mlp_beta = layernorm_mlp_beta

        # TODO: move these to saved config
        self.residual_post_norm = True
        self.layernorm_attention_alpha = 3.5565588200778455
        self.layernorm_attention_beta = 1.0
        self.layernorm_lightning_attention_alpha = 3.5565588200778455
        self.layernorm_lightning_attention_beta = 1.0
        self.layernorm_mlp_alpha = 3.5565588200778455
        self.layernorm_mlp_beta = 1.0

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class MiniMaxText01LightningAttentionDecay(nn.Module):
    def __init__(self, config: MiniMaxText01Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.num_hidden_layers = config.num_hidden_layers
        self.block_size = config.block_size

    def forward(self, x, seq_len):
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        padding = num_blocks * self.block_size - seq_len

        num_heads_range = torch.arange(self.num_heads).to(x) + 1
        block_size_range = torch.arange(self.block_size).to(x) + 1

        slope_rate = ( 1 / (2 ** (8/self.num_heads)) ) ** num_heads_range
        slope_rate *= 1 - self.layer_idx / (self.num_hidden_layers - 1) + 1e-5 # check small addition
        slope_rate = slope_rate[:, None, None]

        query_decay = torch.exp(-slope_rate * block_size_range[:, None])
        query_decay = query_decay[:, None, :, :]

        key_decay = torch.exp(-slope_rate * (self.block_size - block_size_range[:, None]))
        key_decay = key_decay[:, None, :, :]
        key_decay = key_decay.repeat(1, num_blocks, 1, 1)
        key_decay[:, -1, :self.block_size-padding] = key_decay[:, -1, padding:]

        diagonal_decay = block_size_range[:, None] - block_size_range[None, :]
        diagonal_decay = slope_rate * diagonal_decay[None, :, :]
        diagonal_decay = torch.where(diagonal_decay >= 0, -diagonal_decay, float("-inf"))
        diagonal_decay = torch.exp(diagonal_decay)
        diagonal_decay = diagonal_decay[:, None, :, :]

        block_lengths = torch.cat((
            torch.full((num_blocks-1,), self.block_size),
            torch.tensor([self.block_size-padding])
        )).to(x)
        block_decay = torch.exp(-slope_rate[:, None, :, :] * block_lengths[:, None, None])

        return key_decay, query_decay, diagonal_decay, block_decay


class MiniMaxText01LightningAttention(nn.Module):
    def __init__(self, config: MiniMaxText01Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.num_hidden_layers = config.num_hidden_layers
        self.block_size = config.block_size

        self.act_fn = ACT2FN[config.hidden_act]
        self.norm = MixtralRMSNorm(self.head_dim * self.num_heads)
        self.qkv_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim * 3, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.output_gate = nn.Linear(config.hidden_size, self.num_heads * self.num_heads, bias=False)
        self.decay_factors = MiniMaxText01LightningAttentionDecay(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        padding = num_blocks * self.block_size - seq_len

        qkv_states = self.act_fn(self.qkv_proj(hidden_states))
        qkv_states = qkv_states.reshape(batch_size, seq_len, self.num_heads, 3 * self.head_dim)

        query_states, key_states, value_states = torch.split(qkv_states, [self.head_dim] * 3, dim=3)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # TODO: apply attention_mask

        query_states = F.pad(query_states, (0, 0, 0, padding))
        key_states = F.pad(key_states, (0, 0, 0, padding))
        value_states = F.pad(value_states, (0, 0, 0, padding))

        query_states = query_states.reshape(batch_size, self.num_heads, num_blocks, self.block_size, self.head_dim)
        key_states = key_states.reshape(batch_size, self.num_heads, num_blocks, self.block_size, self.head_dim)
        value_states = value_states.reshape(batch_size, self.num_heads, num_blocks, self.block_size, self.head_dim)

        # TODO: get from past_key_value[layer_idx]
        next_cache = torch.zeros(batch_size, self.num_heads, 1, self.head_dim, self.head_dim).to(value_states)

        # get decay factors
        key_decay, query_decay, diagonal_decay, block_decay = self.decay_factors(query_states, seq_len)

        # intra: ( Q @ K.T ) @ V -> QK * V
        attn_weights_intra = torch.matmul(query_states, key_states.transpose(-1, -2))
        attn_output_intra = torch.matmul(attn_weights_intra * diagonal_decay, value_states)

        # inter: Q @ ( K.T @ V ) -> Q * KV
        attn_weights_inter = torch.matmul((key_states * key_decay).transpose(-1, -2), value_states)
        attn_weights_inter = torch.cat([next_cache, attn_weights_inter], dim=2)
        for i in range(num_blocks):
            attn_weights_inter[:, :, i+1, :, :] += attn_weights_inter[:, :, i, :, :] * block_decay[:, i, :, :]
        next_cache = attn_weights_inter[:, :, -1, :, :]
        attn_weights_inter = attn_weights_inter[:, :, :-1, :, :]
        attn_output_inter = torch.matmul(query_states * query_decay, attn_weights_inter)

        # inter + intra
        attn_output = attn_output_inter + attn_output_intra
        attn_output = attn_output.reshape(batch_size, self.num_heads, seq_len+padding, self.head_dim)
        attn_output = attn_output[:, :, :seq_len, :]

        # final output projection
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        attn_output = self.norm(attn_output)
        attn_output = F.sigmoid(self.output_gate(hidden_states)) * attn_output
        attn_output = self.out_proj(attn_output)

        # TODO: put to past_key_value[layer_idx]
        next_cache

        # TODO: remove these
        print()
        print(self.layer_idx)
        print(next_cache)

        return attn_output, None


class MiniMaxText01Attention(MixtralAttention):
    pass


class MiniMaxText01DecoderLayer(MixtralDecoderLayer):
    def __init__(self, config: MiniMaxText01Config, layer_idx: int):
        super().__init__(config, layer_idx)

        self.layer_idx = layer_idx
        self.residual_post_norm = config.residual_post_norm
        self.layernorm_attention_alpha = config.layernorm_attention_alpha
        self.layernorm_attention_beta = config.layernorm_attention_beta
        self.layernorm_lightning_attention_alpha = config.layernorm_lightning_attention_alpha
        self.layernorm_lightning_attention_beta = config.layernorm_lightning_attention_beta
        self.layernorm_mlp_alpha = config.layernorm_mlp_alpha
        self.layernorm_mlp_beta = config.layernorm_mlp_beta

        if config.attn_type_list[layer_idx] == 0:
            self.self_attn = MiniMaxText01LightningAttention(config, layer_idx)
            self.layernorm_alpha = self.layernorm_lightning_attention_alpha
            self.layernorm_beta = self.layernorm_lightning_attention_beta
        else:
            self.self_attn = MiniMaxText01Attention(config, layer_idx)
            self.layernorm_alpha = self.layernorm_attention_alpha
            self.layernorm_beta = self.layernorm_attention_beta

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_router_logits (`bool`, *optional*):
                Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
                should not be returned during inference.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        if self.residual_post_norm:
            residual = hidden_states

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual * self.layernorm_alpha + hidden_states * self.layernorm_beta

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.residual_post_norm:
            residual = hidden_states
        hidden_states, router_logits = self.block_sparse_moe(hidden_states)
        hidden_states = residual * self.layernorm_mlp_alpha + hidden_states * self.layernorm_mlp_beta

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


class MiniMaxText01Model(MixtralModel):
    def __init__(self, config: MiniMaxText01Config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [MiniMaxText01DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )


class MiniMaxText01ForCausalLM(MixtralForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = MiniMaxText01Model(config)


class MiniMaxText01ForSequenceClassification(MixtralForSequenceClassification):
    pass


class MiniMaxText01ForTokenClassification(MixtralForTokenClassification):
    pass


class MiniMaxText01ForQuestionAnswering(MixtralForQuestionAnswering):
    pass
