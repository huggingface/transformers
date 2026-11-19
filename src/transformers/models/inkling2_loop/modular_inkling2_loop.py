# Copyright 2026 the HuggingFace Team. All rights reserved.
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
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...masking_utils import create_causal_mask, create_recurrent_attention_mask, create_sliding_window_causal_mask
from ...modeling_outputs import MoeModelOutputWithPast
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import OutputRecorder, capture_outputs
from ..auto import AutoConfig, AutoModel
from ..inkling2.configuration_inkling2 import Inkling2Config, Inkling2TextConfig
from ..inkling2.modeling_inkling2 import (
    Inkling2BlockAttentionResidual,
    Inkling2Experts,
    Inkling2ForCausalLM,
    Inkling2ForConditionalGeneration,
    Inkling2FullAttention,
    Inkling2FullDecoderLayer,
    Inkling2Model,
    Inkling2PreTrainedModel,
    Inkling2RelativeLogits,
    Inkling2ScaledL2Clip,
    Inkling2SlidingAttention,
    Inkling2SlidingDecoderLayer,
    Inkling2TextModel,
    Inkling2TopKRouter,
    Inkling2UnweightedRMSNorm,
)


# TODO: update before release
@auto_docstring(checkpoint="thinkingmachines/inkling-1116-share")
@strict
class Inkling2LoopTextConfig(Inkling2TextConfig):
    r"""
    unpadded_vocab_size (`int`, *optional*, defaults to 200058):
        Number of rows of the unembedding matrix that hold real tokens. Logits beyond it are dropped.
    swa_num_attention_heads (`int`, *optional*, defaults to 24):
        Number of attention heads in the sliding-window layers.
    swa_num_key_value_heads (`int`, *optional*, defaults to 8):
        Number of key/value heads in the sliding-window layers.
    swa_head_dim (`int`, *optional*, defaults to 128):
        Dimension of query and key heads in the sliding-window layers.
    sliding_window_size (`int`, *optional*, defaults to 1024):
        Size of the sliding attention window used by layers whose `layer_types` entry is `"hybrid_sliding"`.
    d_rel (`int`, *optional*, defaults to 16):
        Per-head dimension of the relative states that are mixed into the relative position bias.
    rel_extent (`int`, *optional*, defaults to 1024):
        Backward distance, in tokens, over which the relative position bias is applied in the full-attention layers.
    log_scaling_n_floor (`int`, *optional*, defaults to 8192):
        Position from which queries and relative states are scaled up logarithmically in the full-attention layers.
    log_scaling_alpha (`float`, *optional*, defaults to 0.1):
        Strength of the logarithmic scaling controlled by `log_scaling_n_floor`.
    qk_l2_clip_threshold (`float`, *optional*, defaults to 22.627416997969522):
        Maximum L2 norm of each query and key head vector. Longer vectors are scaled down to this norm.
    rel_l2_clip_threshold (`float`, *optional*, defaults to 8.0):
        Maximum L2 norm of each relative-state head vector.
    n_layers_per_attn_res_block (`int`, *optional*, defaults to 6):
        Number of decoder layers summed into one block before the block becomes a source of the attention residuals.
    mlp_gate_softcap (`float`, *optional*, defaults to 4.0):
        Softcap applied to the linear part of the experts' gate projection.
    mlp_up_softcap (`float`, *optional*, defaults to 25.0):
        Softcap applied to the experts' up projection.
    final_logit_softcapping (`float`, *optional*, defaults to 30.0):
        Softcap applied to the output logits.
    scoring_func (`str`, *optional*, defaults to `"sigmoid"`):
        Activation turning the router logits into expert scores.
    loop_layer_index_start (`int`, *optional*, defaults to 6):
        Index of the first looped decoder layer. Must be a multiple of `n_layers_per_attn_res_block`.
    loop_layer_index_end (`int`, *optional*, defaults to 18):
        Index after the last looped decoder layer. Layers `loop_layer_index_start` to
        `loop_layer_index_end - 1` are looped. Must be a multiple of `n_layers_per_attn_res_block`.
    loop_num_passes (`int`, *optional*, defaults to 3):
        Number of loops.
    loop_gate_rank (`int`, *optional*, defaults to 64):
        Rank of the message gate projections that mix the residual blocks of the previous pass into the current
        one.
    """

    model_type = "inkling2_loop_text"

    hidden_size: int = 2304
    num_hidden_layers: int = 24
    num_attention_heads: int = 24
    swa_num_attention_heads: int = 24
    log_scaling_n_floor: int = 8192
    moe_intermediate_size: int = 1152
    n_routed_experts: int = 128
    loop_layer_index_start: int = 6
    loop_layer_index_end: int = 18
    loop_num_passes: int = 3
    loop_gate_rank: int = 64

    # TODO: check tp/ep plans

    def __post_init__(self, **kwargs):
        if (loop_config := kwargs.get("loop_config")) is not None:
            self.loop_layer_index_start = loop_config.get("core_start", self.loop_layer_index_start)
            self.loop_layer_index_end = loop_config.get("core_end", self.loop_layer_index_end)
            self.loop_num_passes = loop_config.get("num_passes", self.loop_num_passes)
            self.loop_gate_rank = loop_config.get("gate_rank", self.loop_gate_rank)
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        if self.loop_num_passes < 1:
            raise ValueError(f"`loop_num_passes` ({self.loop_num_passes}) must be at least 1.")
        if self.loop_layer_index_start % self.n_layers_per_attn_res_block != 0:
            raise ValueError(
                f"`loop_layer_index_start` ({self.loop_layer_index_start}) must be a multiple of "
                f"`n_layers_per_attn_res_block` ({self.n_layers_per_attn_res_block})."
            )
        if self.loop_layer_index_end % self.n_layers_per_attn_res_block != 0:
            raise ValueError(
                f"`loop_layer_index_end` ({self.loop_layer_index_end}) must be a multiple of "
                f"`n_layers_per_attn_res_block` ({self.n_layers_per_attn_res_block})."
            )

    @property
    def cache_layer_indices(self) -> list[int]:
        """Returns decoder layer index for each cache layer."""
        looped_layer_indices = range(self.loop_layer_index_start, self.loop_layer_index_end)
        return list(range(self.num_hidden_layers)) + (self.loop_num_passes - 1) * list(looped_layer_indices)

    @property
    def total_num_hidden_layers(self) -> int:
        """Total number of decoder layers visited in each forward pass. This takes looped layers into account."""
        return len(self.cache_layer_indices)


@auto_docstring(checkpoint="thinkingmachines/inkling-1116-share")
@strict
class Inkling2LoopConfig(Inkling2Config):
    r"""
    image_token_id (`int`, *optional*, defaults to 200054):
        Token index of the image placeholders, each replaced by one image feature.
    """

    model_type = "inkling2_loop"
    sub_configs_defaults = {
        "text_config": Inkling2LoopTextConfig,
        "vision_config": SubConfigSpec(
            # TODO: text_hidden_size must be set in hub config to work when loading from hub
            config_class=AutoConfig,
            model_type="inkling2_vision",
            init_kwargs={"text_hidden_size": 2304},
        ),
    }

    vision_config: dict | PreTrainedConfig | None = None

    def __post_init__(self, **kwargs):
        PreTrainedConfig.__post_init__(self, **kwargs)
        # TODO: remove before merge. Currently needed because vision_config.text_hidden_size is missing from
        # the hub config.
        self.vision_config.text_hidden_size = self.text_config.hidden_size


class Inkling2LoopUnweightedRMSNorm(Inkling2UnweightedRMSNorm):
    pass


class Inkling2LoopMessageGate(nn.Module):
    """Mixes hidden states and message as  `(1 - forget_gate) * hidden_states + message_gate * message`"""

    def __init__(self, config: Inkling2LoopTextConfig):
        super().__init__()
        self.norm = Inkling2LoopUnweightedRMSNorm(eps=config.rms_norm_eps)
        self.down_proj = nn.Linear(2 * config.hidden_size, config.loop_gate_rank, bias=False)
        self.forget_gate_proj = nn.Linear(config.loop_gate_rank, config.hidden_size, bias=False)
        self.message_gate_proj = nn.Linear(config.loop_gate_rank, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        hidden_states, message = hidden_states.float(), message.float()
        gate_hidden_states = torch.cat([self.norm(hidden_states), self.norm(message)], dim=-1)
        gate_hidden_states = self.down_proj(gate_hidden_states.to(self.down_proj.weight.dtype)).float()

        forget_gate = torch.tanh(self.forget_gate_proj(gate_hidden_states))
        message_gate = torch.tanh(self.message_gate_proj(gate_hidden_states))
        return (1 - forget_gate) * hidden_states + message_gate * message


class Inkling2LoopMessageAttentionResidual(Inkling2BlockAttentionResidual):
    """Retrieves messages from the block states of the previous block using learned queries."""

    def __init__(self, config: Inkling2LoopTextConfig):
        super().__init__(config)
        # one query per message (query, key, value and mlp) for each loop layer
        self.query = nn.Parameter(
            torch.zeros(config.loop_layer_index_end - config.loop_layer_index_start, 4, config.hidden_size)
        )

    def forward(
        self, block_states: torch.Tensor, partial_block_states: torch.Tensor, loop_layer_idx: int
    ) -> torch.Tensor:
        stacked_states = torch.cat([block_states, partial_block_states.unsqueeze(-2)], dim=-2).float()
        stacked_states = self.norm(stacked_states)

        hidden_size = stacked_states.shape[-1]
        query_states = self.query[loop_layer_idx].float()
        query_states = query_states.expand(*stacked_states.shape[:-2], -1, -1)
        return F.scaled_dot_product_attention(query_states, stacked_states, stacked_states, scale=hidden_size**-0.5)


class Inkling2LoopScaledL2Clip(Inkling2ScaledL2Clip):
    pass


class Inkling2LoopRelativeLogits(Inkling2RelativeLogits):
    pass


class Inkling2LoopSlidingAttention(Inkling2SlidingAttention):
    pass


class Inkling2LoopFullAttention(Inkling2FullAttention):
    pass


class Inkling2LoopExperts(Inkling2Experts):
    pass


class Inkling2LoopBlockAttentionResidual(Inkling2BlockAttentionResidual):
    pass


class Inkling2LoopSlidingDecoderLayer(Inkling2SlidingDecoderLayer):
    pass


class Inkling2LoopSlidingLoopDecoderLayer(Inkling2SlidingDecoderLayer):
    def __init__(self, config: Inkling2LoopTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.q_message_gate = Inkling2LoopMessageGate(config)
        self.k_message_gate = Inkling2LoopMessageGate(config)
        self.v_message_gate = Inkling2LoopMessageGate(config)
        self.mlp_message_gate = Inkling2LoopMessageGate(config)

    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        hidden_states: torch.Tensor,
        block_states: torch.Tensor,
        messages: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        conv_mask: torch.Tensor | None = None,
        cache: Cache | None = None,
        cache_offset: int = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        r"""
        messages (`torch.Tensor` of shape `(batch_size, sequence_length, 4, hidden_size)`, *optional*):
            Messages of the previous pass through the loop layers for the query, key, value and MLP inputs, in the
            passes after the first.
        cache_offset (`int`, *optional*, defaults to 0):
            Offset added to `layer_idx` to get the cache layer holding the keys, values and convolution states of
            this call.
        """
        dtype = hidden_states.dtype
        block_residual = hidden_states
        if self.attention_input is not None:
            hidden_states = self.attention_input(block_states, hidden_states)[0]

        if messages is None:
            query_hidden_states = key_hidden_states = value_hidden_states = self.input_layernorm(hidden_states)
        else:
            query_message, key_message, value_message, mlp_message = messages.unbind(-2)
            query_hidden_states = self.input_layernorm(self.q_message_gate(hidden_states, query_message))
            key_hidden_states = self.input_layernorm(self.k_message_gate(hidden_states, key_message))
            value_hidden_states = self.input_layernorm(self.v_message_gate(hidden_states, value_message))

        hidden_states, _ = self.self_attn(
            hidden_states=query_hidden_states.to(dtype),
            key_hidden_states=key_hidden_states.to(dtype),
            value_hidden_states=value_hidden_states.to(dtype),
            attention_mask=attention_mask,
            conv_mask=conv_mask,
            cache=cache,
            cache_offset=cache_offset,
            **kwargs,
        )
        hidden_states = self.post_attention_convolution(
            hidden_states, cache=cache, conv_mask=conv_mask, cache_offset=cache_offset
        )

        residual = hidden_states = hidden_states.float()
        if self.is_block_boundary:
            block_states = torch.cat([block_states, block_residual.unsqueeze(-2)], dim=-2)
        else:
            residual = hidden_states = block_residual.float() + hidden_states

        hidden_states = self.mlp_input(block_states, hidden_states)[0]
        if messages is not None:
            hidden_states = self.mlp_message_gate(hidden_states, mlp_message)
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states).to(dtype))
        hidden_states = self.post_mlp_convolution(
            hidden_states, cache=cache, conv_mask=conv_mask, cache_offset=cache_offset
        )
        return (residual + hidden_states).to(dtype)


class Inkling2LoopFullDecoderLayer(Inkling2FullDecoderLayer):
    pass


class Inkling2LoopFullLoopDecoderLayer(Inkling2FullDecoderLayer):
    def __init__(self, config: Inkling2LoopTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.q_message_gate = Inkling2LoopMessageGate(config)
        self.k_message_gate = Inkling2LoopMessageGate(config)
        self.v_message_gate = Inkling2LoopMessageGate(config)
        self.mlp_message_gate = Inkling2LoopMessageGate(config)

    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        hidden_states: torch.Tensor,
        block_states: torch.Tensor,
        messages: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        conv_mask: torch.Tensor | None = None,
        cache: Cache | None = None,
        cache_offset: int = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        r"""
        messages (`torch.Tensor` of shape `(batch_size, sequence_length, 4, hidden_size)`, *optional*):
            Messages of the previous pass through the loop layers for the query, key, value and MLP inputs, in the
            passes after the first.
        cache_offset (`int`, *optional*, defaults to 0):
            Offset added to `layer_idx` to get the cache layer holding the keys, values and convolution states of
            this call.
        """
        dtype = hidden_states.dtype
        block_residual = hidden_states
        if self.attention_input is None:
            query_hidden_states = key_hidden_states = value_hidden_states = hidden_states
        else:
            query_hidden_states, key_hidden_states, value_hidden_states = self.attention_input(
                block_states, hidden_states
            )

        if messages is not None:
            query_message, key_message, value_message, mlp_message = messages.unbind(-2)
            query_hidden_states = self.q_message_gate(query_hidden_states, query_message)
            key_hidden_states = self.k_message_gate(key_hidden_states, key_message)
            value_hidden_states = self.v_message_gate(value_hidden_states, value_message)

        hidden_states, _ = self.self_attn(
            hidden_states=self.q_input_layernorm(query_hidden_states).to(dtype),
            key_hidden_states=self.k_input_layernorm(key_hidden_states).to(dtype),
            value_hidden_states=self.v_input_layernorm(value_hidden_states).to(dtype),
            attention_mask=attention_mask,
            conv_mask=conv_mask,
            cache=cache,
            cache_offset=cache_offset,
            **kwargs,
        )
        hidden_states = self.post_attention_convolution(
            hidden_states, cache=cache, conv_mask=conv_mask, cache_offset=cache_offset
        )

        residual = hidden_states = hidden_states.float()
        if self.is_block_boundary:
            block_states = torch.cat([block_states, block_residual.unsqueeze(-2)], dim=-2)
        else:
            residual = hidden_states = block_residual.float() + hidden_states

        hidden_states = self.mlp_input(block_states, hidden_states)[0]
        if messages is not None:
            hidden_states = self.mlp_message_gate(hidden_states, mlp_message)
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states).to(dtype))
        hidden_states = self.post_mlp_convolution(
            hidden_states, cache=cache, conv_mask=conv_mask, cache_offset=cache_offset
        )
        return (residual + hidden_states).to(dtype)


class Inkling2LoopTopKRouter(Inkling2TopKRouter):
    pass


@auto_docstring
class Inkling2LoopPreTrainedModel(Inkling2PreTrainedModel):
    _no_split_modules = [
        "Inkling2LoopSlidingDecoderLayer",
        "Inkling2LoopFullDecoderLayer",
        "Inkling2LoopSlidingLoopDecoderLayer",
        "Inkling2LoopFullLoopDecoderLayer",
    ]
    _keep_in_fp32_modules_strict = [
        "e_score_correction_bias",
        "mlp.gate.weight",
        "post_attention_convolution",
        "post_mlp_convolution",
        "k_sconv.conv1d",
        "v_sconv.conv1d",
        "attention_input",
        "mlp_input",
        "final_input",
        "message_input",
        "message_gate_proj",
        "forget_gate_proj",
    ]
    _can_record_outputs = {
        "hidden_states": OutputRecorder(
            (
                Inkling2LoopSlidingDecoderLayer,
                Inkling2LoopFullDecoderLayer,
                Inkling2LoopSlidingLoopDecoderLayer,
                Inkling2LoopFullLoopDecoderLayer,
            ),
            num_hidden_layers_key="total_num_hidden_layers",
        ),
        "attentions": [Inkling2LoopSlidingAttention, Inkling2LoopFullAttention],
        "router_logits": OutputRecorder(Inkling2LoopTopKRouter, index=0),
    }

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Inkling2LoopMessageAttentionResidual):
            init.zeros_(module.query)


@auto_docstring
class Inkling2LoopTextModel(Inkling2TextModel):
    def __init__(self, config: Inkling2LoopTextConfig):
        super().__init__(config)
        self.layers = nn.ModuleList()
        for layer_idx in range(config.num_hidden_layers):
            is_sliding = config.layer_types[layer_idx] == "hybrid_sliding"
            is_loop_layer = config.loop_layer_index_start <= layer_idx < config.loop_layer_index_end
            if is_loop_layer:
                layer_class = Inkling2LoopSlidingLoopDecoderLayer if is_sliding else Inkling2LoopFullLoopDecoderLayer
            else:
                layer_class = Inkling2LoopSlidingDecoderLayer if is_sliding else Inkling2LoopFullDecoderLayer
            self.layers.append(layer_class(config, layer_idx))
        self.message_input = Inkling2LoopMessageAttentionResidual(config)

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
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
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
                "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
            }

        loop_start = self.config.loop_layer_index_start
        loop_end = self.config.loop_layer_index_end
        num_loop_layers = loop_end - loop_start

        hidden_states = inputs_embeds
        block_states = inputs_embeds.new_zeros(*inputs_embeds.shape[:-1], 0, inputs_embeds.shape[-1])

        # Pre loop layers
        for decoder_layer in self.layers[:loop_start]:
            new_hidden_states = decoder_layer(
                hidden_states,
                block_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                conv_mask=causal_mask_mapping["linear_attention"],
                cache=cache,
                position_ids=position_ids,
                **kwargs,
            )
            if decoder_layer.is_block_boundary:
                block_states = torch.cat([block_states.to(hidden_states.device), hidden_states.unsqueeze(-2)], dim=-2)
            hidden_states = new_hidden_states

        pre_loop_hidden_states, pre_loop_block_states = hidden_states, block_states
        num_pre_loop_blocks = pre_loop_block_states.shape[-2]

        # First pass over loop layers
        for decoder_layer in self.layers[loop_start:loop_end]:
            new_hidden_states = decoder_layer(
                hidden_states,
                block_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                conv_mask=causal_mask_mapping["linear_attention"],
                cache=cache,
                position_ids=position_ids,
                **kwargs,
            )
            if decoder_layer.is_block_boundary:
                block_states = torch.cat([block_states.to(hidden_states.device), hidden_states.unsqueeze(-2)], dim=-2)
            hidden_states = new_hidden_states

        # Remaining passes restart from the pre loop states and receive messages from the previous pass
        for loop_idx in range(1, self.config.loop_num_passes):
            # +1 to skip the states appended by the first loop pass
            previous_block_states = block_states[..., num_pre_loop_blocks + 1 :, :]
            previous_hidden_states = hidden_states
            hidden_states, block_states = pre_loop_hidden_states, pre_loop_block_states
            # Each pass uses its own copies of the loop cache layers, appended after the model's layers
            cache_offset = self.config.num_hidden_layers + (loop_idx - 1) * num_loop_layers - loop_start
            for loop_layer_idx, decoder_layer in enumerate(self.layers[loop_start:loop_end]):
                messages = self.message_input(previous_block_states, previous_hidden_states, loop_layer_idx)
                new_hidden_states = decoder_layer(
                    hidden_states,
                    block_states,
                    messages,
                    attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                    conv_mask=causal_mask_mapping["linear_attention"],
                    cache=cache,
                    cache_offset=cache_offset,
                    position_ids=position_ids,
                    **kwargs,
                )
                if decoder_layer.is_block_boundary:
                    block_states = torch.cat(
                        [block_states.to(hidden_states.device), hidden_states.unsqueeze(-2)], dim=-2
                    )
                hidden_states = new_hidden_states

        # Post loop layers
        for decoder_layer in self.layers[loop_end:]:
            new_hidden_states = decoder_layer(
                hidden_states,
                block_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                conv_mask=causal_mask_mapping["linear_attention"],
                cache=cache,
                position_ids=position_ids,
                **kwargs,
            )
            if decoder_layer.is_block_boundary:
                block_states = torch.cat([block_states.to(hidden_states.device), hidden_states.unsqueeze(-2)], dim=-2)
            hidden_states = new_hidden_states

        hidden_states = self.final_input(block_states, hidden_states)[0]
        hidden_states = self.norm(hidden_states).to(inputs_embeds.dtype)
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            cache=cache,
        )


class Inkling2LoopForCausalLM(Inkling2ForCausalLM):
    def forward(self, **super_kwargs):
        r"""
        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoTokenizer, Inkling2LoopForCausalLM

        >>> tokenizer = AutoTokenizer.from_pretrained("thinkingmachines/inkling-1116-share")
        >>> model = Inkling2LoopForCausalLM.from_pretrained(
        ...     "thinkingmachines/inkling-1116-share", dtype=torch.bfloat16, device_map="auto"
        ... )

        >>> messages = [{"role": "user", "content": "Explain in one sentence why the sky is usually blue."}]
        >>> inputs = tokenizer.apply_chat_template(
        ...     messages,
        ...     tokenize=True,
        ...     add_generation_prompt=True,
        ...     return_dict=True,
        ...     return_tensors="pt",
        ... ).to(model.device)
        >>> input_len = inputs["input_ids"].shape[-1]

        >>> outputs = model.generate(**inputs, max_new_tokens=1024)
        >>> print(tokenizer.decode(outputs[0][input_len:], skip_special_tokens=False))
        ```"""
        return super().forward(**super_kwargs)


class Inkling2LoopModel(Inkling2Model):
    def __init__(self, config: Inkling2LoopConfig):
        super().__init__(config)
        self.vision_tower = AutoModel.from_config(config.vision_config)

    def forward(self, **super_kwargs):
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            Temporal, height and width of each image, counted in language model image tokens.

        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, Inkling2LoopModel

        >>> processor = AutoProcessor.from_pretrained("thinkingmachines/inkling-1116-share")
        >>> model = Inkling2LoopModel.from_pretrained(
        ...     "thinkingmachines/inkling-1116-share", dtype=torch.bfloat16, device_map="auto"
        ... )

        >>> messages = [
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {
        ...                 "type": "image",
        ...                 "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
        ...             },
        ...             {"type": "text", "text": "What animal is in this image, and where is it?"},
        ...         ],
        ...     },
        ... ]

        >>> inputs = processor.apply_chat_template(
        ...     messages,
        ...     tokenize=True,
        ...     add_generation_prompt=True,
        ...     return_dict=True,
        ...     return_tensors="pt",
        ... ).to(model.device)

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        ```
        """
        return super().forward(**super_kwargs)


class Inkling2LoopForConditionalGeneration(Inkling2ForConditionalGeneration):
    def forward(self, **super_kwargs):
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            Temporal, height and width of each image, counted in language model image tokens.

        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, Inkling2LoopForConditionalGeneration

        >>> processor = AutoProcessor.from_pretrained("thinkingmachines/inkling-1116-share")
        >>> model = Inkling2LoopForConditionalGeneration.from_pretrained(
        ...     "thinkingmachines/inkling-1116-share", dtype=torch.bfloat16, device_map="auto"
        ... )

        >>> messages = [
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {
        ...                 "type": "image",
        ...                 "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
        ...             },
        ...             {"type": "text", "text": "What animal is in this image, and where is it?"},
        ...         ],
        ...     },
        ... ]

        >>> inputs = processor.apply_chat_template(
        ...     messages,
        ...     tokenize=True,
        ...     add_generation_prompt=True,
        ...     return_dict=True,
        ...     return_tensors="pt",
        ... ).to(model.device)
        >>> input_len = inputs["input_ids"].shape[-1]

        >>> outputs = model.generate(**inputs, max_new_tokens=1024)
        >>> print(processor.decode(outputs[0][input_len:], skip_special_tokens=False))
        ```
        """
        return super().forward(**super_kwargs)


__all__ = [
    "Inkling2LoopConfig",
    "Inkling2LoopTextConfig",
    "Inkling2LoopPreTrainedModel",
    "Inkling2LoopTextModel",
    "Inkling2LoopForCausalLM",
    "Inkling2LoopModel",
    "Inkling2LoopForConditionalGeneration",
]
