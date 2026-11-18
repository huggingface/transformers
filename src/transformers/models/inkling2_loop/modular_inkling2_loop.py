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

import math
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...integrations import use_kernelized_func
from ...integrations.accelerate import force_accelerate_hooks
from ...masking_utils import create_causal_mask, create_recurrent_attention_mask, create_sliding_window_causal_mask
from ...modeling_outputs import MoeModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import OutputRecorder, capture_outputs
from ..auto import AutoConfig, AutoModel
from ..inkling2.configuration_inkling2 import Inkling2Config, Inkling2TextConfig
from ..inkling2.modeling_inkling2 import (
    Inkling2BlockAttentionResidual,
    Inkling2DynamicShortConvolution,
    Inkling2Experts,
    Inkling2ForCausalLM,
    Inkling2ForConditionalGeneration,
    Inkling2FullAttention,
    Inkling2FullDecoderLayer,
    Inkling2Model,
    Inkling2PreTrainedModel,
    Inkling2RelativeLogits,
    Inkling2ScaledL2Clip,
    Inkling2ShortConvolution,
    Inkling2SlidingAttention,
    Inkling2SlidingDecoderLayer,
    Inkling2TextModel,
    Inkling2TopKRouter,
    Inkling2UnweightedRMSNorm,
    eager_attention_forward,
)
from ..qwen3_next.modeling_qwen3_next import apply_mask_to_padding_states, causal_conv1d_fn, causal_conv1d_update


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
    loop_config (`dict`, *optional*):
        The looped core of the decoder, with keys `core_start`, `core_end`, `num_passes` and `gate_rank`. Layers
        `core_start` to `core_end - 1` run `num_passes` times with the same weights. Each pass after the first
        starts again from the output of the layers before the core and reads the residual blocks of the previous
        pass through message controllers of rank `gate_rank`. The core boundaries must be multiples of
        `n_layers_per_attn_res_block`. Defaults to `{"core_start": 6, "core_end": 18, "num_passes": 3,
        "gate_rank": 64}`.
    """

    model_type = "inkling2_loop_text"

    hidden_size: int = 2304
    num_hidden_layers: int = 24
    num_attention_heads: int = 24
    swa_num_attention_heads: int = 24
    log_scaling_n_floor: int = 8192
    moe_intermediate_size: int = 1152
    n_routed_experts: int = 128
    # TODO: should prob make those flat attributes instead of dict
    loop_config: dict | None = None

    # TODO: check tp/ep plans

    def __post_init__(self, **kwargs):
        if self.loop_config is None:
            # TODO: Better naming: layer_index_start, layer_index_end, num_passes, gate_rank
            self.loop_config = {"core_start": 6, "core_end": 18, "num_passes": 3, "gate_rank": 64}
        super().__post_init__(**kwargs)

    @property
    def cache_layer_indices(self) -> list[int]:
        """Returns decoder layer index for each cache layer."""
        looped_layer_indices = range(self.loop_config["core_start"], self.loop_config["core_end"])
        return list(range(self.num_hidden_layers)) + (self.loop_config["num_passes"] - 1) * list(looped_layer_indices)

    @property
    def total_num_hidden_layers(self) -> int:
        """Total number of decoder layers visited in each forward pass. This takes looped layers into account."""
        return len(self.cache_layer_indices)

    # TODO: move to validate archictecture?
    def validate_loop_config(self):
        keys = {"core_start", "core_end", "num_passes", "gate_rank"}
        if set(self.loop_config) != keys:
            raise ValueError(
                f"`loop_config` must have exactly the keys {sorted(keys)}, got {sorted(self.loop_config)}."
            )
        core_start, core_end = self.loop_config["core_start"], self.loop_config["core_end"]
        if not 0 <= core_start < core_end <= self.num_hidden_layers:
            raise ValueError(
                f"The loop core [{core_start}, {core_end}) must lie within the {self.num_hidden_layers} layers."
            )
        if core_start % self.n_layers_per_attn_res_block or core_end % self.n_layers_per_attn_res_block:
            raise ValueError(
                f"The loop core boundaries ({core_start}, {core_end}) must be multiples of "
                f"`n_layers_per_attn_res_block` ({self.n_layers_per_attn_res_block})."
            )
        if self.loop_config["num_passes"] < 2 or self.loop_config["gate_rank"] < 1:
            raise ValueError("`loop_config` needs `num_passes` >= 2 and `gate_rank` >= 1.")


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


class Inkling2LoopUnweightedRMSNorm(Inkling2UnweightedRMSNorm):
    pass


class Inkling2LoopMessageController(nn.Module):
    """
    Mixes a message from the previous pass into one input of a core layer, per token and channel:
    `(1 - forget) * hidden_states + gain * message`, where `gain` and `forget` are `tanh` of low-rank projections of
    the normalized input and message.
    """

    def __init__(self, config: Inkling2LoopTextConfig):
        super().__init__()
        self.norm = Inkling2LoopUnweightedRMSNorm()
        self.down_proj = nn.Linear(2 * config.hidden_size, config.loop_config["gate_rank"], bias=False)
        # TODO: find better names?
        self.gain_proj = nn.Linear(config.loop_config["gate_rank"], config.hidden_size, bias=False)
        self.forget_proj = nn.Linear(config.loop_config["gate_rank"], config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        hidden_states, message = hidden_states.float(), message.float()
        controls = torch.cat([self.norm(hidden_states), self.norm(message)], dim=-1)
        controls = self.down_proj(controls.to(self.down_proj.weight.dtype)).float()
        gain = torch.tanh(self.gain_proj(controls))
        forget = torch.tanh(self.forget_proj(controls))
        return (1 - forget) * hidden_states + gain * message


class Inkling2LoopCrossPassMessages(nn.Module):
    """
    Reads the messages of a core layer from the residual blocks the previous pass wrote (the message bank): one
    block attention residual per input (query, key, value and MLP), each with its own query.
    """

    def __init__(self, config: Inkling2LoopTextConfig):
        super().__init__()
        # TODO: inline?
        num_core_layers = config.loop_config["core_end"] - config.loop_config["core_start"]
        # rows `2 * i` and `2 * i + 1` belong to core layer `i`: its query and MLP inputs in `queries`, its key and
        # value inputs in `kv_queries`
        self.queries = nn.Parameter(torch.zeros(2 * num_core_layers, config.hidden_size))
        self.kv_queries = nn.Parameter(torch.zeros(2 * num_core_layers, config.hidden_size))
        self.norm = Inkling2LoopUnweightedRMSNorm()

    def forward(self, message_bank: torch.Tensor, loop_layer_idx: int) -> torch.Tensor:
        # TODO: stacking order looks odd
        query_states = torch.stack(
            [
                self.queries[2 * loop_layer_idx],
                self.kv_queries[2 * loop_layer_idx],
                self.kv_queries[2 * loop_layer_idx + 1],
                self.queries[2 * loop_layer_idx + 1],
            ]
        ).float()  # TODO: check dtype, convert early to float?
        bank_states = self.norm(message_bank.float())

        hidden_size = bank_states.shape[-1]
        # (batch, seq_len, 4, hidden_size), the same queries at every position
        query_states = query_states.expand(*bank_states.shape[:-2], -1, -1)
        return F.scaled_dot_product_attention(query_states, bank_states, bank_states, scale=hidden_size**-0.5)


@use_kernelized_func([causal_conv1d_update, causal_conv1d_fn])
class Inkling2LoopShortConvolution(Inkling2ShortConvolution):
    @force_accelerate_hooks("conv1d")
    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        hidden_states: torch.Tensor,
        cache: Cache | None = None,
        conv_mask: torch.Tensor | None = None,
        cache_layer_idx: int | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        r"""
        cache_layer_idx (`int`, *optional*):
            Cache layer holding the state of this call, `layer_idx` by default. Each pass through the loop core has
            its own.
        """
        # Only diff from Inkling2 is the cache_layer_idx
        cache_layer_idx = self.layer_idx if cache_layer_idx is None else cache_layer_idx

        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()

        residual = hidden_states
        hidden_states = apply_mask_to_padding_states(hidden_states, conv_mask)
        seq_len = hidden_states.shape[1]
        hidden_states = hidden_states.transpose(1, 2)

        conv_state = None if cache is None else cache.get_conv_state(cache_layer_idx, state_idx=self.conv_idx)

        if conv_state is not None and seq_len == 1 and not cache.is_recording_past(cache_layer_idx):
            hidden_states = causal_conv1d_update(
                hidden_states, conv_state, self.conv1d.weight.squeeze(1), self.conv1d.bias
            )
        else:
            if cache is not None:
                hidden_states = cache.update_conv_state(
                    hidden_states, cache_layer_idx, state_idx=self.conv_idx, conv_kernel_size=self.conv_kernel_size
                )

            hidden_states = causal_conv1d_fn(
                hidden_states, self.conv1d.weight.squeeze(1), self.conv1d.bias, seq_idx=kwargs.get("seq_idx")
            )

            if cache is not None:
                hidden_states = hidden_states[:, :, -seq_len:]

        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = (hidden_states + residual) / math.sqrt(self.conv_kernel_size + 1)
        return hidden_states.to(dtype=input_dtype)


class Inkling2LoopDynamicShortConvolution(Inkling2DynamicShortConvolution):
    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        hidden_states: torch.Tensor,
        dynamic_input: torch.Tensor,
        cache: Cache | None = None,
        conv_mask: torch.Tensor | None = None,
        cache_layer_idx: int | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        r"""
        cache_layer_idx (`int`, *optional*):
            Cache layer holding the state of this call, `layer_idx` by default. Each pass through the loop core has
            its own.
        """
        # Only diff from Inkling2 is the cache_layer_idx
        cache_layer_idx = self.layer_idx if cache_layer_idx is None else cache_layer_idx

        input_dtype = hidden_states.dtype
        batch_size, seq_len, hidden_size = hidden_states.shape
        dynamic_weights = self.weight_proj(dynamic_input).float()
        dynamic_weights = dynamic_weights.view(batch_size, seq_len, self.num_heads, 1, self.conv_kernel_size)
        weights = self.static_weight.float().T.view(self.num_heads, -1, self.conv_kernel_size) + dynamic_weights

        hidden_states = hidden_states.float()
        residual = hidden_states
        context = apply_mask_to_padding_states(hidden_states, conv_mask).transpose(1, 2)
        if cache is not None:
            context = cache.update_conv_state(
                context, cache_layer_idx, state_idx=self.conv_idx, conv_kernel_size=self.conv_kernel_size
            )
        context = F.pad(context, (self.conv_kernel_size - 1, 0))[..., -(seq_len + self.conv_kernel_size - 1) :]
        # windows[..., lag] is the token `lag` steps back
        windows = context.unfold(-1, self.conv_kernel_size, 1).flip(-1).transpose(1, 2)
        windows = windows.reshape(batch_size, seq_len, self.num_heads, -1, self.conv_kernel_size)
        hidden_states = (windows * weights).sum(-1).reshape(batch_size, seq_len, hidden_size)
        hidden_states = (hidden_states + residual) / math.sqrt(2 * self.conv_kernel_size + 1)
        return hidden_states.to(input_dtype)


class Inkling2LoopScaledL2Clip(Inkling2ScaledL2Clip):
    pass


class Inkling2LoopRelativeLogits(Inkling2RelativeLogits):
    pass


class Inkling2LoopSlidingAttention(Inkling2SlidingAttention):
    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        conv_mask: torch.Tensor | None = None,
        cache: Cache | None = None,
        key_hidden_states: torch.Tensor | None = None,
        value_hidden_states: torch.Tensor | None = None,
        cache_layer_idx: int | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        r"""
        key_hidden_states (`torch.Tensor`, *optional*):
            Input of the key projection, `hidden_states` by default. Differs from it in the passes after the first
            through the loop core.
        value_hidden_states (`torch.Tensor`, *optional*):
            Input of the value projection, `hidden_states` by default.
        cache_layer_idx (`int`, *optional*):
            Cache layer holding the keys, values and convolution states of this call, `layer_idx` by default. Each
            pass through the loop core has its own.
        """
        # Diff from Inkling2
        cache_layer_idx = self.layer_idx if cache_layer_idx is None else cache_layer_idx

        # TODO: move this to Inkling2 which would also match the full attention class?
        key_hidden_states = hidden_states if key_hidden_states is None else key_hidden_states
        value_hidden_states = hidden_states if value_hidden_states is None else value_hidden_states

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_sconv(
            self.k_proj(key_hidden_states), cache=cache, conv_mask=conv_mask, cache_layer_idx=cache_layer_idx
        )
        value_states = self.v_sconv(
            self.v_proj(value_hidden_states), cache=cache, conv_mask=conv_mask, cache_layer_idx=cache_layer_idx
        )
        relative_states = self.r_proj(hidden_states)
        gate = self.gate_proj(hidden_states)
        sinks = self.sink_proj(hidden_states)

        q_length = query_states.shape[1]
        if cache is not None:
            kv_length, kv_offset = cache.get_mask_sizes(q_length, cache_layer_idx)
            q_offset = cache.get_query_offset(cache_layer_idx)
        else:
            kv_length = q_length
            q_offset, kv_offset = 0, 0
        kv_positions = torch.arange(kv_length, device=hidden_states.device) + kv_offset
        q_positions = torch.arange(q_length, device=hidden_states.device) + q_offset

        query_states = self.q_l2_clip(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_l2_clip(key_states.view(hidden_shape)).transpose(1, 2)
        value_states = value_states.view(hidden_shape)
        value_states = self.v_norm(value_states.float()).to(value_states.dtype)
        value_states = value_states.transpose(1, 2)
        relative_states = self.rel_l2_clip(relative_states.view(*input_shape, self.num_heads, -1))

        if cache is not None:
            key_states, value_states = cache.update(key_states, value_states, cache_layer_idx, kwargs=kwargs)

        position_bias = self.rel_logits_proj(relative_states, q_positions, kv_positions)

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
            position_bias=position_bias,
            sinks=sinks.transpose(1, 2).unsqueeze(-1),
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = (attn_output.float() * torch.sigmoid(gate.float())).to(attn_output.dtype)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Inkling2LoopFullAttention(Inkling2FullAttention):
    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_hidden_states: torch.Tensor,
        value_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        conv_mask: torch.Tensor | None = None,
        cache: Cache | None = None,
        cache_layer_idx: int | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        r"""
        key_hidden_states (`torch.Tensor`):
            Input of the key projection and of the key convolution weights.
        value_hidden_states (`torch.Tensor`):
            Input of the value projection and of the value convolution weights.
        cache_layer_idx (`int`, *optional*):
            Cache layer holding the keys, values and convolution states of this call, `layer_idx` by default. Each
            pass through the loop core has its own.
        """
        # Diff from Inkling2
        cache_layer_idx = self.layer_idx if cache_layer_idx is None else cache_layer_idx

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_sconv(
            self.k_proj(key_hidden_states),
            key_hidden_states,
            cache=cache,
            conv_mask=conv_mask,
            cache_layer_idx=cache_layer_idx,
        )
        value_states = self.v_sconv(
            self.v_proj(value_hidden_states),
            value_hidden_states,
            cache=cache,
            conv_mask=conv_mask,
            cache_layer_idx=cache_layer_idx,
        )
        relative_states = self.r_proj(hidden_states)
        gate = self.gate_proj(hidden_states)
        sinks = self.sink_proj(hidden_states)

        q_length = query_states.shape[1]
        if cache is not None:
            kv_length, kv_offset = cache.get_mask_sizes(q_length, cache_layer_idx)
            q_offset = cache.get_query_offset(cache_layer_idx)
        else:
            kv_length = q_length
            q_offset, kv_offset = 0, 0
        kv_positions = torch.arange(kv_length, device=hidden_states.device) + kv_offset
        q_positions = torch.arange(q_length, device=hidden_states.device) + q_offset

        # log scaling of the queries and relative states, in fp32 as in the original
        tau = 1.0 + self.config.log_scaling_alpha * torch.log(
            ((kwargs["position_ids"] + 1).float() / self.config.log_scaling_n_floor).clamp(min=1.0)
        )
        tau = tau[..., None, None]

        query_states = self.q_l2_clip(query_states.view(hidden_shape))
        query_states = (query_states.float() * tau).to(query_states.dtype).transpose(1, 2)
        key_states = self.k_l2_clip(key_states.view(hidden_shape)).transpose(1, 2)
        value_states = value_states.view(hidden_shape)
        value_states = self.v_norm(value_states.float()).to(value_states.dtype)
        value_states = value_states.transpose(1, 2)
        relative_states = self.rel_l2_clip(relative_states.view(*input_shape, self.num_heads, -1))
        relative_states = (relative_states.float() * tau).to(relative_states.dtype)

        if cache is not None:
            key_states, value_states = cache.update(key_states, value_states, cache_layer_idx, kwargs=kwargs)

        position_bias = self.rel_logits_proj(relative_states, q_positions, kv_positions)

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
            position_bias=position_bias,
            sinks=sinks.transpose(1, 2).unsqueeze(-1),
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = (attn_output.float() * torch.sigmoid(gate.float())).to(attn_output.dtype)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Inkling2LoopExperts(Inkling2Experts):
    pass


class Inkling2LoopBlockAttentionResidual(Inkling2BlockAttentionResidual):
    pass


class Inkling2LoopSlidingDecoderLayer(Inkling2SlidingDecoderLayer):
    def __init__(self, config: Inkling2LoopTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        # TODO: split into two classes or assign identity?
        loop_config = config.loop_config
        if loop_config["core_start"] <= layer_idx < loop_config["core_end"]:
            self.q_controller = Inkling2LoopMessageController(config)
            self.k_controller = Inkling2LoopMessageController(config)
            self.v_controller = Inkling2LoopMessageController(config)
            self.mlp_controller = Inkling2LoopMessageController(config)

    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        hidden_states: torch.Tensor,
        block_states: torch.Tensor,
        messages: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        conv_mask: torch.Tensor | None = None,
        cache: Cache | None = None,
        cache_layer_idx: int | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        r"""
        messages (`torch.Tensor` of shape `(batch_size, sequence_length, 4, hidden_size)`, *optional*):
            Messages of the previous pass through the loop core for the query, key, value and MLP inputs, in the
            passes after the first.
        cache_layer_idx (`int`, *optional*):
            Cache layer holding the states of this call, `layer_idx` by default. Each pass through the loop core has
            its own.
        """
        dtype = hidden_states.dtype
        block_residual = hidden_states
        if self.attention_input is not None:
            hidden_states = self.attention_input(block_states, hidden_states)[0]

        if messages is None:
            query_hidden_states = key_hidden_states = value_hidden_states = self.input_layernorm(hidden_states)
        else:
            query_message, key_message, value_message, _ = messages.unbind(-2)
            query_hidden_states = self.input_layernorm(self.q_controller(hidden_states, query_message))
            key_hidden_states = self.input_layernorm(self.k_controller(hidden_states, key_message))
            value_hidden_states = self.input_layernorm(self.v_controller(hidden_states, value_message))

        hidden_states, _ = self.self_attn(
            hidden_states=query_hidden_states.to(dtype),
            key_hidden_states=key_hidden_states.to(dtype),
            value_hidden_states=value_hidden_states.to(dtype),
            attention_mask=attention_mask,
            conv_mask=conv_mask,
            cache=cache,
            cache_layer_idx=cache_layer_idx,
            **kwargs,
        )
        hidden_states = self.post_attention_convolution(
            hidden_states, cache=cache, conv_mask=conv_mask, cache_layer_idx=cache_layer_idx
        )

        residual = hidden_states = hidden_states.float()
        if self.is_block_boundary:
            block_states = torch.cat([block_states, block_residual.unsqueeze(-2)], dim=-2)
        else:
            residual = hidden_states = block_residual.float() + hidden_states

        hidden_states = self.mlp_input(block_states, hidden_states)[0]
        if messages is not None:
            hidden_states = self.mlp_controller(hidden_states, messages[..., 3, :])
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states).to(dtype))
        hidden_states = self.post_mlp_convolution(
            hidden_states, cache=cache, conv_mask=conv_mask, cache_layer_idx=cache_layer_idx
        )
        return (residual + hidden_states).to(dtype)


class Inkling2LoopFullDecoderLayer(Inkling2FullDecoderLayer):
    def __init__(self, config: Inkling2LoopTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        loop_config = config.loop_config

        if loop_config["core_start"] <= layer_idx < loop_config["core_end"]:
            self.q_controller = Inkling2LoopMessageController(config)
            self.k_controller = Inkling2LoopMessageController(config)
            self.v_controller = Inkling2LoopMessageController(config)
            self.mlp_controller = Inkling2LoopMessageController(config)

    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        hidden_states: torch.Tensor,
        block_states: torch.Tensor,
        messages: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        conv_mask: torch.Tensor | None = None,
        cache: Cache | None = None,
        cache_layer_idx: int | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        r"""
        messages (`torch.Tensor` of shape `(batch_size, sequence_length, 4, hidden_size)`, *optional*):
            Messages of the previous pass through the loop core for the query, key, value and MLP inputs, in the
            passes after the first.
        cache_layer_idx (`int`, *optional*):
            Cache layer holding the states of this call, `layer_idx` by default. Each pass through the loop core has
            its own.
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
            query_message, key_message, value_message, _ = messages.unbind(-2)
            query_hidden_states = self.q_controller(query_hidden_states, query_message)
            key_hidden_states = self.k_controller(key_hidden_states, key_message)
            value_hidden_states = self.v_controller(value_hidden_states, value_message)

        hidden_states, _ = self.self_attn(
            hidden_states=self.q_input_layernorm(query_hidden_states).to(dtype),
            key_hidden_states=self.k_input_layernorm(key_hidden_states).to(dtype),
            value_hidden_states=self.v_input_layernorm(value_hidden_states).to(dtype),
            attention_mask=attention_mask,
            conv_mask=conv_mask,
            cache=cache,
            cache_layer_idx=cache_layer_idx,
            **kwargs,
        )
        hidden_states = self.post_attention_convolution(
            hidden_states, cache=cache, conv_mask=conv_mask, cache_layer_idx=cache_layer_idx
        )

        residual = hidden_states = hidden_states.float()
        if self.is_block_boundary:
            block_states = torch.cat([block_states, block_residual.unsqueeze(-2)], dim=-2)
        else:
            residual = hidden_states = block_residual.float() + hidden_states

        hidden_states = self.mlp_input(block_states, hidden_states)[0]
        if messages is not None:
            hidden_states = self.mlp_controller(hidden_states, messages[..., 3, :])
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states).to(dtype))
        hidden_states = self.post_mlp_convolution(
            hidden_states, cache=cache, conv_mask=conv_mask, cache_layer_idx=cache_layer_idx
        )
        return (residual + hidden_states).to(dtype)


class Inkling2LoopTopKRouter(Inkling2TopKRouter):
    pass


@auto_docstring
class Inkling2LoopPreTrainedModel(Inkling2PreTrainedModel):
    _no_split_modules = [
        "Inkling2LoopSlidingDecoderLayer",
        "Inkling2LoopFullDecoderLayer",
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
        "cross_pass_messages",
        "gain_proj",
        "forget_proj",
    ]
    _can_record_outputs = {
        # one hidden state per layer visit: the core layers once per pass
        "hidden_states": OutputRecorder(
            (Inkling2LoopSlidingDecoderLayer, Inkling2LoopFullDecoderLayer),
            num_hidden_layers_key="total_num_hidden_layers",
        ),
        "attentions": [Inkling2LoopSlidingAttention, Inkling2LoopFullAttention],
        "router_logits": OutputRecorder(Inkling2LoopTopKRouter, index=0),
    }

    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        std = self.config.get_text_config().initializer_range
        if isinstance(module, Inkling2LoopRelativeLogits):
            init.normal_(module.proj, mean=0.0, std=std)
        elif isinstance(module, Inkling2LoopExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, Inkling2LoopTopKRouter):
            init.normal_(module.weight, mean=0.0, std=std)
            init.zeros_(module.e_score_correction_bias)
        elif isinstance(module, Inkling2LoopScaledL2Clip):
            init.ones_(module.scale)
        elif isinstance(module, Inkling2LoopDynamicShortConvolution):
            init.zeros_(module.static_weight)
        elif isinstance(module, Inkling2LoopBlockAttentionResidual):
            init.zeros_(module.query)
        elif isinstance(module, Inkling2LoopCrossPassMessages):
            init.zeros_(module.queries)
            init.zeros_(module.kv_queries)


@auto_docstring
class Inkling2LoopTextModel(Inkling2TextModel):
    def __init__(self, config: Inkling2LoopTextConfig):
        super().__init__(config)
        self.cross_pass_messages = Inkling2LoopCrossPassMessages(config)

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

        loop_config = self.config.loop_config
        loop_layer_index_start, loop_layer_index_end = loop_config["core_start"], loop_config["core_end"]
        # pre loop layers, then `num_passes` times the loop layers, then post loop layers
        layer_and_loop_indices = [(layer_idx, 0) for layer_idx in range(loop_layer_index_start)]
        layer_and_loop_indices += [
            (layer_idx, loop_idx)
            for loop_idx in range(loop_config["num_passes"])
            for layer_idx in range(loop_layer_index_start, loop_layer_index_end)
        ]
        layer_and_loop_indices += [
            (layer_idx, 0) for layer_idx in range(loop_layer_index_end, self.config.num_hidden_layers)
        ]

        cache_layer_indices = [[] for _ in range(self.config.num_hidden_layers)]
        for cache_layer_idx, layer_idx in enumerate(self.config.cache_layer_indices):
            cache_layer_indices[layer_idx].append(cache_layer_idx)

        hidden_states = inputs_embeds
        block_states = inputs_embeds.new_zeros(*inputs_embeds.shape[:-1], 0, inputs_embeds.shape[-1])
        for layer_idx, loop_idx in layer_and_loop_indices:
            decoder_layer = self.layers[layer_idx]
            cache_layer_idx = cache_layer_indices[layer_idx][loop_idx]

            if layer_idx == loop_layer_index_start and loop_idx == 0:
                pre_loop_hidden_states, pre_loop_block_states = hidden_states, block_states
            elif layer_idx == loop_layer_index_start:
                num_pre_loop_blocks = pre_loop_block_states.shape[-2] + 1
                message_bank = torch.cat(
                    [block_states[..., num_pre_loop_blocks:, :], hidden_states.unsqueeze(-2)], dim=-2
                )
                hidden_states, block_states = pre_loop_hidden_states, pre_loop_block_states

            messages = None
            if loop_idx > 0:
                messages = self.cross_pass_messages(message_bank, layer_idx - loop_layer_index_start)

            new_hidden_states = decoder_layer(
                hidden_states,
                block_states,
                messages,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                conv_mask=causal_mask_mapping["linear_attention"],
                cache=cache,
                cache_layer_idx=cache_layer_idx,
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
