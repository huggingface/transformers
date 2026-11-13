# Copyright 2026 The HuggingFace Team. All rights reserved.
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
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...masking_utils import (
    create_causal_mask,
    create_recurrent_attention_mask,
    create_sliding_window_causal_mask,
)
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import merge_with_config_defaults, no_inherit_decorator
from ...utils.output_capturing import capture_outputs
from ..lfm2.configuration_lfm2 import Lfm2Config
from ..lfm2.modeling_lfm2 import (
    Lfm2DecoderLayer,
    Lfm2ForCausalLM,
    Lfm2Model,
    Lfm2PreTrainedModel,
    Lfm2RMSNorm,
    Lfm2ShortConv,
)
from ..lfm2_moe.modeling_lfm2_moe import Lfm2MoeMLP
from ..llama.modeling_llama import eager_attention_forward
from ..olmo.modeling_olmo import apply_rotary_pos_emb
from ..olmo3.modeling_olmo3 import Olmo3RotaryEmbedding
from ..qwen3.modeling_qwen3 import Qwen3Attention


@auto_docstring(checkpoint="LiquidAI/LFM3-1.7B")
@strict
class Lfm3Config(Lfm2Config):
    r"""
    conv_bias (`bool`, *optional*, defaults to `False`):
        Whether to use bias in the conv layers.
    conv_L_cache (`int`, *optional*, defaults to 7):
        L_cache dim in the conv layers.
    use_sandwich_norm (`bool`, *optional*, defaults to `True`):
        Whether to apply an RMSNorm to the operator and feed-forward outputs before they enter the residual stream.
    use_attention_gate (`bool`, *optional*, defaults to `True`):
        Whether the attention layers scale each head's output by a sigmoid gate computed from the layer input.

    ```python
    >>> from transformers import Lfm3Model, Lfm3Config

    >>> # Initializing a LFM3 model
    >>> configuration = Lfm3Config()

    >>> # Initializing a model from the LFM3-1.7B style configuration
    >>> model = Lfm3Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "lfm3"
    attribute_map = {"rms_norm_eps": "norm_eps"}
    _rope_type_labels = ("full_attention", "sliding_attention")

    vocab_size: int = 128_000
    hidden_size: int = 2048
    intermediate_size: int = 7680
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 128
    pad_token_id: int | None = 124_893
    bos_token_id: int | None = 124_894
    eos_token_id: int | list[int] | None = 124_900
    rope_parameters: dict | None = None
    conv_L_cache: int = 7
    sliding_window: int | None = 256
    layer_types: list[str] | None = None
    use_sandwich_norm: bool = True
    use_attention_gate: bool = True
    attention_bias: bool = False
    attention_dropout: float = 0.0

    block_multiple_of = AttributeError()
    block_ffn_dim_multiplier = AttributeError()
    block_auto_adjust_ff_dim = AttributeError()
    full_attn_idxs = AttributeError()

    def __post_init__(self, **kwargs):
        if self.rope_parameters is None:
            self.rope_parameters = {
                "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
                "full_attention": {
                    "rope_type": "proportional",
                    "rope_theta": 1_000_000.0,
                    "partial_rotary_factor": 0.5,
                },
            }
        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if i % 4 == 2 else "sliding_attention" if i % 4 == 0 and i > 0 else "conv"
                for i in range(self.num_hidden_layers)
            ]

        PreTrainedConfig.__post_init__(self, **kwargs)


class Lfm3RMSNorm(Lfm2RMSNorm):
    pass


class Lfm3RotaryEmbedding(Olmo3RotaryEmbedding):
    def __init__(self, config: Lfm3Config):
        nn.Module.__init__(self)
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        # diff -> only the attention layer types have a RoPE entry, `conv` layers have none
        self.layer_types = [layer_type for layer_type in config._rope_type_labels if layer_type in config.layer_types]
        self.rope_type = {}
        for layer_type in self.layer_types:
            self.rope_type[layer_type] = config.rope_parameters[layer_type]["rope_type"]
            rope_init_fn: Callable = self.compute_default_rope_parameters
            if self.rope_type[layer_type] != "default":
                rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type[layer_type]]
            curr_inv_freq, curr_attention_scaling = rope_init_fn(config, layer_type=layer_type)
            setattr(self, f"{layer_type}_inv_freq", nn.Buffer(curr_inv_freq, persistent=False))
            setattr(self, f"{layer_type}_original_inv_freq", nn.Buffer(curr_inv_freq.clone(), persistent=False))
            setattr(self, f"{layer_type}_attention_scaling", curr_attention_scaling)


class Lfm3MLP(Lfm2MoeMLP):
    pass


@no_inherit_decorator
class Lfm3Attention(Qwen3Attention):
    """Qwen3 attention with RoPE in float32 and a sigmoid output gate per head."""

    def __init__(self, config: Lfm3Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.g_proj = (
            nn.Linear(config.hidden_size, config.num_attention_heads, bias=False)
            if config.use_attention_gate
            else None
        )

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

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

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
        if self.g_proj is not None:
            attn_output = attn_output * torch.sigmoid(self.g_proj(hidden_states)).unsqueeze(-1)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


# torch only: the hub causal_conv1d kernels serve widths 2-4, LFM3 uses 7
def causal_conv1d_update(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: nn.Parameter,
    bias: nn.Parameter | None = None,
    activation: str | None = None,
):
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]

    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = out[:, :, -seq_len:]
    if activation is not None:
        out = ACT2FN[activation](out)
    return out.to(hidden_states.dtype)


def causal_conv1d_fn(
    hidden_states: torch.Tensor,
    weight: nn.Parameter,
    bias: nn.Parameter | None = None,
    activation: str | None = None,
    **kwargs,
):
    _, hidden_size, seq_len = hidden_states.shape
    padding = weight.shape[-1] - 1

    out = F.conv1d(
        hidden_states.to(weight.dtype),
        weight=weight.unsqueeze(1),
        bias=bias,
        padding=padding,
        groups=hidden_size,
    )[:, :, :seq_len]
    if activation is not None:
        out = ACT2FN[activation](out)
    return out.to(hidden_states.dtype)


@no_inherit_decorator
class Lfm3ShortConv(Lfm2ShortConv):
    pass


class Lfm3DecoderLayer(Lfm2DecoderLayer):
    def __init__(self, config: Lfm3Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.post_operator_norm = (
            Lfm3RMSNorm(config.hidden_size, eps=config.norm_eps) if config.use_sandwich_norm else nn.Identity()
        )
        self.post_ffn_norm = (
            Lfm3RMSNorm(config.hidden_size, eps=config.norm_eps) if config.use_sandwich_norm else nn.Identity()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        if self.is_attention_layer:
            hidden_states, _ = self.self_attn(
                hidden_states=self.operator_norm(hidden_states),
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )
        else:
            hidden_states = self.conv(
                hidden_states=self.operator_norm(hidden_states),
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                **kwargs,
            )
        hidden_states = self.post_operator_norm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = self.post_ffn_norm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Lfm3PreTrainedModel(Lfm2PreTrainedModel):
    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, Lfm3RotaryEmbedding):
            for layer_type in module.layer_types:
                rope_init_fn = module.compute_default_rope_parameters
                if module.rope_type[layer_type] != "default":
                    rope_init_fn = ROPE_INIT_FUNCTIONS[module.rope_type[layer_type]]
                curr_inv_freq, _ = rope_init_fn(module.config, layer_type=layer_type)
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), curr_inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), curr_inv_freq)


class Lfm3Model(Lfm2Model):
    """LFM2 model with a sliding-attention layer type; the mask and rotary embedding are built per layer type."""

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
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            mask_creation_functions = {
                "full_attention": lambda: create_causal_mask(**mask_kwargs),
                "sliding_attention": lambda: create_sliding_window_causal_mask(**mask_kwargs),
                "conv": lambda: create_recurrent_attention_mask(**mask_kwargs),
            }
            causal_mask_mapping = {}
            for layer_type in set(self.config.layer_types):
                causal_mask_mapping[layer_type] = mask_creation_functions[layer_type]()

        hidden_states = inputs_embeds
        position_embeddings = {
            layer_type: self.rotary_emb(hidden_states, position_ids=position_ids, layer_type=layer_type)
            for layer_type in self.rotary_emb.layer_types
        }

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            layer_type = self.config.layer_types[i]
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[layer_type],
                position_embeddings=position_embeddings.get(layer_type),
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )

        hidden_states = self.embedding_norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class Lfm3ForCausalLM(Lfm2ForCausalLM):
    def forward(self, **super_kwargs) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, Lfm3ForCausalLM

        >>> model = Lfm3ForCausalLM.from_pretrained("LiquidAI/LFM3-1.7B")
        >>> tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM3-1.7B")

        >>> inputs = tokenizer.apply_chat_template(
        ...     [{"role": "user", "content": "What is C. elegans?"}], add_generation_prompt=True, return_tensors="pt"
        ... )
        >>> generate_ids = model.generate(**inputs, max_new_tokens=64)
        >>> tokenizer.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        ```"""
        return super().forward(**super_kwargs)


__all__ = ["Lfm3Config", "Lfm3ForCausalLM", "Lfm3Model", "Lfm3PreTrainedModel"]
