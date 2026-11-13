# Copyright 2026 Ricursive Intelligence and the HuggingFace Inc. team. All rights reserved.
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
import torch.nn as nn
from huggingface_hub.dataclasses import strict
from torch.nn import init

from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..gemma3.modeling_gemma3 import Gemma3DecoderLayer
from ..glm5_next.modeling_glm5_next import Glm5NextTextMLP
from ..hunyuan_vl.modeling_hunyuan_vl import apply_rotary_pos_emb
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    eager_attention_forward,
)
from ..olmo3.configuration_olmo3 import Olmo3Config
from ..olmo3.modeling_olmo3 import Olmo3ForCausalLM, Olmo3Model


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="ricursiveAI/Ripple-65B-Preview")
@strict
class RippleConfig(Olmo3Config):
    r"""
    sliding_window (`int`, *optional*, defaults to 1024):
        Window size of the sliding-attention layers. A query attends to the `sliding_window` most recent keys,
        itself included.
    layer_types (`list[str]`, *optional*):
        Per-layer attention type, `"sliding_attention"` (window + RoPE) or `"full_attention"` (global, no positional
        encoding). Defaults to one full-attention layer every 5 layers (indices 4, 9, 14, ...).
    embed_scale (`float`, *optional*, defaults to 71.55417527999327):
        Multiplier applied to the token embeddings before the first layer, `sqrt(hidden_size)` for the released
        checkpoints.
    swiglu_limit (`float`, *optional*, defaults to 20.0):
        Activation clipping in the MLP: `gate_proj` output is clamped to `max=swiglu_limit` and `up_proj` output to
        `[-swiglu_limit, swiglu_limit]` before the SwiGLU product.

    ```python
    >>> from transformers import RippleModel, RippleConfig

    >>> # Initializing a Ripple-65B style configuration
    >>> configuration = RippleConfig()

    >>> # Initializing a model from the Ripple-65B style configuration
    >>> model = RippleModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "ripple"
    default_theta = 10_000.0
    # Reference (`ricursive`) config key -> library name.
    attribute_map = {"ffn_clamp_threshold": "swiglu_limit"}
    # q/k norms are per head, so heads can be sharded column-wise (unlike Olmo3's flat q/k norm).
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.g_proj": "colwise",
        "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    vocab_size: int = 100352
    hidden_size: int = 5120
    intermediate_size: int = 40960
    num_hidden_layers: int = 80
    num_attention_heads: int = 80
    num_key_value_heads: int | None = 8
    head_dim: int = 128
    max_position_embeddings: int = 2097152
    pad_token_id: int | None = 100277
    bos_token_id: int | None = 100257
    eos_token_id: int | list[int] | None = None
    rms_norm_eps: float = 1e-6
    sliding_window: int = 1024
    embed_scale: float = 71.55417527999327  # sqrt(5120)
    swiglu_limit: float = 20.0

    def __post_init__(self, **kwargs):
        if self.eos_token_id is None:
            self.eos_token_id = [100265, 100257]
        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if i % 5 == 4 else "sliding_attention" for i in range(self.num_hidden_layers)
            ]
        PreTrainedConfig.__post_init__(self, **kwargs)

    def convert_rope_params_to_dict(self, **kwargs):
        raise AttributeError()


class RippleRMSNorm(LlamaRMSNorm):
    pass


class RippleMLP(Glm5NextTextMLP):
    pass


class RippleRotaryEmbedding(LlamaRotaryEmbedding):
    pass


class RippleAttention(LlamaAttention):
    """LlamaAttention with:
    - per head Q and K norms applied after the projection
    - some layers are sliding and RoPE is only applied to sliding window layers
    - value residuals: the input of the V projection is a parameterized interpolation between the scaled embeddings
        and the hidden states
    - sigmoid gating before output projection
    """

    def __init__(self, config: RippleConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.layer_type = config.layer_types[layer_idx]
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None
        # Per-head Q/K norms (RMSNorm on head_dim)
        self.q_norm = RippleRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RippleRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        # Learned barycenter for value residual
        self.value_res_lambda = nn.Parameter(torch.zeros(1))
        # Gate projection
        self.g_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        value_residual: torch.Tensor,
        position_embeddings: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Queries and keys are normalized after the projection
        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states).transpose(1, 2)

        key_states = self.k_proj(hidden_states).view(hidden_shape)
        key_states = self.k_norm(key_states).transpose(1, 2)

        # Value states are projected from an interpolation of value residuals and hidden states
        blend = self.value_res_lambda.sigmoid().to(hidden_states.dtype)
        value_states = self.v_proj(torch.lerp(value_residual, hidden_states, blend))
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        # RoPE only on sliding layers
        if self.layer_type == "sliding_attention":
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
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()

        # Apply sigmoid gate before output projection
        gate = self.g_proj(hidden_states).float().sigmoid()
        attn_output = attn_output * gate.to(attn_output.dtype)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class RippleDecoderLayer(Gemma3DecoderLayer):
    def __init__(self, config: RippleConfig, layer_idx: int):
        super().__init__(config, layer_idx)


class RipplePreTrainedModel(LlamaPreTrainedModel):
    """LlamaPreTrainedModel plus zero-init of `value_res_lambda` (the generic init already refills the RoPE table)."""

    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, RippleAttention):
            init.zeros_(module.value_res_lambda)


class RippleModel(Olmo3Model):
    """
    Olmo3Model with `self.embed_scale = config.embed_scale` (python float, not a bf16 buffer) and a forward that:
    - scales the input embeddings by it and keeps the scaled embeddings as `value_residual` for every layer;
    - computes `position_embeddings` once with the flat RoPE and passes it to every layer (full-attention layers skip
      it in the attention, NoPE).
    """

    def __init__(self, config: RippleConfig):
        super().__init__(config)
        self.embed_scale = config.embed_scale

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        # Scale input embeddings and keep them as the values residual for all downstream layers
        inputs_embeds = inputs_embeds * self.embed_scale
        value_residual = inputs_embeds

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                value_residual=value_residual,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_ids=position_ids,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class RippleForCausalLM(Olmo3ForCausalLM):
    pass


__all__ = ["RippleConfig", "RippleForCausalLM", "RippleModel", "RipplePreTrainedModel"]
