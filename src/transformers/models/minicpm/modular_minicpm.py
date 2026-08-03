# Copyright 2026 The OpenBMB Team and the HuggingFace Inc. team. All rights reserved.
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
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...cache_utils import Cache
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_rope_utils import dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import maybe_autocast
from ..gemma3.modeling_gemma3 import Gemma3TextScaledWordEmbedding
from ..llama.configuration_llama import LlamaConfig
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaForSequenceClassification,
    LlamaMLP,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    eager_attention_forward,
    rotate_half,
)


@auto_docstring(checkpoint="openbmb/MiniCPM4-8B")
@strict
class MiniCPMConfig(LlamaConfig):
    r"""
    scale_emb (`int` or `float`, *optional*, defaults to 12):
        Multiplier applied to input embeddings.
    scale_depth (`int` or `float`, *optional*, defaults to 1.4):
        Multiplier for residual connections. The effective scale is
        `scale_depth / sqrt(num_hidden_layers)`.
    dim_model_base (`int`, *optional*, defaults to 256):
        Base model dimension used to scale hidden states before the language model head.
    mup_denominator (`int`, *optional*):
        Width denominator used by compatible speculative decoding heads.
    sparse_config (`dict`, *optional*):
        Configuration for OpenBMB's optional InfLLM-v2 sparse attention implementation. Native Transformers support
        is currently limited to dense attention and raises an error if this is set.

    Example:

    ```python
    >>> from transformers import MiniCPMConfig, MiniCPMModel

    >>> configuration = MiniCPMConfig()
    >>> model = MiniCPMModel(configuration)
    >>> configuration = model.config
    ```
    """

    model_type = "minicpm"

    # Architecture dimensions match MiniCPM4-8B. Compatibility fields omitted by MiniCPM4-0.5B keep their official
    # constructor defaults.
    vocab_size: int = 73448
    hidden_size: int = 4096
    intermediate_size: int = 16384
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int | None = 2
    max_position_embeddings: int = 32768
    initializer_range: float = 0.1
    rms_norm_eps: float = 1e-6
    pad_token_id: int | None = None
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    tie_word_embeddings: bool = True
    scale_emb: int | float = 12
    scale_depth: int | float | None = 1.4
    dim_model_base: int | None = 256
    mup_denominator: int | None = None
    sparse_config: dict | None = None

    def __post_init__(self, **kwargs):
        if self.scale_depth is None:
            self.scale_depth = math.sqrt(self.num_hidden_layers)
        if self.dim_model_base is None:
            self.dim_model_base = self.hidden_size
        super().__post_init__(**kwargs)

    @property
    def logits_scaling(self) -> float:
        return self.hidden_size / self.dim_model_base


class MiniCPMScaledWordEmbedding(Gemma3TextScaledWordEmbedding):
    pass


class MiniCPMRMSNorm(LlamaRMSNorm):
    pass


class MiniCPMRotaryEmbedding(LlamaRotaryEmbedding):
    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor) -> tuple[torch.Tensor, torch.Tensor]:
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):
            if self.rope_type == "longrope":
                rope_parameters = self.config.rope_parameters
                head_dim = getattr(self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads)
                dim = int(head_dim * rope_parameters.get("partial_rotary_factor", 1.0))
                factor_name = (
                    "long_factor"
                    if position_ids.max().item() + 1 > rope_parameters["original_max_position_embeddings"]
                    else "short_factor"
                )
                ext_factors = torch.tensor(rope_parameters[factor_name], dtype=torch.float32, device=x.device)
                inv_freq_shape = torch.arange(0, dim, 2, dtype=torch.int64, device=x.device).float() / dim
                base_inv_freq = 1.0 / (rope_parameters["rope_theta"] ** inv_freq_shape)
                freqs = position_ids.float().unsqueeze(-1) * (1.0 / ext_factors)
                freqs = freqs * base_inv_freq
            else:
                inv_freq_expanded = self.inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1).float()
                position_ids_expanded = position_ids[:, None, :].float()
                freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos, sin


class MiniCPMMLP(LlamaMLP):
    pass


def apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings in float32 and restore the query and key dtypes."""
    query_dtype, key_dtype = query.dtype, key.dtype
    query, key = query.float(), key.float()
    cos, sin = cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)
    query = (query * cos) + (rotate_half(query) * sin)
    key = (key * cos) + (rotate_half(key) * sin)
    return query.to(query_dtype), key.to(key_dtype)


class MiniCPMAttention(LlamaAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
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
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class MiniCPMDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: MiniCPMConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.residual_scale = config.scale_depth / math.sqrt(config.num_hidden_layers)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states * self.residual_scale

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states * self.residual_scale
        return hidden_states


class MiniCPMPreTrainedModel(LlamaPreTrainedModel):
    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, MiniCPMScaledWordEmbedding):
            init.constant_(module.embed_scale, module.scalar_embed_scale)


@auto_docstring
class MiniCPMModel(LlamaModel):
    def __init__(self, config: MiniCPMConfig):
        if config.sparse_config is not None:
            raise NotImplementedError(
                "MiniCPM InfLLM-v2 sparse attention is not implemented in Transformers. Remove `sparse_config` to "
                "use dense attention."
            )
        super().__init__(config)
        self.embed_tokens = MiniCPMScaledWordEmbedding(
            config.vocab_size, config.hidden_size, self.padding_idx, embed_scale=config.scale_emb
        )


@auto_docstring
class MiniCPMForCausalLM(LlamaForCausalLM):
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, MiniCPMForCausalLM

        >>> model = MiniCPMForCausalLM.from_pretrained("openbmb/MiniCPM4-0.5B")
        >>> tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM4-0.5B")

        >>> prompt = "The capital of France is"
        >>> inputs = tokenizer(prompt, return_tensors="pt")
        >>> generate_ids = model.generate(**inputs, max_new_tokens=10)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
        ```
        """
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state / self.config.logits_scaling
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :]).float()

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MiniCPMForSequenceClassification(LlamaForSequenceClassification):
    pass


__all__ = [
    "MiniCPMConfig",
    "MiniCPMPreTrainedModel",
    "MiniCPMModel",
    "MiniCPMForCausalLM",
    "MiniCPMForSequenceClassification",
]
