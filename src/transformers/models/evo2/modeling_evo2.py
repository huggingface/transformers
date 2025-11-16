"""PyTorch Evo2 model."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from ...cache_utils import Cache, DynamicCache
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from ...generation import GenerationMixin
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_evo2 import Evo2Config


logger = logging.get_logger(__name__)

__all__ = ["Evo2Model", "Evo2ForCausalLM", "Evo2PreTrainedModel"]


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Evo2
class Evo2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """Evo2RMSNorm is equivalent to T5LayerNorm."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    del position_ids
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Evo2RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: Evo2Config, device=None):
        super().__init__()
        self.max_seq_len_cached = getattr(config, "max_position_embeddings", 2048)
        self.original_max_seq_len = self.max_seq_len_cached

        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = inv_freq

    @staticmethod
    def compute_default_rope_parameters(
        config: Optional[Evo2Config] = None,
        device: Optional[torch.device] = None,
        seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, float]:
        del seq_len
        rope_params = getattr(config, "rope_parameters", None)
        base = rope_params.get("rope_theta") if rope_params is not None else config.rope_theta
        dim = config.head_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim))
        return inv_freq, 1.0

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Evo2ParallelGatedMLP(nn.Module):
    def __init__(self, config: Evo2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.mlp_dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gated = F.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        hidden_states = self.down_proj(gated * up)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class Evo2Attention(nn.Module):
    def __init__(self, config: Evo2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_head_dim = self.hidden_size // self.num_key_value_heads
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.kv_head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.kv_head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.attn_dropout)

        self.rotary_emb = Evo2RotaryEmbedding(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.kv_head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(
            bsz, q_len, self.num_key_value_heads, self.kv_head_dim
        ).transpose(1, 2)

        cos, sin = self.rotary_emb(query_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        kv_seq_len = key_states.shape[-2]

        if self.num_key_value_heads != self.num_heads:
            key_states = repeat_kv(key_states, self.num_heads // self.num_key_value_heads)
            value_states = repeat_kv(value_states, self.num_heads // self.num_key_value_heads)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        present = past_key_value if use_cache else None
        return attn_output, (attn_weights if output_attentions else None), present


class Evo2HyenaFilter(nn.Module):
    def __init__(self, config: Evo2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.order = config.hyena_order
        self.filter_channels = config.hyena_filters
        self.kernel_size = config.hyena_kernel_size

        self.in_proj = nn.Linear(self.hidden_size, self.filter_channels * self.order, bias=False)
        self.conv = nn.Conv1d(
            in_channels=self.filter_channels,
            out_channels=self.filter_channels,
            kernel_size=self.kernel_size,
            groups=self.filter_channels,
            padding=self.kernel_size - 1,
        )
        self.out_proj = nn.Linear(self.filter_channels * self.order, self.hidden_size, bias=False)
        self.activation = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        projected = self.in_proj(hidden_states)
        projected = projected.view(batch, seq_len, self.order, self.filter_channels).permute(0, 2, 3, 1)
        conv_input = projected.reshape(batch * self.order, self.filter_channels, seq_len)
        conv_output = self.conv(conv_input)
        conv_output = conv_output[:, :, :seq_len]
        conv_output = conv_output.view(batch, self.order, self.filter_channels, seq_len).permute(0, 3, 1, 2)
        conv_output = conv_output.reshape(batch, seq_len, self.order * self.filter_channels)
        conv_output = self.activation(conv_output)
        return self.out_proj(conv_output)


class Evo2AttentionBlock(nn.Module):
    def __init__(self, config: Evo2Config, layer_idx: int):
        super().__init__()
        self.attention = Evo2Attention(config, layer_idx)
        self.input_layernorm = Evo2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Evo2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Evo2ParallelGatedMLP(config)
        self.hidden_dropout = nn.Dropout(config.hidden_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        past_key_value: Optional[Cache],
        output_attentions: bool,
        use_cache: bool,
        cache_position: Optional[torch.LongTensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, attn_weights, present_kv = self.attention(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = residual + self.hidden_dropout(attn_output)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.hidden_dropout(hidden_states)

        return hidden_states, attn_weights, present_kv


class Evo2HyenaBlock(nn.Module):
    def __init__(self, config: Evo2Config):
        super().__init__()
        self.input_layernorm = Evo2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.filter = Evo2HyenaFilter(config)
        self.post_attention_layernorm = Evo2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Evo2ParallelGatedMLP(config)
        self.hidden_dropout = nn.Dropout(config.hidden_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        past_key_value: Optional[Cache],
        output_attentions: bool,
        use_cache: bool,
        cache_position: Optional[torch.LongTensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        del attention_mask, past_key_value, output_attentions, use_cache, cache_position, position_ids
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.filter(hidden_states)
        hidden_states = residual + self.hidden_dropout(hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.hidden_dropout(hidden_states)

        return hidden_states, None, None


class Evo2DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Evo2Config, layer_type: str, layer_idx: int):
        super().__init__()
        self.layer_type = layer_type
        if layer_type == "attention":
            self.block = Evo2AttentionBlock(config, layer_idx)
        else:
            self.block = Evo2HyenaBlock(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        past_key_value: Optional[Cache],
        output_attentions: bool,
        use_cache: bool,
        cache_position: Optional[torch.LongTensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        return self.block(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            cache_position,
        )


class Evo2PreTrainedModel(PreTrainedModel):
    config_class = Evo2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Evo2DecoderLayer"]

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, Evo2RMSNorm):
            module.weight.data.fill_(1.0)



class Evo2Model(Evo2PreTrainedModel):
    def __init__(self, config: Evo2Config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Evo2DecoderLayer(config, layer_type, layer_idx) for layer_idx, layer_type in enumerate(config.layer_types)]
        )
        self.norm = Evo2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.gradient_checkpointing = False

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        output_attentions = (
            output_attentions if output_attentions is not None else getattr(self.config, "output_attentions", False)
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else getattr(self.config, "output_hidden_states", False)
        )
        return_dict = return_dict if return_dict is not None else True
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You must specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache(config=self.config)
            elif not isinstance(past_key_values, Cache):
                raise TypeError("`past_key_values` must be a `Cache` when `use_cache` is True.")
        else:
            past_key_values = None

        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0

        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_length,
        )

        hidden_states = self.dropout(inputs_embeds)

        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + seq_length, device=hidden_states.device)
        else:
            cache_position = cache_position.to(hidden_states.device)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for layer_idx, (decoder_layer, layer_type) in enumerate(zip(self.layers, self.config.layer_types)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_past = past_key_values if layer_type == "attention" else None

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states, attn, present_kv = checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    layer_past,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                hidden_states, attn, present_kv = decoder_layer(
                    hidden_states,
                    attention_mask,
                    position_ids,
                    layer_past,
                    output_attentions,
                    use_cache,
                    cache_position,
                )

            if layer_type == "attention" and present_kv is not None and use_cache:
                past_key_values = present_kv

            if output_attentions:
                all_attentions = all_attentions + (attn,)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            outputs = (hidden_states, past_key_values)
            if output_hidden_states:
                outputs += (all_hidden_states,)
            if output_attentions:
                outputs += (all_attentions,)
            return outputs

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class Evo2ForCausalLM(Evo2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: Evo2Config):
        super().__init__(config)
        self.model = Evo2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        return_dict = return_dict if return_dict is not None else True

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        if isinstance(logits_to_keep, int):
            slice_indices = slice(-logits_to_keep, None) if logits_to_keep > 0 else slice(None)
            logits = self.lm_head(hidden_states[:, slice_indices, :])
        else:
            logits = self.lm_head(hidden_states[:, logits_to_keep, :])

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _reorder_cache(self, past_key_values: Cache, beam_idx: torch.LongTensor) -> Cache:
        past_key_values.reorder_cache(beam_idx)
        return past_key_values
