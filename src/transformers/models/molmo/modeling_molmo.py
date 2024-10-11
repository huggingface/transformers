# coding=utf-8
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
from dataclasses import replace
from typing import List, Optional, Tuple, Union, Dict, Any

import torch
from transformers import PreTrainedModel, add_start_docstrings, GenerationMixin
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_outputs import (
    CausalLMOutputWithPast, ModelOutput, BaseModelOutputWithPast)
from transformers.models.auto import AutoModelForCausalLM
from torch import nn
from transformers.utils import logging, is_flash_attn_greater_or_equal_2_10, \
    add_start_docstrings_to_model_forward, replace_return_docstrings

from .config_molmo import MolmoConfig, MolmoVisionConfig
from torch.nn import functional as F, CrossEntropyLoss

logger = logging.get_logger(__name__)


_CONFIG_FOR_DOC = "MolmoConfig"


MOLMO_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MolmoConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


# Copied from transformers.models.llama.modeling_llama.LlamaPreTrainedModel with Llama->Olmo
class MolmoPreTrainedModel(PreTrainedModel):
    config_class = MolmoConfig
    base_model_prefix = "model"
    _no_split_modules = ["MolmoDecoderLayer", "MolmoeDecoderLayer", "MolmoVisionBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    supports_gradient_checkpointing = True
    _supports_cache_class = True
    _supports_static_cache = False

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear,)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MolmoRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # For BC we register cos and sin cached
        self.max_seq_len_cached = max_position_embeddings

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class MolmoAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # copied from transformers.models.llama.modeling_llama.LlamaAttention.__init__ with Llama->Olmo
    def __init__(self, config: MolmoConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.config.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.rotary_emb = MolmoRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
        self.k_norm: Optional[nn.Module] = None
        self.q_norm: Optional[nn.Module] = None
        if config.qk_layer_norm:
            if config.num_key_value_heads is None:
                config.num_key_value_heads = config.num_attention_heads
            self.q_norm = MolmoRmsLayerNorm(
                config,
                size=config.hidden_size,
                eps=config.layer_norm_eps
            )
            self.k_norm = MolmoRmsLayerNorm(
                config,
                size=config.hidden_size,
                eps=config.layer_norm_eps
            )

        # Attention output projection.
        input_dim = config.hidden_size
        head_dim = config.hidden_size // config.num_attention_heads
        self.fused_dims = (
            config.hidden_size,
            config.num_key_value_heads * head_dim,
            config.num_key_value_heads * head_dim,
        )
        self.att_proj = nn.Linear(
            config.hidden_size, sum(self.fused_dims),
            bias=config.qkv_bias,
        )
        self.attn_out = nn.Linear(
            input_dim, config.hidden_size,
            bias=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        qkv = self.att_proj(hidden_states)
        query_states, key_states, value_states = qkv.split(self.fused_dims, dim=-1)

        if self.q_norm is not None and self.k_norm is not None:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # query_states, k = self.rotary_emb(query_states, key_states, position_ids=position_ids)
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.config.hidden_size)

        attn_output = self.attn_out(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class MolmoFlashAttention2(MolmoAttention):
    """
    Molmo flash attention module. This module inherits from `MolmoAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        qkv = self.att_proj(hidden_states)
        query_states, key_states, value_states = qkv.split(self.fused_dims, dim=-1)

        if self.q_norm is not None and self.k_norm is not None:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (OlmoRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=dropout_rate,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class MolmoSdpaAttention(MolmoAttention):
    """
    OLMo attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `OlmoAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from OlmoAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "OlmoModel is using OlmoSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        bsz, q_len, _ = hidden_states.size()

        qkv = self.att_proj(hidden_states)
        query_states, key_states, value_states = qkv.split(self.fused_dims, dim=-1)

        if self.q_norm is not None and self.k_norm is not None:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        # if attention_mask is not None and cache_position is not None:
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.attn_out(attn_output)

        return attn_output, None, past_key_value


MOLMO_ATTENTION_CLASSES = {
    "eager": MolmoAttention,
    "sdpa": MolmoSdpaAttention,
    "flash_attention_2": MolmoFlashAttention2,
}


class MolmoMlp(nn.Module):
    def __init__(self, input_dim, hidden_size, activation_fn, include_bias=False):
        super().__init__()
        self.ff_proj = nn.Linear(input_dim, hidden_size, bias=include_bias)
        self.ff_out = nn.Linear(hidden_size//2, input_dim, bias=include_bias)
        self.act = ACT2FN[activation_fn]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        x = self.ff_proj(x)
        x, gate = x.chunk(2, dim=-1)
        x = self.act(gate) * x
        x = self.ff_out(x)
        return x


class MolmoDecoderLayer(nn.Module):
    def __init__(self, config: MolmoConfig, layer_idx=None, device=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.intermediate_size
        self.dropout = nn.Dropout(config.residual_dropout)
        self.attn = MOLMO_ATTENTION_CLASSES[self.config._attn_implementation](config, layer_idx)
        self.attn_norm = MolmoRmsLayerNorm(config, size=config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = MolmoMlp(config.hidden_size, config.intermediate_size, config.activation_type)
        self.ff_norm = MolmoRmsLayerNorm(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        cache_position=None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        if not self.config.norm_after:
            atten_in = self.attn_norm(x)
        else:
            atten_in = x

        att, self_attn_weights, present_key_value = self.attn(
            atten_in,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        if self.config.norm_after:
            att = self.attn_norm(att)

        x = x + self.dropout(att)

        og_x = x

        if not self.config.norm_after:
            x = self.ff_norm(x)

        x = self.mlp(x)

        if self.config.norm_after:
            x = self.ff_norm(x)

        x = self.dropout(x)
        x = og_x + x

        outputs = (x,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class MolmoeMlp(nn.Module):
    def __init__(self, input_dim, hidden_size, activation):
        super().__init__()
        self.gate_proj = nn.Linear(input_dim, hidden_size, bias=False)
        self.up_proj = nn.Linear(input_dim, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, input_dim, bias=False)
        self.act_fn = ACT2FN[activation]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MolmoeMlpExpert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.moe_num_experts
        self.top_k = config.moe_top_k
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([MolmoeMlp(config.hidden_size, config.intermediate_size // 2, config.activation_type)
                                      for _ in range(self.num_experts)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states = self.ff_norm(hidden_states)
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be selected
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


class MolmoeDecoderLayer(nn.Module):
    """Decoder for MolmoE with MoE MLP layers

    Note this layers does not expose the router logits, so it is suitable for inference but no
    for training.
    """

    def __init__(self, config: MolmoConfig, layer_idx=None):
        super().__init__()
        self.config = config
        self.attn = MOLMO_ATTENTION_CLASSES[self.config._attn_implementation](config, layer_idx)
        self.attn_norm = MolmoRmsLayerNorm(config, size=config.hidden_size, eps=config.layer_norm_eps)
        assert config.moe_num_experts > 0
        self.ff_norm = MolmoRmsLayerNorm(config, size=config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = MolmoeMlpExpert(config)
        self.hidden_size = config.intermediate_size
        self.dropout = nn.Dropout(config.residual_dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        cache_position=None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        if not self.config.norm_after:
            atten_in = self.attn_norm(x)
        else:
            atten_in = x

        att, cache = self.attn(
            atten_in,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position
        )

        if self.config.norm_after:
            att = self.attn_norm(att)

        x = x + self.dropout(att)
        og_x = x

        if not self.config.norm_after:
            x = self.ff_norm(x)

        x, _ = self.mlp(x)

        if self.config.norm_after:
            x = self.ff_norm(x)

        x = self.dropout(x)
        x = og_x + x
        return x, cache


class MolmoEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        num_new_embeddings: int,
        features: int,
        device: Union[str, torch.device] = None,
        initializer_range: float = 0.02,
        new_embed_initializer_range: float = 0.02,
    ):
        super().__init__()
        self.initializer_range = initializer_range
        self.new_embed_initializer_range = new_embed_initializer_range
        self.embedding = nn.Parameter(
            torch.zeros(num_embeddings, features, device=device),
        )
        # We keep the special token embedding separate from the embedding from the LM so we can
        # put a separate learning rate of them during training
        self.new_embedding = nn.Parameter(torch.zeros(num_new_embeddings, features, device=device))

    def reset_parameters(self):
        nn.init.normal_(self.embedding, std=self.initializer_range)
        nn.init.normal_(self.new_embedding, std=self.new_embed_initializer_range)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.embedding(x, torch.cat([self.embedding, self.new_embedding], dim=0))


class VisionMlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, hidden_act: str, device=None):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=True, device=device)
        self.act = ACT2FN[hidden_act]
        self.w2 = nn.Linear(hidden_dim, dim, bias=True, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act(self.w1(x)))


class MolmoVisionBlock(nn.Module):

    def __init__(self, config: MolmoVisionConfig, attention_type, device=None):
        super().__init__()
        self.attention = VisionAttention(config, device=device, attention_type=attention_type)
        self.feed_forward = VisionMlp(
            config.image_emb_dim, config.image_mlp_dim, config.image_mlp_activations, device)
        self.attention_norm = nn.LayerNorm(
            config.image_emb_dim,
            eps=config.image_norm_eps,
            device=device,
        )
        self.ffn_norm = nn.LayerNorm(
            config.image_emb_dim,
            eps=config.image_norm_eps,
            device=device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class VisionTransformer(nn.Module):

    def __init__(self, config: MolmoVisionConfig, attention_type, device=None):
        super().__init__()
        self.config = config

        # class embeddings and positional embeddings
        self.scale = config.image_emb_dim ** -0.5
        self.class_embedding = nn.Parameter(
            torch.zeros(config.image_emb_dim, device=device))
        self.positional_embedding = nn.Parameter(
            torch.zeros(config.image_num_pos, config.image_emb_dim, device=device))

        image_patch_size = config.image_patch_size
        self.patch_embedding = nn.Linear(
            image_patch_size * image_patch_size * 3,
            config.image_emb_dim,
            bias=False,
            device=device
        )

        self.pre_ln = nn.LayerNorm(config.image_emb_dim, eps=config.image_norm_eps)
        self.layers = nn.ModuleList([
            MolmoVisionBlock(config, attention_type=attention_type, device=device)
            for _ in range(config.image_num_layers)
        ])

    def add_pos_emb(self, x: torch.Tensor, patch_num: int) -> torch.Tensor:
        cls_emb = self.positional_embedding[0:1]
        pos_emb = self.positional_embedding[1:]

        pos_emb = pos_emb.reshape(
            (int(math.sqrt(pos_emb.shape[0])), int(math.sqrt(pos_emb.shape[0])), pos_emb.shape[1])
        )

        (patch_num_0, patch_num_1) = patch_num

        if pos_emb.shape[0] != patch_num_0 or pos_emb.shape[1] != patch_num_1:
            # Dervied from https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
            # antialias: default True in jax.image.resize
            pos_emb = pos_emb.unsqueeze(0).permute(0, 3, 1, 2)
            pos_emb = F.interpolate(
                pos_emb, size=(patch_num_0, patch_num_1), mode="bicubic", align_corners=False, antialias=True,
            )
            pos_emb = pos_emb.permute(0, 2, 3, 1).squeeze(0)

        pos_emb = pos_emb.reshape(-1, pos_emb.shape[-1])
        x = x + torch.cat([cls_emb[None, :, :], pos_emb[None, :, :]], dim=1).to(x.dtype)
        return x

    def forward(self, x: torch.Tensor, patch_num: int = None) -> List[torch.Tensor]:
        if patch_num is None:
            patch_num = self.config.image_num_patch
        B, N, D = x.shape

        x = self.patch_embedding(x)

        # class embeddings and positional embeddings
        x = torch.cat([
            self.class_embedding.view(1,  1, -1).expand(x.shape[0], -1, -1).to(x.dtype),
            x
        ], dim=1)
        x = self.add_pos_emb(x, patch_num)

        x = self.pre_ln(x)

        hidden_states = []
        for r in self.layers:
            x = r(x)
            hidden_states.append(x)
        return hidden_states


class VisionAttention(nn.Module):
    def __init__(self, config: MolmoVisionConfig, use_bias: bool =True,
                 embed_dim: int=None, device=None, attention_type: str="sdpa"):
        super().__init__()
        self.config = config
        self.embed_dim = config.image_emb_dim
        self.num_heads = config.image_num_heads
        self.head_dim = config.image_head_dim
        self.num_key_value_heads = config.image_num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.initializer_range = config.initializer_range
        self.attention_type = attention_type

        embed_dim = embed_dim if embed_dim else config.image_emb_dim

        self.wq = nn.Linear(
            embed_dim,
            self.num_heads * self.head_dim,
            bias=use_bias,
            device=device,
        )
        self.wk = nn.Linear(
            embed_dim,
            self.num_key_value_heads * self.head_dim,
            bias=use_bias,
            device=device,
        )
        self.wv = nn.Linear(
            embed_dim,
            self.num_key_value_heads * self.head_dim,
            bias=use_bias,
            device=device,
        )
        self.wo = nn.Linear(
            self.num_heads * self.head_dim,
            self.embed_dim,
            bias=use_bias,
            device=device,
        )
        self.residual_dropout = nn.Dropout(config.residual_dropout)

    def _split_heads(self, hidden_states, num_heads) -> torch.Tensor:
        return hidden_states.reshape(hidden_states.shape[:2] + (num_heads, self.head_dim))

    def _merge_heads(self, hidden_states) -> torch.Tensor:
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    def forward(self, inputs_q: torch.Tensor, inputs_kv: Optional[torch.Tensor] = None) -> torch.Tensor:
        if inputs_kv is not None:
            inputs_k = inputs_kv
            inputs_v = inputs_kv
        else:
            inputs_k = inputs_q
            inputs_v = inputs_q

        xq, xk, xv = self.wq(inputs_q), self.wk(inputs_k), self.wv(inputs_v)

        xq = self._split_heads(xq, self.num_heads)
        xk = self._split_heads(xk, self.num_key_value_heads)
        xv = self._split_heads(xv, self.num_key_value_heads)

        if self.num_heads != self.num_key_value_heads:
            xk = xk.repeat_interleave(self.num_key_value_groups, dim=2, output_size=self.num_heads)
            xv = xv.repeat_interleave(self.num_key_value_groups, dim=2, output_size=self.num_heads)

        og_dtype = xq.dtype

        if self.config.float32_attention:
            xq = xq.to(torch.float)
            xk = xk.to(torch.float)

        if self.attention_type == "eager":
            attn_weights = torch.einsum("...qhd,...khd->...hqk", xq / math.sqrt(xq.size(-1)), xk)
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_output = torch.einsum("...hqk,...khd->...qhd", attn_weights.to(xv.dtype), xv)

        elif self.attention_type == "sdpa":
            if self.config.float32_attention and not torch.is_autocast_enabled():
                xv = xv.to(torch.float32)
            attn_output = F.scaled_dot_product_attention(
                xq.transpose(1, 2).contiguous(),
                xk.transpose(1, 2).contiguous(),
                xv.transpose(1, 2).contiguous(),
                is_causal=False,
            ).transpose(1, 2)

        elif self.attention_type == "flash_attention_2":
            assert not self.config.float32_attention
            # Downcast in case we are running with fp32 hidden states
            attn_output = _flash_attention_forward(
                xq.transpose(1, 2).to(torch.bfloat16),
                xk.transpose(1, 2).to(torch.bfloat16),
                xv.transpose(1, 2).to(torch.bfloat16),
                attention_mask=None,
                query_length=inputs_q.shape[1],
                is_causal=False,
            )
        else:
            raise NotImplementedError(self.attention_type)
        attn_output = attn_output.to(og_dtype)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.wo(attn_output)
        attn_output = self.residual_dropout(attn_output)
        return attn_output


class MolmoImageProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim, output_dim,  act_fn="silu", device=None):
        super().__init__()
        self.w1 = nn.Linear(input_dim, hidden_dim, bias=False, device=device)
        self.w2 = nn.Linear(hidden_dim, output_dim, bias=False, device=device)
        self.w3 = nn.Linear(input_dim, hidden_dim, bias=False, device=device)
        self.act_fn = ACT2FN[act_fn]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act_fn(self.w1(x))*self.w3(x))


class MolmoVisionBackbone(nn.Module):
    def __init__(self, config: MolmoConfig):
        super().__init__()
        self.config = config
        self.image_vit = VisionTransformer(config.vision_config, config.attention_type)

        self.image_pooling_2d = VisionAttention(
            config.vision_config,
            embed_dim=len(config.vit_layers)*config.vision_config.image_emb_dim,
            attention_type=config.attention_type
        )

        # `MLP` assume the activation takes two inputs, so it must be a 'llama' version
        if config.activation_type == "swiglu":
            mlp_config = replace(config, activation_type="llama_swiglu")
        elif config.activation_type == "gelu":
            raise NotImplementedError()
        else:
            mlp_config = config

        self.image_projector = MolmoImageProjector(
            config.vision_config.image_emb_dim,
            config.intermediate_size//2,  # //2 since `mlp_hidden_size` includes the gate and parts
            config.hidden_size,
            act_fn=config.activation_type
        )
        self.image_feature_dropout = nn.Dropout(config.image_feature_dropout)
        self.num_prefix_tokens = 1

        self.pad_embed = None
        if config.image_padding_embed:
            image_dim = config.vision_config.image_emb_dim*len(self.config.vit_layers)
            if config.image_padding_embed == "pad_and_partial_pad":
                self.pad_embed = nn.Parameter(torch.zeros((2, image_dim)))
            else:
                raise ValueError(config.image_padding_embed)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        v_cfg = self.config.vision_config
        B, T, N, D = images.shape

        mask = ~torch.all(images.view(B * T, N, D) == -1, dim=(1, 2), keepdim=True)

        # Output all hidden states
        # n_layers x (batch_num_crops, (1+)n_tokens, image_emb_dim)
        images = images.view(B * T, N, D)
        image_features = self.image_vit(images)

        if cfg.vit_layers is not None:
            features = []
            for layer in cfg.vit_layers:
                features.append(image_features[layer])
            image_features = torch.cat(features, dim=-1)
        else:
            image_features = image_features[-1]

        cls_embed: torch.Tensor = None
        if self.num_prefix_tokens > 0:
            cls_embed = image_features[:, 0]
            image_features = image_features[:, 1:]

        image_features = image_features * mask
        image_features = image_features.view(B, T, N, -1)

        cls_embed = cls_embed.view(B, T, -1) if cls_embed is not None else None

        return image_features, cls_embed

    def forward(self, images: torch.Tensor, image_masks: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        cfg = self.config

        # image_features: (batch_size, num_crops(=num_image), num_patch, nximage_emb_dim)
        batch_size, num_image = images.shape[:2]
        image_features, cls_embed = self.encode_image(images)

        if cfg.image_padding_embed:
            assert image_masks is not None
            if cfg.image_padding_embed == "pad_embed":
                all_pad = (image_masks == 0).to(dtype=torch.float32)
                pad_embed = self.pad_embed[None, None, None, :]
                image_features = image_features + pad_embed * torch.unsqueeze(all_pad, -1)
            elif cfg.image_padding_embed == "regress":
                pad_embed = self.pad_embed[None, None, None, :]
                image_features = image_features + pad_embed * torch.unsqueeze(torch.maximum(image_masks, torch.zeros_like(image_masks)), -1)
            elif cfg.image_padding_embed == "pad_and_partial_pad":
                pad_embed = self.pad_embed[:, None, None, None, :]
                all_pad = image_masks == 0
                partial_pad = torch.logical_and(image_masks < 1, torch.logical_not(all_pad)).to(dtype=image_features.dtype)
                all_pad = all_pad.to(dtype=image_features.dtype)
                image_features = image_features + pad_embed[0] * torch.unsqueeze(all_pad, -1)
                image_features = image_features + pad_embed[1] * torch.unsqueeze(partial_pad, -1)
            else:
                raise ValueError(cfg.image_padding_embed)

        image_features = self.image_feature_dropout(image_features)
        if cls_embed is not None:
            cls_embed = self.image_feature_dropout(cls_embed)

        image_features = image_features.reshape(
            (batch_size, num_image) + cfg.image_num_patch + (-1,))

        # transpose to get 2x2 feature squares [n_patches, 4, n_features]
        batch, n_crops, h, w, c = image_features.shape
        image_features = torch.reshape(image_features, [batch*n_crops, h//2, 2, w//2, 2, c])
        image_features = torch.permute(image_features, [0, 1, 3, 2, 4, 5])
        image_features = torch.reshape(image_features, [batch*n_crops*h//2*w//2, 2*2, c])

        query = image_features.mean(-2, keepdim=True)
        image_features = self.image_pooling_2d(query, image_features)

        h = self.config.vision_config.image_num_patch[0]//2
        w = self.config.vision_config.image_num_patch[1]//2
        image_features = image_features.reshape(batch_size, num_image, h * w, -1)

        # MLP layer to map the feature.
        image_features = self.image_projector(image_features)

        # image_features: (batch_size, num_image, num_patch, hidden_size)
        # cls_embed: (batch_size, num_image, hidden_size)
        return image_features, cls_embed


class MolmoRmsLayerNorm(nn.Module):

    def __init__(
        self,
        config: MolmoConfig,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = None,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.config = config
        self.eps = self.config.layer_norm_eps or eps
        self.normalized_shape = (size or config.hidden_size,)
        if elementwise_affine or (elementwise_affine is None):
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            use_bias = self.config.bias_for_layer_norm
            if use_bias:
                self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("bias", None)
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=x.device.type):
            og_dtype = x.dtype
            x = x.to(torch.float32)
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            x = x.to(og_dtype)

        if self.weight is not None:
            if self.bias is not None:
                return self.weight * x + self.bias
            else:
                return self.weight * x
        else:
            return x


MOLMO_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            
            [What are input IDs?](../glossary#input-ids)
        images (`torch.FloatTensor` of shape `(batch_size, n_crops, n_patches, n_features)`, *optional*):
            The input crops in with pixel values between 0 and 1 and normalized with CLIP mean/std
            
            Each crop contains 24x24 patches with 14*14*3 pixel values
        image_masks  (`torch.FloatTensor` of shape `(batch_size, n_crops, n_patches, n_features)`, *optional*):
            Image masks showing what percent of each patch is paddding
        image_input_idx  (`torch.LongTensor` of shape `(batch_size, n_crops)`):
            For each patch, the token index to add its features to, -1 to ignore the patch            
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance, see our
            [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache);
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare Olmo Model outputting raw hidden-states without any specific head on top.",
    MOLMO_START_DOCSTRING,
)
class MolmoModel(MolmoPreTrainedModel):
    def __init__(self, config: MolmoConfig):
        super().__init__(config)
        if self.config.additional_vocab_size is not None:
            self.wte = MolmoEmbedding(
                config.vocab_size,
                config.additional_vocab_size,
                config.hidden_size,
            )
        else:
            self.wte = nn.Embedding(config.vocab_size, config.hidden_size)

        self.emb_drop = nn.Dropout(config.embedding_dropout)
        self.norm = MolmoRmsLayerNorm(config)

        if config.moe_num_experts > 0:
            layers = [MolmoeDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        else:
            layers = [MolmoDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        self.layers = nn.ModuleList(layers)

        self.vision_backbone: Optional[MolmoVisionBackbone] = None
        if config.vision_config is not None:
            self.vision_backbone = MolmoVisionBackbone(config)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def build_input_embeddings(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        images: torch.FloatTensor = None,  # image inputs
        image_masks: Optional[torch.Tensor] = None,
        image_input_idx: torch.LongTensor = None,
    ):
        if input_ids is not None:
            input_ids = input_ids * (input_ids != -1).to(input_ids.dtype)
        x = self.wte(input_ids) if inputs_embeds is None else inputs_embeds  # type: ignore

        num_image: Optional[int] = None
        if images is not None:
            batch_size = x.shape[0]
            # shape: (batch_size, num_image, num_patch, hidden_size)
            # cls_embed: (batch_size, num_image, hidden_size)
            image_features, cls_embed = self.vision_backbone(images, image_masks)
            num_image, num_patch = image_features.shape[1:3]
            assert image_input_idx.shape == (batch_size, num_image, num_patch)

            # insert the image feature into the input embeddings.
            image_features = image_features.view(batch_size, num_image * num_patch, -1)
            image_input_idx = image_input_idx.view(batch_size, num_image * num_patch)

            valid = image_input_idx >= 0
            batch_idx = torch.arange(batch_size, device=x.device)
            batch_idx = torch.tile(batch_idx[:, None], [1, image_features.shape[1]])

            image_features = image_features.to(x.device)
            x[batch_idx[valid], image_input_idx[valid]] += image_features[valid]

        x = self.emb_drop(x)

        # normalized
        if self.config.normalize_input_embeds:
            x = x * (self.config.hidden_size ** 0.5)
        return x

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        images: torch.FloatTensor = None,  # image inputs
        image_masks: Optional[torch.Tensor] = None,
        image_input_idx: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.build_input_embeddings(input_ids, inputs_embeds, images, image_masks, image_input_idx)

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    # Copied from transformers.models.llama.modeling_llama.LlamaModel._update_causal_mask
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    # Copied from transformers.models.llama.modeling_llama.LlamaModel._prepare_4d_causal_attention_mask_with_cache_position
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


# Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM with LLAMA->OLMO,Llama->Olmo
class MolmoForCausalLM(MolmoPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = MolmoModel(config)
        self.vocab_size = config.vocab_size
        if not self.config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(MOLMO_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        images: torch.FloatTensor = None,  # image inputs
        image_masks: Optional[torch.Tensor] = None,
        image_input_idx: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, OlmoForCausalLM

        >>> model = OlmoForCausalLM.from_pretrained("allenai/OLMo-1B-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        'Hey, are you conscious? Can you talk to me?\nI’m not sure if you’re conscious of this, but I’m'
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            images=images,
            image_masks=image_masks,
            image_input_idx=image_input_idx,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        if self.config.weight_tying:
            logits = F.linear(hidden_states, self.model.wte.weight, None)  # type: ignore
        else:
            logits = self.lm_head(hidden_states)  # type: ignore
        if self.config.scale_logits:
            logits.mul_(1 / math.sqrt(self.config.hidden_size))

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        if model_kwargs["use_cache"] and "images" in model_kwargs:
            # After the first step, no long pass the images into forward since the images tokens
            # are already cached
            for k in ["images", "image_masks", "image_input_idx"]:
                del model_kwargs[k]
        return super()._update_model_kwargs_for_generation(outputs, model_kwargs,is_encoder_decoder, num_new_tokens)