# Copyright 2025 The HuggingFace Team. All rights reserved.
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
"""PyTorch Hindi Causal Language Model."""

import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_hindi_causal_lm import HindiCausalLMConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "convaiinnovations/hindi-foundational-model-base"
_CONFIG_FOR_DOC = "HindiCausalLMConfig"

HINDI_CAUSAL_LM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "convaiinnovations/hindi-foundational-model-base",
]


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight * hidden_states).to(input_dtype)


class HindiCausalLMRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(seq_len=max_position_embeddings, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=torch.int64).type_as(
            self.inv_freq
        )
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]
            
        if (
            seq_len > self.max_seq_len_cached
            or self.cos_cached.device != x.device
            or self.cos_cached.dtype != x.dtype
        ):
            self._set_cos_sin_cache(seq_len=seq_len, dtype=x.dtype)
            
        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
        )


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def _prepare_4d_causal_attention_mask(
    attention_mask: Optional[torch.Tensor],
    input_shape: Tuple[int, int],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    is_causal: bool = True,
):
    bsz, tgt_len = input_shape
    dtype = inputs_embeds.dtype
    src_len = past_key_values_length + tgt_len

    if attention_mask is not None and attention_mask.dim() == 4:
        expected_shape = (bsz, 1, tgt_len, src_len)
        if attention_mask.shape != expected_shape:
            raise ValueError(
                f"Provided 4D attention mask shape {attention_mask.shape} does not match expected {expected_shape}"
            )
        if not torch.is_floating_point(attention_mask) and attention_mask.dtype != torch.bool:
            raise ValueError("4D attention mask must be additive (float type with 0.0/-inf) or boolean")
        if torch.is_floating_point(attention_mask):
            attention_mask = attention_mask.to(dtype)
        return attention_mask

    if attention_mask is None and not is_causal:
        return None

    if is_causal:
        mask = torch.full((tgt_len, src_len), torch.finfo(dtype).min, dtype=dtype, device=inputs_embeds.device)
        mask_cond = torch.arange(mask.size(-1), device=inputs_embeds.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0.0)
        if past_key_values_length > 0:
            mask[:, :past_key_values_length] = 0.0
        expanded_mask = mask[None, None, :, :].expand(bsz, 1, tgt_len, src_len)
    else:
        expanded_mask = torch.zeros((bsz, 1, tgt_len, src_len), dtype=dtype, device=inputs_embeds.device)

    if attention_mask is not None:
        if attention_mask.dim() != 2:
            raise ValueError(f"Attention mask should be 2D but is {attention_mask.dim()}D")
        padding_mask = attention_mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len)
        expanded_mask = expanded_mask.masked_fill(padding_mask == 0, torch.finfo(dtype).min)

    return expanded_mask


class HindiCausalLMAttention(nn.Module):
    def __init__(self, config: HindiCausalLMConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(f"Instantiating {self.__class__.__name__} without layer_idx is not recommended...")
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.is_causal = True

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.attention_dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.positional_encoding_type = getattr(config, "positional_encoding_type", "rope")
        if self.positional_encoding_type == "rope":
            self.rotary_emb = HindiCausalLMRotaryEmbedding(
                self.head_dim, max_position_embeddings=self.max_position_embeddings
            )
        else:
            self.rotary_emb = None

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn("Argument `padding_mask` is deprecated and will be removed", FutureWarning)

        bsz, q_len, _ = hidden_states.size()
        query_states = self._shape(self.q_proj(hidden_states), q_len, bsz)
        key_states = self._shape(self.k_proj(hidden_states), q_len, bsz)
        value_states = self._shape(self.v_proj(hidden_states), q_len, bsz)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError("Layer index needed for cache")
            # If past_key_value is a tuple from a legacy cache
            kv_seq_len += past_key_value[0].shape[-2]  # Adjust the key_states shape

        if self.rotary_emb is not None and position_ids is not None:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            cos = cos[position_ids]
            sin = sin[position_ids]
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # Legacy cache handling
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(f"Attention weights shape error {attn_weights.size()} vs expected")
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask shape error {attention_mask.size()} vs expected {(bsz, 1, q_len, kv_seq_len)}"
                )
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.attention_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value_states)
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(f"Attention output shape error {attn_output.size()} vs expected")

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        present_key_value = (key_states, value_states) if use_cache else None
        attn_weights_output = attn_weights if output_attentions else None

        return attn_output, attn_weights_output, present_key_value


class HindiCausalLMMLP(nn.Module):
    def __init__(self, config: HindiCausalLMConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = (
            config.intermediate_size if hasattr(config, "intermediate_size") else 4 * config.hidden_size
        )

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        self.act_fn = nn.SiLU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_output = self.act_fn(self.gate_proj(hidden_states))
        up_output = self.up_proj(hidden_states)
        intermediate_output = gate_output * up_output
        output = self.down_proj(intermediate_output)
        output = self.dropout(output)
        return output


class HindiCausalLMLayer(nn.Module):
    def __init__(self, config: HindiCausalLMConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = HindiCausalLMAttention(config=config, layer_idx=layer_idx)
        self.mlp = HindiCausalLMMLP(config)
        norm_class = RMSNorm if getattr(config, "normalization_layer", "rmsnorm") == "rmsnorm" else nn.LayerNorm
        self.input_layernorm = norm_class(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = norm_class(config.hidden_size, eps=config.layer_norm_eps)
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = attn_outputs[0]
        attn_weights = attn_outputs[1]
        present_key_value = attn_outputs[2]
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


class HindiCausalLMPreTrainedModel(PreTrainedModel):
    config_class = HindiCausalLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["HindiCausalLMLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = False
    _supports_sdpa = True
    _supports_cache_class = False  # Use legacy cache handling for compatibility
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        std = self.config.initializer_range if hasattr(self.config, "initializer_range") else 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, HindiCausalLMModel):
            module.gradient_checkpointing = value


class HindiCausalLMModel(HindiCausalLMPreTrainedModel):
    def __init__(self, config: HindiCausalLMConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [HindiCausalLMLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        norm_class = RMSNorm if getattr(config, "normalization_layer", "rmsnorm") == "rmsnorm" else nn.LayerNorm
        self.norm = norm_class(config.hidden_size, eps=config.layer_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.FloatTensor, torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # Remove token_type_ids if present
        if 'token_type_ids' in kwargs:
            kwargs.pop('token_type_ids')
        
        # Remove cache_position if present
        if 'cache_position' in kwargs:
            kwargs.pop('cache_position')
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once("use_cache=True is incompatible with gradient checkpointing...")
                use_cache = False

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]  # Get length from first layer's keys

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        batch_size, seq_length = inputs_embeds.shape[:2]

        if position_ids is None:
            device = inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Standard causal mask handling
        _attention_mask = attention_mask
        if seq_length > 1:
            _attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value=None, output_attentions=output_attentions, use_cache=False)
                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    _attention_mask,
                    position_ids,
                    use_reentrant=False,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=_attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
                
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attns] if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class HindiCausalLMForCausalLM(HindiCausalLMPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight", r"position_ids"]
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: HindiCausalLMConfig):
        super().__init__(config)
        self.model = HindiCausalLMModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def tie_weights(self):
        if self.config.tie_word_embeddings:
            output_embeddings, input_embeddings = self.get_output_embeddings(), self.get_input_embeddings()
            if output_embeddings is not None and input_embeddings is not None:
                output_embeddings.weight = input_embeddings.weight
            super().tie_weights()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.FloatTensor, torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # Remove unsupported arguments
        if 'token_type_ids' in kwargs:
            kwargs.pop('token_type_ids')
        
        if 'cache_position' in kwargs:
            kwargs.pop('cache_position')

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values if return_dict else outputs[1],
            hidden_states=outputs.hidden_states if return_dict else outputs[2],
            attentions=outputs.attentions if return_dict else outputs[3],
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        
        # Include position_ids if provided
        if "position_ids" in kwargs:
            model_inputs["position_ids"] = kwargs["position_ids"]
            
        return model_inputs

    def generate(self, **kwargs):
        # Handle default generation parameters
        if "max_length" not in kwargs:
            kwargs["max_length"] = 50
        # Custom generation handling if needed
        return super().generate(**kwargs)
        
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_layer_past = tuple(
                past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past
            )
            reordered_past += (reordered_layer_past,)
        return reordered_past