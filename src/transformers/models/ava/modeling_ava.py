# -*- coding: utf-8 -*-
# Copyright 2025 Nika Kudukhashvili <nikakuduxashvili0@gmail.com>. All rights reserved.
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
import math

from .configuration_ava import AvaConfig

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]

    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos[:, :, :q.shape[2], :]
    sin = sin[:, :, :q.shape[2], :]

    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin

    return q_embed, k_embed

class AvaAttention(nn.Module):
    def __init__(self, config: AvaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_heads = getattr(config, "kv_heads", self.num_heads)
        self.kv_dim = self.head_dim * self.kv_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.dropout = nn.Dropout(config.attention_dropout)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        rotary_emb=None,
        position_ids=None
    ):
        B, T, _ = hidden_states.shape

        query = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        key = self.k_proj(hidden_states).view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)    # (B, kvh, T, hs)
        value = self.v_proj(hidden_states).view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)  # (B, kvh, T, hs)

        if past_key_value is not None:
            key = torch.cat([past_key_value[0], key], dim=2)
            value = torch.cat([past_key_value[1], value], dim=2)

        past_key_value = (key, value) if use_cache else None

        if rotary_emb is not None:
            cos, sin = rotary_emb(query, seq_len=query.shape[2])
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        if self.kv_heads != self.num_heads:
            repeat_factor = self.num_heads // self.kv_heads
            key = key.repeat_interleave(repeat_factor, dim=1)
            value = value.repeat_interleave(repeat_factor, dim=1)

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_scores += attention_mask

        attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_probs = self.dropout(attn_probs)

        context = torch.matmul(attn_probs, value)
        context = context.transpose(1, 2).contiguous().view(B, T, self.hidden_size)

        output = self.o_proj(context)

        if output_attentions:
            return output, past_key_value, attn_probs

        return output, past_key_value

class AvaRMSNorm(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            epsilon: float = 1e-5
        ):

        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.epsilon = epsilon

    def forward(self, hidden_states):
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.epsilon)

        return self.weight * hidden_states.to(hidden_states.dtype)

class AvaRotaryEmbedding(nn.Module):
    """Rotary Position Embeddings"""

    def __init__(self,
                 dim,
                 max_position_embeddings = 2048,
                 base                    = 10000.0
        ):

        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len_cached = max_position_embeddings

        t = torch.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer('cos_cached', emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len

            t = torch.arange(
                self.max_seq_len_cached,
                device = x.device,
                dtype  = self.inv_freq.dtype
            )

            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self.register_buffer('cos_cached', emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer('sin_cached', emb.sin()[None, None, :, :], persistent=False)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype, device=x.device),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype, device=x.device),
        )

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class AvaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(
            self.hidden_size,
            self.intermediate_size,
            bias = False
        )

        self.up_proj = nn.Linear(
            self.hidden_size,
            self.intermediate_size,
            bias = False
        )

        self.down_proj = nn.Linear(
            self.intermediate_size,
            self.hidden_size,
            bias = False
        )

        self.act_fn = SiLU()

    def forward(self, x):
        return self.down_proj(
            self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        )

class AvaDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.self_attn = AvaAttention(config)

        self.mlp = AvaMLP(config)
        self.input_layernorm = AvaRMSNorm(
            config.hidden_size,
            epsilon = config.rms_norm_eps
        )

        self.post_attention_layernorm = AvaRMSNorm(
            config.hidden_size,
            epsilon = config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        rotary_emb=None,
    ):

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_outputs = self.self_attn(
            hidden_states     = hidden_states,
            attention_mask    = attention_mask,
            position_ids      = position_ids,
            past_key_value    = past_key_value,
            output_attentions = output_attentions,
            use_cache         = use_cache,
            rotary_emb        = rotary_emb,
        )

        hidden_states = attn_outputs[0]
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if use_cache:
            outputs += (attn_outputs[1],)

        if output_attentions:
            outputs += (attn_outputs[2],)

        return outputs

class AvaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([AvaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = AvaRMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)
        self.rotary_emb = AvaRotaryEmbedding(
            config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )


        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)

            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        batch_size, seq_length = input_ids.shape if input_ids is not None else inputs_embeds.shape[:2]

        if position_ids is None:
            position_ids = torch.arange(
                seq_length,
                dtype = torch.long,
                device = input_ids.device
            ).unsqueeze(0)

        past_length = 0
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
            position_ids = position_ids[:, past_length:]


        if attention_mask is not None:
            causal_mask = torch.triu(
                torch.full((seq_length, seq_length), -float('inf'), device=attention_mask.device),
                diagonal=1,
            )

            expanded_attn_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * torch.finfo(torch.float32).min
            expanded_attn_mask = expanded_attn_mask + causal_mask.unsqueeze(0)
        else:
            causal_mask = torch.triu(
                torch.full(
                    (seq_length, seq_length),
                    -float('inf'),
                    device = input_ids.device
                ),

                diagonal=1,
            )

            expanded_attn_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)


        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_past_key_values = () if use_cache else None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs = layer(
                hidden_states,
                attention_mask=expanded_attn_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                rotary_emb=self.rotary_emb,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_past_key_values += (layer_outputs[1],)

            if output_attentions:
                all_self_attns += (layer_outputs[2],)


        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return {
            'last_hidden_state': hidden_states,
            'past_key_values': next_past_key_values,
            'hidden_states': all_hidden_states,
            'attentions': all_self_attns,
        }

class AvaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = AvaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.apply(self._init_weights)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)

            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output = self.model(
            input_ids            = input_ids,
            attention_mask       = attention_mask,
            position_ids         = position_ids,
            past_key_values      = past_key_values,
            inputs_embeds        = inputs_embeds,
            use_cache            = use_cache,
            output_attentions    = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict          = return_dict,
        )

        hidden_states = output['last_hidden_state']
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)


            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return {
            'loss': loss,
            'logits': logits,
            'past_key_values': output.get('past_key_values', None),
            'hidden_states': output.get('hidden_states', None),
            'attentions': output.get('attentions', None),
        }

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        input_shape = input_ids.shape
        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)


        position_ids = kwargs.get('position_ids', None)
        if attention_mask is not None and position_ids is None:

            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'past_key_values': past_key_values,
            'use_cache': kwargs.get('use_cache', True),
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_length=None,
        temperature=1.0,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.0,
        do_sample=True,
        num_return_sequences=1,
        pad_token_id=None,
        eos_token_id=None,
        use_cache=True,
        streamer=None,
        early_stopping=True,
    ):
        """
        Improved generate method with streamer support and better caching
        """
        
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        max_length = max_length if max_length is not None else self.config.max_position_embeddings
        batch_size = input_ids.shape[0]

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        input_ids_seq_length = input_ids.shape[-1]
        generated_tokens = input_ids.clone()
        cached_position_ids = torch.arange(input_ids_seq_length, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        past_key_values = None

        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        self.eval()

        for current_length in range(input_ids_seq_length, max_length):
            if past_key_values is not None:
                inputs = generated_tokens[:, -1].unsqueeze(-1)
            else:
                inputs = generated_tokens

            if past_key_values is not None:
                position_ids = cached_position_ids[:, -1].unsqueeze(-1) + 1
                cached_position_ids = torch.cat([cached_position_ids, position_ids], dim=-1)
            else:
                position_ids = cached_position_ids

            if attention_mask is not None and past_key_values is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    unfinished_sequences.unsqueeze(-1)
                ], dim = -1)

            model_inputs = self.prepare_inputs_for_generation(
                inputs,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=use_cache,
            )

            outputs = self.forward(**model_inputs)
            next_token_logits = outputs['logits'][:, -1, :]
            past_key_values = outputs['past_key_values']
            next_token_logits = next_token_logits / temperature

            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for previous_token in generated_tokens[i]:
                        if previous_token in [pad_token_id, eos_token_id]:
                            continue

                        next_token_logits[i, previous_token] /= repetition_penalty

            if top_k > 0:
                top_k_values, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits.scatter_(1, top_k_indices, top_k_values)

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                for i in range(batch_size):
                    indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                    next_token_logits[i, indices_to_remove] = float('-inf')

            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)

            if eos_token_id is not None:
                next_tokens = next_tokens * unfinished_sequences + eos_token_id * (1 - unfinished_sequences)

            generated_tokens = torch.cat([generated_tokens, next_tokens.unsqueeze(-1)], dim=-1)

            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

            if streamer is not None:
                streamer.put(next_tokens.unsqueeze(-1))

            if unfinished_sequences.max() == 0 or (early_stopping and current_length > input_ids_seq_length + 50):
                break

        if streamer is not None:
            streamer.end()

        return generated_tokens
