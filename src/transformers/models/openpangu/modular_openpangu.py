# coding=utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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

from typing import Callable, Optional, Tuple

import torch
from torch import nn

from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils import logging
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaForSequenceClassification,
    LlamaMLP,
    LlamaModel,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from .configuration_openpangu import PanguConfig


logger = logging.get_logger(__name__)


class OpenPanguMLP(LlamaMLP):
    def __init__(self, config):
        super().__init__(config)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

class MLP(nn.Module):
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        output = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return output

class MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.routed_scaling_factor = config.routed_scaling_factor

        self.norm_topk_prob = config.norm_topk_prob
        self.weight = nn.Parameter(
            torch.empty((config.num_routed_experts, config.hidden_size))
        )

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(
            hidden_states.to(torch.float32), self.weight.to(torch.float32), None
        )
        scores = logits.sigmoid()
        scores_for_choice = scores.view(bsz * seq_len, -1)
        _, topk_idx = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)
        topk_weight = scores.gather(1, topk_idx)

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        topk_weight = topk_weight * self.routed_scaling_factor

        return topk_idx, topk_weight


class OpenPanguMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_shared_experts = config.num_shared_experts
        self.num_routed_experts = config.num_routed_experts
        self.experts = nn.ModuleList(
            [
                MLP(config, intermediate_size=config.moe_intermediate_size)
                for i in range(self.num_routed_experts)
            ]
        )
        self.gate = MoEGate(config)
        if self.num_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * self.num_shared_experts
            self.shared_experts = MLP(
                config=config, intermediate_size=intermediate_size
            )

    def forward(self, hidden_states):
        if self.num_shared_experts is not None:
            shared_output = self.shared_experts(hidden_states)
        input_shape = hidden_states.shape
        topk_ids, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        counts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        counts.scatter_(1, topk_ids, 1)
        tokens_per_expert = counts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = hidden_states[idxs // topk_ids.shape[1]]
        tokens_per_expert = tokens_per_expert.cpu().numpy()

        output_hidden_states = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            output_hidden_states.append(expert_out)
            start_idx = end_idx

        if len(output_hidden_states) > 0:
            cat_hidden_states = torch.cat(output_hidden_states, dim=0)
        else:
            cat_hidden_states = sorted_tokens.new_empty(0)

        final_hidden_states = torch.empty_like(cat_hidden_states)
        final_hidden_states[idxs] = cat_hidden_states
        final_out = final_hidden_states.view(*topk_ids.shape, -1).to(topk_weight.dtype)
        final_out = (
            final_out.mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .to(final_hidden_states.dtype)
        ).view(*input_shape)
        if self.num_shared_experts is not None:
            final_out = final_out + shared_output
        return final_out


class OpenPanguEmbeddedAttention(LlamaAttention):
    def __init__(self, config: PanguUltraMoEConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.attention_q_lora_dim = config.attention_q_lora_dim
        self.attention_qk_rope_dim = config.attention_qk_rope_dim
        self.attention_kv_lora_dim = config.attention_kv_lora_dim
        self.attention_v_dim = config.attention_v_dim
        self.attention_qk_dim = config.attention_qk_dim
        self.q_head_dim = config.attention_qk_dim + config.attention_qk_rope_dim

        if self.attention_q_lora_dim is None:
            self.q_proj = nn.Linear(
                self.hidden_size, self.num_heads * self.q_head_dim, bias=False
            )
        else:
            self.q_a_proj = nn.Linear(
                self.hidden_size, config.attention_q_lora_dim, bias=False
            )
            self.q_a_layernorm = PanguUltraMoERMSNorm(config.attention_q_lora_dim)
            self.q_b_proj = nn.Linear(
                config.attention_q_lora_dim,
                self.num_heads * self.q_head_dim,
                bias=False,
            )
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            config.attention_kv_lora_dim + config.attention_qk_rope_dim,
            bias=False,
        )
        self.kv_a_layernorm = PanguUltraMoERMSNorm(config.attention_kv_lora_dim)
        self.kv_b_proj = nn.Linear(
            config.attention_kv_lora_dim,
            self.num_heads * (config.attention_qk_dim + self.attention_v_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.attention_v_dim,
            self.hidden_size,
            bias=False,
        )
        self.rotary_emb = PanguUltraMoERotaryEmbedding(
            self.attention_qk_rope_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        self.softmax_scale = self.q_head_dim ** (-0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()

        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(
            q, [self.attention_qk_dim, self.attention_qk_rope_dim], dim=-1
        )

        latent_kv = self.kv_a_proj_with_mqa(hidden_states)
        kv_a, k_pe = torch.split(
            latent_kv, [self.attention_kv_lora_dim, self.attention_qk_rope_dim], dim=-1
        )
        k_pe = k_pe.view(bsz, q_len, 1, self.attention_qk_rope_dim).transpose(1, 2)
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(kv_a))
            .view(
                bsz, q_len, self.num_heads, self.attention_qk_dim + self.attention_v_dim
            )
            .transpose(1, 2)
        )
        kv_seq_len = kv.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(kv, kv_seq_len)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        k_nope, value = torch.split(
            kv, [self.attention_qk_dim, self.attention_v_dim], dim=-1
        )

        def concat_nope_pe(nope, pe):
            states = torch.empty(
                [bsz, self.num_heads, q_len, self.q_head_dim],
                dtype=nope.dtype,
                device=nope.device,
            )
            states[:, :, :, : self.attention_qk_dim] = nope
            states[:, :, :, self.attention_qk_dim :] = pe
            return states

        query = concat_nope_pe(q_nope, q_pe)
        key = concat_nope_pe(k_nope, k_pe)

        if past_key_value is not None:
            key, value = past_key_value.update(
                key, value, self.layer_idx, {"sin": sin, "cos": cos}
            )

        attn_weights = (
            torch.matmul(query, key.transpose(2, 3)) * self.softmax_scale
            + attention_mask
        )
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value

class OpenPanguMLAAttention(LlamaAttention):
    pass         

class OpenPanguDecoderLayer(nn.Module):
    def __init__(self, config: PanguConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = OpenPanguMoEAttention(config=config, layer_idx=layer_idx)

        self.mlp = (
            OpenPanguMoE(config)
            if (
                config.num_routed_experts is not None
                and layer_idx >= config.num_dense_layers
            )
            else MLP(config)
        )
        self.input_layernorm = OpenPanguMoERMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps
        )
        self.post_attention_layernorm = OpenPanguMoERMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps
        )
        if getattr(config, "sandwich_norm", False):
            self.sandwich_norm = True
            self.pre_mlp_layernorm = OpenPanguMoERMSNorm(
                config.hidden_size, epsilon=config.rms_norm_eps
            )
            self.post_mlp_layernorm = OpenPanguMoERMSNorm(
                config.hidden_size, epsilon=config.rms_norm_eps
            )
        else:
            self.sandwich_norm = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            **kwargs,
        )
        if self.sandwich_norm:
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.pre_mlp_layernorm(hidden_states)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)

        if self.sandwich_norm:
            hidden_states = self.post_mlp_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return (hidden_states, present_key_value)
    

class OpenPanguEmbeddedModel(LlamaModel):
    pass

class PanguUltraMoEPreTrainedModel(PreTrainedModel):
    config_class = PanguUltraMoEConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["PanguUltraMoEDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        self._initialize_linear(module, std)
        self._initialize_embedding(module, std)

    def _initialize_linear(self, module, std):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()

    def _initialize_embedding(self, module, std):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
                
class OpenPanguMoEModel(LlamaModel):
    def __init__(self, config: PanguUltraMoEConfig):
        super().__init__(config)

        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.padding_idx = config.pad_token_id
        self.layer_num = config.num_hidden_layers
        self.epsilon = config.rms_norm_eps

        self.embed_tokens = nn.Embedding(
            self.vocab_size, self.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [PanguUltraMoEDecoderLayer(config, idx) for idx in range(self.layer_num)]
        )
        self.norm = PanguUltraMoERMSNorm(self.hidden_size, epsilon=self.epsilon)
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
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You have to specify input_ids or inputs_embeds.")

        if input_ids is not None:
            hidden_states = self.embed_tokens(input_ids)
            batch_size, seq_length = input_ids.size()
        else:
            hidden_states = inputs_embeds
            batch_size, seq_length = inputs_embeds.size()

        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.device
            ).unsqueeze(0)

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)
            position_ids += past_key_values_length

        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            hidden_states,
            past_key_values_length,
        )

        for decoder_layer in self.layers:
            hidden_states, present_key_value = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
            )

        hidden_states = self.norm(hidden_states)

        if use_cache and use_legacy_cache:
            present_key_value = present_key_value.to_legacy_cache()

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value,
        )

class OpenPanguEmbeddedForCausalLM(LlamaForCausalLM):
    pass

class OpenPanguMoEForCausalLM(LlamaForCausalLM):
   _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = PanguUltraMoEModel(config)
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

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
        )

        logits = self.lm_head(outputs[0])
        logits = logits.float()

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.vocab_size
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
