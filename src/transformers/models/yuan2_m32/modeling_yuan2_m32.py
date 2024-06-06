# coding=utf-8
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
""" PyTorch Yuan model."""
import math
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings

from .configuration_yuan2_m32 import Yuan2M32Config
from einops import rearrange


import copy

try:
    from flash_attn import flash_attn_varlen_func as flash_attn_unpadded_func
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_unpadded_func = None


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "Yuan2M32Config"


class LocalizedFiltering(torch.nn.Module):
    """
    Mega's Exponential Moving Average layer, largely left unmodified from the original repo with the exception of
    variable names and moving away from the stateful representation of incremental decoding state. See
    "https://arxiv.org/abs/2209.10655" for more details.
    """

    def __init__(self, hidden_size):
        super().__init__()

        self.embed_dim = hidden_size
        self.lf_conv2d_group = 1
        self.lf_conv2d_num_pad = 1

        self.conv1 = torch.nn.Conv2d(self.embed_dim, self.embed_dim // 2, (2, 1), stride=(1, 1), padding=(self.lf_conv2d_num_pad, 0), groups=self.lf_conv2d_group)
        self.conv2 = torch.nn.Conv2d(self.embed_dim // 2, self.embed_dim, (2, 1), stride=(1, 1), padding=(self.lf_conv2d_num_pad, 0), groups=self.lf_conv2d_group)
        self.output_layernorm = YuanRMSNorm(self.embed_dim)

    def _train_forward(self, inputs):
        inputs = inputs.transpose(0,1)
        seq_len, bsz, embed_dim = inputs.size()
        if embed_dim != self.embed_dim:
            raise ValueError(
                f"Unexpected embedding dimension received: input is {embed_dim}, model expects {self.embed_dim}"
            )
        residual = inputs

        inputs = inputs.view(seq_len, 1, bsz, embed_dim).permute(2, 3, 0, 1)
        output1 = self.conv1(inputs)
        output1 = output1[:, :, :seq_len, :]

        output2 = self.conv2(output1)
        output2 = output2[:, :, :seq_len, :].permute(2, 3, 0, 1).contiguous()
        output2 = output2.view(seq_len, bsz, embed_dim)
        assert output2.shape == residual.shape

        lf_output = self.output_layernorm(output2 + residual)
        lf_output = lf_output.transpose(0,1)
        return lf_output

    def _inference_forward(self, inputs, before_hidden_states):

        if before_hidden_states is None:
            inputs = inputs.transpose(0,1)
            seq_len, bsz, embed_dim = inputs.size()
            if embed_dim != self.embed_dim:
                raise ValueError(
                    f"Unexpected embedding dimension received: input is {embed_dim}, model expects {self.embed_dim}"
                )
            residual = inputs

            inputs = inputs.view(seq_len, 1, bsz, embed_dim).permute(2, 3, 0, 1)
            output1 = self.conv1(inputs)
            output1 = output1[:, :, :seq_len, :]

            output2 = self.conv2(output1)
            output2 = output2[:, :, :seq_len, :].permute(2, 3, 0, 1).contiguous()
            output2 = output2.view(seq_len, bsz, embed_dim)
            assert output2.shape == residual.shape

            lf_output = self.output_layernorm(output2 + residual)
            lf_output = lf_output.transpose(0,1)
            return lf_output
        else:
            inputs = inputs.transpose(0,1)
            before_hidden_states = before_hidden_states.transpose(0,1)
            residual = inputs

            seq_len, bsz, embed_dim = inputs.size()
            seq_len_before, _, _ = before_hidden_states.size()

            assert seq_len == 1 and seq_len_before == 2

            inputs = torch.cat((before_hidden_states, inputs), dim=0)
            inputs = inputs.view(3, 1, bsz, embed_dim).permute(2, 3, 0, 1)

            output1 = self.conv1(inputs)
            output2 = self.conv2(output1[:,:,1:-1,:])
            output2 = output2[:,:,1:-1,:]
            output2 = output2.view(1, bsz, embed_dim)
            assert output2.shape == residual.shape

            lf_output = self.output_layernorm(output2 + residual)
            lf_output = lf_output.transpose(0,1)

        return lf_output



    def forward(
        self,
        inputs,
        before_hidden_states
    ) -> torch.Tensor:
        assert self.lf_conv2d_num_pad == 1
        if self.training:
            lf_output = self._train_forward(inputs)
        else:
            lf_output = self._inference_forward(inputs, before_hidden_states)

        return lf_output


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb_0(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    rot_dim = sin.shape[-1]

    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]

    q, q_pass = q[..., :rot_dim], q[..., rot_dim:]
    k, k_pass = k[..., :rot_dim], k[..., rot_dim:]

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return torch.cat((q_embed, q_pass), dim=-1), torch.cat((k_embed, k_pass), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class YuanRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        YuanRMSNorm is equivalent to LlamaRMSNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class YuanRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):

        """
        YuanRotaryEmbedding is equivalent to LlamaRotaryEmbedding in transformers v4.36
        """

        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        inv_freq = inv_freq.to(torch.bfloat16)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

# flash attn
class FlashSelfAttention(torch.nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """
    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0,
                 device=None, dtype=None):
        super().__init__()
        assert flash_attn_unpadded_func is not None, ('Please install FlashAttention first, '
                                                      'e.g., with pip install flash-attn')
        assert rearrange is not None, 'Please install einops first, e.g., with pip install einops'
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, q, k, v):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q, k, v: The tensor containing the query, key, and value. (B, S, H, D)
        """

        assert all((i.dtype in [torch.float16, torch.bfloat16] for i in (q,k,v)))
        assert all((i.is_cuda for i in (q,k,v)))

        batch_size, seqlen_q = q.shape[0], q.shape[1]
        seqlen_k = k.shape[1]

        q, k, v = [rearrange(x, 'b s ... -> (b s) ...') for x in [q, k, v]]
        cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32,
                                    device=q.device)

        if self.training:
            # during training q,k,v always have same seqlen
            assert seqlen_k == seqlen_q

            is_causal = self.causal
            cu_seqlens_k = cu_seqlens_q
            dropout_p = self.dropout_p
        else:
            # turn off FA causal mask after first inference autoregressive iteration
            # only on first autoregressive step q,k,v have same seqlen
            is_causal = seqlen_q == seqlen_k
            cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32,
                        device=q.device)
            dropout_p = 0

        output = flash_attn_unpadded_func(
            q, k, v, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen_k,
            dropout_p,
            softmax_scale=self.softmax_scale, causal=is_causal
        )

        output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
        return output


class ParallelAttention_router(nn.Module):
    def __init__(self, config):
        super(ParallelAttention_router, self).__init__()
        layer_number=0
        self.layer_number = max(1, layer_number)
        

        self.flash_attn_drop = 0.01
        self.hidden_size = config.hidden_size
        self.projection_size = config.moe_config['moe_num_experts']            
            
        self.query = nn.Linear(self.hidden_size, self.projection_size, bias=False)
        self.key = nn.Linear(self.hidden_size, self.projection_size, bias=False)
        self.value = nn.Linear(self.hidden_size, self.projection_size, bias=False)


    def forward(self, hidden_states, attention_mask=None, enc_position_ids=None,
                encoder_output=None, inference_params=None,
                rotary_pos_emb=None):
        is_first_step = False
        before_hidden_states = None
        
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)
        
        b = query_layer.size(0)
        s = query_layer.size(1) # seq*batch = token_num
        z = query_layer.size(2) # expert_num
        
        # use fp32 router
        query_layer = query_layer.float().view(b,s,z,1)
        key_layer = key_layer.float().view(b,s,z,1)
        value_layer = value_layer.float().view(b,s,z,1)
        
        
        attn_weights = torch.matmul(query_layer, key_layer.transpose(2, 3))
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        
        attn_output = torch.matmul(attn_weights, value_layer)
        
        router_output = attn_output.view(b*s, z)
        
        return router_output

class YuanExpertMLP(nn.Module):
    def __init__(self, config):
        super(YuanExpertMLP, self).__init__()
        
        self.gated_linear_unit = config.moe_config['gated_linear_unit']
        self.ffn_hidden_size = config.moe_config['ffn_hidden_size']


        if self.gated_linear_unit:
            self.w1 = nn.Linear(config.hidden_size, self.ffn_hidden_size*2, bias=False)
            
              
        else:
            self.w1 = nn.Linear(config.hidden_size, self.ffn_hidden_size, bias=False)
            
        self.act_fn = ACT2FN[config.hidden_act]
        self.w2 = nn.Linear(self.ffn_hidden_size, config.hidden_size, bias=False)
        

    def forward(self, x):
        x = self.w1(x)
        if self.gated_linear_unit:
            x = torch.chunk(x, 2, dim=-1)
            x = self.act_fn(x[0]) * x[1]
        else:
            x = self.act_fn(x)  
        x = self.w2(x)  
        return x
    


class YuanMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str
    ):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.gate_proj(x) * self.act_fn(self.up_proj(x)))
        

class YuanAttention(nn.Module):
    """Localized Filtering-based Attention 'YUAN 2.0: A Large Language Model with Localized Filtering-based Attention' paper"""

    def __init__(self, config: Yuan2M32Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        try:
            self.attention_projection_size = config.attention_projection_size
        except:
            self.attention_projection_size = None
        
        if self.attention_projection_size is None:
            self.head_dim = self.hidden_size // self.num_heads
        else:
            self.head_dim = self.attention_projection_size // self.num_heads

        self.max_position_embeddings = config.max_position_embeddings
        self.causal_mask = config.causal_mask
        self.softmax_scale = 1.0 / math.sqrt(self.head_dim)
        self.use_flash_attention = config.use_flash_attention
        try:
            self.use_shareqk = config.use_shareqk
        except Exception as e:
            self.use_shareqk=False
        self.dropout = 0.0
        
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        if self.head_dim == self.hidden_size // self.num_heads:
            self.rotary_emb = YuanRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

        else:
            self.rotary_emb = YuanRotaryEmbedding(self.hidden_size // self.num_heads, max_position_embeddings=self.max_position_embeddings)

        if self.use_shareqk:
            self.qk_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
            self.qk_weight = nn.Parameter(torch.Tensor(2, self.hidden_size))
            self.qk_bias = nn.Parameter(torch.Tensor(2, self.hidden_size))
        else:
            self.lf_gate = LocalizedFiltering(self.hidden_size)
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
            self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)


    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        bsz, q_len, _ = hidden_states.size()
        before_hidden_states = None
        is_first_step = False
        if use_cache:
            if past_key_value is None:
                inference_hidden_states_memory = torch.empty(bsz, 2, hidden_states.shape[2], dtype=hidden_states.dtype)
                is_first_step = True
            else:
                before_hidden_states = past_key_value[2]

        if use_cache:
            if is_first_step:
                if q_len >= 2:
                    inference_hidden_states_memory = hidden_states[ :, -2:, :]
                else:
                    inference_hidden_states_memory[:, :, :] = 0
                    inference_hidden_states_memory[:, -1:, :] = hidden_states[:, -1:, :]
            else:
                hidden_states_tmp = before_hidden_states[:, -1:, :]
                inference_hidden_states_memory = copy.deepcopy(torch.cat((hidden_states_tmp, hidden_states), dim=1))
        
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        if self.use_shareqk:
            qk_states = self.qk_proj(hidden_states).view(bsz, q_len, self.num_heads*self.head_dim)
            query_key = qk_states.unsqueeze(2) * self.qk_weight + self.qk_bias
            query_states, key_states = torch.unbind(query_key, dim=2)

            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            hidden_states = self.lf_gate(hidden_states,before_hidden_states)
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            qk_states = torch.cat([query_states, key_states], dim=-1)
            qk_states = qk_states.view(bsz,q_len,self.num_heads,int(qk_states.shape[-1]//self.num_heads))
            (query_states,key_states) =  torch.chunk(qk_states, 2, dim=-1)
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        
        query_states, key_states = apply_rotary_pos_emb_0(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states,inference_hidden_states_memory) if use_cache else None 
        if self.use_flash_attention:
            attn_weights = None
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

            batch_size, seqlen_q = query_states.shape[0], query_states.shape[1]
            seqlen_k = key_states.shape[1]

            q, k, v = [rearrange(x, 'b s ... -> (b s) ...') for x in [query_states, key_states, value_states]]

            cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int,
                                    device=q.device)
                                    
            if self.training:                        
                assert seqlen_k == seqlen_q
                cu_seqlens_k = cu_seqlens_q
                is_causal = self.causal_mask
            else:
                is_causal = seqlen_q == seqlen_k
                cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int,
                        device=q.device)
                self.dropout=0

            output = flash_attn_unpadded_func(
                q, k, v, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen_k, self.dropout, causal=is_causal
            )

            attn_output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
        else:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )
            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

            attn_output = attn_output.transpose(1, 2)

        if self.attention_projection_size is None:
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        else:
            attn_output = attn_output.reshape(bsz, q_len, self.attention_projection_size)
        
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value



class YuanMoeLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.moe_config['moe_num_experts']
        self.top_k = config.moe_config['moe_top_k']
        self.norm_topk_prob = config.moe_config['norm_topk_prob']
        self.hidden_size = config.hidden_size
        
        
        self.gate = ParallelAttention_router(config)
        self.experts = nn.ModuleList(
            [YuanExpertMLP(config) for _ in range(self.num_experts)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_dim)
        
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        
        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


class YuanDecoderLayer(nn.Module):
    def __init__(self, config: Yuan2M32Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = YuanAttention(config=config)
        
        if config.moe_config['moe_num_experts'] > 0:
            self.mlp = YuanMoeLayer(config)
        else:
            self.mlp = YuanMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
            )
        
        
        self.input_layernorm = YuanRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = YuanRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)
        
        hidden_states, router_logits = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)
    
        return outputs


YUAN_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Yuan2M32Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Yuan Model outputting raw hidden-states without any specific head on top.",
   YUAN_START_DOCSTRING,
)
class YuanPreTrainedModel(PreTrainedModel):
    config_class = Yuan2M32Config 
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["YuanDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, YuanModel):
            module.gradient_checkpointing = value


YUAN_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
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
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
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
"""


@add_start_docstrings(
    "The bare Yuan Model outputting raw hidden-states without any specific head on top.",
    YUAN_START_DOCSTRING,
)
class YuanModel(YuanPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`YuanDecoderLayer`]

    Args:
        config: Yuan2M32Config
    """

    def __init__(self, config: Yuan2M32Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        #TODO: control it by config
        self.eod_token = config.eod_token
        self.reset_attention_mask = config.reset_attention_mask
        self.reset_position_ids = config.reset_position_ids
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([YuanDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = YuanRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def _prepare_decoder_attention_mask_training(self, input_id, inputs_embeds, eod_token, reset_mask_flag ,reset_attention_mask=True, reset_position_ids=True):
    
        micro_batch_size, seq_length = input_id.size()
        
        attention_mask = torch.tril(torch.ones(
            (micro_batch_size, seq_length, seq_length), device=inputs_embeds.device)).view(
                micro_batch_size, 1, seq_length, seq_length)    
                
        position_ids = torch.arange(seq_length, dtype=torch.long,
                                    device=inputs_embeds.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_id)
               
        if reset_position_ids:
            position_ids = position_ids.clone()
        
        if reset_position_ids or reset_attention_mask:
            # Loop through the batches:
            for b in range(micro_batch_size):

                # Find indecies where EOD token is.
                eod_index = position_ids[b, input_id[b] == eod_token]

                # Detach indecies from positions if going to modify positions.
                if reset_position_ids:
                    eod_index = eod_index.clone()
                # Loop through EOD indecies:
                prev_index = 0
                for j in range(eod_index.size()[0]):
                    i = eod_index[j]
                    # Mask attention loss.
                    if reset_attention_mask:
                        attention_mask[b, 0, (i + 1):, :(i + 1)] = 0
                    # Reset positions.
                    if reset_position_ids:
                        position_ids[b, (i + 1):] -= (i + 1 - prev_index)
                        prev_index = i + 1
                        
        inverted_mask = 1 - attention_mask   
        output_attn_mask = inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(inputs_embeds.dtype).min)
        if reset_mask_flag:
            output_attn_mask = output_attn_mask[:,:,-1:,:]
        return output_attn_mask, position_ids

    @add_start_docstrings_to_model_forward(YUAN_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        input_ids1 = copy.deepcopy(input_ids)
        reset_mask_flag = False
        if past_key_values:
            input_ids = input_ids[:, -1:]
            if use_cache:
                reset_mask_flag = True
        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if self.training or self.reset_position_ids:
            attention_mask, _ = self._prepare_decoder_attention_mask_training(input_ids1, inputs_embeds, self.eod_token, reset_mask_flag, self.reset_attention_mask, self.reset_position_ids)
        
        else: 
            if attention_mask is None:
                attention_mask = torch.ones(
                    (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
                )
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
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
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class YuanForCausalLM(YuanPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.eod_token = config.eod_token
        self.sep_token = config.sep_token
        self.use_loss_mask = config.use_loss_mask
        self.model = YuanModel(config)

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

    def get_loss_mask(self, input_ids, labels, eod_token, sep_token):
        micro_batch_size, seq_length = input_ids.size()
        loss_mask = torch.ones(input_ids.size(), dtype=torch.float, device=input_ids.device)

        position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids) 


        """modify loss_mask to only calculate the loss of the answer (separated with [SEP])"""

        for b in range(micro_batch_size):
            eod_indexs = position_ids[b, input_ids[b] == eod_token]
            sep_indexs = position_ids[b, input_ids[b] == sep_token]

            if len(eod_indexs) == 0 or len(sep_indexs) == 0:
                loss_mask[b] = 1.0
            else:
                if eod_indexs[0] > sep_indexs[0]:
                    loss_mask[b, 0:sep_indexs[0]] = 0

                    if len(eod_indexs) == len(sep_indexs):
                        for ii, eod_index in enumerate(eod_indexs):
                            start_index = eod_index
                            if ii == (len(sep_indexs) - 1):
                                stop_index = seq_length
                            else:
                                stop_index = sep_indexs[ii + 1]
                            loss_mask[b, start_index:stop_index] = 0.0
                    else:
                        if len(eod_indexs) > len(sep_indexs):
                            loss_mask[b,:] = 1.0
                        else:
                            for ii, eod_index in enumerate(eod_indexs):
                                start_index = eod_index
                                stop_index = sep_indexs[ii + 1]

                                loss_mask[b, start_index:stop_index] = 0.0

                elif eod_indexs[0] < sep_indexs[0]:

                    if len(eod_indexs) == len(sep_indexs):
                        for ii, eod_index in enumerate(eod_indexs):
                            start_index = eod_index
                            stop_index = sep_indexs[ii]
                            loss_mask[b, start_index:stop_index] = 0.0

                    else:
                        if len(eod_indexs) < len(sep_indexs):
                            loss_mask[b,:] = 1.0
                        else:
                            for ii, eod_index in enumerate(eod_indexs):
                                start_index = eod_index
                                if ii >= len(sep_indexs):
                                    stop_index = seq_length
                                else:
                                    stop_index = sep_indexs[ii]
                                loss_mask[b, start_index:stop_index] = 0.0

        loss_mask[input_ids == eod_token] = 1.0
        return loss_mask                       
    @add_start_docstrings_to_model_forward(YUAN_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, YuanForCausalLM

        >>> model = YuanForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
        ```"""

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
        )

        hidden_states = outputs[0]
        
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            if self.use_loss_mask:
                loss_mask = self.get_loss_mask(input_ids, labels, self.eod_token, self.sep_token)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            if self.use_loss_mask:
                loss_fct = CrossEntropyLoss(reduction='none')
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
                loss = torch.sum(loss * loss_mask) / loss_mask.sum()
            else:
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
            hidden_states=hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past


@add_start_docstrings(
    """
    The Yuan Model transformer with a sequence classification head on top (linear layer).

    [`YuanForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    YUAN_START_DOCSTRING,
)
class YuanForSequenceClassification(YuanPreTrainedModel):
    #_keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = YuanModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(YUAN_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )



