# coding=utf-8
# Copyright 2023 The Mega Authors and The HuggingFace Inc. team.
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
"""PyTorch Mega model."""

import math
from typing import List, Optional, Tuple, Union, Dict, Callable
import uuid

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F

from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_mega import MegaConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "mnaylor/mega-base-wikitext"
_CONFIG_FOR_DOC = "MegaConfig"

MEGA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "mnaylor/mega-base-wikitext",
    # See all Mega models at https://huggingface.co/models?filter=mega
]

# Mega source code converted to pure PyTorch
# resources
#   - paper: https://arxiv.org/abs/2209.10655
#   - original implementation: https://github.com/facebookresearch/mega 

# starting with activation functions
def relu2(x):
    return torch.square(F.relu(x))

def laplace(x, mu=0.707107, sigma=0.282095):
    x = (x - mu).div(sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + torch.erf(x))

def gelu_accurate(x):
    if not hasattr(gelu_accurate, "_a"):
        gelu_accurate._a = math.sqrt(2 / math.pi)
    return (
        0.5 * x * (1 + torch.tanh(gelu_accurate._a * (x + 0.044715 * torch.pow(x, 3))))
    )

def get_activation_fn(activation: str) -> Callable:
    """ Returns the activation function corresponding to `activation` """
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "gelu_accurate":
        return gelu_accurate
    elif activation == 'silu':
        return F.silu
    elif activation == "linear":
        return lambda x: x
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))

# utility for causal LM masking in the format that Mega expects
def generate_causal_mask(seq_len):
    '''
    Tiny utility to generate a `seq_len` by `seq_len` causal mask,
    where 1 corresponds to *not masked* and 0 corresponds to *masked*

    causal_mask[i][j] corresponds to whether token `i` can attend to token `j`
    '''
    seq_ids = torch.arange(seq_len)
    causal_mask = seq_ids[None, :].repeat(seq_len, 1) <= seq_ids[:, None]
    return causal_mask.to(torch.long)

# EMA attention module
class MultiHeadEMA(nn.Module):
    """Exponential Moving Average Layer.
    See "https://arxiv.org/abs/2209.10655" for more details.
    """

    def __init__(
        self,
        embed_dim,
        ndim=2,
        bidirectional=False,
        truncation=None,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.ndim = ndim
        self.bidirectional = bidirectional
        self.truncation = truncation
        self.scale = math.sqrt(1.0 / self.ndim)

        kernel_dim = 2 * embed_dim if self.bidirectional else embed_dim
        self.delta = nn.Parameter(torch.Tensor(kernel_dim, ndim, 1))
        self.alpha = nn.Parameter(torch.Tensor(kernel_dim, ndim, 1))
        self.beta = nn.Parameter(torch.Tensor(kernel_dim, ndim, 1))
        self.gamma = nn.Parameter(torch.Tensor(kernel_dim, ndim))
        self.omega = nn.Parameter(torch.Tensor(embed_dim))
        self._kernel = None
        self._coeffs = None

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            # delta & alpha
            nn.init.normal_(self.delta, mean=0.0, std=0.2)
            nn.init.normal_(self.alpha, mean=0.0, std=0.2)
            # beta [1, -1, 1, -1, ...] seems more stable.
            val = torch.ones(self.ndim, 1)
            if self.ndim > 1:
                idx = torch.tensor(list(range(1, self.ndim, 2)))
                val.index_fill_(0, idx, -1.0)
            self.beta.normal_(mean=0.0, std=0.02).add_(val)
            # gamma & omega
            nn.init.normal_(self.gamma, mean=0.0, std=1.0)
            nn.init.normal_(self.omega, mean=0.0, std=1.0)

    def _calc_coeffs(self):
        self._coeffs = None
        # D x N x 1
        p = torch.sigmoid(self.delta)
        alpha = torch.sigmoid(self.alpha)
        q = 1.0 - p * alpha
        return p, q

    def _compute_kernel(self, length: int):
        self._kernel = None
        # D x N x 1
        p, q = self._calc_coeffs()
        # D x N x L
        vander = torch.arange(length).to(p).view(1, 1, length) * torch.log(q)
        kernel = (p * self.beta) * torch.exp(vander)
        # D x L
        return torch.einsum('dnl,dn->dl', kernel, self.gamma * self.scale)

    def coeffs(self):
        if self.training:
            return self._calc_coeffs()
        else:
            if self._coeffs is None:
                self._coeffs = self._calc_coeffs()
            return self._coeffs

    def kernel(self, length: int):
        kernel_size = length if self.truncation is None else min(self.truncation, length)
        if self.training:
            return self._compute_kernel(kernel_size)
        else:
            if self._kernel is None or self._kernel.size(-1) < kernel_size:
                self._kernel = self._compute_kernel(kernel_size)
            return self._kernel[..., :kernel_size]

    def step(self, x, length, hx=None):
        if length == 1:
            return self.one_step(x, hx=hx)

        # D x N x 1
        p, q = self.coeffs()
        # D x N x L+1
        vander = torch.arange(length + 1).to(p).view(1, 1, length + 1) * torch.log(q)
        vander = torch.exp(vander)
        if hx is not None:
            # D x N x L * D x N x 1 -> D x N x L
            k = vander[:, :, 1:] * (self.gamma * self.scale).unsqueeze(-1)
            ox = torch.einsum('bdn,dnl->bdl', hx, k)
            # D x N * B x D x N -> B x D x N
            hh = vander[:, :, -1] * hx
        else:
            ox = None
            hh = None

        # D x N x L
        vander = vander[:, :, :-1]
        kernel = (p * self.beta) * vander
        k = torch.einsum('dnl,dn->dl', kernel, self.gamma * self.scale)

        k_f = torch.fft.rfft(k.float(), n=2 * length)
        x_f = torch.fft.rfft(x.float(), n=2 * length)
        # B x D x L
        out = torch.fft.irfft(x_f * k_f, n=2 * length)[..., 0:length]
        out = out.type_as(x)
        if ox is not None:
            out = out + ox

        h = torch.einsum('bdl,dnl->bdn', x, torch.flip(kernel, dims=[2]))
        if hh is not None:
            h = h + hh
        # L x B x D, B x D x N
        return out.permute(2, 0, 1), h

    def one_step(self, x, hx=None):
        p, q = self.coeffs()
        # (D x N) x (B x D x 1) -> B x D x N
        h = (p * self.beta).squeeze(-1) * x
        if hx is not None:
            h = h + q.squeeze(-1) * hx
        # B x D
        out = torch.einsum('bdn,dn->bd', h, self.gamma * self.scale)
        # 1 x B x D, B x D x N
        return out.unsqueeze(0), h

    def forward(
        self,
        x,
        attention_mask: Optional[torch.Tensor] = None,
        # incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]] = None,
        prev_state: Optional[torch.Tensor] = None,
        use_cache: bool = False
    ) -> torch.Tensor:
        """Input shape: Time x Batch x Channel
        Args:
            padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.

        Refactoring:
            attention_mask: 1 for unmasked, 0 for masked
            prev_state (Tensor, optional): the hidden state returned from 
                the previous timestep if performing incremental decoding
            use_cache (boolean): whether to perform incremental decoding; 
                if True, the hidden state output will be returned
        """

        seq_len, bsz, embed_dim = x.size()
        assert embed_dim == self.embed_dim

        # L x B x D
        residual = x * self.omega

        # L x B x D -> B x D x L
        x = x.permute(1, 2, 0)
        # mask the input: output is a tensor with 0 in the masked positions
        if attention_mask is not None:
            x = x * (attention_mask.unsqueeze(1).type_as(x))

        if self.bidirectional and use_cache:
            raise ValueError('Bidirectional EMA does not support incremental state')

        if use_cache:
            out, updated_state = self.step(x, seq_len, hx=prev_state)

            # B x D -> 1 x B x D
            out = F.silu(out + residual)

            # if incremental decoding, return the new state along with the output
            return out, updated_state
        else:
            # D x L
            k = self.kernel(seq_len)
            fft_len = seq_len
            s = 0
            kernel_size = k.size(1)
            if self.bidirectional:
                k1, k2 = torch.split(k, [self.embed_dim, self.embed_dim], dim=0)
                # D x 2*L-1
                k = F.pad(k1, (kernel_size - 1, 0)) + F.pad(k2.flip(-1), (0, kernel_size - 1))
                x = F.pad(x, (kernel_size - 1, 0))
                fft_len = fft_len + kernel_size - 1
                s = 2 * kernel_size - 2

            k_f = torch.fft.rfft(k.float(), n=2 * fft_len)
            x_f = torch.fft.rfft(x.float(), n=2 * fft_len)
            # B x D x L
            out = torch.fft.irfft(x_f * k_f, n=2 * fft_len)[..., s:s + seq_len]
            out = out.type_as(x)
            # B x D x L -> L x B x D
            out = F.silu(out.permute(2, 0, 1) + residual)

            return out, None

# Gated cross-attention
class GatedCrossAttention(nn.Module):
    """Gated Structured State Attention.
    See "" for more details.
    """

    def __init__(self, config: MegaConfig):
        super().__init__()

        self.config = config
        self.activation = get_activation_fn(activation=self.config.activation)
        self.attention_activation = self.config.attention_activation
        self.scaling = self.config.shared_representation_size ** -0.5 if self.attention_activation == 'softmax' else None

        dropout_module = MegaFeatureDropout if self.config.use_feature_dropout else MegaDropout
        self.dropout = dropout_module(self.config.dropout_prob, module_name=self.__class__.__name__)
        self.hidden_dropout = dropout_module(self.config.hidden_dropout_prob, module_name=self.__class__.__name__)
        # Attention dropout is standard dropout
        self.attention_dropout = MegaDropout(self.config.attention_probs_dropout_prob, module_name=self.__class__.__name__)

        self.prenorm = self.config.normalize_before_mega
        self.norm = SequenceNorm(self.config.normalization_type, self.config.hidden_size, affine=self.config.norm_affine)

        self.k_proj = nn.Linear(self.config.hidden_size, self.config.shared_representation_size)
        self.v_proj = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.q_proj = nn.Linear(self.config.hidden_size, 2 * self.config.hidden_size + self.config.shared_representation_size)
        self.h_proj = nn.Linear(self.config.hidden_size, self.config.hidden_size)

        if self.config.relative_positional_bias == 'simple':
            self.rel_pos_bias = SimpleRelativePositionalBias(self.config.max_positions)
        elif self.config.relative_positional_bias == 'rotary':
            self.rel_pos_bias = RotaryRelativePositionalBias(self.config.shared_representation_size, self.config.max_positions)
        else:
            raise ValueError('unknown relative position bias: {}'.format(self.config.relative_positional_bias))

        self.reset_parameters()
        self.softmax = nn.Softmax(dim=-1)


    def reset_parameters(self):
        std = 0.02
        nn.init.normal_(self.k_proj.weight, mean=0.0, std=std)
        nn.init.constant_(self.k_proj.bias, 0.0)

        nn.init.normal_(self.v_proj.weight, mean=0.0, std=std)
        nn.init.constant_(self.v_proj.bias, 0.0)

        nn.init.normal_(self.q_proj.weight, mean=0.0, std=std)
        nn.init.constant_(self.q_proj.bias, 0.0)

        nn.init.normal_(self.h_proj.weight, mean=0.0, std=std)
        nn.init.constant_(self.h_proj.bias, 0.0)

    def element_attention(self, q, k, key_padding_mask, pidx, before_attn_fn):
        bsz, clen, _ = k.size()
        slen = q.size(1) if pidx is None else pidx + 1
        if key_padding_mask is not None:
            # B x L1 --> B x 1 x 1
            lengths = key_padding_mask.sum(dim=-1).view(bsz, 1, 1)
        else:
            lengths = clen

        # L x L1
        bias = self.rel_pos_bias(max(slen, clen))[:, :clen]
        if pidx is not None:
            assert q.size(1) == 1
            # L1
            bias = bias[pidx]
        else:
            # L2 x L1
            bias = bias[:slen]

        # B x L2 x L1
        qk = torch.bmm(q, k.transpose(1, 2)) / lengths + bias

        if before_attn_fn:
            return qk

        if self.attention_activation == 'relu2':
            attn_weights = relu2(qk).type_as(qk)
        elif self.attention_activation == 'laplace':
            attn_weights = laplace(qk).type_as(qk)
        else:
            raise ValueError('Unknown attention activation function: {}'.format(self.attention_activation))

        if key_padding_mask is not None:
            attn_weights = attn_weights * key_padding_mask.unsqueeze(1)

        return attn_weights

    def softmax_attention(self, q, k, key_padding_mask, pidx, before_attn_fn):
        bsz, clen, _ = k.size()
        slen = q.size(1) if pidx is None else pidx + 1

        # L x L1
        bias = self.rel_pos_bias(max(slen, clen))[:, :clen]
        if pidx is not None:
            assert q.size(1) == 1
            # L1
            bias = bias[pidx]
        else:
            # L2 x L1
            bias = bias[:slen]

        # scaled attention
        q = q * self.scaling
        # B x L2 x L1
        qk = torch.bmm(q, k.transpose(1, 2)) + bias

        if key_padding_mask is not None:
            qk = qk.masked_fill((1 - key_padding_mask).unsqueeze(1).to(torch.bool), float('-inf'))

        if before_attn_fn:
            return qk

        attn_weights = self.softmax(qk).type_as(qk)
        return attn_weights

    def forward(
        self,
        query,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        padding_mask: Optional[torch.Tensor] = None, ## NOT USED
        key_padding_mask: Optional[torch.Tensor] = None,
        prev_key_values: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        static_kv: bool = False,
        before_attn_fn: bool = False,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Time x Batch x Channel
        Args:
            padding_mask (ByteTensor, optional): mask to exclude
                queries that are pads, of shape `(batch, tgt_len)`, where
                padding elements are indicated by 1s. 
                (NOTE: this argument is not used in the original Mega repo and is left with original functionality)
            output_attentions (bool, optional): return the attention weights,
                averaged over heads (default: False).
            static_kv (bool, optional): static key and value pair.
            before_attn_fn (bool, optional): return the raw attention
                weights and values before the attention softmax.

        Refactoring:
            key_padding_mask (optional Tensor): 1 for *not masked*, 0 for *masked*; size (batch size, source length)
            prev_key_values: tuple of tensors for hidden state caching in incremental decoding
            use_cache: whether to return keys and values for incremental decoding
        """

        seq_len, bsz, embed_dim = query.size()
        assert embed_dim == self.config.hidden_size

        if prev_key_values is not None:
            # make sure the inputs only have a sequence length of 1 if we're doing incremental decoding
            if seq_len != 1:
                raise ValueError(f"Incremental decoding requested with self-sequence length > 1: {seq_len}")
            # expect prev_key_values to have (self_key, self_value, self_ema, cross_key, cross_value)
            prev_cross_key, prev_cross_value = prev_key_values[-2:]
            assert static_kv, "Incremental decoding requires static_kv in cross-attention"
            key = value = None
            
            # use the self-attention cache to get the position id of the current step
            prev_self_key = prev_key_values[0]
            num_incremental_steps = prev_self_key.size(1) + 1
        else:
            prev_cross_key = prev_cross_value = None
            # we still need the position id if we're doing incremental decoding (past_key_values will be None for the first step)
            num_incremental_steps = 0 if use_cache else None

        q = query
        if self.prenorm:
            q = self.norm(q)

        # L2 x B x (2*D+S)
        base = self.q_proj(q)
        u, r, q = torch.split(base, [self.config.hidden_size, self.config.hidden_size, self.config.shared_representation_size], dim=-1)

        # L2 x B x D
        u = torch.sigmoid(u)
        r = F.silu(r)

        if key is None:
            assert value is None
            k = v = None
        else:
            # L1 x B x S
            k = self.k_proj(key)
            v = self.activation(self.v_proj(key))

        # L2 x B x S -> B x L2 x S
        q = q.transpose(0, 1)
        if k is not None:
            k = k.transpose(0, 1)
        if v is not None:
            v = v.transpose(0, 1)

        # if we're doing incremental decoding, k and v are None and need to be overwritten with past values
        if prev_key_values is not None:
            k = prev_cross_key
            v = prev_cross_value

        # if we're returning the cache for later use, store these now for later return (can be done without having prev_key_values provided)
        if use_cache:
            updated_cross_key = k 
            updated_cross_value = v

        ctx_len = k.size(1)
        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == ctx_len

        if self.attention_activation == 'softmax':
            attn_weights = self.softmax_attention(q, k, key_padding_mask, num_incremental_steps, before_attn_fn)
        else:
            attn_weights = self.element_attention(q, k, key_padding_mask, num_incremental_steps, before_attn_fn)

        if before_attn_fn:
            return attn_weights, v

        v = self.hidden_dropout(v, batch_first=True)
        kernel = self.attention_dropout(attn_weights)
        # B x L2 x D -> L2 x B x D
        h = torch.bmm(kernel, v).transpose(0, 1)
        # L2 x B x D
        h = self.activation(self.h_proj(h * r))
        h = self.dropout(h)
        out = torch.addcmul(query, u, h - query)

        if not self.prenorm:
            out = self.norm(out)

        outputs = (out, attn_weights) if output_attentions else (out, )
        if use_cache:
            outputs = outputs + (updated_cross_key, updated_cross_value)
        
        return outputs


# Positional embeddings
class SimpleRelativePositionalBias(nn.Module):

    def __init__(self, max_positions):
        super().__init__()
        self.max_positions = max_positions
        self.rel_pos_bias = nn.Parameter(torch.Tensor(2 * max_positions - 1))
        self.reset_parameters()

    def reset_parameters(self):
        std = 0.02
        nn.init.normal_(self.rel_pos_bias, mean=0.0, std=std)

    def forward(self, seq_len):
        if seq_len > self.max_positions:
            raise ValueError('Sequence length {} going beyond max length {}'.format(seq_len, self.max_positions))

        # seq_len * 2 -1
        b = self.rel_pos_bias[(self.max_positions - seq_len):(self.max_positions + seq_len - 1)]
        # seq_len * 3 - 1
        t = F.pad(b, (0, seq_len))
        # (seq_len * 3 - 1) * seq_len
        t = torch.tile(t, (seq_len,))
        t = t[:-seq_len]
        # seq_len x (3 * seq_len - 2)
        t = t.view(seq_len, 3 * seq_len - 2)
        r = (2 * seq_len - 1) // 2
        start = r
        end = t.size(1) - r
        t = t[:, start:end]
        return t

class RotaryRelativePositionalBias(nn.Module):
    def __init__(self, embed_dim, max_positions):
        super().__init__()
        assert embed_dim % 2 == 0
        self.embed_dim = embed_dim
        self.max_positions = max_positions
        self.sine, self.cosine = RotaryRelativePositionalBias.get_sinusoid_embeddings(max_positions, embed_dim)
        self.alpha = nn.Parameter(torch.Tensor(1, embed_dim))
        self.beta = nn.Parameter(torch.Tensor(1, embed_dim))
        self.register_buffer("_float_tensor", torch.FloatTensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        std = 0.02
        nn.init.normal_(self.alpha, mean=0.0, std=std)
        nn.init.normal_(self.beta, mean=0.0, std=std)

    @staticmethod
    def get_sinusoid_embeddings(max_positions: int, embedding_dim: int):
        half_dim = embedding_dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(max_positions, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        return torch.sin(emb), torch.cos(emb)

    def rotary(self, x):
        n, d = x.size()
        x1, x2 = torch.chunk(x, 2, dim=-1)
        if self.sine is None or n > self.sine.size(0):
            self.sine, self.cosine = RotaryRelativePositionalBias.get_sinusoid_embeddings(n, d)
            self.max_positions = n
        self.sine = self.sine.to(self._float_tensor)
        self.cosine = self.cosine.to(self._float_tensor)

        sin = self.sine[:n]
        cos = self.cosine[:n]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=1)

    def forward(self, seq_len):
        a = self.rotary(self.alpha.expand(seq_len, self.embed_dim))
        b = self.rotary(self.beta.expand(seq_len, self.embed_dim))
        t = torch.einsum('mk,nk->mn', a, b)
        return t

# Normalization modules
class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, affine=True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.affine = affine
        if affine:
            self.scalar = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('scalar', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.constant_(self.scalar, 1.0)

    def forward(self, x):
        mean_square = torch.mean(torch.square(x), dim=self.dim, keepdim=True)
        if self.scalar is not None:
            x = self.scalar * x

        x = x * torch.rsqrt(mean_square + self.eps)
        return x

class RMSNorm(nn.Module):
    def __init__(self, number_features, eps=1e-6, affine=True):
        super().__init__()
        self.num_features = number_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.Tensor(self.num_features))
        else:
            self.register_parameter('weight', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.constant_(self.weight, 1.0)

    def forward(self, x):
        mean_square = torch.mean(torch.square(x), dim=-1, keepdim=True)
        if self.weight is not None:
            x = x * self.weight

        x = x * torch.rsqrt(mean_square + self.eps)
        return x

class SequenceNorm(nn.Module):
    def __init__(self, norm_type, embedding_dim, eps=1e-5, affine=True, export=False):
        super().__init__()
        if norm_type == 'layernorm':
          self.norm = nn.LayerNorm(embedding_dim, eps, elementwise_affine=affine)
        elif norm_type == 'scalenorm':
            self.norm = ScaleNorm(dim=-1, eps=eps, affine=affine)
        elif norm_type == 'rmsnorm':
            self.norm = RMSNorm(embedding_dim, eps=eps, affine=affine)
        elif norm_type == 'batchnorm':
            self.norm = nn.BatchNorm1d(embedding_dim, eps=eps, affine=affine)
        elif norm_type == 'syncbatchnorm':
            self.norm = nn.SyncBatchNorm(embedding_dim, eps=eps, affine=affine)
        else:
            raise ValueError('Unknown norm type: {}'.format(norm_type))

    def normalize(self, x):
        if isinstance(self.norm, nn.modules.batchnorm._BatchNorm):
            assert x.dim() == 3
            x = x.permute(1, 2, 0)
            x = self.norm(x)
            return x.permute(2, 0, 1)
        else:
            return self.norm(x)

    def forward(self, x):
        return self.normalize(x)

# Dropout: standard dropout + feature dropout
class MegaDropout(nn.Module):

    def __init__(self, p, module_name=None):
        super().__init__()
        self.p = p
        self.module_name = module_name
        self.apply_during_inference = False

    def forward(self, x, batch_first: bool = False, inplace: bool = False):
        if self.training or self.apply_during_inference:
            return F.dropout(x, p=self.p, training=True, inplace=inplace)
        else:
            return x

    def make_generation_fast_(
        self,
        name: str,
        retain_dropout: bool = False,
        retain_dropout_modules: Optional[List[str]] = None,
        **kwargs
    ):
        if retain_dropout:
            if retain_dropout_modules is not None and self.module_name is None:
                logger.warning(
                    'Cannot enable dropout during inference for module {} '
                    'because module_name was not set'.format(name)
                )
            elif (
                retain_dropout_modules is None  # if None, apply to all modules
                or self.module_name in retain_dropout_modules
            ):
                logger.info(
                    'Enabling dropout during inference for module: {}'.format(name)
                )
                self.apply_during_inference = True
            else:
                logger.info('Disabling dropout for module: {}'.format(name))

class MegaFeatureDropout(nn.Module):

    def __init__(self, p, module_name=None):
        super().__init__()
        self.p = p
        self.module_name = module_name
        self.apply_during_inference = False

    def forward(self, x, batch_first: bool = False, inplace: bool = False):
        if self.training or self.apply_during_inference:
            if batch_first:
                # B x L x D -> B x D x L -> B x L x D
                return F.dropout2d(x.transpose(-1, -2), p=self.p, training=True, inplace=inplace).transpose(-1, -2)
            else:
                assert x.dim() == 3
                # L x B x D -> B x D x L -> L x B x D
                return F.dropout2d(x.permute(1, 2, 0), p=self.p, training=True, inplace=inplace).permute(2, 0, 1)
        else:
            return x

    def make_generation_fast_(
        self,
        name: str,
        retain_dropout: bool = False,
        retain_dropout_modules: Optional[List[str]] = None,
        **kwargs
    ):
        if retain_dropout:
            if retain_dropout_modules is not None and self.module_name is None:
                logger.warning(
                    'Cannot enable dropout during inference for module {} '
                    'because module_name was not set'.format(name)
                )
            elif (
                retain_dropout_modules is None  # if None, apply to all modules
                or self.module_name in retain_dropout_modules
            ):
                logger.info(
                    'Enabling dropout during inference for module: {}'.format(name)
                )
                self.apply_during_inference = True
            else:
                logger.info('Disabling dropout for module: {}'.format(name))

# Mega attention: EMA + self-attention
class MovingAverageGatedAttention(nn.Module):
    """
    Pure PyTorch implementation of Mega block; see https://arxiv.org/abs/2209.10655 
    and original fairseq implementation at https://github.com/facebookresearch/mega
    (copyright Meta Research, licensed under MIT License)
    """
    def __init__(self, config : MegaConfig):
        super().__init__()
        self.config = config
        self.activation = get_activation_fn(self.config.activation)
        self.scaling = self.config.shared_representation_size ** -0.5 if self.config.attention_activation == 'softmax' else None 
        dropout_module = MegaFeatureDropout if self.config.use_feature_dropout else MegaDropout
        self.dropout = dropout_module(self.config.dropout_prob, module_name=self.__class__.__name__)
        self.hidden_dropout = dropout_module(self.config.hidden_dropout_prob, module_name=self.__class__.__name__)
        # attention dropout is standard dropout
        self.attention_dropout = MegaDropout(self.config.attention_probs_dropout_prob, module_name=self.__class__.__name__)

        self.norm = SequenceNorm(self.config.normalization_type, self.config.hidden_size, affine=self.config.norm_affine)
        self.move = MultiHeadEMA(self.config.hidden_size, 
                                ndim=self.config.ema_projection_size, 
                                bidirectional=self.config.bidirectional, 
                                truncation=self.config.truncation)
        
        self.v_proj = nn.Linear(self.config.hidden_size, self.config.intermediate_size)
        self.mx_proj = nn.Linear(self.config.hidden_size, self.config.shared_representation_size + self.config.intermediate_size + 2 * self.config.hidden_size)
        self.h_proj = nn.Linear(self.config.intermediate_size, self.config.hidden_size)

        self.gamma = nn.Parameter(torch.Tensor(2, self.config.shared_representation_size))
        self.beta = nn.Parameter(torch.Tensor(2, self.config.shared_representation_size))

        max_positions = self.config.max_positions if self.config.chunk_size < 0 else self.config.chunk_size 
        if self.config.relative_positional_bias == 'simple':
            self.rel_pos_bias = SimpleRelativePositionalBias(max_positions)
        elif self.config.relative_positional_bias == 'rotary':
            self.rel_pos_bias = RotaryRelativePositionalBias(self.config.shared_representation_size, max_positions)
        else:
            raise ValueError(f"Unknown relative positional bias: {self.config.relative_positional_bias}")

        self.softmax = nn.Softmax(dim=-1)
        self.attention_function = self.softmax_attention if self.config.attention_activation == 'softmax' else self.element_attention

        self.reset_parameters()

    def reset_parameters(self):
        # possibly remove this in favor of automated initialization by HF classes
        std = 0.02
        mean = 0.0
        nn.init.normal_(self.v_proj.weight, mean=mean, std=std)
        nn.init.constant_(self.v_proj.bias, 0.0)
        
        nn.init.normal_(self.mx_proj.weight, mean=mean, std=std)
        nn.init.constant_(self.mx_proj.bias, 0.0)
        
        nn.init.normal_(self.h_proj.weight, mean=mean, std=std)
        nn.init.constant_(self.h_proj.bias, 0.0)
        
        nn.init.normal_(self.gamma, mean=mean, std=std)
        nn.init.constant_(self.beta, 0.0)
    
    def element_attention(self, q, k, padding_mask, causal_mask, before_attn_fn):
        '''
        Apply element-wise attention via relu^2 or laplace. Same as original 
        implementation but with standardized causal attention mask

        padding_mask: tensor with shape (batch x no. chunks x chunk size x sequence length), or None
        causal_mask: tensor with shape (sequence length x sequence length), or None

        Both masks expect 1 to indicate *not masked* and 0 to indicate *masked*
        '''
        slen = k.size(2)
        if padding_mask is not None:
            # 1 for *not masked*
            # 0 for *masked*

            # B x K x 1
            lengths = padding_mask.sum(-1, keepdim=True)
            # B x K x 1 x 1
            lengths = lengths.clamp(min=1.0).unsqueeze(-1)
        
        if causal_mask is not None:
            lengths = causal_mask.sum(dim=-1, keepdim=True)

        # C x C
        bias = self.rel_pos_bias(slen)
        if slen != q.size(2):
            assert q.size(2) == 1, "Size mismatch between Q and K in element attention"
            # 1 x C
            bias = bias[-1:]
    
        # B x K x C x C
        qk = torch.matmul(q, k.transpose(2, 3)) / lengths + bias 

        if before_attn_fn:
            return qk 
    
        if self.config.attention_activation == 'relu2':
            attn_weights = relu2(qk).type_as(qk)
        elif self.config.attention_activation == 'laplace':
            attn_weights = laplace(qk).type_as(qk)
        else:
            raise ValueError(f"Unknown attention activation function: {self.config.attention_activation}")
    
        if padding_mask is not None:
            attn_weights = attn_weights * padding_mask.unsqueeze(2) 

        if causal_mask is not None:
            attn_weights = attn_weights * causal_mask
    
        return attn_weights

    def softmax_attention(self, q, k, padding_mask, causal_mask, before_attn_fn):
        'Standard softmax attention with combined padding/attention mask'
        slen = k.size(2)
        # C x C
        bias = self.rel_pos_bias(slen)
        if slen != q.size(2):
            assert q.size(2) == 1, "Size mismatch between Q and K in softmax attention"
            # 1 x C
            bias = bias[-1:] 

        # scaled attention
        q = q * self.scaling 

        # B x K x C x C (if chunking)
        # B x 1 x S x S (otherwise)
        qk = torch.matmul(q, k.transpose(2, 3)) + bias 

        # apply causal mask (presumed to be 1/0 for not masked / masked)
        # additive, but convert to 0/-inf (which is not explicitly in the Mega source code)
        if causal_mask is not None:
            additive_causal_mask = torch.zeros_like(causal_mask, dtype=torch.float)
            additive_causal_mask = additive_causal_mask.masked_fill((1 - causal_mask).bool(), float("-inf"))
            qk = qk + additive_causal_mask

        if padding_mask is not None:
            # 1 for tokens which are *not masked*
            # 0 for tokens which are *masked*
            # replace masked tokens with -inf to make softmax ignore them
            padding_mask_all = padding_mask.all(dim=-1, keepdim=True)
            padding_mask = torch.logical_and(padding_mask, ~padding_mask_all)
            qk = qk.masked_fill(padding_mask.unsqueeze(2).to(torch.bool), float('-inf'))

        if before_attn_fn:
            return qk
    
        attn_weights = self.softmax(qk).type_as(qk)
        return attn_weights 

    def forward(
        self, 
        x, 
        padding_mask: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None,
        prev_key_values: Optional[Tuple[torch.Tensor]] = None,
        output_attentions = False,
        before_attn_fn = False,
        use_cache=False
    ):
        '''
        Mega self-attention block, combining multi-headed EMA with traditional attention

        x: embeddings on which to perform Mega attention, shape = (time, batch, embedding size)
        padding_mask (optional): 1 if unmasked, 0 if masked; shape = (batch size, time)
        causal_mask (optional): 1 if unmasked, 0 if masked; shape = (time, time)
        prev_key_values (optional): a tuple of tensors representing past state for
            incremental decoding - (self_key, self_value, self_ema_state)
        output_attentions: if True, attention weights will be returned 
        before_attn_fn: whether to return attention values before softmax (retained from original Mega)
        use_cache: whether you are performing incremental decoding - if True, 
            states will be returned for use in the next step
        '''
        seq_len, bsz, embed_dim = x.size()
        assert embed_dim == self.config.hidden_size, f"Input embedding dimension should be {self.config.hidden_size}; received {embed_dim}"

        # store inputs for residual connection and handle pre-norm if requested
        residual = x 
        if self.config.normalize_before_mega:
            x = self.norm(x)
    
        # L x B x E
        v = self.activation(self.v_proj(x))

        # unpack the incremental state if provided
        # assumed to be (self K, self V, self EMA state, cross K, cross V)
        # also assumes that incremental decoding is working one token at a time, so input sequence length must be 1
        if self.config.is_decoder and (prev_key_values is not None):
            if seq_len > 1:
                raise ValueError(f"Incremental decoding only supports self sequence length of 1; received {seq_len}")
            # the first 3 items in the saved states will be these regardless of whether cross-attention is present
            prev_self_key, prev_self_value, prev_ema_state = prev_key_values[0:3]
        else:
            prev_self_key = prev_self_value = prev_ema_state = None 

        # L x B x D
        # updated_ema_state will be None if use_cache=False
        mx, updated_ema_state = self.move(x, attention_mask=padding_mask, prev_state=prev_ema_state, use_cache=use_cache)
        mx = self.dropout(mx)

        # L x B x D -> L x B x (2*D+S+E)
        base = self.mx_proj(mx)
        u, zr, hx = torch.split(
            base, [
                self.config.hidden_size, 
                self.config.shared_representation_size + self.config.intermediate_size, 
                self.config.hidden_size
            ], 
            dim=-1
        )

        # L x B x D
        u = torch.sigmoid(u)

        # L x B x (E + S)
        z, r = torch.split(F.silu(zr), [self.config.shared_representation_size, self.config.intermediate_size], dim=-1)

        # L x B x S -> L x B x 1 x S -> L x B x 2 X S 
        z = z.unsqueeze(2) * self.gamma + self.beta
        
        # L x B x 2 x S -> L x B x S 
        q, k = torch.unbind(z, dim=2)

        # L x B x D -> B x L x D
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        if self.config.is_decoder:
            # combine history and current to save updated state (if history is provided)
            # when chunking is applied, the past states will be None at the end of the chunk, in 
            # which case, proceed as if no K/V history had been provided
            # saved states are stored with shape (bsz, seq_len, dim)
            if prev_self_key is not None:
                k = torch.cat([prev_self_key, k], dim=1)
            if prev_self_value is not None:
                v = torch.cat([prev_self_value, v], dim=1)

            # if not chunking, store as-is
            if not self.config.use_chunking:
                updated_self_key = k 
                updated_self_value = v
            else:
                curr_len = k.size(1) % self.config.chunk_size 
                if curr_len == 0:
                    # if we're chunking and have reached the end of a chunk, wipe out the saved state
                    updated_self_key = None 
                    updated_self_value = None
                else:
                    updated_self_key = k 
                    updated_self_value = v
    
        ctx_len = k.size(1)
        if not self.config.use_chunking:
            # if we're not chunking, treat the entire sequence as one long chunk
            # B x L x S -> B x 1 x L x S
            q = q.unsqueeze(1)
            k = k.unsqueeze(1)
            v = v.unsqueeze(1)
            if padding_mask is not None:
                # B x L -> B x 1 x L
                padding_mask = padding_mask.unsqueeze(1)
        else:
            # otherwise, split the sequences in the batch into K chunks of size C
            if seq_len < self.config.chunk_size:
                q = q.unsqueeze(1)
            else:
                # B x L x S -> B x K x C x S
                nc = seq_len // self.config.chunk_size 
                q = q.reshape(bsz, nc, self.config.chunk_size, self.config.shared_representation_size)
      
            if ctx_len < self.config.chunk_size:
                k = k.unsqueeze(1)
                v = v.unsqueeze(1)
                if padding_mask is not None:
                    padding_mask = padding_mask.unsqueeze(1)
            else:
                # B x L x S -> B x K x C x S
                nc = ctx_len // self.config.chunk_size 
                k = k.reshape(bsz, nc, self.config.chunk_size, self.config.shared_representation_size)
                v = v.reshape(bsz, nc, self.config.chunk_size, self.config.intermediate_size)
                if padding_mask is not None:
                    padding_mask = padding_mask.view(bsz, nc, self.config.chunk_size)
        
        # this is in the original Mega implementation to work around fork/join parallelism not supporting optional types
        if padding_mask is not None and padding_mask.dim() == 0:
            padding_mask = None 

        attn_weights = self.attention_function(q, k, 
                                               padding_mask=padding_mask, 
                                               causal_mask=causal_mask,
                                               before_attn_fn=before_attn_fn)
    
        v = self.hidden_dropout(v, batch_first=True)
        kernel = self.attention_dropout(attn_weights)

        # B x K x C x E -> B x L x E -> L x B x E
        h = torch.matmul(kernel, v).view(bsz, seq_len, self.config.intermediate_size).transpose(0, 1)

        # L x B x E -> L x B x D
        h = self.activation(hx + self.h_proj(h * r))
        h = self.dropout(h)
        # L x B x D
        out = torch.addcmul(residual, u, h - residual)

        if not self.config.normalize_before_mega:
            out = self.norm(out)

        return_values = (out, attn_weights) if output_attentions else (out, )

        if self.config.is_decoder:
            return_values = return_values + (updated_self_key, updated_self_value, updated_ema_state)
    
        return return_values

# Normalized feed-forward network
class NormalizedFeedForwardNetwork(nn.Module):
    def __init__(self, config : MegaConfig):
        super().__init__()

        self.config = config
        self.hidden_dim = config.nffn_hidden_size
        self.act_fn = config.activation
        self.activation = get_activation_fn(config.activation)

        dropout_module = MegaFeatureDropout if self.config.use_feature_dropout else MegaDropout
        self.dropout = dropout_module(self.config.dropout_prob, module_name=self.__class__.__name__)
        self.hidden_dropout = dropout_module(self.config.nffn_activation_dropout_prob, module_name=self.__class__.__name__)

        self.prenorm = self.config.normalize_before_ffn
        self.norm = SequenceNorm(self.config.normalization_type, self.config.hidden_size, affine=self.config.norm_affine)

        self.fc1 = nn.Linear(self.config.hidden_size, self.config.nffn_hidden_size)
        self.fc2 = nn.Linear(self.config.nffn_hidden_size, self.config.hidden_size)

        self.reset_parameters()

    def reset_parameters(self):
        std = 0.02
        nn.init.normal_(self.fc1.weight, mean=0.0, std=std)
        nn.init.constant_(self.fc1.bias, 0.0)

        nn.init.normal_(self.fc2.weight, mean=0.0, std=std)
        nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, x):
        residual = x

        if self.prenorm:
            x = self.norm(x)

        x = self.activation(self.fc1(x))
        x = self.hidden_dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = x + residual

        if not self.prenorm:
            x = self.norm(x)

        return x

# Mega Encoder Layer
# Mega block + NFFN wtih dropout and normalization
# NOTE: this can probably be combined with the decoder layer into a single 
class MegaEncoderLayer(nn.Module):
    """Encoder layer block.
    Args:
        args (argparse.Namespace): parsed command-line arguments
    """
    def __init__(self, config: MegaConfig):
        super().__init__()
        self.mega_layer = MovingAverageGatedAttention(config)
        self.nffn = NormalizedFeedForwardNetwork(config) if config.use_normalized_ffn else None

    def forward(self, x, encoder_padding_mask):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        x, _ = self.mega_layer(x, encoder_padding_mask)
        if self.nffn is not None:
            x = self.nffn(x)
        return x

# Mega Decoder Layer
# same thing, but includes optional cross-attention
class MegaDecoderLayer(nn.Module):
    """
    Mega layer block for decoder

    Accepts a `MegaConfig` which could be different from the config used for
    any relevant encoders.
    """

    def __init__(self, config: MegaConfig, no_cross_attention: bool = False):
        super().__init__()
        self.mega_layer = MovingAverageGatedAttention(config)
        self.cross_attn = None if no_cross_attention else GatedCrossAttention(config)
        self.nffn = NormalizedFeedForwardNetwork(config) if config.use_normalized_ffn else None


    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]] = None,
        attn_mask: Optional[torch.Tensor] = None,
        decoder_padding_mask: Optional[torch.Tensor] = None,
        return_attention_weights = False
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_out (Tensor): encoder out for cross attention `(src_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary ByteTensor of shape `(batch, src_len)` where padding elements are indicated by ``1``.
            incremental_state: dictionary for caching incremental states.
            attn_mask (Tensor): attention mask for autoregressive decoding.
            decoder_padding_mask: padding mask for target sequence.
            need_attn (bool, optional): return attention weights.
        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """

        x, attn_weights = self.mega_layer(x=x, 
                                        padding_mask=decoder_padding_mask, 
                                        incremental_state=incremental_state,
                                        return_attention_weights=False,
                                        attn_mask=attn_mask)
        
        if self.cross_attn is not None:
            x, attn_weights = self.cross_attn(query=x,
                                                key=encoder_out, 
                                                value=encoder_out,
                                                padding_mask=decoder_padding_mask,
                                                key_padding_mask=encoder_padding_mask,
                                                incremental_state=incremental_state,
                                                static_kv=True, 
                                                need_weights=return_attention_weights)
        
        if self.nffn is not None:
            x = self.nffn(x)
            
        return x, attn_weights, None 

# NOTE: ROBERTA STUFF BEGINS HERE !!!
class MegaEmbeddings(nn.Module):
    """
    Mega's basic implementation does not incorporate token type embeddings, so this is
    a stripped-down version of RoBERTa's embeddings which optionally includes token types
    """

    def __init__(self, config: MegaConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.use_token_types = config.add_token_type_embeddings
        if self.use_token_types:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
            self.register_buffer(
                "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
            )

        # End copy
        self.padding_idx = config.pad_token_id

    def forward(
        self, input_ids=None, token_type_ids=None, inputs_embeds=None
    ):
        if (input_ids is None) and (inputs_embeds is None):
            raise ValueError("Must provide one of input_ids or inputs_embeds")
        elif input_ids is not None:
            input_shape = input_ids.size()
            
            # get the word embeddings if only IDs are provided
            inputs_embeds = self.word_embeddings(input_ids)
        else:
            input_shape = inputs_embeds.size()[:-1]

        # the original Mega implementation did not include token type embeddings, so we add
        # an option to use them if desired
        if self.use_token_types:
            # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
            # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
            # issue #5664
            if token_type_ids is None:
                if hasattr(self, "token_type_ids"):
                    seq_length = input_shape[1]
                    buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                    buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                    token_type_ids = buffered_token_type_ids_expanded
                else:
                    token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

            # access token type embeddings
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            # add the token type embeddings to the word embeddings
            embeddings = inputs_embeds + token_type_embeddings
        else:
            embeddings = inputs_embeds
        return embeddings

# Copied from transformers.models.bert.modeling_bert.BertLayer with Bert->Mega
class MegaLayer(nn.Module):
    def __init__(self, config: MegaConfig):
        super().__init__()
        self.seq_len_dim = 1
        self.mega_layer = MovingAverageGatedAttention(config)
        self.nffn = NormalizedFeedForwardNetwork(config) if config.use_normalized_ffn else None
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.cross_attn = GatedCrossAttention(config)
        else:
            self.cross_attn = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[torch.FloatTensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor]:

        # Mega self-attention
        # create a causal mask for self-attention if we're decoding
        if self.is_decoder:
            sequence_length = hidden_states.size(0)
            causal_mask = generate_causal_mask(sequence_length)
        else:
            causal_mask = None
        mega_outputs = self.mega_layer(x=hidden_states, 
                                       padding_mask=attention_mask, 
                                       causal_mask=causal_mask,
                                       prev_key_values=past_key_value, 
                                       output_attentions=output_attentions, 
                                       before_attn_fn=False, 
                                       use_cache=use_cache)

        new_hidden_states = mega_outputs[0]
        self_key, self_value, self_ema_state = mega_outputs[-3:] if use_cache else (None, None, None)
        self_attention_weights = mega_outputs[1] if output_attentions else None
        
        # optional cross attention
        if self.cross_attn is not None:
            if encoder_hidden_states is None:
                raise ValueError(f"Requested cross-attention without providing encoder hidden states")
            
            cross_attn_outputs = self.cross_attn(query=new_hidden_states,
                                                 key=encoder_hidden_states,
                                                 value=encoder_hidden_states,
                                                 padding_mask=None,
                                                 key_padding_mask=encoder_attention_mask,
                                                 prev_key_values=past_key_value,
                                                 output_attentions=output_attentions,
                                                 static_kv=True,
                                                 before_attn_fn=False,
                                                 use_cache=use_cache)
            # update the hidden state from cross attention
            new_hidden_states = cross_attn_outputs[0]
            # store cross-attention k/v if caching
            cross_key, cross_value = cross_attn_outputs[-2:] if use_cache else (None, None)
            cross_attention_weights = cross_attn_outputs[1] if output_attentions else None

        # optional NFFN follows cross attention
        if self.nffn is not None:
            new_hidden_states = self.nffn(new_hidden_states)

        outs = (new_hidden_states,) 
        if output_attentions:
            outs = outs + (self_attention_weights,)
            if self.cross_attn is not None:
                outs = outs + (cross_attention_weights, )
        
        if use_cache:
            new_key_values = (self_key, self_value, self_ema_state, )
            if self.cross_attn is not None:
                new_key_values = new_key_values + (cross_key, cross_value)
            
            outs = outs + (new_key_values,)
        
        return outs


# Copied from transformers.models.bert.modeling_bert.BertEncoder with Bert->Mega
class MegaEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([MegaLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


# Copied from transformers.models.bert.modeling_bert.BertPooler
class MegaPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class MegaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MegaConfig
    base_model_prefix = "mega"
    supports_gradient_checkpointing = False
    _no_split_modules = []

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, MegaEncoder):
            module.gradient_checkpointing = value

    def update_keys_to_ignore(self, config, del_keys_to_ignore):
        """Remove some keys from ignore list"""
        if not config.tie_word_embeddings:
            # must make a new list, or the class variable gets modified!
            self._keys_to_ignore_on_save = [k for k in self._keys_to_ignore_on_save if k not in del_keys_to_ignore]
            self._keys_to_ignore_on_load_missing = [
                k for k in self._keys_to_ignore_on_load_missing if k not in del_keys_to_ignore
            ]


MEGA_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MegaConfig`]): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

MEGA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.
            This parameter can only be used when the model is initialized with `type_vocab_size` parameter with value
            >= 2. All the value in this tensor should be always < type_vocab_size.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
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
    "The bare Mega Model transformer outputting raw hidden-states without any specific head on top.",
    MEGA_START_DOCSTRING,
)
class MegaModel(MegaPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in *Attention is
    all you need*_ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
    Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.

    .. _*Attention is all you need*: https://arxiv.org/abs/1706.03762

    """

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = MegaEmbeddings(config)
        self.encoder = MegaEncoder(config)

        self.pooler = MegaPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(MEGA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


@add_start_docstrings(
    """Mega Model with a `language modeling` head on top for CLM fine-tuning.""", MEGA_START_DOCSTRING
)
class MegaForCausalLM(MegaPreTrainedModel):
    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        if not config.is_decoder:
            logger.warning("If you want to use `MegaLMHeadModel` as a standalone, add `is_decoder=True.`")

        self.mega = MegaModel(config, add_pooling_layer=False)
        self.lm_head = MegaLMHead(config)

        # The LM head weights require special treatment only when they are tied with the word embeddings
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(MEGA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Tuple[Tuple[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, MegaForCausalLM, AutoConfig
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("mnaylor/mega-base-wikitext")
        >>> config = AutoConfig.from_pretrained("mnaylor/mega-base-wikitext")
        >>> config.is_decoder = True
        >>> model = MegaForCausalLM.from_pretrained("mnaylor/mega-base-wikitext", config=config)

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> prediction_logits = outputs.logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        outputs = self.mega(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=prediction_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values}

    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past


@add_start_docstrings("""Mega Model with a `language modeling` head on top.""", MEGA_START_DOCSTRING)
class MegaForMaskedLM(MegaPreTrainedModel):
    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `MegaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.mega = MegaModel(config, add_pooling_layer=False)
        self.lm_head = MegaLMHead(config)

        # The LM head weights require special treatment only when they are tied with the word embeddings
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(MEGA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        mask="<mask>",
        expected_output="' Paris'",
        expected_loss=0.1,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.mega(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MegaLMHead(nn.Module):
    """Mega Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        # For accelerate compatibility and to not break backward compatibility
        if self.decoder.bias.device.type == "meta":
            self.decoder.bias = self.bias
        else:
            self.bias = self.decoder.bias


@add_start_docstrings(
    """
    Mega Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    MEGA_START_DOCSTRING,
)
class MegaForSequenceClassification(MegaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.mega = MegaModel(config, add_pooling_layer=False)
        self.classifier = MegaClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(MEGA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="cardiffnlp/twitter-mnaylor/mega-base-wikitext-emotion",
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="'optimism'",
        expected_loss=0.08,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.mega(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
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
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Mega Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    MEGA_START_DOCSTRING,
)
class MegaForMultipleChoice(MegaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)

        self.mega = MegaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(MEGA_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.mega(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Mega Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    MEGA_START_DOCSTRING,
)
class MegaForTokenClassification(MegaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.mega = MegaModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(MEGA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="Jean-Baptiste/mega-large-ner-english",
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="['O', 'ORG', 'ORG', 'O', 'O', 'O', 'O', 'O', 'LOC', 'O', 'LOC', 'LOC']",
        expected_loss=0.01,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.mega(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MegaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


@add_start_docstrings(
    """
    Mega Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    MEGA_START_DOCSTRING,
)
class MegaForQuestionAnswering(MegaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.mega = MegaModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(MEGA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="deepset/mnaylor/mega-base-wikitext-squad2",
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="' puppet'",
        expected_loss=0.86,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.mega(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )