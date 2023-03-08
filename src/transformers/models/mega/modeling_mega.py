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
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...activations import ACT2FN
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
# notable differences from the original implementation:
#   - refactored away from stateful representation of incremental decoding
#     state in favor of Hugging Face's typical `past_key_values`
#   - fixed inconsistency in how causal masks are expected by `softmax_attention`
#     and `element_attention` in MovingAverageGatedAttention (see https://github.com/facebookresearch/mega/issues/11)
#   - added support for token type embeddings (not specifically included
#     or excluded in the original implementation/paper)


# starting with activation functions
# squared-relu and laplace are alternatives to softmax for attention activation
def relu2(x):
    relu_x = F.relu(x)
    squared = torch.square(relu_x)
    return squared

def laplace(x, mu=0.707107, sigma=0.282095):
    x = (x - mu).div(sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + torch.erf(x))

# gelu-accurate is an alternative to gelu and is used with the remaining hidden activation functions in the 
# original MEGA repo, which all have exact equivalents in ACT2FN
def gelu_accurate(x):
    if not hasattr(gelu_accurate, "_a"):
        gelu_accurate._a = math.sqrt(2 / math.pi)
    return 0.5 * x * (1 + torch.tanh(gelu_accurate._a * (x + 0.044715 * torch.pow(x, 3))))

ACT2FN['gelu_accurate'] = gelu_accurate

# utility for causal LM masking in the format that Mega expects
def generate_causal_mask(seq_len):
    """
    Tiny utility to generate a `seq_len` by `seq_len` causal mask, where 1 corresponds to *not masked* and 0
    corresponds to *masked*

    causal_mask[i][j] corresponds to whether token `i` can attend to token `j`
    """
    seq_ids = torch.arange(seq_len)
    causal_mask = seq_ids[None, :].repeat(seq_len, 1) <= seq_ids[:, None]
    return causal_mask.to(torch.long)


# EMA attention module
# largely left unmodified except the incremental state
class MultiHeadEMA(nn.Module):
    """Exponential Moving Average Layer.
    See "https://arxiv.org/abs/2209.10655" for more details.
    """

    def __init__(self, config: MegaConfig):
        super().__init__()

        self.config = config

        self.embed_dim = config.hidden_size
        self.ndim = config.ema_projection_size
        self.bidirectional = config.bidirectional
        self.truncation = config.truncation
        self.scale = math.sqrt(1.0 / self.ndim)

        kernel_dim = 2 * config.hidden_size if self.bidirectional else config.hidden_size
        self.delta = nn.Parameter(torch.Tensor(kernel_dim, self.ndim, 1))
        self.alpha = nn.Parameter(torch.Tensor(kernel_dim, self.ndim, 1))
        # renamed gamma and beta to g_param and b_param respectively to avoid HF renaming things
        self.b_param = nn.Parameter(torch.Tensor(kernel_dim, self.ndim, 1))
        self.g_param = nn.Parameter(torch.Tensor(kernel_dim, self.ndim))
        self.omega = nn.Parameter(torch.Tensor(config.hidden_size))
        self._kernel = None
        self._coeffs = None

    def _calc_coeffs(self):
        self._coeffs = None
        # convert the alpha and delta parameters (kernel_dim x EMA projection size x 1) to [0, 1] with sigmoid
        p_coeff = torch.sigmoid(self.delta)
        alpha = torch.sigmoid(self.alpha)
        q_coeff = 1.0 - p_coeff * alpha
        return p_coeff, q_coeff

    def _compute_kernel(self, length: int):
        self._kernel = None
        # p and q have shape (kernel_dim x ema_projection_size x 1)
        p_coeff, q_coeff = self._calc_coeffs()
        # extend the kernel to (kernel_dim X ema_projection_size X sequence_length) and 
        # multiply q by sequential ints up to the sequence length
        vander = torch.arange(length).to(p_coeff).view(1, 1, length) * torch.log(q_coeff)
        kernel = (p_coeff* self.b_param) * torch.exp(vander)
        # (kernel_dim X ema_projection_size X sequence_length) -> (kernel_dim, sequence_length)
        return torch.einsum("dnl,dn->dl", kernel, self.g_param * self.scale)

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

    def step(self, inputs, length, past_state=None):
        if length == 1:
            return self.one_step(inputs, past_state=past_state)

        # (kernel_dim X ema_projection_size X 1)
        p_coeff, q_coeff = self.coeffs()
        # (kernel_dim X ema_projection_size X 1+sequence_length)
        vander = torch.arange(length + 1).to(p_coeff).view(1, 1, length + 1) * torch.log(q_coeff)
        vander = torch.exp(vander)
        if past_state is not None:
            # (kernel_dim X ema_projection_size X sequence_length) * (kernel_dim X ema_projection_size X 1) 
            # -> (kernel_dim X ema_projection_size X sequence_length)
            past_ema_proj = vander[:, :, 1:] * (self.g_param * self.scale).unsqueeze(-1)
            # past_state will be (batch_size, kernel_dim, ema_projection_size)
            past_ema_state = torch.einsum("bdn,dnl->bdl", past_state, past_ema_proj)
            # (kernel_dim X ema_projection_size) * (batch_size X kernel_dim X ema_projection_size) 
            # -> (batch_size X kernel_dim X ema_projection_size)
            past_vandermonde = vander[:, :, -1] * past_state
        else:
            past_ema_state = None
            past_vandermonde = None

        # (kernel_dim X ema_projection_size X sequence_length)
        vander = vander[:, :, :-1]
        kernel = (p_coeff * self.b_param) * vander
        kernel_proj = torch.einsum("dnl,dn->dl", kernel, self.g_param * self.scale)

        kernel_fourier = torch.fft.rfft(kernel_proj.float(), n=2 * length)
        inputs_fourier = torch.fft.rfft(inputs.float(), n=2 * length)
        # (batch_size X kernel_dim X sequence_length)
        out = torch.fft.irfft(inputs_fourier * kernel_fourier, n=2 * length)[..., 0:length]
        out = out.type_as(inputs)
        if past_ema_state is not None:
            out = out + past_ema_state

        updated_hidden_state = torch.einsum("bdl,dnl->bdn", inputs, torch.flip(kernel, dims=[2]))
        if past_vandermonde is not None:
            updated_hidden_state = updated_hidden_state + past_vandermonde
        # return a tuple:
        # (sequence_length, batch_size, kernel_dim)
        # (batch_size, kernel_dim, ema_projection_size)
        return out.permute(2, 0, 1), updated_hidden_state

    def one_step(self, inputs, past_state=None):
        p_coeff, q_coeff = self.coeffs()
        # (kernel_dim X ema_projection_size) x (batch_size X kernel_dim X 1) 
        # -> (batch_size X kernel_dim X ema_projection_size)
        updated_state = (p_coeff * self.b_param).squeeze(-1) * inputs
        if past_state is not None:
            updated_state = updated_state + q_coeff.squeeze(-1) * past_state
        # (batch_size X kernel_dim)
        out = torch.einsum("bdn,dn->bd", updated_state, self.g_param * self.scale)
        # (1 X batch_size X kernel_dim), (batch_size X kernel_dim X ema_projection_size)
        return out.unsqueeze(0), updated_state

    def forward(
        self,
        inputs,
        attention_mask: Optional[torch.Tensor] = None,
        prev_state: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """
        Mega's self-attention mechanism based on exponential moving average (EMA)

        Args:
            inputs (`torch.Tensor` of shape `(sequence_length, batch_size, hidden_size)`): 
                Hidden state / embedding input on which to perform self-attention
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*): 
                Indicates which inputs are to be ignored (mostly due to padding), where 
                elements are either 1 for *not masked* or 0 for *masked*
            prev_state (`torch.Tensor` of shape `(batch_size, config.ndim)`, *optional*): 
                The hidden state returned from the previous timestep during incremental 
                decoding.
            use_cache (`bool`, default `False`): 
                Whether to perfom incremental decoding; uses `prev_state` as the prior
                timestep, and returns the updated EMA hidden state for use in the next step

        Returns:
            `tuple(torch.FloatTensor)` containing various elements depending on configuration ([`MegaConfig`]) and inputs:
            - **hidden_states** (`torch.FloatTensor` of shape `(sequence_length, batch_size, hidden_size)`) -- 
              Hidden states updated by EMA self-attention, with same shapes as inputs
            - **updated_state** (*optional*, returned when `use_cache=True`) `torch.FloatTensor of shape `(batch_size, config.ndim)` --
              The incremental EMA state for use in the next step of incremental decoding
        """

        seq_len, bsz, embed_dim = inputs.size()
        if embed_dim != self.embed_dim:
            raise ValueError(f"Unexpected embedding dimension received: input is {embed_dim}, model expects {self.embed_dim}")

        # sequence_length X batch_size X hidden_size
        residual = inputs * self.omega

        # (sequence_length x batch_size x hidden_size) -> (batch_size x hidden_size x sequence_length)
        inputs = inputs.permute(1, 2, 0)
        # mask the input: output is a tensor with 0 in the masked positions
        if attention_mask is not None:
            inputs = inputs * (attention_mask.unsqueeze(1).type_as(inputs))

        if self.bidirectional and use_cache:
            raise RuntimeError("Bidirectional EMA does not support incremental state")

        if use_cache:
            out, updated_state = self.step(inputs, seq_len, past_state=prev_state)

            # (batch_size X hidden_size) -> (1 x batch_size x hidden_size)
            out = F.silu(out + residual)

            # if incremental decoding, return the new state along with the output
            return out, updated_state
        else:
            # (hidden_size x sequence_length)
            kernel = self.kernel(seq_len)
            fft_len = seq_len
            s_index = 0
            kernel_size = kernel.size(1)
            if self.bidirectional:
                # split the kernel for each direction of EMA
                k1, k2 = torch.split(kernel, [self.embed_dim, self.embed_dim], dim=0)
                # (hidden_size X 2*sequence_length - 1)
                kernel = F.pad(k1, (kernel_size - 1, 0)) + F.pad(k2.flip(-1), (0, kernel_size - 1))
                inputs = F.pad(inputs, (kernel_size - 1, 0))
                fft_len = fft_len + kernel_size - 1
                s_index = 2 * kernel_size - 2

            kernel_fourier = torch.fft.rfft(kernel.float(), n=2 * fft_len)
            inputs_fourier = torch.fft.rfft(inputs.float(), n=2 * fft_len)
            # (batch_size X hidden_size X sequence_length)
            out = torch.fft.irfft(inputs_fourier * kernel_fourier, n=2 * fft_len)[..., s_index : s_index + seq_len]
            out = out.type_as(inputs)
            # (batch_size X hidden_size X sequence_length) -> (sequence_length X batch_size X hidden_size)
            out = F.silu(out.permute(2, 0, 1) + residual)

            return out, None


# Gated cross-attention
# removed before_attn_fn argument and unused static_kv
# otherwise left as-is with the exception of hidden states
class MegaGatedCrossAttention(nn.Module):
    """Gated Structured State Attention for use in encoder-decoder model
    See Mega paper for more details.
    """

    def __init__(self, config: MegaConfig):
        super().__init__()

        self.config = config
        self.activation = ACT2FN[self.config.activation]
        self.attention_activation = self.config.attention_activation
        self.scaling = (
            self.config.shared_representation_size**-0.5 if self.attention_activation == "softmax" else None
        )

        dropout_module = MegaFeatureDropout if self.config.use_feature_dropout else MegaDropout
        self.dropout = dropout_module(self.config.dropout_prob)
        self.hidden_dropout = dropout_module(self.config.hidden_dropout_prob)
        # Attention dropout is standard dropout
        self.attention_dropout = MegaDropout(self.config.attention_probs_dropout_prob)

        self.prenorm = self.config.normalize_before_mega
        self.norm = MegaSequenceNorm(
            self.config.normalization_type, self.config.hidden_size, affine=self.config.norm_affine
        )

        self.k_proj = nn.Linear(self.config.hidden_size, self.config.shared_representation_size)
        self.v_proj = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.q_proj = nn.Linear(
            self.config.hidden_size, 2 * self.config.hidden_size + self.config.shared_representation_size
        )
        self.h_proj = nn.Linear(self.config.hidden_size, self.config.hidden_size)

        if self.config.relative_positional_bias == "simple":
            self.rel_pos_bias = SimpleRelativePositionalBias(config)
        elif self.config.relative_positional_bias == "rotary":
            self.rel_pos_bias = RotaryRelativePositionalBias(config)
        else:
            raise ValueError("unknown relative position bias: {}".format(self.config.relative_positional_bias))

        self.softmax = nn.Softmax(dim=-1)

    def element_attention(self, query, key, key_padding_mask, pidx):
        bsz, clen, _ = key.size()
        slen = query.size(1) if pidx is None else pidx + 1
        if key_padding_mask is not None:
            # (batch_size X source_sequence_length) --> (batch_size X 1 X 1)
            lengths = key_padding_mask.sum(dim=-1).view(bsz, 1, 1)
        else:
            lengths = clen

        # (target_sequence_length X source_sequence_length)
        bias = self.rel_pos_bias(max(slen, clen))[:, :clen]
        if pidx is not None:
            if query.size(1) != 1:
                raise ValueError("Position offset provided with queries longer than 1 token")
            # source_sequence_length
            bias = bias[pidx]
        else:
            # (target_sequence_length X source_sequence_length)
            bias = bias[:slen]

        # (batch_size X target_sequence_length X source_sequence_length)
        qk = torch.bmm(query, key.transpose(1, 2)) / lengths + bias

        if self.attention_activation == "relu2":
            attn_weights = relu2(qk).type_as(qk)
        elif self.attention_activation == "laplace":
            attn_weights = laplace(qk).type_as(qk)
        else:
            raise ValueError("Unknown attention activation function: {}".format(self.attention_activation))

        if key_padding_mask is not None:
            attn_weights = attn_weights * key_padding_mask.unsqueeze(1)

        return attn_weights

    def softmax_attention(self, query, key, key_padding_mask, pidx):
        bsz, clen, _ = key.size()
        slen = query.size(1) if pidx is None else pidx + 1

        # (target_sequence_length X source_sequence_length)
        bias = self.rel_pos_bias(max(slen, clen))[:, :clen]
        if pidx is not None:
            if query.size(1) != 1:
                raise ValueError("Position offset provided with queries longer than 1 token")
            # source_sequence_length
            bias = bias[pidx]
        else:
            # (target_sequence_length X source_sequence_length)
            bias = bias[:slen]

        # scaled attention
        query = query * self.scaling
        # (batch_size X target_sequence_length X source_sequence_length)
        qk = torch.bmm(query, key.transpose(1, 2)) + bias

        if key_padding_mask is not None:
            qk = qk.masked_fill((1 - key_padding_mask).unsqueeze(1).to(torch.bool), float("-inf"))

        attn_weights = self.softmax(qk).type_as(qk)
        return attn_weights

    def forward(
        self,
        query,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor] = None,
        prev_key_values: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Gated cross-attention used in Mega

        Args:
            query (`torch.Tensor` of shape `(target_sequence_length, batch_size, hidden_size)`): 
                The self (or target) sequence input used as query inputs for cross-attention
            key (`torch.Tensor` of shape `(source_sequence_length, batch_size, hidden_size)`): 
                The cross (or source) sequence input with shape used as keys in cross-attention
            value (`torch.Tensor` of shape `(source_sequence_length, batch_size, hidden_size)`): 
                The cross (or source) sequence input with shape used as values in cross-attention
            key_padding_mask (`torch.LongTensor` of shape `(batch_size, source_sequence_length)`, *optional*): 
                Padding mask corresponding to the source sequence, where entries are 1 for *not masked* and 
                0 for *masked* tokens
            prev_key_values (`tuple(torch.FloatTensor)`, *optional*): 
                If provided, the hidden state returned from the previous timestep during incremental decoding; 
                expects that prior cross-attention keys and values will be the last two items in the tuple
            output_attentions (`bool`, defaults to `False`): if true, cross-attention weights will be returned 
            use_cache (`bool`, defaults to `False`): 
                Whether to perfom incremental decoding; uses `prev_state` as the prior timestep, and returns the 
                updated EMA hidden state for use in the next step

        Returns:
            `tuple(torch.FloatTensor)` containing various elements depending on configuration ([`MegaConfig`]) and inputs:
            - **hidden_states** (`torch.FloatTensor` of shape `(target_sequence_length, batch_size, hidden_size)`) -- 
              Hidden states from target sequence updated by gated cross-attention
            - **attn_weights** (*optional*, returned when `output_attentions=True`) `torch.FloatTensor` of shape `(batch_size, source_sequence_length, target_sequence_length)` --
              The pairwise cross-attention weights corresponding to each token in the source and target sequences
            - **cross_key** (*optional*, returned when `use_cache=True`) `torch.FloatTensor` of shape `(batch_size, source_sequence_length, config.shared_representation_size)` --
              The cross-attention key state for use in the next step of incremental decoding
            - **cross_value** (*optional*, returned when `use_cache=True`) `torch.FloatTensor` of shape `(batch_size, source_sequence_length, config.hidden_size)` --
              The cross-attention value state for use in the next step of incremental decoding
        """

        seq_len, bsz, embed_dim = query.size()
        if embed_dim != self.config.hidden_size:
            raise ValueError(f"Unexpected embedding dimension received: input is {embed_dim} but expected {self.config.hidden_size}")

        if prev_key_values is not None:
            # make sure the inputs only have a sequence length of 1 if we're doing incremental decoding
            if seq_len != 1:
                raise ValueError(f"Incremental decoding requested with self-sequence length > 1: {seq_len}")
            # expect prev_key_values to have (self_key, self_value, self_ema, cross_key, cross_value)
            prev_cross_key, prev_cross_value = prev_key_values[-2:]
            key = value = None

            # use the self-attention cache to get the position id of the current step
            prev_self_key = prev_key_values[0]
            num_incremental_steps = prev_self_key.size(1) + 1
        else:
            prev_cross_key = prev_cross_value = None
            # we still need the position id if we're doing incremental decoding (past_key_values will be None for the first step)
            num_incremental_steps = 0 if use_cache and (seq_len == 1) else None

        full_query = query
        if self.prenorm:
            full_query = self.norm(full_query)

        # (target_sequence_length X batch_size X 2*hidden_size + shared_representation_size)
        query_projected = self.q_proj(full_query)
        # split the query projections into separate components
        # - residual_weight is passed through sigmoid and sent through elementwise multiplication to the gated/weighted targets prior to being added to the query directly
        # - target_gate is a silu-gated tensor that is multiplied by the attention-weighted target below prior to residual connection
        # - attention_query is the part that is passed to the attention function
        residual_weight, target_gate, attention_query = torch.split(
            query_projected, [self.config.hidden_size, self.config.hidden_size, self.config.shared_representation_size], dim=-1
        )

        # (target_sequence_length X batch_size X hidden_size)
        residual_weight = torch.sigmoid(residual_weight)
        target_gate = F.silu(target_gate)

        if key is None:
            if value is not None:
                raise ValueError("Key and value must be `None` simultaneously")
            projected_key = projected_value = None
        else:
            # (source_sequence_length X batch_size X shared_representation_size)
            projected_key = self.k_proj(key)
            # (source_sequence_length X batch_size X hidden_size)
            projected_value = self.activation(self.v_proj(key))

        # (target_sequence_length X batch_size X shared_representation_size) 
        # -> (batch_size X target_sequence_length X shared_representation_size)
        attention_query = attention_query.transpose(0, 1)
        if projected_key is not None:
            projected_key = projected_key.transpose(0, 1)
        if projected_value is not None:
            projected_value = projected_value.transpose(0, 1)

        # if we're doing incremental decoding, k and v are None and need to be overwritten with past values
        if prev_key_values is not None:
            projected_key = prev_cross_key
            projected_value = prev_cross_value

        # if we're returning the cache for later use, store these now for later return (can be done without having prev_key_values provided)
        if use_cache:
            updated_cross_key = projected_key
            updated_cross_value = projected_value

        ctx_len = projected_key.size(1)
        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            if key_padding_mask.size(0) != bsz:
                raise ValueError("Key padding mask does not align on the batch dimension")
            if key_padding_mask.size(1) != ctx_len:
                raise ValueError("Key padding mask does not align on the sequence length dimension")

        if self.attention_activation == "softmax":
            attn_weights = self.softmax_attention(attention_query, projected_key, key_padding_mask, num_incremental_steps)
        else:
            attn_weights = self.element_attention(attention_query, projected_key, key_padding_mask, num_incremental_steps)

        projected_value = self.hidden_dropout(projected_value, batch_first=True)
        kernel = self.attention_dropout(attn_weights)
        # (batch_size X target_sequence_length X hidden_size) 
        # -> (target_sequence_length X batch_size X hidden_size)
        weighted_targets = torch.bmm(kernel, projected_value).transpose(0, 1)
        # (target_sequence_length X batch_size X hidden_size)
        weighted_targets = self.activation(self.h_proj(weighted_targets * target_gate))
        weighted_targets = self.dropout(weighted_targets)
        out = torch.addcmul(query, residual_weight, weighted_targets - query)

        if not self.prenorm:
            out = self.norm(out)

        outputs = (out, attn_weights) if output_attentions else (out,)
        if use_cache:
            outputs = outputs + (updated_cross_key, updated_cross_value)

        return outputs


# Positional embeddings
# copied from original Mega code and renamed variables for better readability
class SimpleRelativePositionalBias(nn.Module):
    def __init__(self, config: MegaConfig):
        super().__init__()
        self.config = config
        self.max_positions = self.config.max_positions if self.config.chunk_size < 0 else self.config.chunk_size
        self.rel_pos_bias = nn.Parameter(torch.Tensor(2 * config.max_positions - 1))

    def forward(self, seq_len):
        if seq_len > self.max_positions:
            raise ValueError("Sequence length {} going beyond max length {}".format(seq_len, self.max_positions))

        # seq_len * 2 - 1
        bias = self.rel_pos_bias[(self.max_positions - seq_len) : (self.max_positions + seq_len - 1)]
        # seq_len * 3 - 1
        tile = F.pad(bias, (0, seq_len))
        # (seq_len * 3 - 1) * seq_len
        tile = torch.tile(tile, (seq_len,))
        tile = tile[:-seq_len]
        # seq_len x (3 * seq_len - 2)
        tile = tile.view(seq_len, 3 * seq_len - 2)
        start = (2 * seq_len - 1) // 2
        end = tile.size(1) - start
        tile = tile[:, start:end]
        return tile


class RotaryRelativePositionalBias(nn.Module):
    def __init__(self, config: MegaConfig):
        super().__init__()
        if config.hidden_size % 2 != 0:
            raise RuntimeError("Rotary positional bias requires `hidden_size` to be a multiple of 2")
        self.config = config
        self.embed_dim = config.shared_representation_size
        self.max_positions = self.config.max_positions if self.config.chunk_size < 0 else self.config.chunk_size
        self.sine, self.cosine = RotaryRelativePositionalBias.get_sinusoid_embeddings(
            config.max_positions, self.embed_dim
        )
        self.alpha = nn.Parameter(torch.Tensor(1, self.embed_dim))
        self.b_param = nn.Parameter(torch.Tensor(1, self.embed_dim))
        self.register_buffer("_float_tensor", torch.FloatTensor([0.0]))

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
        rotary_alpha = self.rotary(self.alpha.expand(seq_len, self.embed_dim))
        rotary_beta = self.rotary(self.b_param.expand(seq_len, self.embed_dim))
        bias = torch.einsum("mk,nk->mn", rotary_alpha, rotary_beta)
        return bias


# Normalization modules
# copied from original Mega repo without modification except variable names
class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, affine=True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.affine = affine
        if affine:
            self.scalar = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter("scalar", None)

    def forward(self, input):
        mean_square = torch.mean(torch.square(input), dim=self.dim, keepdim=True)
        if self.scalar is not None:
            input = self.scalar * input

        output = input * torch.rsqrt(mean_square + self.eps)
        return output


class RMSNorm(nn.Module):
    def __init__(self, number_features, eps=1e-6, affine=True):
        super().__init__()
        self.num_features = number_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.Tensor(self.num_features))
        else:
            self.register_parameter("weight", None)

    def forward(self, input):
        mean_square = torch.mean(torch.square(input), dim=-1, keepdim=True)
        if self.weight is not None:
            input = input * self.weight

        output = input * torch.rsqrt(mean_square + self.eps)
        return input


class MegaSequenceNorm(nn.Module):
    def __init__(self, norm_type, embedding_dim, eps=1e-5, affine=True, export=False):
        super().__init__()
        if norm_type == "layernorm":
            self.norm = nn.LayerNorm(embedding_dim, eps, elementwise_affine=affine)
        elif norm_type == "scalenorm":
            self.norm = ScaleNorm(dim=-1, eps=eps, affine=affine)
        elif norm_type == "rmsnorm":
            self.norm = RMSNorm(embedding_dim, eps=eps, affine=affine)
        elif norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(embedding_dim, eps=eps, affine=affine)
        elif norm_type == "syncbatchnorm":
            self.norm = nn.SyncBatchNorm(embedding_dim, eps=eps, affine=affine)
        else:
            raise ValueError("Unknown norm type: {}".format(norm_type))

    def normalize(self, x):
        if isinstance(self.norm, nn.modules.batchnorm._BatchNorm):
            if x.dim() != 3:
                raise ValueError("BatchNorm inputs must be exactly 3-dimensional")
            x = x.permute(1, 2, 0)
            x = self.norm(x)
            return x.permute(2, 0, 1)
        else:
            return self.norm(x)

    def forward(self, input):
        return self.normalize(input)

# add this layernorm class to ALL_LAYERNORM_LAYERS
ALL_LAYERNORM_LAYERS.append(MegaSequenceNorm)

# Dropout: standard dropout + feature dropout
# copied from Mega repo, but changed name from Fairseq->Mega,
# modified variable names, and removed unused items:
# - `make_generation_fast_` method 
# - `apply_during_inference` attribute 
# - `module_name` arg
class MegaDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, input, batch_first: bool = False, inplace: bool = False):
        if self.training:
            return F.dropout(input, p=self.p, training=True, inplace=inplace)
        else:
            return input

class MegaFeatureDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, input, batch_first: bool = False, inplace: bool = False):
        if self.training:
            if batch_first:
                # (batch_size X sequence_length X feature_dimension) 
                # -> (batch_size X feature_dimension X sequence_length) 
                # -> (batch_size X sequence_length X feature_dimension) 
                return F.dropout2d(input.transpose(-1, -2), p=self.p, training=True, inplace=inplace).transpose(-1, -2)
            else:
                if input.dim() != 3:
                    raise ValueError("Feature dropout inputs must be exactly 3-dimensional if inputs are ordered [sequence length, batch size, hidden dimension]")
                # (sequence_length X batch_size X feature_dimension) 
                # -> (batch_size X feature_dimension X sequence_length) 
                # -> (sequence_length X batch_size X feature_dimension)
                return F.dropout2d(input.permute(1, 2, 0), p=self.p, training=True, inplace=inplace).permute(2, 0, 1)
        else:
            return input


# Mega attention: EMA + self-attention
# differences from original include hidden state refactor and fixed inconsistency with additive/multiplicative attention masks
class MovingAverageGatedAttention(nn.Module):
    """
    Pure PyTorch implementation of Mega block; see https://arxiv.org/abs/2209.10655 and original fairseq implementation
    at https://github.com/facebookresearch/mega (copyright Meta Research, licensed under MIT License)
    """

    def __init__(self, config: MegaConfig):
        super().__init__()
        self.config = config
        self.activation = ACT2FN[self.config.activation]
        self.scaling = (
            self.config.shared_representation_size**-0.5 if self.config.attention_activation == "softmax" else None
        )
        dropout_module = MegaFeatureDropout if self.config.use_feature_dropout else MegaDropout
        self.dropout = dropout_module(self.config.dropout_prob)
        self.hidden_dropout = dropout_module(self.config.hidden_dropout_prob)
        # attention dropout is standard dropout
        self.attention_dropout = MegaDropout(self.config.attention_probs_dropout_prob)

        self.norm = MegaSequenceNorm(
            self.config.normalization_type, self.config.hidden_size, affine=self.config.norm_affine
        )
        self.move = MultiHeadEMA(config)

        self.v_proj = nn.Linear(self.config.hidden_size, self.config.intermediate_size)
        self.mx_proj = nn.Linear(
            self.config.hidden_size,
            self.config.shared_representation_size + self.config.intermediate_size + 2 * self.config.hidden_size,
        )
        self.h_proj = nn.Linear(self.config.intermediate_size, self.config.hidden_size)

        # renamed gamma and beta to g_param and b_param respectively due to Hugging Face renaming weights upon `.from_pretrained`
        self.g_param = nn.Parameter(torch.Tensor(2, self.config.shared_representation_size))
        self.b_param = nn.Parameter(torch.Tensor(2, self.config.shared_representation_size))

        if self.config.relative_positional_bias == "simple":
            self.rel_pos_bias = SimpleRelativePositionalBias(config)
        elif self.config.relative_positional_bias == "rotary":
            self.rel_pos_bias = RotaryRelativePositionalBias(config)
        else:
            raise ValueError(f"Unknown relative positional bias: {self.config.relative_positional_bias}")

        self.softmax = nn.Softmax(dim=-1)
        self.attention_function = (
            self.softmax_attention if self.config.attention_activation == "softmax" else self.element_attention
        )

    def element_attention(self, query, key, padding_mask, causal_mask):
        """
        Apply element-wise attention via relu^2 or laplace. Same as original implementation but with standardized
        causal attention mask
        """
        slen = key.size(2)
        if padding_mask is not None:
            # 1 for *not masked*
            # 0 for *masked*

            # (batch_size X number of chunks X 1)
            lengths = padding_mask.sum(-1, keepdim=True)
            # (batch_size X number of chunks X 1 X 1)
            lengths = lengths.clamp(min=1.0).unsqueeze(-1)
        else:
            lengths = slen

        if causal_mask is not None:
            lengths = causal_mask.sum(dim=-1, keepdim=True)

        # (sequence_length X sequence_length)
        bias = self.rel_pos_bias(slen)
        if slen != query.size(2):
            if query.size(2) != 1:
                raise ValueError("Size mismatch between Q and K in element attention")
            # (1 X sequence_length)
            bias = bias[-1:]

        # (batch_size X number of chunks X sequence_length X sequence_length)
        qk = torch.matmul(query, key.transpose(2, 3)) / lengths + bias

        if self.config.attention_activation == "relu2":
            attn_weights = relu2(qk).type_as(qk)
        elif self.config.attention_activation == "laplace":
            attn_weights = laplace(qk).type_as(qk)
        else:
            raise ValueError(f"Unknown attention activation function: {self.config.attention_activation}")

        if padding_mask is not None:
            attn_weights = attn_weights * padding_mask.unsqueeze(2)

        if causal_mask is not None:
            attn_weights = attn_weights * causal_mask

        return attn_weights

    def softmax_attention(self, query, key, padding_mask, causal_mask):
        "Standard softmax self-attention, as in the original Transformer paper"
        slen = key.size(2)
        # (sequence_length X sequence_length)
        bias = self.rel_pos_bias(slen)
        if slen != query.size(2):
            if query.size(2) != 1:
                raise ValueError("Size mismatch between Q and K in softmax attention")
            # (1 X sequence_length)
            bias = bias[-1:]

        # scaled attention
        query = query * self.scaling

        # (batch_size x number of chunks x chunk_size x chunk_size) if chunking
        # (batch_size x 1 x sequence_length x sequence_length) otherwise
        qk = torch.matmul(query, key.transpose(2, 3)) + bias

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
            # need to invert the padding mask to match what mega original did
            padding_mask = 1 - padding_mask
            padding_mask_all = padding_mask.all(dim=-1, keepdim=True)
            padding_mask = torch.logical_and(padding_mask, ~padding_mask_all)
            qk = qk.masked_fill(padding_mask.unsqueeze(2).to(torch.bool), float("-inf"))

        attn_weights = self.softmax(qk).type_as(qk)
        return attn_weights

    def forward(
        self,
        input,
        padding_mask: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None,
        prev_key_values: Optional[Tuple[torch.Tensor]] = None,
        output_attentions=False,
        use_cache=False,
    ):
        """
        Mega's self-attention block, which combines multi-headed EMA with traditional self-attention

        Args:
            input (`torch.Tensor`` of shape `(sequence_length, batch_size, hidden_size)`):
                Hidden states to be updated by Mega's self-attention
            padding_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*): 
                Indicates which inputs are to be ignored due to padding, where elements are either 
                1 for *not masked* or 0 for *masked*
            causal_mask (`torch.LongTensor` of shape `(sequence_length, sequence_length)`, *optional*): 
                Indicates which inputs are to be ignored due to causal attention, where elements 
                are either 1 for *not masked* or 0 for *masked*
            prev_key_values (`tuple(torch.Tensor)`, *optional*): 
                The hidden states returned from the previous timestep during incremental decoding; 
                expects that self-attention key, value, and EMA states are the first 3 entries in 
                the tuple
            output_attentions (`bool`, default `False`): 
                Whether to return self-attention weights 
            use_cache (`bool`, default `False`): 
                Whether to perfom incremental decoding; uses `prev_key_values` as prior state, and 
                returns the updated states for use in the next step

        Returns:
            `tuple(torch.FloatTensor)` containing various elements depending on configuration ([`MegaConfig`]) and inputs:
            - **hidden_states** (`torch.FloatTensor` of shape `(sequence_length, batch_size, hidden_size)`) -- 
              Hidden states from target sequence updated by Mega's self-attention
            - **attn_weights** (*optional*, returned when `output_attentions=True`) `torch.FloatTensor` of shape `(batch_size, 1, sequence_length, sequence_length)` --
              The self-attention weights corresponding to how each token in the input sequence 
              attends to every other token
            - **self_key** (*optional*, returned when `use_cache=True`) `torch.FloatTensor` of shape `(batch_size, sequence_length, config.shared_representation_size)` --
              The self-attention key state for use in the next step of incremental decoding
            - **self_value** (*optional*, returned when `use_cache=True`) `torch.FloatTensor` of shape `(batch_size, sequence_length, config.hidden_size)` --
              The self-attention value state for use in the next step of incremental decoding
            - **self_ema_state** (*optional*, returned when `use_cache=True`) `torch.FloatTensor` of shape `(batch_size, config.ndim)`
              The incremental EMA state for use in the next step of incremental decoding.
        """

        seq_len, bsz, embed_dim = input.size()
        if embed_dim != self.config.hidden_size:
            raise ValueError(f"Input embedding dimension should be {self.config.hidden_size}; received {embed_dim}")

        # store inputs for residual connection and handle pre-norm if requested
        residual = input
        if self.config.normalize_before_mega:
            input = self.norm(input)

        # (sequence_length X batch_size X hidden_size) -> (sequence_length X batch_size X intermediate_size)
        value = self.activation(self.v_proj(input))

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

        # ema output is (sequence_length x batch_size x hidden_size)
        # updated_ema_state will be None if use_cache=False; otherwise (batch_size, config.ndim)
        ema_out, updated_ema_state = self.move(
            input, attention_mask=padding_mask, prev_state=prev_ema_state, use_cache=use_cache
        )
        ema_out = self.dropout(ema_out)

        # (sequence_length X batch_size X hidden_size)
        # -> (sequence_length X batch_size X 2*hidden_size + config.shared_representation_size + config.intermediate_size) 
        # - residual_weight -> sigmoid -> applied to residual connection in torch.addcmul
        # - query_key_gates -> split into two components: query_key becomes query and key for attention input, gates becomes gating for self-attention output
        # - intermediate_state -> added to weighted attention output, sent through activation, and has inputs subtracted during 
        #   torch.addcmul to create the final layer output
        base = self.mx_proj(ema_out)
        residual_weight, query_key_gates, intermediate_state = torch.split(
            base,
            [
                self.config.hidden_size,
                self.config.shared_representation_size + self.config.intermediate_size,
                self.config.hidden_size,
            ],
            dim=-1,
        )

        # (sequence_length X batch_size X hidden_size)
        residual_weight = torch.sigmoid(residual_weight)

        # (sequence_length X batch_size X shared_representation_size + intermediate_size)
        # split into two different tensors: one for Q/K usage and the other for gating self-attention
        query_key, attention_gate = torch.split(F.silu(query_key_gates), [self.config.shared_representation_size, self.config.intermediate_size], dim=-1)

        # (sequence_length X batch_size X shared_representation_size) 
        # -> (sequence_length X batch_size X 1 X shared_representation_size) 
        # -> (sequence_length X batch_size X 2 X shared_representation_size) 
        query_key = query_key.unsqueeze(2) * self.g_param + self.b_param

        # (sequence_length X batch_size X 2 X shared_representation_size) 
        # -> 2 tensors of (sequence_length X batch_size X shared_representation_size)
        query, key = torch.unbind(query_key, dim=2)

        # (sequence_length X batch_size X dimension) 
        # -> (batch_size X sequence_length X dimension)
        # where `dimension` is either shared_representation_size (queries and keys) or intermediate_size (values)
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)

        if self.config.is_decoder:
            # combine history and current to save updated state (if history is provided)
            # when chunking is applied, the past states will be None at the end of the chunk, in
            # which case, proceed as if no K/V history had been provided
            # saved states are stored with shape (batch_size X sequence_length X dimension)
            if prev_self_key is not None:
                key = torch.cat([prev_self_key, key], dim=1)
            if prev_self_value is not None:
                value = torch.cat([prev_self_value, value], dim=1)

            # if not chunking, store as-is
            if not self.config.use_chunking:
                updated_self_key = key
                updated_self_value = value
            else:
                curr_len = key.size(1) % self.config.chunk_size
                if curr_len == 0:
                    # if we're chunking and have reached the end of a chunk, wipe out the saved state
                    updated_self_key = None
                    updated_self_value = None
                else:
                    updated_self_key = key
                    updated_self_value = value

        ctx_len = key.size(1) # potentially differs from seq_len because of incremental decoding
        if not self.config.use_chunking:
            # if we're not chunking, treat the entire sequence as one long chunk
            # (batch_size X sequence_length X dimension) -> (batch_size X 1 X sequence_length X dimension)
            query = query.unsqueeze(1)
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)
            if padding_mask is not None:
                # (batch_size X sequence_length) -> (batch_size X 1 X sequence_length)
                padding_mask = padding_mask.unsqueeze(1)
        else:
            # otherwise, split the sequences in the batch into `n_chunks` chunks of size `chunk_size`
            if seq_len < self.config.chunk_size:
                query = query.unsqueeze(1)
            else:
                # (batch_size X sequence_length X dimension) -> (batch_size X n_chunks X chunk_size X dimension)
                n_chunks = seq_len // self.config.chunk_size
                query = query.reshape(bsz, n_chunks, self.config.chunk_size, self.config.shared_representation_size)

            if ctx_len < self.config.chunk_size:
                key = key.unsqueeze(1)
                value = value.unsqueeze(1)
                if padding_mask is not None:
                    padding_mask = padding_mask.unsqueeze(1)
            else:
                # (batch_size X sequence_length X dimension) -> (batch_size X n_chunks X chunk_size X dimension)
                n_chunks = ctx_len // self.config.chunk_size
                key = key.reshape(bsz, n_chunks, self.config.chunk_size, self.config.shared_representation_size)
                value = value.reshape(bsz, n_chunks, self.config.chunk_size, self.config.intermediate_size)
                if padding_mask is not None:
                    padding_mask = padding_mask.view(bsz, n_chunks, self.config.chunk_size)

        # this is in the original Mega implementation to work around fork/join parallelism not supporting optional types
        if padding_mask is not None and padding_mask.dim() == 0:
            padding_mask = None

        attn_weights = self.attention_function(query, key, padding_mask=padding_mask, causal_mask=causal_mask)

        value = self.hidden_dropout(value, batch_first=True)
        kernel = self.attention_dropout(attn_weights)

        # (batch_size x n_chunks x chunk_size x intermediate_size) -> (sequence_length X batch_size X intermediate_size)
        weighted_self_output = torch.matmul(kernel, value).view(bsz, seq_len, self.config.intermediate_size).transpose(0, 1)

        # (sequence_length X batch_size X intermediate_size) -> (sequence_length X batch_size X hidden_size)
        weighted_self_output = self.activation(intermediate_state + self.h_proj(weighted_self_output * attention_gate))
        weighted_self_output = self.dropout(weighted_self_output)
        # (sequence_length X batch_size X hidden_size)
        out = torch.addcmul(residual, residual_weight, weighted_self_output - residual)

        if not self.config.normalize_before_mega:
            out = self.norm(out)

        return_values = (out, attn_weights) if output_attentions else (out,)

        if self.config.is_decoder:
            return_values = return_values + (updated_self_key, updated_self_value, updated_ema_state)

        return return_values


# Normalized feed-forward network
# left as-is from original Mega repo aside from retrieving args from Hugging Face config
class MegaNormalizedFeedForwardNetwork(nn.Module):
    """
    Normalized feed-forward network used in Mega blocks
    """

    def __init__(self, config: MegaConfig):
        super().__init__()

        self.config = config
        self.hidden_dim = config.nffn_hidden_size
        self.act_fn = config.activation
        self.activation = ACT2FN[config.activation]

        dropout_module = MegaFeatureDropout if self.config.use_feature_dropout else MegaDropout
        self.dropout = dropout_module(self.config.dropout_prob)
        self.hidden_dropout = dropout_module(self.config.nffn_activation_dropout_prob)

        self.prenorm = self.config.normalize_before_ffn
        self.norm = MegaSequenceNorm(
            self.config.normalization_type, self.config.hidden_size, affine=self.config.norm_affine
        )

        self.fc1 = nn.Linear(self.config.hidden_size, self.config.nffn_hidden_size)
        self.fc2 = nn.Linear(self.config.nffn_hidden_size, self.config.hidden_size)

    def forward(self, inputs):
        residual = inputs

        if self.prenorm:
            inputs = self.norm(inputs)

        hidden = self.activation(self.fc1(inputs))
        hidden = self.hidden_dropout(hidden)
        output = self.fc2(hidden)
        output = self.dropout(output)
        output = output + residual

        if not self.prenorm:
            output = self.norm(output)

        return output


class MegaEmbeddings(nn.Module):
    """
    Mega's basic implementation does not incorporate token type embeddings, so this is a stripped-down version of
    RoBERTa's embeddings which optionally includes token types
    """

    def __init__(self, config: MegaConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.use_token_types = config.add_token_type_embeddings
        if self.use_token_types:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # End copy
        self.padding_idx = config.pad_token_id

    def forward(self, input_ids=None, token_type_ids=None, inputs_embeds=None):
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
            if token_type_ids is None:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)

            # access token type embeddings
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            # add the token type embeddings to the word embeddings
            embeddings = inputs_embeds + token_type_embeddings
        else:
            embeddings = inputs_embeds
        return embeddings


class MegaLayer(nn.Module):
    def __init__(self, config: MegaConfig):
        super().__init__()
        self.seq_len_dim = 1
        self.mega_layer = MovingAverageGatedAttention(config)
        self.nffn = MegaNormalizedFeedForwardNetwork(config) if config.use_normalized_ffn else None
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.cross_attn = MegaGatedCrossAttention(config)
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
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor]:
        """
        A single Mega layer: either encoder or decoder, with optional cross-attention and optional normalized
        feed-forward layer

        Args:
            hidden_states (`torch.Tensor`` of shape `(target_sequence_length, batch_size, hidden_size)`):
                Hidden states to be updated by the Mega block
            attention_mask (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*): 
                Indicates which entries in the self/target sequence are to be ignored (mostly due to 
                padding), where elements are either 1 for *not masked* or 0 for *masked*. Causal 
                attention is enforced internally.
            encoder_hidden_states (`torch.Tensor`, of shape `(source_sequence_length, batch_size, hidden_size)`, *optional*):
                Encoder hidden states to be used for cross-attention (and required for encoder-decoder
                model setup)
            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, source_sequence_length)`, *optional*): 
                Indicates which entries in the cross/source sequence are to be ignored (mostly due 
                to padding), where elements are either 1 for *not masked* or 0 for *masked*.
            past_key_value (`tuple(torch.Tensor)`, *optional*): 
                The hidden states returned from the previous timestep during incremental decoding; 
                expects that self-attention key, value, and EMA states are the first 3 entries in 
                the tuple, and (if doing cross-attention) cross-attention key and value are the last
                2 entries in the tuple
            output_attentions (`bool`, default `False`): 
                Whether to return self-attention weights 
            use_cache (`bool`, default `False`): 
                Whether to perfom incremental decoding; uses `past_key_value` as prior state, and 
                returns the updated states for use in the next step

        Returns:
            `tuple(torch.FloatTensor)` containing various elements depending on configuration ([`MegaConfig`]) and inputs:
            - **hidden_states** (`torch.FloatTensor` of shape `(target_sequence_length, batch_size, hidden_size)`) -- 
              Hidden states from target sequence updated by Mega
            - **self_attn_weights** (*optional*, returned when `output_attentions=True`) `torch.FloatTensor` of shape `(batch_size, 1, target_sequence_length, target_sequence_length)` --
              The self-attention weights corresponding to how each token in the input sequence 
              attends to every other token
            - **cross_attn_weights** (*optional*, returned when `output_attentions=True` and `config.add_cross_attention=True`) `torch.FloatTensor` of shape `(batch_size, source_sequence_length, target_sequence_length)` --
              Pairwise cross-attention weights between every entry in the source sequence and target sequence
            - **self_key** (*optional*, returned when `use_cache=True`) `torch.FloatTensor` of shape `(batch_size, sequence_length, config.shared_representation_size)` --
              The self-attention key state for use in the next step of incremental decoding
            - **self_value** (*optional*, returned when `use_cache=True`) `torch.FloatTensor` of shape `(batch_size, sequence_length, config.hidden_size)` --
              The self-attention value state for use in the next step of incremental decoding
            - **self_ema_state** (*optional*, returned when `use_cache=True`) `torch.FloatTensor` of shape `(batch_size, config.ndim)`
              The incremental EMA state for use in the next step of incremental decoding.
            - **cross_key** (*optional*, returned when `use_cache=True` and `config.is_decoder=True`) `torch.FloatTensor` of shape `(batch_size, source_sequence_length, config.shared_representation_size)` --
              The cross-attention key state for use in the next step of incremental decoding
            - **cross_value** (*optional*, returned when `use_cache=True` and `config.is_decoder=True`) `torch.FloatTensor` of shape `(batch_size, source_sequence_length, config.hidden_size)` --
              The cross-attention value state for use in the next step of incremental decoding
        """

        # Mega self-attention
        # create a causal mask for self-attention if we're decoding
        # note that the Mega code does not account for any past key values in the causal mask - only the input sequence
        if self.is_decoder:
            sequence_length = hidden_states.size(0)
            causal_mask = generate_causal_mask(sequence_length)
        else:
            causal_mask = None

        # incremental decoding in the MultiHeadEMA module requires that the attention mask has the same
        # sequence length as the input tensor; if we're caching incremental states, we assume the input
        # sequence length is 1 (Mega will break otherwise), so we take the padding mask for the final
        # token in the input (mask is received as [batch X sequence length])
        if use_cache and (past_key_value is not None) and (attention_mask is not None):
            mega_padding_mask = attention_mask[:, -1].unsqueeze(-1)
        else:
            mega_padding_mask = attention_mask

        mega_outputs = self.mega_layer(
            input=hidden_states,
            padding_mask=mega_padding_mask,
            causal_mask=causal_mask,
            prev_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        new_hidden_states = mega_outputs[0]
        self_key, self_value, self_ema_state = mega_outputs[-3:] if use_cache else (None, None, None)
        self_attention_weights = mega_outputs[1] if output_attentions else None

        # optional cross attention
        if self.cross_attn is not None:
            if encoder_hidden_states is None:
                raise ValueError("Requested cross-attention without providing encoder hidden states")

            cross_attn_outputs = self.cross_attn(
                query=new_hidden_states,
                key=encoder_hidden_states,
                value=encoder_hidden_states,
                key_padding_mask=encoder_attention_mask,
                prev_key_values=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

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
                outs = outs + (cross_attention_weights,)

        if use_cache:
            new_key_values = (
                self_key,
                self_value,
                self_ema_state,
            )
            if self.cross_attn is not None:
                new_key_values = new_key_values + (cross_key, cross_value)

            outs = outs + (new_key_values,)

        return outs


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
        if isinstance(module, MultiHeadEMA):
            with torch.no_grad():
                # delta & alpha
                nn.init.normal_(module.delta, mean=0.0, std=self.config.ema_delta_alpha_range)
                nn.init.normal_(module.alpha, mean=0.0, std=self.config.ema_delta_alpha_range)
                # beta [1, -1, 1, -1, ...] seems more stable.
                val = torch.ones(self.config.ema_projection_size, 1)
                if self.config.ema_projection_size > 1:
                    idx = torch.tensor(list(range(1, self.config.ema_projection_size, 2)))
                    val.index_fill_(0, idx, -1.0)
                module.b_param.normal_(mean=0.0, std=self.config.ema_beta_range).add_(val)
                # gamma & omega
                nn.init.normal_(module.g_param, mean=0.0, std=self.config.ema_gamma_omega_range)
                nn.init.normal_(module.omega, mean=0.0, std=self.config.ema_gamma_omega_range)
        elif isinstance(module, SimpleRelativePositionalBias):
            nn.init.normal_(module.rel_pos_bias, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, RotaryRelativePositionalBias):
            nn.init.normal_(module.alpha, mean=0.0, std=self.config.initializer_range)
            nn.init.normal_(module.b_param, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, ScaleNorm):
            if self.config.norm_affine:
                nn.init.constant_(module.scalar, 1.0)
        elif isinstance(module, RMSNorm):
            if self.config.norm_affine:
                nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, MovingAverageGatedAttention):
            # linear layers covered separately by the generic nn.Linear init below
            nn.init.normal_(module.g_param, mean=0.0, std=self.config.initializer_range)
            nn.init.constant_(module.b_param, 0.0)
        elif isinstance(module, nn.Linear):
            # initializes all linear layers in the entire network
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
            This parameter can only be used when the model is initialized with `add_token_type_embeddings` parameter
            set to `True`. All the value in this tensor should be always < config.type_vocab_size.

            [What are token type IDs?](../glossary#token-type-ids)
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
    cross-attention is added after self-attention, following the architecture described in *Mega: Moving Average
    Equipped Gated Attention*_ by Xuezhe Ma, Chunting Zhou, Xiang Kong, Junxian He, Liangke Gui, Graham Neubig,
    Jonathan May, and Luke Zettlemoyer

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True` and `bidirectional` set to `False`. To be used in a Seq2Seq model, the model needs to initialized with
    both `is_decoder=True` and `bidirectional=False` argument as well as `add_cross_attention` set to `True`; an
    `encoder_hidden_states` is then expected as an input to the forward pass.

    .. _*Mega: Moving Average Equipped Gated Attention*: https://arxiv.org/abs/2209.10655

    """

    _keys_to_ignore_on_load_missing = []

    def __init__(self, config: MegaConfig, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embedding_layer = MegaEmbeddings(config)
        self.encoders = nn.ModuleList([MegaLayer(config) for _ in range(config.num_hidden_layers)])

        self.pooler = MegaPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing (retained from RoBERTa code)
        self.post_init()

    def get_input_embeddings(self):
        return self.embedding_layer.word_embeddings

    def set_input_embeddings(self, value):
        self.embedding_layer.word_embeddings = value

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

        if self.config.use_chunking and (input_ids.size(1) > self.config.chunk_size):
            if input_ids.size(1) % self.config.chunk_size != 0:
                raise ValueError(
                    f"config.use_chunking is activated; input sequence length must be shorter than or a multiple of config.chunk_size\nreceived sequence length of {input_ids.size(1)} with chunk size {self.config.chunk_size}"
                )

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

        batch_size, sequence_length = input_shape

        # if using cache, make sure we have a tuple of tuples which matches the length of our hidden layers
        if (past_key_values is not None) and (len(past_key_values) != self.config.num_hidden_layers):
            raise ValueError(
                f"Received past key/value cache with size mismatch; expected {self.config.num_hidden_layers}, received {len(past_key_values)}"
            )

        # get embeddings (batch X sequence length X embed dim)
        embedding_output = self.embedding_layer(
            input_ids=input_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        # transpose for Mega --> (seq len X batch X embed dim)
        hidden_states = embedding_output.transpose(0, 1)

        # we expect encoder hidden states to also have batch first in line
        # with typical Hugging Face behavior (which is also how we return them)
        # Mega expects sequence length first, so do the same transpose here
        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.transpose(0, 1)

        # pass through mega layers
        all_hidden_states = (embedding_output,) if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        next_decoder_cache = () if use_cache else None
        for i, mega_layer in enumerate(self.encoders):
            current_decoder_cache = past_key_values[i] if past_key_values is not None else None
            mega_outputs = mega_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=current_decoder_cache,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = mega_outputs[0]
            if output_hidden_states:
                # store layer-wise hidden states in the way that the user expects
                # (seq len X batch X embed dim) --> (batch X seq len X embed dim)
                all_hidden_states += (hidden_states.transpose(0, 1), )
            if output_attentions:
                self_attn_weights = mega_outputs[1]
                all_self_attentions += (self_attn_weights,)
                if self.config.add_cross_attention:
                    cross_attn_weights = mega_outputs[2]
                    all_cross_attentions += (cross_attn_weights,)
            if use_cache:
                updated_cache = mega_outputs[-1]
                next_decoder_cache += (updated_cache,)

        # transpose final hidden states
        hidden_states = hidden_states.transpose(0, 1)

        # optional pooling layer
        pooled_output = self.pooler(hidden_states) if self.pooler is not None else None

        if not return_dict:
            return (hidden_states, pooled_output) + (
                all_hidden_states,
                next_decoder_cache,
                all_self_attentions,
                all_cross_attentions,
            )

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=hidden_states,
            pooler_output=pooled_output,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


@add_start_docstrings(
    """Mega Model with a `language modeling` head on top for CLM fine-tuning.""", MEGA_START_DOCSTRING
)
class MegaForCausalLM(MegaPreTrainedModel):
    _keys_to_ignore_on_save = [r"lm_head.weight", r"lm_head.bias"]
    _keys_to_ignore_on_load_missing = [r"lm_head.weight", r"lm_head.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config: MegaConfig):
        super().__init__(config)

        if not config.is_decoder:
            logger.warning("If you want to use `MegaForCausalLM` as a standalone, add `is_decoder=True.`")

        self.mega = MegaModel(config, add_pooling_layer=False)

        if config.add_lm_hidden_dense_layer:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.hidden_activation = nn.Tanh()
        else:
            self.dense = None
            self.hidden_activation = None

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        # The LM head weights require special treatment only when they are tied with the word embeddings
        self.update_keys_to_ignore(config, ["lm_head.weight"])

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @add_start_docstrings_to_model_forward(MEGA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
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
        >>> config.bidirectional = False
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

        if self.dense is not None:
            sequence_output = self.dense(sequence_output)
            sequence_output = self.hidden_activation(sequence_output)

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
    _keys_to_ignore_on_save = [r"mlm_head.weight", r"mlm_head.bias"]
    _keys_to_ignore_on_load_missing = [r"mlm_head.weight", r"mlm_head.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config: MegaConfig):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `MegaForMaskedLM`, set `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.mega = MegaModel(config, add_pooling_layer=False)
        if config.add_lm_hidden_dense_layer:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.hidden_activation = nn.Tanh()
        else:
            self.dense = None
            self.hidden_activation = None
        self.mlm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.dropout = nn.Dropout(config.dropout_prob)

        # The LM head weights require special treatment only when they are tied with the word embeddings
        self.update_keys_to_ignore(config, ["mlm_head.weight"])

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.mlm_head

    def set_output_embeddings(self, new_embeddings):
        self.mlm_head = new_embeddings

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
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        if self.dense is not None:
            sequence_output = self.dense(sequence_output)
            sequence_output = self.hidden_activation(sequence_output)
        prediction_scores = self.mlm_head(sequence_output)

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


@add_start_docstrings(
    """
    Mega Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    MEGA_START_DOCSTRING,
)
class MegaForSequenceClassification(MegaPreTrainedModel):
    _keys_to_ignore_on_load_missing = []

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
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
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
    _keys_to_ignore_on_load_missing = []

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
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.mega(
            flat_input_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
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
    _keys_to_ignore_on_load_missing = []

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
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
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
    _keys_to_ignore_on_load_missing = []

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.mega = MegaModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(MEGA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
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
