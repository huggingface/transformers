# coding=utf-8
# Copyright 2020 The Microsoft Authors and The HuggingFace Inc. team.
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
""" PyTorch ProphetNet model, ported from ProphetNet repo(fairseq version). """

import logging
import os

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from .activations import ACT2FN
from .configuration_prophetnet import ProphetNetConfig
from .file_utils import add_start_docstrings
from .modeling_utils import PreTrainedModel
from typing import Dict, List, Optional, Tuple
import numpy as np
import math
from .file_utils import add_start_docstrings, add_start_docstrings_to_callable, add_code_sample_docstrings
import warnings

logger = logging.getLogger(__name__)

_TOKENIZER_FOR_DOC = "ProphetNetTokenizer"

PROPHETNET_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/prophetnet-large-uncased",
    "microsoft/xprophetnet-large-wiki100-cased",
    # See all ProphetNet models at https://huggingface.co/models?filter=prophetnet
]


PROPHETNET_START_DOCSTRING = r"""

    Model and checkpoints are converted from ProphetNet and xProphetNet original Fairseq version repo. 
    Details can be found from <https://github.com/microsoft/ProphetNet>
    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class. Use it as a regular PyTorch Module and
    refer to the PyTorch documentation for all matters related to general usage and behavior.

    Parameters:
        config (:class:`~transformers.ProphetNetConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.

"""
PROPHETNET_GENERATION_EXAMPLE = r"""
    ProphetNet Summarization example::

        from transformers import ProphetNetTokenizer, ProphetNetForConditionalGeneration, ProphetNetConfig

        model = ProphetNetForConditionalGeneration.from_pretrained('microsoft/prophetnet-large-uncased-cnndm')
        tokenizer = ProphetNetTokenizer.from_pretrained('microsoft/prophetnet-large-uncased-cnndm')

        ARTICLE_TO_SUMMARIZE = "USTC was founded in Beijing by the Chinese Academy of Sciences (CAS) in September 1958. The Director of CAS, Mr. Guo Moruo was appointed the first president of USTC. USTC's founding mission was to develop a high-level science and technology workforce, as deemed critical for development of China's economy, defense, and science and technology education. The establishment was hailed as \"A Major Event in the History of Chinese Education and Science.\" CAS has supported USTC by combining most of its institutes with the departments of the university. USTC is listed in the top 16 national key universities, becoming the youngest national key university.".lower()
        inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=100, return_tensors='pt')

        # Generate Summary
        summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=512, early_stopping=True)
        print([tokenizer.decode(g) for g in summary_ids])
    
    xProphetNet xGLUE News Title Generation example:
    
        from transformers import ProphetNetTokenizer, ProphetNetForConditionalGeneration, ProphetNetConfig

        model = ProphetNetForConditionalGeneration.from_pretrained('microsoft/xprophetnet-large-wiki100-cased-xglue-ntg')
        tokenizer = ProphetNetTokenizer.from_pretrained('microsoft/xprophetnet-large-wiki100-cased-xglue-ntg')

        EN_SENTENCE = "Microsoft Corporation intends to officially end free support for the Windows 7 operating system after January 14, 2020, according to the official portal of the organization. From that day, users of this system will not be able to receive security updates, which could make their computers vulnerable to cyber attacks."
        RU_SENTENCE = "орпорация Microsoft намерена официально прекратить бесплатную поддержку операционной системы Windows 7 после 14 января 2020 года, сообщается на официальном портале организации . С указанного дня пользователи этой системы не смогут получать обновления безопасности, из-за чего их компьютеры могут стать уязвимыми к кибератакам."
        ZH_SENTENCE = "根据该组织的官方门户网站，微软公司打算在2020年1月14日之后正式终止对Windows 7操作系统的免费支持。从那时起，该系统的用户将无法接收安全更新，这可能会使他们的计算机容易受到网络攻击。"
        inputs = tokenizer([EN_SENTENCE, RU_SENTENCE, ZH_SENTENCE], padding=True, max_length=256, return_tensors='pt')

        # Generate Summary
        summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=100, early_stopping=True)
        print([tokenizer.decode(g) for g in summary_ids])    
"""

PROPHETNET_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
               Indices of input sequence tokens in the vocabulary. Use ProphetNetTokenizer.encode to produce them.
            Padding will be ignored by default should you provide it.
            Indices can be obtained using :class:`transformers.ProphetNetTokenizer.encode(text)`.
        attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices in input_ids.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        encoder_outputs (:obj:`tuple(tuple(torch.FloatTensor)`, `optional`, defaults to :obj:`None`):
            Tuple consists of (`last_hidden_state`, `optional`: `hidden_states`, `optional`: `attentions`)
            `last_hidden_state` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`) is a sequence of hidden-states at the output of the last layer of the encoder.
            Used in the cross-attention of the decoder.
        decoder_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`, defaults to :obj:`None`):
            Provide for translation and summarization training. By default, the model will create this tensor by shifting the input_ids right, following the paper.
        decoder_attention_mask (:obj:`torch.BoolTensor` of shape :obj:`(batch_size, tgt_seq_len)`, `optional`, defaults to :obj:`None`):
            Default behavior: generate a tensor that ignores pad tokens in decoder_input_ids. Causal mask will also be used by default.
            If you want to change padding behavior, you should read :func:`~transformers.modeling_bart._prepare_decoder_inputs` and modify.
            See diagram 1 in the paper for more info on the default strategy
        output_attentions (:obj:`bool`, `optional`, defaults to :obj:`None`):
            If set to ``True``, the attentions tensors of all attention layers are returned. See ``attentions`` under returned tensors for more detail.
"""

class ProphetNetPreTrainedModel(PreTrainedModel):
    config_class = ProphetNetConfig
    base_model_prefix = "model"

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            padding_idx: int,
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.onnx_trace = False

    def forward(self, input, use_cache=False, positions=None):
        """Input is expected to be of size [bsz x seqlen]."""
        assert (
            (positions is None) or (self.padding_idx is None)
        ), "If positions is pre-computed then padding_idx should not be set."

        if positions is None:
            if use_cache:
                # positions is the same for every token when decoding a single step
                # Without the int() cast, it doesn't work in some cases when exporting to ONNX
                positions = input.data.new(1, 1).fill_(int(self.padding_idx + input.size(1)))
            else:
                mask = input.data.ne(self.padding_idx).int()
                positions = ( torch.cumsum(mask, dim=1).type_as(mask) * mask ).long() + self.padding_idx
            real_positions = positions
        else:
            real_positions = positions
        return super().forward(positions), real_positions

    def max_positions(self):
        """Maximum number of supported positions."""
        if self.padding_idx is not None:
            return self.num_embeddings - self.padding_idx - 1
        else:
            return self.num_embeddings

    def _forward(self, positions):
        return super().forward(positions)

def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    if not export and torch.cuda.is_available():
        try:
            from apex.normalization import FusedLayerNorm
            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)

def invert_mask(attention_mask):
    assert attention_mask.dim() == 2
    return attention_mask.eq(0)


class SelfAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        encoder_decoder_attention=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.encoder_decoder_attention = encoder_decoder_attention
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.cache_key = "encoder_decoder" if self.encoder_decoder_attention else "self"

    def _shape(self, tensor, dim_0, bsz):
        return tensor.contiguous().view(dim_0, bsz * self.num_heads, self.head_dim).transpose(0, 1)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        layer_state: Optional[Dict[str, Optional[Tensor]]] = None,
        attn_mask: Optional[Tensor] = None,
        output_attentions=False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time(SeqLen) x Batch x Channel"""
        static_kv: bool = self.encoder_decoder_attention
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        # get here for encoder decoder cause of static_kv
        if layer_state is not None:  # reuse k,v and encoder_padding_mask
            saved_state = layer_state.get(self.cache_key, {})
            if "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute key and value if they are static
                if static_kv:
                    key = None
        else:
            saved_state = None
            layer_state = {}

        q = self.q_proj(query) * self.scaling
        if static_kv:
            if key is None:
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)
        else:
            k = self.k_proj(query)
            v = self.v_proj(query)

        q = self._shape(q, tgt_len, bsz)
        if k is not None:
            k = self._shape(k, -1, bsz)
        if v is not None:
            v = self._shape(v, -1, bsz)

        if saved_state is not None:
            k, v, key_padding_mask = self._use_saved_state(k, v, saved_state, key_padding_mask, static_kv, bsz)

        # Update cache
        layer_state[self.cache_key] = {
            "prev_key": k.view(bsz, self.num_heads, -1, self.head_dim),
            "prev_value": v.view(bsz, self.num_heads, -1, self.head_dim),
            "prev_key_padding_mask": key_padding_mask if not static_kv else None,
        }

        assert k is not None
        src_len = k.size(1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert attn_weights.size() == (bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attn_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # This is part of a workaround to get around fork/join parallelism not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None
        assert key_padding_mask is None or key_padding_mask.size()[:2] == (bsz, src_len,)

        if key_padding_mask is not None:  # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            reshaped = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(reshaped, float("-inf"))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training,)

        assert v is not None
        attn_output = torch.bmm(attn_probs, v)
        assert attn_output.size() == (bsz * self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)
        if output_attentions:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        else:
            attn_weights = None
        return attn_output, attn_weights

    def _use_saved_state(self, k, v, saved_state, key_padding_mask, static_kv, bsz):
        # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
        if "prev_key" in saved_state:
            _prev_key = saved_state["prev_key"]
            assert _prev_key is not None
            prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                k = prev_key
            else:
                assert k is not None
                k = torch.cat([prev_key, k], dim=1)
        if "prev_value" in saved_state:
            _prev_value = saved_state["prev_value"]
            assert _prev_value is not None
            prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                v = prev_value
            else:
                assert v is not None
                v = torch.cat([prev_value, v], dim=1)
        assert k is not None and v is not None
        prev_key_padding_mask: Optional[Tensor] = saved_state.get("prev_key_padding_mask", None)
        key_padding_mask = self._cat_prev_key_padding_mask(
            key_padding_mask, prev_key_padding_mask, bsz, k.size(1), static_kv
        )
        return k, v, key_padding_mask

    @staticmethod
    def _cat_prev_key_padding_mask(
        key_padding_mask: Optional[Tensor],
        prev_key_padding_mask: Optional[Tensor],
        batch_size: int,
        src_len: int,
        static_kv: bool,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None:
            if static_kv:
                new_key_padding_mask = prev_key_padding_mask
            else:
                new_key_padding_mask = torch.cat([prev_key_padding_mask, key_padding_mask], dim=1)

        elif key_padding_mask is not None:
            filler = torch.zeros(
                batch_size,
                src_len - key_padding_mask.size(1),
                dtype=key_padding_mask.dtype,
                device=key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat([filler, key_padding_mask], dim=1)
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask


class EncoderLayer(nn.Module):
    def __init__(self, config: ProphetNetConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = SelfAttention(
            self.embed_dim, config.encoder_attention_heads, dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(self, x, encoder_padding_mask, output_attentions=False):
        residual = x
        x, attn_weights = self.self_attn(
            query=x, key=x, key_padding_mask=encoder_padding_mask, output_attentions=output_attentions
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.self_attn_layer_norm(x)
        residual = x
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)
        return x, attn_weights



class ProphetNetEncoder(nn.Module):
    """
    ProphetNet Encoder is same to traditional Transformer Encoder.
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer
    is a :class:`EncoderLayer`.

    Args:
        config: ProphetNetConfig
    """
    def __init__(self, config: ProphetNetConfig, embed_tokens):
        super().__init__()

        self.output_hidden_states = config.output_hidden_states

        self.dropout = config.dropout
        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = config.max_position_embeddings

        self.embed_tokens = embed_tokens
        self.embed_scale = None
        self.embed_positions = LearnedPositionalEmbedding(config.max_position_embeddings+1+self.padding_idx, embed_dim, self.padding_idx)

        self.layers = nn.ModuleList([])
        self.layers.extend([
            EncoderLayer(config)
            for i in range(config.encoder_layers)
        ])
        self.emb_layer_norm = LayerNorm(embed_dim)

    def forward(self, input_ids, attention_mask=None, output_attentions=False):
        """

        Args:
            input_ids (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            attention_mask (torch.LongTensor): indicating which indices are padding tokens.
        Returns:
            Tuple comprised of:
                - **x** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *output_hidden_states:* is True.
                - **all_attentions** (List[Tensor]): Attention weights for each layer.
                During training might not be of length n_layers because of layer dropout.

        """

        # remove bos to be consistent with fairseq version
        input_ids = input_ids[:, 1:]
        if attention_mask is not None:
            attention_mask = attention_mask[:, 1:]

        if attention_mask is not None:
            attention_mask = invert_mask(attention_mask)

        inputs_embeds =self.embed_tokens(input_ids)
        embed_pos, real_positions = self.embed_positions(input_ids)
        x = inputs_embeds + embed_pos
        x = self.emb_layer_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states, all_attentions = [], []
        for encoder_layer in self.layers:
            if self.output_hidden_states:
                encoder_states.append(x)
            x, attn = encoder_layer(x, attention_mask, output_attentions=output_attentions)
            if output_attentions:
                all_attentions.append(attn)

        if self.output_hidden_states:
            encoder_states.append(x)

        # T x B x C -> B x T x C
        encoder_states = [hidden_state.transpose(0, 1) for hidden_state in encoder_states]
        x = x.transpose(0, 1)
        results = (x, encoder_states, all_attentions)
        return results

    def ngram_attention_bias(length, num_skip):
        bias_result = []
        for n_skip in range(num_skip):
            bias_n_skip = []
            for i in range(length):
                bias_this = [float('-inf')] * (2 * length)
                bias_this[length + i] = 0
                first_k = i - n_skip
                first_k = first_k if first_k > 0 else 0
                for j in range(first_k + 1):
                    bias_this[j] = 0
                bias_n_skip.append(bias_this)
            bias_result.append(bias_n_skip)
        return torch.from_numpy(np.array(bias_result, dtype=np.float32))

def softmax(x, dim, onnx_trace=False):
    if onnx_trace:
        return F.softmax(x.float(), dim=dim)
    else:
        return F.softmax(x, dim=dim, dtype=torch.float32)

class NgramMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None, dropout=0., bias=True,
                 add_bias_kv=False, add_zero_attn=False, self_attention=False,
                 encoder_decoder_attention=False, ngram=2, num_buckets=32, relative_max_distance=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_buckets = num_buckets
        self.relative_max_distance = relative_max_distance
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.ngram = ngram

        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, 'Self-attention requires query, key and ' \
                                                             'value to be of the same size'

        self.relative_linear = nn.Linear(embed_dim, num_buckets * num_heads)
        if self.qkv_same_dim:
            self.in_proj_weight = nn.Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        else:
            self.k_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.vdim))
            self.q_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))

        if bias:
            self.in_proj_bias = nn.Parameter(torch.Tensor(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.onnx_trace = False
        self.cache_key = "encoder_decoder" if self.encoder_decoder_attention else "self"
        assert self.cache_key == "self"

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)
            nn.init.xavier_uniform_(self.q_proj_weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def _relative_positions_bucket(self, relative_positions, bidirectional=False):
        num_buckets = self.num_buckets
        max_distance = self.relative_max_distance
        n = -relative_positions
        result = 0
        if bidirectional:
            num_buckets = num_buckets // 2
            result = result + torch.lt(n, torch.zeros_like(n)).int() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))
        max_exact = num_buckets // 2
        is_small = torch.lt(n, max_exact)
        val_if_large = max_exact + torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (
                num_buckets - max_exact)
        val_if_large = torch.min(val_if_large, torch.ones_like(val_if_large) * (num_buckets - 1))
        val_if_large = val_if_large.int()
        result = result + torch.where(is_small, n.int(), val_if_large)
        return result

    def main_stream_relative_logits(self, query, attn_weights, real_positions, i_bucket_main_stream):
        # input query [T,B,C]
        # input attn_weights [T*head,T,S]
        # input real_positions [B,T] or [1,1]

        T, B, _ = query.size()
        S = attn_weights.size(-1)

        if i_bucket_main_stream is not None:
            i_buckets = i_bucket_main_stream
        else:
            # [B,T,S]
            relative_positions = torch.arange(1, S + 1).unsqueeze(0).unsqueeze(0).repeat(B, T, 1).to(
                real_positions.device)
            # [B,T,1]
            real_positions = real_positions.unsqueeze(0).repeat(B, T, 1)
            # [B,T,S]
            relative_positions = relative_positions - real_positions
            # [B,T,T]
            i_buckets = self._relative_positions_bucket(relative_positions, False)

        # [B,T,C]
        query = query.transpose(0, 1)
        # [B,T,Buckets*head]
        values = self.relative_linear(query)
        # [B,T,Buckets,head]
        values = values.view(values.size(0), values.size(1), self.num_buckets, self.num_heads)
        # [B,head,Buckets,T]
        values = values.transpose(1, 3)
        # [B,head,T,Buckets]
        values = values.transpose(2, 3)
        # [B*head,T,Buckets]
        values = values.reshape(attn_weights.size(0), attn_weights.size(1), -1)

        # => [B,head*T,T] => [B*head,T,T]
        i_buckets = i_buckets.repeat(1, self.num_heads, 1).view(attn_weights.size(0), attn_weights.size(1), -1)
        # [B*head*T,Buckets]
        values = values.reshape(-1, values.size(-1))
        # [B*head*T,T]
        i_buckets = i_buckets.view(-1, i_buckets.size(-1)).long()
        # [B*head*T,T]
        result = torch.gather(values, dim=1, index=i_buckets)
        # [B*head,T,T]
        result = result.view(attn_weights.size(0), attn_weights.size(1), -1)

        return result

    def ngram_relative_logits(self, query, attn_weights, real_positions, i_bucket_relative_stream):
        # input query [ngram, T,B,C]
        # input attn_weights [ngram, B*head,T,S]
        # input real_positions [B,T] or [1,1]
        # input i_bucket_relative_stream [B,T, 2*T] or None

        N, T, B, _ = query.size()
        _, BH, _, S = attn_weights.size()

        if i_bucket_relative_stream is not None:
            i_buckets = i_bucket_relative_stream
        else:
            # [B,T,S]
            assert real_positions[0][0] == S - 1, 'memory position is 1 2 3 4 5(S-1)'
            relative_positions = torch.arange(0, S).unsqueeze(0).unsqueeze(0).repeat(B, T, 1).to(real_positions.device)
            # [B,T,1]
            real_positions = real_positions.unsqueeze(0).repeat(B, T, 1)
            relative_positions = relative_positions
            # [B,T,2*T] or [B,T,S]
            relative_positions = relative_positions - real_positions
            i_buckets = self._relative_positions_bucket(relative_positions, False)

        # [ngram, B, T, C]
        query = query.transpose(1, 2)
        # [ngram, B, T, bucket*head]
        values = self.relative_linear(query)
        # [ngram, B, T, bucket, head]
        values = values.view(*values.size()[:-1], self.num_buckets, self.num_heads)
        # [ngram, B, head, T, bucket]
        values = values.permute(0, 1, 4, 2, 3)
        # [ngram*B*head, T, bucket]
        values = values.reshape(N * BH, T, -1)

        # [ngram, B, head*T, S]
        i_buckets = i_buckets.unsqueeze(0).repeat(N, 1, self.num_heads, 1)

        values = values.reshape(-1, values.size(-1))
        i_buckets = i_buckets.view(-1, i_buckets.size(-1)).long()
        # [ngram*B*head*T, S]
        result = torch.gather(values, dim=1, index=i_buckets)
        # [ngram, B*head, T, S]
        result = result.view(N, BH, T, -1)

        return result

    def forward(self, query, key, value, key_padding_mask=None, layer_state=None,
                need_weights=True, static_kv=False,
                self_attn_mask=None,
                ngram_mask_matrix=None,
                i_buckets_main_stream=None,
                i_bucket_relative_stream=None,
                real_positions=None,
                output_attentions = False
                ):

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        if layer_state is not None:  # reuse k,v and encoder_padding_mask
            saved_state = layer_state.get(self.cache_key, {})
            if 'prev_key' in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None
            layer_state = {}

        q, k, v = self.in_proj_qkv(query)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        h_list = query.chunk(1 + self.ngram, dim=0)

        q_list = q.chunk(1 + self.ngram, dim=1)
        k_list = k.chunk(1 + self.ngram, dim=1)
        v_list = v.chunk(1 + self.ngram, dim=1)

        h_main, h_predict_list = h_list[0], h_list[1:]
        q_main, q_predict_list = q_list[0], q_list[1:]
        k_main, k_predict_list = k_list[0], k_list[1:]
        v_main, v_predict_list = v_list[0], v_list[1:]

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if 'prev_key' in saved_state:
                prev_key = saved_state['prev_key'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    assert False, 'static_kv not supprt in ngram decoder'
                    k = prev_key
                else:
                    k_main = torch.cat((prev_key, k_main), dim=1)
            if 'prev_value' in saved_state:
                prev_value = saved_state['prev_value'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    v_main = torch.cat((prev_value, v_main), dim=1)
            # Update cache
            layer_state[self.cache_key] = {
                "prev_key": k_main.view(bsz, self.num_heads, -1, self.head_dim),
                "prev_value": v_main.view(bsz, self.num_heads, -1, self.head_dim)
            }

        real_tgt_len = tgt_len // (1 + self.ngram)

        attn_weights_main = torch.bmm(q_main, k_main.transpose(1, 2))

        main_relative_logits = self.main_stream_relative_logits(h_main, attn_weights_main, real_positions,
                                                                i_buckets_main_stream)
        attn_weights_main = attn_weights_main + main_relative_logits

        if self_attn_mask is not None:
            self_attn_mask = self_attn_mask.unsqueeze(0)
            attn_weights_main = attn_weights_main + self_attn_mask

        attn_weights_main = softmax(
            attn_weights_main, dim=-1, onnx_trace=self.onnx_trace,
        ).type_as(attn_weights_main)
        attn_weights_main = F.dropout(attn_weights_main, p=self.dropout, training=self.training)

        attn_main = torch.bmm(attn_weights_main, v_main)
        attn_main = attn_main.transpose(0, 1).contiguous().view(1, real_tgt_len, bsz, embed_dim)
        attn_main = self.out_proj(attn_main)

        # [ngram, B*head, T, c]
        q_ngram = torch.cat(q_predict_list, 0).view(self.ngram, -1, real_tgt_len, self.head_dim)
        # [ngram, B*head, 2*T, c]
        k_ngram = torch.cat([torch.cat([k_main, k_p], 1).unsqueeze(0) for k_p in k_predict_list], 0)
        # below code slower than above for loop
        # k_ngram = torch.cat([k_main.unsqueeze(0).repeat(self.ngram, 1, 1, 1) , torch.cat(k_predict_list).view(self.ngram, -1, real_tgt_len, self.head_dim)], 2)

        # [ngram, T, B, C]
        h_ngram = torch.cat(h_predict_list, 0).view(self.ngram, real_tgt_len, bsz, embed_dim)

        # [ngram, B*head, 2*T, c]
        v_ngram = torch.cat([torch.cat([v_main, v_p], 1).unsqueeze(0) for v_p in v_predict_list], 0)
        # below code slower than above for loop
        # v_ngram = torch.cat([v_main.unsqueeze(0).repeat(self.ngram, 1, 1, 1) , torch.cat(v_predict_list).view(self.ngram, -1, real_tgt_len, self.head_dim)], 2)

        # [ngram, B*head, T, 2*T]
        attn_weights_ngram = torch.einsum('nbtc,nbsc->nbts', (q_ngram, k_ngram))

        # [ngram, B*head, T, S]
        predict_relative_logits = self.ngram_relative_logits(h_ngram, attn_weights_ngram, real_positions,
                                                             i_bucket_relative_stream)
        # [ngram, B*head, T, 2*T]
        attn_weights_ngram = attn_weights_ngram + predict_relative_logits

        if ngram_mask_matrix is not None:
            ngram_mask_matrix = ngram_mask_matrix.unsqueeze(1)
            attn_weights_ngram = attn_weights_ngram + ngram_mask_matrix

        attn_weights_ngram = softmax(
            attn_weights_ngram, dim=-1, onnx_trace=self.onnx_trace,
        ).type_as(attn_weights_ngram)
        attn_weights_ngram = F.dropout(attn_weights_ngram, p=self.dropout, training=self.training)

        # [ngram, B*head, T, c]
        attn_ngram = torch.einsum('nbts,nbsc->nbtc', (attn_weights_ngram, v_ngram))
        # [ngram, T, B, C]
        attn_ngram = attn_ngram.transpose(1, 2).contiguous().view(self.ngram, real_tgt_len, bsz, embed_dim)
        attn_ngram = self.out_proj(attn_ngram)

        attn_result = []
        attn_result.append(attn_main)
        attn_result.append(attn_ngram)

        # [1+ngram*T, B, C]
        attn = torch.cat(attn_result, 0).view(-1, bsz, embed_dim)

        if output_attentions:
            attn_weights = attn_weights_ngram#.view(bsz, self.num_heads, tgt_len, src_len)
        else:
            attn_weights = None

        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_q(self, query):
        if self.qkv_same_dim:
            return self._in_proj(query, end=self.embed_dim)
        else:
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[:self.embed_dim]
            return F.linear(query, self.q_proj_weight, bias)

    def in_proj_k(self, key):
        if self.qkv_same_dim:
            return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)
        else:
            weight = self.k_proj_weight
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[self.embed_dim:2 * self.embed_dim]
            return F.linear(key, weight, bias)

    def in_proj_v(self, value):
        if self.qkv_same_dim:
            return self._in_proj(value, start=2 * self.embed_dim)
        else:
            weight = self.v_proj_weight
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[2 * self.embed_dim:]
            return F.linear(value, weight, bias)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer[k] = input_buffer[k].index_select(0, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    def _get_input_buffer(self, incremental_state):
        return {}

    def _set_input_buffer(self, incremental_state, buffer):
        pass


class ProphetNetDecoderLayer(nn.Module):
    def __init__(self, config: ProphetNetConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.dropout = config.dropout
        self.activation_dropout = config.activation_dropout
        self.activation_fn = ACT2FN[config.activation_function]

        self.ngram_self_attn = NgramMultiheadAttention(
            self.embed_dim,
            config.num_attention_heads,
            dropout=config.attention_dropout,
            add_bias_kv=False,
            add_zero_attn=False,
            self_attention=True,
            ngram=config.ngram
        )
        self.encoder_attn = SelfAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            encoder_decoder_attention=True,
        )

        self.ngram = config.ngram
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(
        self,
        x,
        encoder_hidden_states,
        encoder_attn_mask=None,
        layer_state=None,
        self_attn_mask=None,
        output_attentions=False,
        ngram_mask_matrix=None,
        i_buckets_main_stream=None,
        i_bucket_relative_stream=None,
        real_positions=None
    ):
        # one main stream and ngram predicting streams
        residual = x

        if layer_state is None:
            layer_state = {}

        x, self_attn_weights = self.ngram_self_attn(
            query=x,
            key=x,
            value=x,
            layer_state=layer_state,
            need_weights=False,
            self_attn_mask=self_attn_mask,
            ngram_mask_matrix=ngram_mask_matrix,
            i_buckets_main_stream=i_buckets_main_stream,
            i_bucket_relative_stream=i_bucket_relative_stream,
            real_positions=real_positions,
            output_attentions=output_attentions
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.self_attn_layer_norm(x)
        residual = x
        x, _ = self.encoder_attn(
            query=x,
            key=encoder_hidden_states,
            key_padding_mask=encoder_attn_mask,
            layer_state=layer_state,  # mutates layer state
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.encoder_attn_layer_norm(x)

        # Fully Connected
        residual = x
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)

        return (
            x,
            self_attn_weights,
            layer_state,
        )  # just self_attn weights for now, following t5, layer_state = cache for decoding

def ngram_attention_bias(length, num_skip):
        bias_result = []
        for n_skip in range(num_skip):
            bias_n_skip = []
            for i in range(length):
                bias_this = [float('-inf')] * (2 * length)
                bias_this[length + i] = 0
                first_k = i - n_skip
                first_k = first_k if first_k > 0 else 0
                for j in range(first_k + 1):
                    bias_this[j] = 0
                bias_n_skip.append(bias_this)
            bias_result.append(bias_n_skip)
        return torch.from_numpy(np.array(bias_result, dtype=np.float32))

class ProphetNetDecoder(nn.Module):
    """
    N-stream decoder. One main stream, self.ngram predicting streams.
    Next self.ngram tokens are predicted.

    N-stream decoder consisting of *config.decoder_layers* layers. Each layer
    is a :class:`ProphetNetDecoderLayer`.
    Args:
        config: ProphetNetConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """
    def __init__(self, config: ProphetNetConfig, embed_tokens: nn.Embedding):
        super().__init__()
        self.ngram = config.ngram
        self.num_buckets = config.num_buckets
        self.relative_max_distance = config.relative_max_distance

        # for hugging face
        self.output_hidden_states = config.output_hidden_states

        self.dropout = config.dropout
        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = None
        self.embed_tokens = embed_tokens
        embed_dim = config.d_model
        self.embed_positions = LearnedPositionalEmbedding(config.max_position_embeddings + 2 + self.padding_idx,
                                                          embed_dim, self.padding_idx)
        self.ngram_input_embed = nn.Embedding(self.ngram, embed_dim, None)
        self.layers = nn.ModuleList([])
        self.layers.extend([
            ProphetNetDecoderLayer(config)
            for _ in range(config.decoder_layers)
        ])
        self.emb_layer_norm = LayerNorm(embed_dim)

    def _relative_positions_bucket(self, relative_positions, bidirectional=False):
        num_buckets = self.num_buckets
        max_distance = self.relative_max_distance
        n = -relative_positions
        result = 0
        if bidirectional:
            num_buckets = num_buckets // 2
            result = result + torch.lt(n, torch.zeros_like(n)).int() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))
        max_exact = num_buckets // 2
        is_small = torch.lt(n, max_exact)
        val_if_large = max_exact + torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (
                num_buckets - max_exact)
        val_if_large = torch.min(val_if_large, torch.ones_like(val_if_large) * (num_buckets - 1))
        val_if_large = val_if_large.int()
        result = result + torch.where(is_small, n.int(), val_if_large)
        return result

    def cal_pretrain_relative_positions(self, real_positions):
        # main stream
        main_stream_relative_positions = real_positions.unsqueeze(1)
        # [B,T,T/S]
        main_stream_relative_positions = main_stream_relative_positions.repeat(1, real_positions.size(-1), 1)
        # [B,T,1]
        real_positions_main = real_positions.unsqueeze(-1)
        main_stream_relative_positions = main_stream_relative_positions - real_positions_main

        # predicting stream
        # input shift
        real_positions_shift_predicting_stream = real_positions - 1
        # [B,1, 2*T]
        predicting_stream_relative_positions = torch.cat((real_positions_shift_predicting_stream, real_positions),
                                                         dim=-1).unsqueeze(1)
        # [B,T, 2*T]
        predicting_stream_relative_positions = predicting_stream_relative_positions.repeat(1, real_positions.size(-1),
                                                                                           1)
        # [B,T, 1]
        real_positions_predicting_stream = real_positions.unsqueeze(-1)
        predicting_stream_relative_positions = predicting_stream_relative_positions - real_positions_predicting_stream
        i_buckets_main_stream = self._relative_positions_bucket(main_stream_relative_positions, bidirectional=False)
        i_bucket_relative_stream = self._relative_positions_bucket(predicting_stream_relative_positions,
                                                                   bidirectional=False)
        return i_buckets_main_stream, i_bucket_relative_stream

    def cal_finetune_relative_positions(self, real_positions):
        n_tokens = real_positions.size(-1)
        batch_size = real_positions.size(0)
        if not hasattr(self,
                       '_finetune_i_bucket_main_stream') or \
                self._finetune_i_bucket_main_stream is None \
                or self._finetune_i_bucket_main_stream.device != real_positions.device:
            fake_positions = torch.arange(1, self.max_target_positions + 1).repeat(1, 1)
            finetune_i_bucket_main_stream, finetune_i_bucket_predicting_stream = \
                self.cal_pretrain_relative_positions(fake_positions)
            self._finetune_i_bucket_main_stream = finetune_i_bucket_main_stream.to(real_positions.device)
            self._finetune_i_bucket_predicting_stream = finetune_i_bucket_predicting_stream.to(real_positions.device)
        finetune_i_bucket_main_stream = self._finetune_i_bucket_main_stream[:, :n_tokens, :n_tokens].repeat(batch_size,
                                                                                                            1, 1)
        finetune_i_bucket_predicting_stream = torch.cat([
            self._finetune_i_bucket_predicting_stream[:, :n_tokens, :n_tokens],
            self._finetune_i_bucket_predicting_stream[:, :n_tokens,
            self.max_target_positions:self.max_target_positions + n_tokens]
        ], 2).repeat(batch_size, 1, 1)
        return finetune_i_bucket_main_stream, finetune_i_bucket_predicting_stream

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self,
                       '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device or self._future_mask.size(
            0) < dim:
            self._future_mask = torch.triu(fill_with_neg_inf(tensor.new(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def buffered_future_mask_ngram(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self,
                       '_ngram_future_mask') or self._ngram_future_mask is None or self._ngram_future_mask.device != tensor.device:
            self._ngram_future_mask = ngram_attention_bias(self.max_target_positions, self.ngram).type(tensor.dtype).to(
                tensor.device)
        ngram_future_mask = torch.cat([self._ngram_future_mask[:, :dim, :dim],
                                       self._ngram_future_mask[:, :dim,
                                       self.max_target_positions: self.max_target_positions + dim]
                                       ], 2)
        return ngram_future_mask

    def forward(
            self,
            input_ids,
            encoder_hidden_states,
            encoder_padding_mask,
            decoder_cached_states=None,
            use_cache=False,
            output_attentions=False,
            **unused,
    ):
        """

        Args:
            input_ids (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_hidden_states: output from the encoder, used for
                encoder-side attention
            encoder_padding_mask: for ignoring pad tokens
            decoder_cached_states (dict or None): dictionary used for storing state during generation
            use_cache: inference or training procedure.

        Returns:
            tuple:
                - the decoder's features of next n-grams, with shape `(batch, self.ngram * tgt_len, embed_dim)`
                - hidden states
                - attentions

        """
        if encoder_padding_mask is not None:
            # remove bos to be consistent with fairseq version
            encoder_padding_mask = encoder_padding_mask[:, 1:]
            encoder_padding_mask = invert_mask(encoder_padding_mask)
        main_stream_pos_embed, real_positions = self.embed_positions(
            input_ids,
            use_cache=use_cache
        )
        if use_cache:
            input_ids = input_ids[:, -1:]
            main_stream_pos_embed = main_stream_pos_embed[:, -1:]  # happens after we embed them
            i_buckets_main_stream, i_bucket_relative_stream = None, None
        else:
            i_buckets_main_stream, i_bucket_relative_stream = \
                self.cal_finetune_relative_positions(real_positions)
        predicting_stream_pos_embed = self.embed_positions._forward(real_positions + 1)
        x = self.embed_tokens(input_ids)
        if self.embed_scale is not None:
            x *= self.embed_scal
        x += main_stream_pos_embed
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        ngram_input_embed = self.ngram_input_embed.weight
        if use_cache:
            B = x.size(1)
            ngram_masks = [
                (ngram_input_embed[ngram - 1] + predicting_stream_pos_embed).transpose(0, 1).repeat(1, B, 1)
                for ngram in range(self.ngram)]
            self_attn_mask = None
            ngram_mask_matrix = None
        else:
            ngram_masks = [(ngram_input_embed[ngram - 1] + predicting_stream_pos_embed).transpose(0, 1) for
                           ngram in range(self.ngram)]
            self_attn_mask = self.buffered_future_mask(x)
            ngram_mask_matrix = self.buffered_future_mask_ngram(x)
        # TODO in train [(1+ngram)*T, B, C], in inference [T+ngram, B, C]
        x = torch.cat([x] + ngram_masks, 0)
        if self.emb_layer_norm:
            x = self.emb_layer_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)

        # decoder layers
        all_hidden_states = ()
        all_self_attns = ()
        next_decoder_cache = []
        for idx, decoder_layer in enumerate(self.layers):
            if self.output_hidden_states:
                all_hidden_states += (x,)
            layer_state = decoder_cached_states[idx] if decoder_cached_states is not None else None
            x, layer_self_attn, layer_past = decoder_layer(
                x,
                encoder_hidden_states,
                encoder_attn_mask=encoder_padding_mask,
                layer_state=layer_state,
                self_attn_mask=self_attn_mask,
                output_attentions=output_attentions,
                ngram_mask_matrix=ngram_mask_matrix,
                i_buckets_main_stream=i_buckets_main_stream,
                i_bucket_relative_stream=i_bucket_relative_stream,
                real_positions=real_positions
            )
            if use_cache:
                next_decoder_cache.append(layer_past.copy())
            if output_attentions:
                all_self_attns += (layer_self_attn,)
        # TODO [(1+ngram)*T, B, C] -> [B, (1+ngram)*T, C]
        x_list = x.transpose(0, 1).chunk(1 + self.ngram, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)
        if use_cache:
            next_cache = ((encoder_hidden_states, encoder_padding_mask), next_decoder_cache)
        else:
            next_cache = None
        return x_list, next_cache, all_hidden_states, list(all_self_attns)







def shift_tokens_right(input_ids, pad_token_id):
    """Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens

def make_padding_mask(input_ids, padding_idx=1):
    """True for pad tokens"""
    padding_mask = input_ids.eq(padding_idx)
    if not padding_mask.any():
        padding_mask = None
    return padding_mask

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a input_ids with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)

def _filter_out_falsey_values(tup) -> Tuple:
    """Remove entries that are None or [] from an iterable."""
    return tuple(x for x in tup if isinstance(x, torch.Tensor) or x)

@add_start_docstrings(
    "The bare ProphetNet Model transformer outputting raw hidden-states without any specific head on top.",
    PROPHETNET_START_DOCSTRING,
    PROPHETNET_INPUTS_DOCSTRING,
)
class ProphetNetModel(ProphetNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        padding_idx, vocab_size, dim_size = config.pad_token_id, config.vocab_size, config.d_model
        self.embed_tokens = nn.Embedding(vocab_size, dim_size, padding_idx=padding_idx)
        nn.init.normal_(self.embed_tokens.weight, mean=0, std=dim_size ** -0.5)
        nn.init.constant_(self.embed_tokens.weight[padding_idx], 0)
        self.encoder = ProphetNetEncoder(config, self.embed_tokens)
        self.decoder = ProphetNetDecoder(config, self.embed_tokens)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value
        self.encoder.embed_tokens = self.embed_tokens
        self.decoder.embed_tokens = self.embed_tokens

    def get_output_embeddings(self):
        return _make_linear_from_emb(self.embed_tokens)  # make it on the fly

    @add_start_docstrings_to_callable(PROPHETNET_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint="microsoft/prophetnet-large-uncased")
    def forward(
            self,
            input_ids,
            attention_mask=None,
            decoder_input_ids=None,
            encoder_outputs: Optional[Tuple] = None,
            decoder_cached_states=None,
            use_cache=False,
            output_attentions=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions


        assert decoder_input_ids is not None

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask, output_attentions=output_attentions,
            )
        assert isinstance(encoder_outputs, tuple)
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            decoder_input_ids,
            encoder_outputs[0],
            attention_mask,
            decoder_cached_states=decoder_cached_states,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        # Attention and hidden_states will be [] or None if they aren't needed
        decoder_outputs: Tuple = _filter_out_falsey_values(decoder_outputs)
        assert isinstance(decoder_outputs[0], tuple) # last layer hidden state of main stream and predicting streams
        encoder_outputs: Tuple = _filter_out_falsey_values(encoder_outputs)
        return decoder_outputs + encoder_outputs


def _reorder_buffer(attn_cache, new_order):
    for k, input_buffer_k in attn_cache.items():
        if input_buffer_k is not None:
            attn_cache[k] = input_buffer_k.index_select(0, new_order)
    return attn_cache

def _make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer

@add_start_docstrings(
    "The ProphetNet Model with a language modeling head. Can be used for summarization.", PROPHETNET_START_DOCSTRING
)
class ProphetNetForConditionalGeneration(ProphetNetPreTrainedModel):
    base_model_prefix = "model"

    def __init__(self, config: ProphetNetConfig):
        super().__init__(config)
        base_model = ProphetNetModel(config)
        self.model = base_model
        #self.padding_idx = config.pad_token_id
        self.padding_idx = -100
        self.disable_ngram_loss = config.disable_ngram_loss

    def output_layer(self, features, **kwargs):
        return F.linear(features, self.model.embed_tokens.weight)

    @add_start_docstrings_to_callable(PROPHETNET_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_cached_states=None,
        labels=None,
        use_cache=False,
        output_attentions=None,
        **unused,
    ):
        if "lm_labels" in unused:
            warnings.warn(
                "The `lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                DeprecationWarning,
            )
            labels = unused.pop("lm_labels")

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        predicting_streams = outputs[0][1:]
        predicting_streams_logits = [self.output_layer(x) for x in predicting_streams]
        #lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)

        if labels is not None:
            # fine-tune
            expend_targets = labels.new_zeros(self.config.ngram, labels.size(0), labels.size(1)).fill_(self.padding_idx)
            for i in range(self.config.ngram):
                if i > 0 and self.disable_ngram_loss:
                    break
                expend_targets[i, :, :] = labels
            logits = torch.cat(predicting_streams_logits, dim=0)
            lprobs = F.log_softmax(
                logits.view(-1, logits.size(-1)),
                dim=-1,
                dtype=torch.float32,
            )
            loss = F.nll_loss(
                lprobs,
                expend_targets.view(-1),
                reduction='sum',
                ignore_index=self.padding_idx,
            )
            if self.config.eps > 0.:
                smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
                non_pad_mask = expend_targets.ne(self.padding_idx).view(-1)
                smooth_loss = smooth_loss[non_pad_mask]
                smooth_loss = smooth_loss.sum()

                eps_i = self.config.eps / lprobs.size(-1)
                loss = (1. - self.config.eps) * loss + eps_i * smooth_loss
            outputs_loss = (loss,) + outputs
            return outputs_loss
        else:
            # inference
            outputs_logits = (predicting_streams_logits[0],) + outputs[1:]  # Add cache, hidden states and attention if they are here
            return outputs_logits

    def prepare_inputs_for_generation(self, decoder_input_ids, past, attention_mask, use_cache, **kwargs):
        assert past is not None, "past has to be defined for encoder_outputs"

        # first step, decoder_cached_states are empty
        encoder_outputs, decoder_cached_states = past
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "decoder_cached_states": decoder_cached_states,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }

    def prepare_logits_for_generation(self, logits, cur_len, max_length):
        if cur_len == 1:
            self._force_token_ids_generation(logits, self.config.bos_token_id)
        if cur_len == max_length - 1 and self.config.eos_token_id is not None:
            self._force_token_ids_generation(logits, self.config.eos_token_id)
        return logits

    def _force_token_ids_generation(self, scores, token_ids) -> None:
        """force one of token_ids to be generated by setting prob of all other tokens to 0"""
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        all_but_token_ids_mask = torch.tensor(
            [x for x in range(self.config.vocab_size) if x not in token_ids],
            dtype=torch.long,
            device=next(self.parameters()).device,
        )
        assert len(scores.shape) == 2, "scores should be of rank 2 with shape: [batch_size, vocab_size]"
        scores[:, all_but_token_ids_mask] = -float("inf")

    @staticmethod
    def _reorder_cache(past, beam_idx):
        ((enc_out, enc_mask), decoder_cached_states) = past
        reordered_past = []
        for layer_past in decoder_cached_states:
            # get the correct batch idx from decoder layer's batch dim for cross and self-attn
            layer_past_new = {
                attn_key: _reorder_buffer(attn_cache, beam_idx) for attn_key, attn_cache in layer_past.items()
            }
            reordered_past.append(layer_past_new)

        new_enc_out = enc_out if enc_out is None else enc_out.index_select(0, beam_idx)
        new_enc_mask = enc_mask if enc_mask is None else enc_mask.index_select(0, beam_idx)

        past = ((new_enc_out, new_enc_mask), reordered_past)
        return past

    def get_encoder(self):
        return self.model.encoder

    def get_output_embeddings(self):
        return _make_linear_from_emb(self.model.embed_tokens)  # make it on the fly
