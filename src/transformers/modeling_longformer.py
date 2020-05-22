# coding=utf-8
# Copyright 2020 The Allen Institute for AI team and The HuggingFace Inc. team.
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
"""PyTorch Longformer model. """

import logging
import math

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F

from .configuration_longformer import LongformerConfig
from .file_utils import add_start_docstrings, add_start_docstrings_to_callable
from .modeling_bert import BertPreTrainedModel
from .modeling_roberta import RobertaLMHead, RobertaModel


logger = logging.getLogger(__name__)

LONGFORMER_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "longformer-base-4096": "https://s3.amazonaws.com/models.huggingface.co/bert/allenai/longformer-base-4096/pytorch_model.bin",
    "longformer-large-4096": "https://s3.amazonaws.com/models.huggingface.co/bert/allenai/longformer-large-4096/pytorch_model.bin",
}


class LongformerSelfAttention(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.output_attentions = config.output_attentions
        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size / config.num_attention_heads)
        self.embed_dim = config.hidden_size

        self.query = nn.Linear(config.hidden_size, self.embed_dim)
        self.key = nn.Linear(config.hidden_size, self.embed_dim)
        self.value = nn.Linear(config.hidden_size, self.embed_dim)

        # separate projection layers for tokens with global attention
        self.query_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.key_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.value_global = nn.Linear(config.hidden_size, self.embed_dim)

        self.dropout = config.attention_probs_dropout_prob

        self.layer_id = layer_id
        attention_window = config.attention_window[self.layer_id]
        assert (
            attention_window % 2 == 0
        ), f"`attention_window` for layer {self.layer_id} has to be an even value. Given {attention_window}"
        assert (
            attention_window > 0
        ), f"`attention_window` for layer {self.layer_id} has to be positive. Given {attention_window}"

        self.one_sided_attention_window_size = attention_window // 2

    @staticmethod
    def _skew(x, direction):
        """Convert diagonals into columns (or columns into diagonals depending on `direction`"""
        x_padded = F.pad(x, direction)  # padding value is not important because it will be overwritten
        x_padded = x_padded.view(*x_padded.size()[:-2], x_padded.size(-1), x_padded.size(-2))
        return x_padded

    @staticmethod
    def _skew2(x):
        """shift every row 1 step to right converting columns into diagonals"""
        # X = B x C x M x L
        B, C, M, L = x.size()
        x = F.pad(x, (0, M + 1))  # B x C x M x (L+M+1). Padding value is not important because it'll be overwritten
        x = x.view(B, C, -1)  # B x C x ML+MM+M
        x = x[:, :, :-M]  # B x C x ML+MM
        x = x.view(B, C, M, M + L)  # B x C, M x L+M
        x = x[:, :, :, :-1]
        return x

    @staticmethod
    def _chunk(x, w):
        """convert into overlapping chunkings. Chunk size = 2w, overlap size = w"""

        # non-overlapping chunks of size = 2w
        x = x.view(x.size(0), x.size(1) // (w * 2), w * 2, x.size(2))

        # use `as_strided` to make the chunks overlap with an overlap size = w
        chunk_size = list(x.size())
        chunk_size[1] = chunk_size[1] * 2 - 1

        chunk_stride = list(x.stride())
        chunk_stride[1] = chunk_stride[1] // 2
        return x.as_strided(size=chunk_size, stride=chunk_stride)

    def _mask_invalid_locations(self, input_tensor, w) -> torch.Tensor:
        affected_seqlen = w
        beginning_mask_2d = input_tensor.new_ones(w, w + 1).tril().flip(dims=[0])
        beginning_mask = beginning_mask_2d[None, :, None, :]
        ending_mask = beginning_mask.flip(dims=(1, 3))
        seqlen = input_tensor.size(1)
        beginning_input = input_tensor[:, :affected_seqlen, :, : w + 1]
        beginning_mask = beginning_mask[:, :seqlen].expand(beginning_input.size())
        beginning_input.masked_fill_(beginning_mask == 1, -float("inf"))  # `== 1` converts to bool or uint8
        ending_input = input_tensor[:, -affected_seqlen:, :, -(w + 1) :]
        ending_mask = ending_mask[:, -seqlen:].expand(ending_input.size())
        ending_input.masked_fill_(ending_mask == 1, -float("inf"))  # `== 1` converts to bool or uint8

    def _sliding_chunks_matmul_qk(self, q: torch.Tensor, k: torch.Tensor, w: int):
        """Matrix multiplicatio of query x key tensors using with a sliding window attention pattern.
        This implementation splits the input into overlapping chunks of size 2w (e.g. 512 for pretrained Longformer)
        with an overlap of size w"""
        batch_size, seqlen, num_heads, head_dim = q.size()
        assert seqlen % (w * 2) == 0, f"Sequence length should be multiple of {w * 2}. Given {seqlen}"
        assert q.size() == k.size()

        chunks_count = seqlen // w - 1

        # group batch_size and num_heads dimensions into one, then chunk seqlen into chunks of size w * 2
        q = q.transpose(1, 2).reshape(batch_size * num_heads, seqlen, head_dim)
        k = k.transpose(1, 2).reshape(batch_size * num_heads, seqlen, head_dim)

        chunk_q = self._chunk(q, w)
        chunk_k = self._chunk(k, w)

        # matrix multipication
        # bcxd: batch_size * num_heads x chunks x 2w x head_dim
        # bcyd: batch_size * num_heads x chunks x 2w x head_dim
        # bcxy: batch_size * num_heads x chunks x 2w x 2w
        chunk_attn = torch.einsum("bcxd,bcyd->bcxy", (chunk_q, chunk_k))  # multiply

        # convert diagonals into columns
        diagonal_chunk_attn = self._skew(chunk_attn, direction=(0, 0, 0, 1))

        # allocate space for the overall attention matrix where the chunks are compined. The last dimension
        # has (w * 2 + 1) columns. The first (w) columns are the w lower triangles (attention from a word to
        # w previous words). The following column is attention score from each word to itself, then
        # followed by w columns for the upper triangle.

        diagonal_attn = diagonal_chunk_attn.new_empty((batch_size * num_heads, chunks_count + 1, w, w * 2 + 1))

        # copy parts from diagonal_chunk_attn into the compined matrix of attentions
        # - copying the main diagonal and the upper triangle
        diagonal_attn[:, :-1, :, w:] = diagonal_chunk_attn[:, :, :w, : w + 1]
        diagonal_attn[:, -1, :, w:] = diagonal_chunk_attn[:, -1, w:, : w + 1]
        # - copying the lower triangle
        diagonal_attn[:, 1:, :, :w] = diagonal_chunk_attn[:, :, -(w + 1) : -1, w + 1 :]
        diagonal_attn[:, 0, 1:w, 1:w] = diagonal_chunk_attn[:, 0, : w - 1, 1 - w :]

        # separate batch_size and num_heads dimensions again
        diagonal_attn = diagonal_attn.view(batch_size, num_heads, seqlen, 2 * w + 1).transpose(2, 1)

        self._mask_invalid_locations(diagonal_attn, w)
        return diagonal_attn

    def _sliding_chunks_matmul_pv(self, prob: torch.Tensor, v: torch.Tensor, w: int):
        """Same as _sliding_chunks_matmul_qk but for prob and value tensors. It is expecting the same output
        format from _sliding_chunks_matmul_qk"""
        batch_size, seqlen, num_heads, head_dim = v.size()
        assert seqlen % (w * 2) == 0
        assert prob.size()[:3] == v.size()[:3]
        assert prob.size(3) == 2 * w + 1
        chunks_count = seqlen // w - 1
        # group batch_size and num_heads dimensions into one, then chunk seqlen into chunks of size 2w
        chunk_prob = prob.transpose(1, 2).reshape(batch_size * num_heads, seqlen // w, w, 2 * w + 1)

        # group batch_size and num_heads dimensions into one
        v = v.transpose(1, 2).reshape(batch_size * num_heads, seqlen, head_dim)

        # pad seqlen with w at the beginning of the sequence and another w at the end
        padded_v = F.pad(v, (0, 0, w, w), value=-1)

        # chunk padded_v into chunks of size 3w and an overlap of size w
        chunk_v_size = (batch_size * num_heads, chunks_count + 1, 3 * w, head_dim)
        chunk_v_stride = padded_v.stride()
        chunk_v_stride = chunk_v_stride[0], w * chunk_v_stride[1], chunk_v_stride[1], chunk_v_stride[2]
        chunk_v = padded_v.as_strided(size=chunk_v_size, stride=chunk_v_stride)

        skewed_prob = self._skew2(chunk_prob)

        context = torch.einsum("bcwd,bcdh->bcwh", (skewed_prob, chunk_v))
        return context.view(batch_size, num_heads, seqlen, head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        """
        LongformerSelfAttention expects `len(hidden_states)` to be multiple of `attention_window`.
        Padding to `attention_window` happens in LongformerModel.forward to avoid redoing the padding on each layer.

        The `attention_mask` is changed in `BertModel.forward` from 0, 1, 2 to
            -ve: no attention
              0: local attention
            +ve: global attention

        `encoder_hidden_states` and `encoder_attention_mask` are not supported and should be None
        """
        # TODO: add support for `encoder_hidden_states` and `encoder_attention_mask`
        assert encoder_hidden_states is None, "`encoder_hidden_states` is not supported and should be None"
        assert encoder_attention_mask is None, "`encoder_attention_mask` is not supported and shiould be None"

        if attention_mask is not None:
            attention_mask = attention_mask.squeeze(dim=2).squeeze(dim=1)
            key_padding_mask = attention_mask < 0
            extra_attention_mask = attention_mask > 0
            remove_from_windowed_attention_mask = attention_mask != 0

            num_extra_indices_per_batch = extra_attention_mask.long().sum(dim=1)
            max_num_extra_indices_per_batch = num_extra_indices_per_batch.max()
            if max_num_extra_indices_per_batch <= 0:
                extra_attention_mask = None
            else:
                # To support the case of variable number of global attention in the rows of a batch,
                # we use the following three selection masks to select global attention embeddings
                # in a 3d tensor and pad it to `max_num_extra_indices_per_batch`
                # 1) selecting embeddings that correspond to global attention
                extra_attention_mask_nonzeros = extra_attention_mask.nonzero(as_tuple=True)
                zero_to_max_range = torch.arange(
                    0, max_num_extra_indices_per_batch, device=num_extra_indices_per_batch.device
                )
                # mask indicating which values are actually going to be padding
                selection_padding_mask = zero_to_max_range < num_extra_indices_per_batch.unsqueeze(dim=-1)
                # 2) location of the non-padding values in the selected global attention
                selection_padding_mask_nonzeros = selection_padding_mask.nonzero(as_tuple=True)
                # 3) location of the padding values in the selected global attention
                selection_padding_mask_zeros = (selection_padding_mask == 0).nonzero(as_tuple=True)
        else:
            remove_from_windowed_attention_mask = None
            extra_attention_mask = None
            key_padding_mask = None

        hidden_states = hidden_states.transpose(0, 1)
        seqlen, batch_size, embed_dim = hidden_states.size()
        assert embed_dim == self.embed_dim
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)
        q /= math.sqrt(self.head_dim)

        q = q.view(seqlen, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(seqlen, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        # attn_weights = (batch_size, seqlen, num_heads, window*2+1)
        attn_weights = self._sliding_chunks_matmul_qk(q, k, self.one_sided_attention_window_size)
        self._mask_invalid_locations(attn_weights, self.one_sided_attention_window_size)
        if remove_from_windowed_attention_mask is not None:
            # This implementation is fast and takes very little memory because num_heads x hidden_size = 1
            # from (batch_size x seqlen) to (batch_size x seqlen x num_heads x hidden_size)
            remove_from_windowed_attention_mask = remove_from_windowed_attention_mask.unsqueeze(dim=-1).unsqueeze(
                dim=-1
            )
            # cast to fp32/fp16 then replace 1's with -inf
            float_mask = remove_from_windowed_attention_mask.type_as(q).masked_fill(
                remove_from_windowed_attention_mask, -10000.0
            )
            ones = float_mask.new_ones(size=float_mask.size())  # tensor of ones
            # diagonal mask with zeros everywhere and -inf inplace of padding
            d_mask = self._sliding_chunks_matmul_qk(ones, float_mask, self.one_sided_attention_window_size)
            attn_weights += d_mask
        assert list(attn_weights.size()) == [
            batch_size,
            seqlen,
            self.num_heads,
            self.one_sided_attention_window_size * 2 + 1,
        ]

        # the extra attention
        if extra_attention_mask is not None:
            selected_k = k.new_zeros(batch_size, max_num_extra_indices_per_batch, self.num_heads, self.head_dim)
            selected_k[selection_padding_mask_nonzeros] = k[extra_attention_mask_nonzeros]
            # (batch_size, seqlen, num_heads, max_num_extra_indices_per_batch)
            selected_attn_weights = torch.einsum("blhd,bshd->blhs", (q, selected_k))
            selected_attn_weights[selection_padding_mask_zeros[0], :, :, selection_padding_mask_zeros[1]] = -10000
            # concat to attn_weights
            # (batch_size, seqlen, num_heads, extra attention count + 2*window+1)
            attn_weights = torch.cat((selected_attn_weights, attn_weights), dim=-1)

        attn_weights_fp32 = F.softmax(attn_weights, dim=-1, dtype=torch.float32)  # use fp32 for numerical stability
        attn_weights = attn_weights_fp32.type_as(attn_weights)

        if key_padding_mask is not None:
            # softmax sometimes inserts NaN if all positions are masked, replace them with 0
            attn_weights = torch.masked_fill(attn_weights, key_padding_mask.unsqueeze(-1).unsqueeze(-1), 0.0)

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)
        v = v.view(seqlen, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        attn = None
        if extra_attention_mask is not None:
            selected_attn_probs = attn_probs.narrow(-1, 0, max_num_extra_indices_per_batch)
            selected_v = v.new_zeros(batch_size, max_num_extra_indices_per_batch, self.num_heads, self.head_dim)
            selected_v[selection_padding_mask_nonzeros] = v[extra_attention_mask_nonzeros]
            # use `matmul` because `einsum` crashes sometimes with fp16
            # attn = torch.einsum('blhs,bshd->blhd', (selected_attn_probs, selected_v))
            attn = torch.matmul(
                selected_attn_probs.transpose(1, 2), selected_v.transpose(1, 2).type_as(selected_attn_probs)
            ).transpose(1, 2)
            attn_probs = attn_probs.narrow(
                -1, max_num_extra_indices_per_batch, attn_probs.size(-1) - max_num_extra_indices_per_batch
            ).contiguous()
        if attn is None:
            attn = self._sliding_chunks_matmul_pv(attn_probs, v, self.one_sided_attention_window_size)
        else:
            attn += self._sliding_chunks_matmul_pv(attn_probs, v, self.one_sided_attention_window_size)

        assert attn.size() == (batch_size, seqlen, self.num_heads, self.head_dim), "Unexpected size"
        attn = attn.transpose(0, 1).reshape(seqlen, batch_size, embed_dim).contiguous()

        # For this case, we'll just recompute the attention for these indices
        # and overwrite the attn tensor.
        # TODO: remove the redundant computation
        if extra_attention_mask is not None:
            selected_hidden_states = hidden_states.new_zeros(max_num_extra_indices_per_batch, batch_size, embed_dim)
            selected_hidden_states[selection_padding_mask_nonzeros[::-1]] = hidden_states[
                extra_attention_mask_nonzeros[::-1]
            ]

            q = self.query_global(selected_hidden_states)
            k = self.key_global(hidden_states)
            v = self.value_global(hidden_states)
            q /= math.sqrt(self.head_dim)

            q = (
                q.contiguous()
                .view(max_num_extra_indices_per_batch, batch_size * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )  # (batch_size * self.num_heads, max_num_extra_indices_per_batch, head_dim)
            k = (
                k.contiguous().view(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
            )  # batch_size * self.num_heads, seqlen, head_dim)
            v = (
                v.contiguous().view(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
            )  # batch_size * self.num_heads, seqlen, head_dim)
            attn_weights = torch.bmm(q, k.transpose(1, 2))
            assert list(attn_weights.size()) == [batch_size * self.num_heads, max_num_extra_indices_per_batch, seqlen]

            attn_weights = attn_weights.view(batch_size, self.num_heads, max_num_extra_indices_per_batch, seqlen)
            attn_weights[selection_padding_mask_zeros[0], :, selection_padding_mask_zeros[1], :] = -10000.0
            if key_padding_mask is not None:
                attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), -10000.0,)
            attn_weights = attn_weights.view(batch_size * self.num_heads, max_num_extra_indices_per_batch, seqlen)
            attn_weights_float = F.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            )  # use fp32 for numerical stability
            attn_probs = F.dropout(attn_weights_float.type_as(attn_weights), p=self.dropout, training=self.training)
            selected_attn = torch.bmm(attn_probs, v)
            assert list(selected_attn.size()) == [
                batch_size * self.num_heads,
                max_num_extra_indices_per_batch,
                self.head_dim,
            ]

            selected_attn_4d = selected_attn.view(
                batch_size, self.num_heads, max_num_extra_indices_per_batch, self.head_dim
            )
            nonzero_selected_attn = selected_attn_4d[
                selection_padding_mask_nonzeros[0], :, selection_padding_mask_nonzeros[1]
            ]
            attn[extra_attention_mask_nonzeros[::-1]] = nonzero_selected_attn.view(
                len(selection_padding_mask_nonzeros[0]), -1
            ).type_as(hidden_states)

        context_layer = attn.transpose(0, 1)
        if self.output_attentions:
            if extra_attention_mask is not None:
                # With global attention, return global attention probabilities only
                # batch_size x num_heads x max_num_global_attention_tokens x sequence_length
                # which is the attention weights from tokens with global attention to all tokens
                # It doesn't not return local attention
                # In case of variable number of global attantion in the rows of a batch,
                # attn_weights are padded with -10000.0 attention scores
                attn_weights = attn_weights.view(batch_size, self.num_heads, max_num_extra_indices_per_batch, seqlen)
            else:
                # without global attention, return local attention probabilities
                # batch_size x num_heads x sequence_length x window_size
                # which is the attention weights of every token attending to its neighbours
                attn_weights = attn_weights.permute(0, 2, 1, 3)
        outputs = (context_layer, attn_weights) if self.output_attentions else (context_layer,)
        return outputs


LONGFORMER_START_DOCSTRING = r"""

    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config (:class:`~transformers.LongformerConfig`): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

LONGFORMER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`transformers.LonmgformerTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.encode_plus` for details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to decide the attention given on each token, local attention, global attenion, or no attention (for padding tokens).
            Tokens with global attention attends to all other tokens, and all other tokens attend to them. This is important for
            task-specific finetuning because it makes the model more flexible at representing the task. For example,
            for classification, the <s> token should be given global attention. For QA, all question tokens should also have
            global attention. Please refer to the Longformer paper https://arxiv.org/abs/2004.05150 for more details.
            Mask values selected in ``[0, 1, 2]``:
            ``0`` for no attention (padding tokens),
            ``1`` for local attention (a sliding window attention),
            ``2`` for global attention (tokens that attend to all other tokens, and all other tokens attend to them).

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
"""


@add_start_docstrings(
    "The bare Longformer Model outputting raw hidden-states without any specific head on top.",
    LONGFORMER_START_DOCSTRING,
)
class LongformerModel(RobertaModel):
    """
    This class overrides :class:`~transformers.RobertaModel` to provide the ability to process
    long sequences following the selfattention approach described in `Longformer: the Long-Document Transformer`_by
    Iz Beltagy, Matthew E. Peters, and Arman Cohan. Longformer selfattention combines a local (sliding window)
    and global attention to extend to long documents without the O(n^2) increase in memory and compute.

    The selfattention module `LongformerSelfAttention` implemented here supports the combination of local and
    global attention but it lacks support for autoregressive attention and dilated attention. Autoregressive
    and dilated attention are more relevant for autoregressive language modeling than finetuning on downstream
    tasks. Future release will add support for autoregressive attention, but the support for dilated attention
    requires a custom CUDA kernel to be memory and compute efficient.

    .. _`Longformer: the Long-Document Transformer`:
        https://arxiv.org/abs/2004.05150

    """

    config_class = LongformerConfig
    pretrained_model_archive_map = LONGFORMER_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "longformer"

    def __init__(self, config):
        super().__init__(config)

        if isinstance(config.attention_window, int):
            assert config.attention_window % 2 == 0, "`config.attention_window` has to be an even value"
            assert config.attention_window > 0, "`config.attention_window` has to be positive"
            config.attention_window = [config.attention_window] * config.num_hidden_layers  # one value per layer
        else:
            assert len(config.attention_window) == config.num_hidden_layers, (
                "`len(config.attention_window)` should equal `config.num_hidden_layers`. "
                f"Expected {config.num_hidden_layers}, given {len(config.attention_window)}"
            )

        for i, layer in enumerate(self.encoder.layer):
            # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
            layer.attention.self = LongformerSelfAttention(config, layer_id=i)

        self.init_weights()

    def _pad_to_window_size(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        position_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_window: int,
        pad_token_id: int,
    ):
        """A helper function to pad tokens and mask to work with implementation of Longformer selfattention."""

        assert attention_window % 2 == 0, f"`attention_window` should be an even value. Given {attention_window}"
        input_shape = input_ids.shape if input_ids is not None else inputs_embeds.shape
        batch_size, seqlen = input_shape[:2]

        padding_len = (attention_window - seqlen % attention_window) % attention_window
        if padding_len > 0:
            logger.info(
                "Input ids are automatically padded from {} to {} to be a multiple of `config.attention_window`: {}".format(
                    seqlen, seqlen + padding_len, attention_window
                )
            )
            if input_ids is not None:
                input_ids = F.pad(input_ids, (0, padding_len), value=pad_token_id)
            if attention_mask is not None:
                attention_mask = F.pad(
                    attention_mask, (0, padding_len), value=False
                )  # no attention on the padding tokens
            if token_type_ids is not None:
                token_type_ids = F.pad(token_type_ids, (0, padding_len), value=0)  # pad with token_type_id = 0
            if position_ids is not None:
                # pad with position_id = pad_token_id as in modeling_roberta.RobertaEmbeddings
                position_ids = F.pad(position_ids, (0, padding_len), value=pad_token_id)
            if inputs_embeds is not None:
                input_ids_padding = inputs_embeds.new_full(
                    (batch_size, padding_len), self.config.pad_token_id, dtype=torch.long,
                )
                inputs_embeds_padding = self.embeddings(input_ids_padding)
                inputs_embeds = torch.cat([inputs_embeds, inputs_embeds_padding], dim=-2)

        return padding_len, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds

    @add_start_docstrings_to_callable(LONGFORMER_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        masked_lm_labels=None,
    ):
        r"""

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        masked_lm_loss (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        import torch
        from transformers import LongformerModel, LongformerTokenizer

        model = LongformerModel.from_pretrained('longformer-base-4096')
        tokenizer = LongformerTokenizer.from_pretrained('longformer-base-4096')

        SAMPLE_TEXT = ' '.join(['Hello world! '] * 1000)  # long input document
        input_ids = torch.tensor(tokenizer.encode(SAMPLE_TEXT)).unsqueeze(0)  # batch of size 1

        # Attention mask values -- 0: no attention, 1: local attention, 2: global attention
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device) # initialize to local attention
        attention_mask[:, [1, 4, 21,]] = 2  # Set global attention based on the task. For example,
                                            # classification: the <s> token
                                            # QA: question tokens
                                            # LM: potentially on the beginning of sentences and paragraphs
        sequence_output, pooled_output = model(input_ids, attention_mask=attention_mask)
        """

        # padding
        attention_window = (
            self.config.attention_window
            if isinstance(self.config.attention_window, int)
            else max(self.config.attention_window)
        )
        padding_len, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds = self._pad_to_window_size(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            attention_window=attention_window,
            pad_token_id=self.config.pad_token_id,
        )

        # embed
        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=None,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
        )

        # undo padding
        if padding_len > 0:
            # `output` has the following tensors: sequence_output, pooled_output, (hidden_states), (attentions)
            # `sequence_output`: unpad because the calling function is expecting a length == input_ids.size(1)
            # `pooled_output`: independent of the sequence length
            # `hidden_states`: mainly used for debugging and analysis, so keep the padding
            # `attentions`: mainly used for debugging and analysis, so keep the padding
            output = output[0][:, :-padding_len], *output[1:]

        return output


@add_start_docstrings("""Longformer Model with a `language modeling` head on top. """, LONGFORMER_START_DOCSTRING)
class LongformerForMaskedLM(BertPreTrainedModel):
    config_class = LongformerConfig
    pretrained_model_archive_map = LONGFORMER_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "longformer"

    def __init__(self, config):
        super().__init__(config)

        self.longformer = LongformerModel(config)
        self.lm_head = RobertaLMHead(config)

        self.init_weights()

    @add_start_docstrings_to_callable(LONGFORMER_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        masked_lm_labels=None,
    ):
        r"""
        masked_lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        masked_lm_loss (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        import torch
        from transformers import LongformerForMaskedLM, LongformerTokenizer

        model = LongformerForMaskedLM.from_pretrained('longformer-base-4096')
        tokenizer = LongformerTokenizer.from_pretrained('longformer-base-4096')

        SAMPLE_TEXT = ' '.join(['Hello world! '] * 1000)  # long input document
        input_ids = torch.tensor(tokenizer.encode(SAMPLE_TEXT)).unsqueeze(0)  # batch of size 1

        attention_mask = None  # default is local attention everywhere, which is a good choice for MaskedLM
                               # check ``LongformerModel.forward`` for more details how to set `attention_mask`
        loss, prediction_scores = model(input_ids, attention_mask=attention_mask, masked_lm_labels=input_ids)
        """

        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)
