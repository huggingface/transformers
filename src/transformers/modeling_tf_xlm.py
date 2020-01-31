# coding=utf-8
# Copyright 2019-present, Facebook, Inc and the HuggingFace Inc. team.
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
""" TF 2.0 XLM model.
"""


import itertools
import logging
import math

import numpy as np
import tensorflow as tf

from .configuration_xlm import XLMConfig
from .file_utils import add_start_docstrings, add_start_docstrings_to_callable
from .modeling_tf_utils import TFPreTrainedModel, TFSequenceSummary, TFSharedEmbeddings, get_initializer, shape_list


logger = logging.getLogger(__name__)

TF_XLM_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "xlm-mlm-en-2048": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-en-2048-tf_model.h5",
    "xlm-mlm-ende-1024": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-ende-1024-tf_model.h5",
    "xlm-mlm-enfr-1024": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-enfr-1024-tf_model.h5",
    "xlm-mlm-enro-1024": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-enro-1024-tf_model.h5",
    "xlm-mlm-tlm-xnli15-1024": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-tlm-xnli15-1024-tf_model.h5",
    "xlm-mlm-xnli15-1024": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-xnli15-1024-tf_model.h5",
    "xlm-clm-enfr-1024": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-clm-enfr-1024-tf_model.h5",
    "xlm-clm-ende-1024": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-clm-ende-1024-tf_model.h5",
    "xlm-mlm-17-1280": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-17-1280-tf_model.h5",
    "xlm-mlm-100-1280": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-100-1280-tf_model.h5",
}


def create_sinusoidal_embeddings(n_pos, dim, out):
    position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
    out[:, 0::2] = tf.constant(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = tf.constant(np.cos(position_enc[:, 1::2]))


def gelu(x):
    """ Gaussian Error Linear Unit.
    Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    cdf = 0.5 * (1.0 + tf.math.erf(x / tf.math.sqrt(2.0)))
    return x * cdf


def get_masks(slen, lengths, causal, padding_mask=None, dtype=tf.float32):
    """
    Generate hidden states mask, and optionally an attention mask.
    """
    bs = shape_list(lengths)[0]
    if padding_mask is not None:
        mask = padding_mask
    else:
        # assert lengths.max().item() <= slen
        alen = tf.range(slen)
        mask = tf.math.less(alen, lengths[:, tf.newaxis])

    # attention mask is the same as mask, or triangular inferior attention (causal)
    if causal:
        attn_mask = tf.less_equal(
            tf.tile(alen[tf.newaxis, tf.newaxis, :], (bs, slen, 1)), alen[tf.newaxis, :, tf.newaxis]
        )
    else:
        attn_mask = mask

    # sanity check
    # assert shape_list(mask) == [bs, slen]
    tf.debugging.assert_equal(shape_list(mask), [bs, slen])
    assert causal is False or shape_list(attn_mask) == [bs, slen, slen]

    mask = tf.cast(mask, dtype=dtype)
    attn_mask = tf.cast(attn_mask, dtype=dtype)

    return mask, attn_mask


class TFMultiHeadAttention(tf.keras.layers.Layer):

    NEW_ID = itertools.count()

    def __init__(self, n_heads, dim, config, **kwargs):
        super().__init__(**kwargs)
        self.layer_id = next(TFMultiHeadAttention.NEW_ID)
        self.output_attentions = config.output_attentions
        self.dim = dim
        self.n_heads = n_heads
        assert self.dim % self.n_heads == 0

        self.q_lin = tf.keras.layers.Dense(dim, kernel_initializer=get_initializer(config.init_std), name="q_lin")
        self.k_lin = tf.keras.layers.Dense(dim, kernel_initializer=get_initializer(config.init_std), name="k_lin")
        self.v_lin = tf.keras.layers.Dense(dim, kernel_initializer=get_initializer(config.init_std), name="v_lin")
        self.out_lin = tf.keras.layers.Dense(dim, kernel_initializer=get_initializer(config.init_std), name="out_lin")
        self.dropout = tf.keras.layers.Dropout(config.attention_dropout)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        raise NotImplementedError

    def call(self, inputs, training=False):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        input, mask, kv, cache, head_mask = inputs
        # Input is (bs, qlen, dim)
        # Mask is (bs, klen) (non-causal) or (bs, klen, klen)
        bs, qlen, dim = shape_list(input)
        if kv is None:
            klen = qlen if cache is None else cache["slen"] + qlen
        else:
            klen = shape_list(kv)[1]
        # assert dim == self.dim, 'Dimensions do not match: %s input vs %s configured' % (dim, self.dim)
        n_heads = self.n_heads
        dim_per_head = self.dim // n_heads
        mask_reshape = (bs, 1, qlen, klen) if len(shape_list(mask)) == 3 else (bs, 1, 1, klen)

        def shape(x):
            """  projection """
            return tf.transpose(tf.reshape(x, (bs, -1, self.n_heads, dim_per_head)), perm=(0, 2, 1, 3))

        def unshape(x):
            """  compute context """
            return tf.reshape(tf.transpose(x, perm=(0, 2, 1, 3)), (bs, -1, self.n_heads * dim_per_head))

        q = shape(self.q_lin(input))  # (bs, n_heads, qlen, dim_per_head)
        if kv is None:
            k = shape(self.k_lin(input))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v_lin(input))  # (bs, n_heads, qlen, dim_per_head)
        elif cache is None or self.layer_id not in cache:
            k = v = kv
            k = shape(self.k_lin(k))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v_lin(v))  # (bs, n_heads, qlen, dim_per_head)

        if cache is not None:
            if self.layer_id in cache:
                if kv is None:
                    k_, v_ = cache[self.layer_id]
                    k = tf.concat([k_, k], axis=2)  # (bs, n_heads, klen, dim_per_head)
                    v = tf.concat([v_, v], axis=2)  # (bs, n_heads, klen, dim_per_head)
                else:
                    k, v = cache[self.layer_id]
            cache[self.layer_id] = (k, v)

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, qlen, dim_per_head)
        scores = tf.matmul(q, k, transpose_b=True)  # (bs, n_heads, qlen, klen)
        mask = tf.reshape(mask, mask_reshape)  # (bs, n_heads, qlen, klen)
        # scores.masked_fill_(mask, -float('inf'))                            # (bs, n_heads, qlen, klen)
        scores = scores - 1e30 * (1.0 - mask)

        weights = tf.nn.softmax(scores, axis=-1)  # (bs, n_heads, qlen, klen)
        weights = self.dropout(weights, training=training)  # (bs, n_heads, qlen, klen)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = tf.matmul(weights, v)  # (bs, n_heads, qlen, dim_per_head)
        context = unshape(context)  # (bs, qlen, dim)

        outputs = (self.out_lin(context),)
        if self.output_attentions:
            outputs = outputs + (weights,)
        return outputs


class TFTransformerFFN(tf.keras.layers.Layer):
    def __init__(self, in_dim, dim_hidden, out_dim, config, **kwargs):
        super().__init__(**kwargs)
        self.lin1 = tf.keras.layers.Dense(dim_hidden, kernel_initializer=get_initializer(config.init_std), name="lin1")
        self.lin2 = tf.keras.layers.Dense(out_dim, kernel_initializer=get_initializer(config.init_std), name="lin2")
        self.act = tf.keras.layers.Activation(gelu) if config.gelu_activation else tf.keras.activations.relu
        self.dropout = tf.keras.layers.Dropout(config.dropout)

    def call(self, input, training=False):
        x = self.lin1(input)
        x = self.act(x)
        x = self.lin2(x)
        x = self.dropout(x, training=training)
        return x


class TFXLMMainLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states

        # encoder / decoder, output layer
        self.is_encoder = config.is_encoder
        self.is_decoder = not config.is_encoder
        if self.is_decoder:
            raise NotImplementedError("Currently XLM can only be used as an encoder")
        # self.with_output = with_output
        self.causal = config.causal

        # dictionary / languages
        self.n_langs = config.n_langs
        self.use_lang_emb = config.use_lang_emb
        self.n_words = config.n_words
        self.eos_index = config.eos_index
        self.pad_index = config.pad_index
        # self.dico = dico
        # self.id2lang = config.id2lang
        # self.lang2id = config.lang2id
        # assert len(self.dico) == self.n_words
        # assert len(self.id2lang) == len(self.lang2id) == self.n_langs

        # model parameters
        self.dim = config.emb_dim  # 512 by default
        self.hidden_dim = self.dim * 4  # 2048 by default
        self.n_heads = config.n_heads  # 8 by default
        self.n_layers = config.n_layers
        assert self.dim % self.n_heads == 0, "transformer dim must be a multiple of n_heads"

        # embeddings
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.attention_dropout = tf.keras.layers.Dropout(config.attention_dropout)

        self.position_embeddings = tf.keras.layers.Embedding(
            config.max_position_embeddings,
            self.dim,
            embeddings_initializer=get_initializer(config.embed_init_std),
            name="position_embeddings",
        )
        if config.sinusoidal_embeddings:
            raise NotImplementedError
            # create_sinusoidal_embeddings(config.max_position_embeddings, self.dim, out=self.position_embeddings.weight)
        if config.n_langs > 1 and config.use_lang_emb:
            self.lang_embeddings = tf.keras.layers.Embedding(
                self.n_langs,
                self.dim,
                embeddings_initializer=get_initializer(config.embed_init_std),
                name="lang_embeddings",
            )
        self.embeddings = TFSharedEmbeddings(
            self.n_words, self.dim, initializer_range=config.embed_init_std, name="embeddings"
        )  # padding_idx=self.pad_index)
        self.layer_norm_emb = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm_emb")

        # transformer layers
        self.attentions = []
        self.layer_norm1 = []
        self.ffns = []
        self.layer_norm2 = []
        # if self.is_decoder:
        #     self.layer_norm15 = []
        #     self.encoder_attn = []

        for i in range(self.n_layers):
            self.attentions.append(
                TFMultiHeadAttention(self.n_heads, self.dim, config=config, name="attentions_._{}".format(i))
            )
            self.layer_norm1.append(
                tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm1_._{}".format(i))
            )
            # if self.is_decoder:
            #     self.layer_norm15.append(nn.LayerNorm(self.dim, eps=config.layer_norm_eps))
            #     self.encoder_attn.append(MultiHeadAttention(self.n_heads, self.dim, dropout=self.attention_dropout))
            self.ffns.append(
                TFTransformerFFN(self.dim, self.hidden_dim, self.dim, config=config, name="ffns_._{}".format(i))
            )
            self.layer_norm2.append(
                tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm2_._{}".format(i))
            )

        if hasattr(config, "pruned_heads"):
            pruned_heads = config.pruned_heads.copy().items()
            config.pruned_heads = {}
            for layer, heads in pruned_heads:
                if self.attentions[int(layer)].n_heads == config.n_heads:
                    self.prune_heads({int(layer): list(map(int, heads))})

    def get_input_embeddings(self):
        return self.embeddings

    def _resize_token_embeddings(self, new_num_tokens):
        raise NotImplementedError

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        raise NotImplementedError

    def call(
        self,
        inputs,
        attention_mask=None,
        langs=None,
        token_type_ids=None,
        position_ids=None,
        lengths=None,
        cache=None,
        head_mask=None,
        inputs_embeds=None,
        training=False,
    ):  # removed: src_enc=None, src_len=None
        if isinstance(inputs, (tuple, list)):
            input_ids = inputs[0]
            attention_mask = inputs[1] if len(inputs) > 1 else attention_mask
            langs = inputs[2] if len(inputs) > 2 else langs
            token_type_ids = inputs[3] if len(inputs) > 3 else token_type_ids
            position_ids = inputs[4] if len(inputs) > 4 else position_ids
            lengths = inputs[5] if len(inputs) > 5 else lengths
            cache = inputs[6] if len(inputs) > 6 else cache
            head_mask = inputs[7] if len(inputs) > 7 else head_mask
            inputs_embeds = inputs[8] if len(inputs) > 8 else inputs_embeds
            assert len(inputs) <= 9, "Too many inputs."
        elif isinstance(inputs, dict):
            input_ids = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask", attention_mask)
            langs = inputs.get("langs", langs)
            token_type_ids = inputs.get("token_type_ids", token_type_ids)
            position_ids = inputs.get("position_ids", position_ids)
            lengths = inputs.get("lengths", lengths)
            cache = inputs.get("cache", cache)
            head_mask = inputs.get("head_mask", head_mask)
            inputs_embeds = inputs.get("inputs_embeds", inputs_embeds)
            assert len(inputs) <= 9, "Too many inputs."
        else:
            input_ids = inputs

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            bs, slen = shape_list(input_ids)
        elif inputs_embeds is not None:
            bs, slen = shape_list(inputs_embeds)[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if lengths is None:
            if input_ids is not None:
                lengths = tf.reduce_sum(tf.cast(tf.not_equal(input_ids, self.pad_index), dtype=tf.int32), axis=1)
            else:
                lengths = tf.convert_to_tensor([slen] * bs, tf.int32)
        # mask = input_ids != self.pad_index

        # check inputs
        # assert shape_list(lengths)[0] == bs
        tf.debugging.assert_equal(shape_list(lengths)[0], bs)
        # assert lengths.max().item() <= slen
        # input_ids = input_ids.transpose(0, 1)  # batch size as dimension 0
        # assert (src_enc is None) == (src_len is None)
        # if src_enc is not None:
        #     assert self.is_decoder
        #     assert src_enc.size(0) == bs

        # generate masks
        mask, attn_mask = get_masks(slen, lengths, self.causal, padding_mask=attention_mask)
        # if self.is_decoder and src_enc is not None:
        #     src_mask = torch.arange(src_len.max(), dtype=torch.long, device=lengths.device) < src_len[:, None]

        # position_ids
        if position_ids is None:
            position_ids = tf.expand_dims(tf.range(slen), axis=0)
        else:
            # assert shape_list(position_ids) == [bs, slen]  # (slen, bs)
            tf.debugging.assert_equal(shape_list(position_ids), [bs, slen])
            # position_ids = position_ids.transpose(0, 1)

        # langs
        if langs is not None:
            # assert shape_list(langs) == [bs, slen]  # (slen, bs)
            tf.debugging.assert_equal(shape_list(langs), [bs, slen])
            # langs = langs.transpose(0, 1)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x qlen x klen]
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.n_layers

        # do not recompute cached elements
        if cache is not None and input_ids is not None:
            _slen = slen - cache["slen"]
            input_ids = input_ids[:, -_slen:]
            position_ids = position_ids[:, -_slen:]
            if langs is not None:
                langs = langs[:, -_slen:]
            mask = mask[:, -_slen:]
            attn_mask = attn_mask[:, -_slen:]

        # embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        tensor = inputs_embeds + self.position_embeddings(position_ids)
        if langs is not None and self.use_lang_emb:
            tensor = tensor + self.lang_embeddings(langs)
        if token_type_ids is not None:
            tensor = tensor + self.embeddings(token_type_ids)
        tensor = self.layer_norm_emb(tensor)
        tensor = self.dropout(tensor, training=training)
        tensor = tensor * mask[..., tf.newaxis]

        # transformer layers
        hidden_states = ()
        attentions = ()
        for i in range(self.n_layers):
            if self.output_hidden_states:
                hidden_states = hidden_states + (tensor,)

            # self attention
            attn_outputs = self.attentions[i]([tensor, attn_mask, None, cache, head_mask[i]], training=training)
            attn = attn_outputs[0]
            if self.output_attentions:
                attentions = attentions + (attn_outputs[1],)
            attn = self.dropout(attn, training=training)
            tensor = tensor + attn
            tensor = self.layer_norm1[i](tensor)

            # encoder attention (for decoder only)
            # if self.is_decoder and src_enc is not None:
            #     attn = self.encoder_attn[i](tensor, src_mask, kv=src_enc, cache=cache)
            #     attn = F.dropout(attn, p=self.dropout, training=self.training)
            #     tensor = tensor + attn
            #     tensor = self.layer_norm15[i](tensor)

            # FFN
            tensor = tensor + self.ffns[i](tensor)
            tensor = self.layer_norm2[i](tensor)
            tensor = tensor * mask[..., tf.newaxis]

        # Add last hidden state
        if self.output_hidden_states:
            hidden_states = hidden_states + (tensor,)

        # update cache length
        if cache is not None:
            cache["slen"] += tensor.size(1)

        # move back sequence length to dimension 0
        # tensor = tensor.transpose(0, 1)

        outputs = (tensor,)
        if self.output_hidden_states:
            outputs = outputs + (hidden_states,)
        if self.output_attentions:
            outputs = outputs + (attentions,)
        return outputs  # outputs, (hidden_states), (attentions)


class TFXLMPreTrainedModel(TFPreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = XLMConfig
    pretrained_model_archive_map = TF_XLM_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "transformer"

    @property
    def dummy_inputs(self):
        # Sometimes XLM has language embeddings so don't forget to build them as well if needed
        inputs_list = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
        attns_list = tf.constant([[1, 1, 0, 0, 1], [1, 1, 1, 0, 0], [1, 0, 0, 1, 1]])
        if self.config.use_lang_emb and self.config.n_langs > 1:
            langs_list = tf.constant([[1, 1, 0, 0, 1], [1, 1, 1, 0, 0], [1, 0, 0, 1, 1]])
        else:
            langs_list = None
        return {"input_ids": inputs_list, "attention_mask": attns_list, "langs": langs_list}


XLM_START_DOCSTRING = r"""

    .. note::

        TF 2.0 models accepts two formats as inputs:

            - having all inputs as keyword arguments (like PyTorch models), or
            - having all inputs as a list, tuple or dict in the first positional arguments.

        This second option is useful when using :obj:`tf.keras.Model.fit()` method which currently requires having
        all the tensors in the first argument of the model call function: :obj:`model(inputs)`.

        If you choose this second option, there are three possibilities you can use to gather all the input Tensors
        in the first positional argument :

        - a single Tensor with input_ids only and nothing else: :obj:`model(inputs_ids)`
        - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
          :obj:`model([input_ids, attention_mask])` or :obj:`model([input_ids, attention_mask, token_type_ids])`
        - a dictionary with one or several input Tensors associated to the input names given in the docstring:
          :obj:`model({'input_ids': input_ids, 'token_type_ids': token_type_ids})`

    Parameters:
        config (:class:`~transformers.XLMConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

XLM_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`tf.Tensor` or :obj:`Numpy array` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`transformers.BertTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.encode_plus` for details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`tf.Tensor` or :obj:`Numpy array` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.

            `What are attention masks? <../glossary.html#attention-mask>`__
        langs (:obj:`tf.Tensor` or :obj:`Numpy array` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            A parallel sequence of tokens to be used to indicate the language of each token in the input.
            Indices are languages ids which can be obtained from the language names by using two conversion mappings
            provided in the configuration of the model (only provided for multilingual models).
            More precisely, the `language name -> language id` mapping is in `model.config.lang2id` (dict str -> int) and
            the `language id -> language name` mapping is `model.config.id2lang` (dict int -> str).

            See usage examples detailed in the `multilingual documentation <https://huggingface.co/transformers/multilingual.html>`__.
        token_type_ids (:obj:`tf.Tensor` or :obj:`Numpy array` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`tf.Tensor` or :obj:`Numpy array` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        lengths (:obj:`tf.Tensor` or :obj:`Numpy array` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Length of each sentence that can be used to avoid performing attention on padding token indices.
            You can also use `attention_mask` for the same result (see above), kept here for compatbility.
            Indices selected in ``[0, ..., input_ids.size(-1)]``:
        cache (:obj:`Dict[str, tf.Tensor]`, `optional`, defaults to :obj:`None`):
            dictionary with ``tf.Tensor`` that contains pre-computed
            hidden-states (key and values in the attention blocks) as computed by the model
            (see `cache` output below). Can be used to speed up sequential decoding.
            The dictionary object will be modified in-place during the forward pass to add newly computed hidden-states.
        head_mask (:obj:`tf.Tensor` or :obj:`Numpy array` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            :obj:`1` indicates the head is **not masked**, :obj:`0` indicates the head is **masked**.
        input_embeds (:obj:`tf.Tensor` or :obj:`Numpy array` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
"""


@add_start_docstrings(
    "The bare XLM Model transformer outputing raw hidden-states without any specific head on top.",
    XLM_START_DOCSTRING,
)
class TFXLMModel(TFXLMPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFXLMMainLayer(config, name="transformer")

    @add_start_docstrings_to_callable(XLM_INPUTS_DOCSTRING)
    def call(self, inputs, **kwargs):
        r"""
    Return:
        :obj:`tuple(tf.Tensor)` comprising various elements depending on the configuration (:class:`~transformers.XLMConfig`) and inputs:
        last_hidden_state (:obj:`tf.Tensor` or :obj:`Numpy array` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` or :obj:`Numpy array` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` or :obj:`Numpy array` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        import tensorflow as tf
        from transformers import XLMTokenizer, TFXLMModel

        tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-en-2048')
        model = TFXLMModel.from_pretrained('xlm-mlm-en-2048')
        input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True))[None, :]  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

        """
        outputs = self.transformer(inputs, **kwargs)
        return outputs


class TFXLMPredLayer(tf.keras.layers.Layer):
    """
    Prediction layer (cross_entropy or adaptive_softmax).
    """

    def __init__(self, config, input_embeddings, **kwargs):
        super().__init__(**kwargs)
        self.asm = config.asm
        self.n_words = config.n_words
        self.pad_index = config.pad_index
        if config.asm is False:
            self.input_embeddings = input_embeddings
        else:
            raise NotImplementedError
            # self.proj = nn.AdaptiveLogSoftmaxWithLoss(
            #     in_features=dim,
            #     n_classes=config.n_words,
            #     cutoffs=config.asm_cutoffs,
            #     div_value=config.asm_div_value,
            #     head_bias=True,  # default is False
            # )

    def build(self, input_shape):
        # The output weights are the same as the input embeddings, but there is an output-only bias for each token.
        self.bias = self.add_weight(shape=(self.n_words,), initializer="zeros", trainable=True, name="bias")
        super().build(input_shape)

    def call(self, hidden_states):
        hidden_states = self.input_embeddings(hidden_states, mode="linear")
        hidden_states = hidden_states + self.bias
        return hidden_states


@add_start_docstrings(
    """The XLM Model transformer with a language modeling head on top
    (linear layer with weights tied to the input embeddings). """,
    XLM_START_DOCSTRING,
)
class TFXLMWithLMHeadModel(TFXLMPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFXLMMainLayer(config, name="transformer")
        self.pred_layer = TFXLMPredLayer(config, self.transformer.embeddings, name="pred_layer_._proj")

    def get_output_embeddings(self):
        return self.pred_layer.input_embeddings

    @add_start_docstrings_to_callable(XLM_INPUTS_DOCSTRING)
    def call(self, inputs, **kwargs):
        r"""
    Return:
        :obj:`tuple(tf.Tensor)` comprising various elements depending on the configuration (:class:`~transformers.XLMConfig`) and inputs:
        prediction_scores (:obj:`tf.Tensor` or :obj:`Numpy array` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` or :obj:`Numpy array` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` or :obj:`Numpy array` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        import tensorflow as tf
        from transformers import XLMTokenizer, TFXLMWithLMHeadModel

        tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-en-2048')
        model = TFXLMWithLMHeadModel.from_pretrained('xlm-mlm-en-2048')
        input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True))[None, :]  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

        """
        transformer_outputs = self.transformer(inputs, **kwargs)

        output = transformer_outputs[0]
        outputs = self.pred_layer(output)
        outputs = (outputs,) + transformer_outputs[1:]  # Keep new_mems and attention/hidden states if they are here

        return outputs


@add_start_docstrings(
    """XLM Model with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks. """,
    XLM_START_DOCSTRING,
)
class TFXLMForSequenceClassification(TFXLMPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        self.transformer = TFXLMMainLayer(config, name="transformer")
        self.sequence_summary = TFSequenceSummary(config, initializer_range=config.init_std, name="sequence_summary")

    @add_start_docstrings_to_callable(XLM_INPUTS_DOCSTRING)
    def call(self, inputs, **kwargs):
        r"""
    Returns:
        :obj:`tuple(tf.Tensor)` comprising various elements depending on the configuration (:class:`~transformers.XLMConfig`) and inputs:
        logits (:obj:`tf.Tensor` or :obj:`Numpy array` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` or :obj:`Numpy array` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` or :obj:`Numpy array` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        import tensorflow as tf
        from transformers import XLMTokenizer, TFXLMForSequenceClassification

        tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-en-2048')
        model = TFXLMForSequenceClassification.from_pretrained('xlm-mlm-en-2048')
        input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True))[None, :]  # Batch size 1
        labels = tf.constant([1])[None, :]  # Batch size 1
        outputs = model(input_ids)
        logits = outputs[0]

        """
        transformer_outputs = self.transformer(inputs, **kwargs)
        output = transformer_outputs[0]

        logits = self.sequence_summary(output)

        outputs = (logits,) + transformer_outputs[1:]  # Keep new_mems and attention/hidden states if they are here
        return outputs


@add_start_docstrings(
    """XLM Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of
    the hidden-states output to compute `span start logits` and `span end logits`). """,
    XLM_START_DOCSTRING,
)
class TFXLMForQuestionAnsweringSimple(TFXLMPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFXLMMainLayer(config, name="transformer")
        self.qa_outputs = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.init_std), name="qa_outputs"
        )

    @add_start_docstrings_to_callable(XLM_INPUTS_DOCSTRING)
    def call(self, inputs, **kwargs):
        r"""
    Returns:
        :obj:`tuple(tf.Tensor)` comprising various elements depending on the configuration (:class:`~transformers.XLMConfig`) and inputs:
        start_scores (:obj:`tf.Tensor` or :obj:`Numpy array` of shape :obj:`(batch_size, sequence_length,)`):
            Span-start scores (before SoftMax).
        end_scores (:obj:`tf.Tensor` or :obj:`Numpy array` of shape :obj:`(batch_size, sequence_length,)`):
            Span-end scores (before SoftMax).
        hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` or :obj:`Numpy array` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` or :obj:`Numpy array` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        import tensorflow as tf
        from transformers import XLMTokenizer, TFXLMForQuestionAnsweringSimple

        tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-en-2048')
        model = TFXLMForQuestionAnsweringSimple.from_pretrained('xlm-mlm-en-2048')
        input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True))[None, :]  # Batch size 1
        outputs = model(input_ids)
        start_scores, end_scores = outputs[:2]

        """
        transformer_outputs = self.transformer(inputs, **kwargs)

        sequence_output = transformer_outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)

        outputs = (start_logits, end_logits,) + transformer_outputs[
            1:
        ]  # Keep mems, hidden states, attentions if there are in it

        return outputs  # start_logits, end_logits, (hidden_states), (attentions)
