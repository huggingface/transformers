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
"""
 TF 2.0 XLM model.
"""

import itertools
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...file_utils import (
    MULTIPLE_CHOICE_DUMMY_INPUTS,
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFMultipleChoiceModelOutput,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
)
from ...modeling_tf_utils import (
    TFMultipleChoiceLoss,
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    TFSequenceSummary,
    TFSharedEmbeddings,
    TFTokenClassificationLoss,
    get_initializer,
    input_processing,
    keras_serializable,
    shape_list,
)
from ...utils import logging
from .configuration_xlm import XLMConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "XLMConfig"
_TOKENIZER_FOR_DOC = "XLMTokenizer"

TF_XLM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "xlm-mlm-en-2048",
    "xlm-mlm-ende-1024",
    "xlm-mlm-enfr-1024",
    "xlm-mlm-enro-1024",
    "xlm-mlm-tlm-xnli15-1024",
    "xlm-mlm-xnli15-1024",
    "xlm-clm-enfr-1024",
    "xlm-clm-ende-1024",
    "xlm-mlm-17-1280",
    "xlm-mlm-100-1280",
    # See all XLM models at https://huggingface.co/models?filter=xlm
]


def create_sinusoidal_embeddings(n_pos, dim, out):
    position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
    out[:, 0::2] = tf.constant(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = tf.constant(np.cos(position_enc[:, 1::2]))


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


class TFXLMMultiHeadAttention(tf.keras.layers.Layer):
    NEW_ID = itertools.count()

    def __init__(self, n_heads, dim, config, **kwargs):
        super().__init__(**kwargs)
        self.layer_id = next(TFXLMMultiHeadAttention.NEW_ID)
        self.dim = dim
        self.n_heads = n_heads
        self.output_attentions = config.output_attentions
        assert self.dim % self.n_heads == 0

        self.q_lin = tf.keras.layers.Dense(dim, kernel_initializer=get_initializer(config.init_std), name="q_lin")
        self.k_lin = tf.keras.layers.Dense(dim, kernel_initializer=get_initializer(config.init_std), name="k_lin")
        self.v_lin = tf.keras.layers.Dense(dim, kernel_initializer=get_initializer(config.init_std), name="v_lin")
        self.out_lin = tf.keras.layers.Dense(dim, kernel_initializer=get_initializer(config.init_std), name="out_lin")
        self.dropout = tf.keras.layers.Dropout(config.attention_dropout)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        raise NotImplementedError

    def call(self, input, mask, kv, cache, head_mask, output_attentions, training=False):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        # Input is (bs, qlen, dim)
        # Mask is (bs, klen) (non-causal) or (bs, klen, klen)
        bs, qlen, dim = shape_list(input)

        if kv is None:
            klen = qlen if cache is None else cache["slen"] + qlen
        else:
            klen = shape_list(kv)[1]

        # assert dim == self.dim, 'Dimensions do not match: %s input vs %s configured' % (dim, self.dim)
        dim_per_head = tf.math.divide(self.dim, self.n_heads)
        dim_per_head = tf.cast(dim_per_head, dtype=tf.int32)
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

        q = tf.cast(q, dtype=tf.float32)
        q = tf.multiply(q, tf.math.rsqrt(tf.cast(dim_per_head, dtype=tf.float32)))  # (bs, n_heads, qlen, dim_per_head)
        k = tf.cast(k, dtype=q.dtype)
        scores = tf.matmul(q, k, transpose_b=True)  # (bs, n_heads, qlen, klen)
        mask = tf.reshape(mask, mask_reshape)  # (bs, n_heads, qlen, klen)
        # scores.masked_fill_(mask, -float('inf'))                            # (bs, n_heads, qlen, klen)
        mask = tf.cast(mask, dtype=scores.dtype)
        scores = scores - 1e30 * (1.0 - mask)
        weights = tf.nn.softmax(scores, axis=-1)  # (bs, n_heads, qlen, klen)
        weights = self.dropout(weights, training=training)  # (bs, n_heads, qlen, klen)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = tf.matmul(weights, v)  # (bs, n_heads, qlen, dim_per_head)
        context = unshape(context)  # (bs, qlen, dim)
        outputs = (self.out_lin(context),)

        if output_attentions:
            outputs = outputs + (weights,)

        return outputs


class TFXLMTransformerFFN(tf.keras.layers.Layer):
    def __init__(self, in_dim, dim_hidden, out_dim, config, **kwargs):
        super().__init__(**kwargs)

        self.lin1 = tf.keras.layers.Dense(dim_hidden, kernel_initializer=get_initializer(config.init_std), name="lin1")
        self.lin2 = tf.keras.layers.Dense(out_dim, kernel_initializer=get_initializer(config.init_std), name="lin2")
        self.act = get_tf_activation("gelu") if config.gelu_activation else get_tf_activation("relu")
        self.dropout = tf.keras.layers.Dropout(config.dropout)

    def call(self, input, training=False):
        x = self.lin1(input)
        x = self.act(x)
        x = self.lin2(x)
        x = self.dropout(x, training=training)

        return x


@keras_serializable
class TFXLMMainLayer(tf.keras.layers.Layer):
    config_class = XLMConfig

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions
        self.return_dict = config.use_return_dict

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
                TFXLMMultiHeadAttention(self.n_heads, self.dim, config=config, name="attentions_._{}".format(i))
            )
            self.layer_norm1.append(
                tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm1_._{}".format(i))
            )
            # if self.is_decoder:
            #     self.layer_norm15.append(nn.LayerNorm(self.dim, eps=config.layer_norm_eps))
            #     self.encoder_attn.append(MultiHeadAttention(self.n_heads, self.dim, dropout=self.attention_dropout))
            self.ffns.append(
                TFXLMTransformerFFN(self.dim, self.hidden_dim, self.dim, config=config, name="ffns_._{}".format(i))
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

    def set_input_embeddings(self, value):
        self.embeddings.weight = value
        self.embeddings.vocab_size = value.shape[0]

    def _resize_token_embeddings(self, new_num_tokens):
        raise NotImplementedError

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    def call(
        self,
        input_ids=None,
        attention_mask=None,
        langs=None,
        token_type_ids=None,
        position_ids=None,
        lengths=None,
        cache=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
        **kwargs,
    ):
        # removed: src_enc=None, src_len=None
        inputs = input_processing(
            func=self.call,
            input_ids=input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
            kwargs_call=kwargs,
        )
        output_attentions = (
            inputs["output_attentions"] if inputs["output_attentions"] is not None else self.output_attentions
        )
        output_hidden_states = (
            inputs["output_hidden_states"] if inputs["output_hidden_states"] is not None else self.output_hidden_states
        )
        return_dict = inputs["return_dict"] if inputs["return_dict"] is not None else self.return_dict

        if inputs["input_ids"] is not None and inputs["inputs_embeds"] is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif inputs["input_ids"] is not None:
            bs, slen = shape_list(inputs["input_ids"])
        elif inputs["inputs_embeds"] is not None:
            bs, slen = shape_list(inputs["inputs_embeds"])[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs["lengths"] is None:
            if inputs["input_ids"] is not None:
                inputs["lengths"] = tf.reduce_sum(
                    tf.cast(tf.not_equal(inputs["input_ids"], self.pad_index), dtype=tf.int32), axis=1
                )
            else:
                inputs["lengths"] = tf.convert_to_tensor([slen] * bs, tf.int32)
        # mask = input_ids != self.pad_index

        # check inputs
        # assert shape_list(lengths)[0] == bs
        tf.debugging.assert_equal(
            shape_list(inputs["lengths"])[0], bs
        ), f"Expected batch size {shape_list(inputs['lengths'])[0]} and received batch size {bs} mismatched"
        # assert lengths.max().item() <= slen
        # input_ids = input_ids.transpose(0, 1)  # batch size as dimension 0
        # assert (src_enc is None) == (src_len is None)
        # if src_enc is not None:
        #     assert self.is_decoder
        #     assert src_enc.size(0) == bs

        # generate masks
        mask, attn_mask = get_masks(slen, inputs["lengths"], self.causal, padding_mask=inputs["attention_mask"])
        # if self.is_decoder and src_enc is not None:
        #     src_mask = torch.arange(src_len.max(), dtype=torch.long, device=lengths.device) < src_len[:, None]

        # position_ids
        if inputs["position_ids"] is None:
            inputs["position_ids"] = tf.expand_dims(tf.range(slen), axis=0)
        else:
            # assert shape_list(position_ids) == [bs, slen]  # (slen, bs)
            tf.debugging.assert_equal(
                shape_list(inputs["position_ids"]), [bs, slen]
            ), f"Position id shape {shape_list(inputs['position_ids'])} and input shape {[bs, slen]} mismatched"
            # position_ids = position_ids.transpose(0, 1)

        # langs
        if inputs["langs"] is not None:
            # assert shape_list(langs) == [bs, slen]  # (slen, bs)
            tf.debugging.assert_equal(
                shape_list(inputs["langs"]), [bs, slen]
            ), f"Lang shape {shape_list(inputs['langs'])} and input shape {[bs, slen]} mismatched"
            # langs = langs.transpose(0, 1)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x qlen x klen]
        if inputs["head_mask"] is not None:
            raise NotImplementedError
        else:
            inputs["head_mask"] = [None] * self.n_layers

        # do not recompute cached elements
        if inputs["cache"] is not None and inputs["input_ids"] is not None:
            _slen = slen - inputs["cache"]["slen"]
            inputs["input_ids"] = inputs["input_ids"][:, -_slen:]
            inputs["position_ids"] = inputs["position_ids"][:, -_slen:]
            if inputs["langs"] is not None:
                inputs["langs"] = inputs["langs"][:, -_slen:]
            mask = mask[:, -_slen:]
            attn_mask = attn_mask[:, -_slen:]

        # embeddings
        if inputs["inputs_embeds"] is None:
            inputs["inputs_embeds"] = self.embeddings(inputs["input_ids"])

        tensor = inputs["inputs_embeds"] + self.position_embeddings(inputs["position_ids"])

        if inputs["langs"] is not None and self.use_lang_emb and self.n_langs > 1:
            tensor = tensor + self.lang_embeddings(inputs["langs"])
        if inputs["token_type_ids"] is not None:
            tensor = tensor + self.embeddings(inputs["token_type_ids"])

        tensor = self.layer_norm_emb(tensor)
        tensor = self.dropout(tensor, training=inputs["training"])
        tensor = tensor * mask[..., tf.newaxis]

        # transformer layers
        hidden_states = () if output_hidden_states else None
        attentions = () if output_attentions else None

        for i in range(self.n_layers):
            if output_hidden_states:
                hidden_states = hidden_states + (tensor,)

            # self attention
            attn_outputs = self.attentions[i](
                tensor,
                attn_mask,
                None,
                inputs["cache"],
                inputs["head_mask"][i],
                output_attentions,
                training=inputs["training"],
            )
            attn = attn_outputs[0]

            if output_attentions:
                attentions = attentions + (attn_outputs[1],)

            attn = self.dropout(attn, training=inputs["training"])
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
        if output_hidden_states:
            hidden_states = hidden_states + (tensor,)

        # update cache length
        if inputs["cache"] is not None:
            inputs["cache"]["slen"] += tensor.size(1)

        # move back sequence length to dimension 0
        # tensor = tensor.transpose(0, 1)

        if not return_dict:
            return tuple(v for v in [tensor, hidden_states, attentions] if v is not None)

        return TFBaseModelOutput(last_hidden_state=tensor, hidden_states=hidden_states, attentions=attentions)


class TFXLMPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = XLMConfig
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


# Remove when XLMWithLMHead computes loss like other LM models
@dataclass
class TFXLMWithLMHeadModelOutput(ModelOutput):
    """
    Base class for :class:`~transformers.TFXLMWithLMHeadModel` outputs.

    Args:
        logits (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    logits: tf.Tensor = None
    hidden_states: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[tf.Tensor]] = None


XLM_START_DOCSTRING = r"""

    This model inherits from :class:`~transformers.TFPreTrainedModel`. Check the superclass documentation for the
    generic methods the library implements for all its model (such as downloading or saving, resizing the input
    embeddings, pruning heads etc.)

    This model is also a `tf.keras.Model <https://www.tensorflow.org/api_docs/python/tf/keras/Model>`__ subclass. Use
    it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage
    and behavior.

    .. note::

        TF 2.0 models accepts two formats as inputs:

        - having all inputs as keyword arguments (like PyTorch models), or
        - having all inputs as a list, tuple or dict in the first positional arguments.

        This second option is useful when using :meth:`tf.keras.Model.fit` method which currently requires having all
        the tensors in the first argument of the model call function: :obj:`model(inputs)`.

        If you choose this second option, there are three possibilities you can use to gather all the input Tensors in
        the first positional argument :

        - a single Tensor with :obj:`input_ids` only and nothing else: :obj:`model(inputs_ids)`
        - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
          :obj:`model([input_ids, attention_mask])` or :obj:`model([input_ids, attention_mask, token_type_ids])`
        - a dictionary with one or several input Tensors associated to the input names given in the docstring:
          :obj:`model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

    Parameters:
        config (:class:`~transformers.XLMConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

XLM_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.BertTokenizer`. See
            :func:`transformers.PreTrainedTokenizer.__call__` and :func:`transformers.PreTrainedTokenizer.encode` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        langs (:obj:`tf.Tensor` or :obj:`Numpy array` of shape :obj:`({0})`, `optional`):
            A parallel sequence of tokens to be used to indicate the language of each token in the input. Indices are
            languages ids which can be obtained from the language names by using two conversion mappings provided in
            the configuration of the model (only provided for multilingual models). More precisely, the `language name
            to language id` mapping is in :obj:`model.config.lang2id` (which is a dictionary string to int) and the
            `language id to language name` mapping is in :obj:`model.config.id2lang` (dictionary int to string).

            See usage examples detailed in the :doc:`multilingual documentation <../multilingual>`.
        token_type_ids (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`({0})`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`__
        position_ids (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`({0})`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`__
        lengths (:obj:`tf.Tensor` or :obj:`Numpy array` of shape :obj:`(batch_size,)`, `optional`):
            Length of each sentence that can be used to avoid performing attention on padding token indices. You can
            also use `attention_mask` for the same result (see above), kept here for compatibility. Indices selected in
            ``[0, ..., input_ids.size(-1)]``.
        cache (:obj:`Dict[str, tf.Tensor]`, `optional`):
            Dictionary string to ``torch.FloatTensor`` that contains precomputed hidden states (key and values in the
            attention blocks) as computed by the model (see :obj:`cache` output below). Can be used to speed up
            sequential decoding.

            The dictionary object will be modified in-place during the forward pass to add newly computed
            hidden-states.
        head_mask (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (:obj:`tf.Tensor` of shape :obj:`({0}, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        training (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""


@add_start_docstrings(
    "The bare XLM Model transformer outputting raw hidden-states without any specific head on top.",
    XLM_START_DOCSTRING,
)
class TFXLMModel(TFXLMPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFXLMMainLayer(config, name="transformer")

    @add_start_docstrings_to_model_forward(XLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="xlm-mlm-en-2048",
        output_type=TFBaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        langs=None,
        token_type_ids=None,
        position_ids=None,
        lengths=None,
        cache=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
        **kwargs,
    ):
        inputs = input_processing(
            func=self.call,
            input_ids=input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
            kwargs_call=kwargs,
        )
        return_dict = inputs["return_dict"] if inputs["return_dict"] is not None else self.transformer.return_dict
        outputs = self.transformer(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            langs=inputs["langs"],
            token_type_ids=inputs["token_type_ids"],
            position_ids=inputs["position_ids"],
            lengths=inputs["lengths"],
            cache=inputs["cache"],
            head_mask=inputs["head_mask"],
            inputs_embeds=inputs["inputs_embeds"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=return_dict,
            training=inputs["training"],
        )

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
    """
    The XLM Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    XLM_START_DOCSTRING,
)
class TFXLMWithLMHeadModel(TFXLMPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFXLMMainLayer(config, name="transformer")
        self.pred_layer = TFXLMPredLayer(config, self.transformer.embeddings, name="pred_layer_._proj")

    def get_output_embeddings(self):
        return self.pred_layer.input_embeddings

    def prepare_inputs_for_generation(self, inputs, **kwargs):
        mask_token_id = self.config.mask_token_id
        lang_id = self.config.lang_id

        effective_batch_size = inputs.shape[0]
        mask_token = tf.ones((effective_batch_size, 1), dtype=tf.int32) * mask_token_id
        inputs = tf.concat([inputs, mask_token], axis=1)

        if lang_id is not None:
            langs = tf.ones_like(inputs) * lang_id
        else:
            langs = None
        return {"input_ids": inputs, "langs": langs}

    @add_start_docstrings_to_model_forward(XLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="xlm-mlm-en-2048",
        output_type=TFXLMWithLMHeadModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        langs=None,
        token_type_ids=None,
        position_ids=None,
        lengths=None,
        cache=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
        **kwargs,
    ):
        inputs = input_processing(
            func=self.call,
            input_ids=input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
            kwargs_call=kwargs,
        )
        return_dict = inputs["return_dict"] if inputs["return_dict"] is not None else self.transformer.return_dict
        transformer_outputs = self.transformer(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            langs=inputs["langs"],
            token_type_ids=inputs["token_type_ids"],
            position_ids=inputs["position_ids"],
            lengths=inputs["lengths"],
            cache=inputs["cache"],
            head_mask=inputs["head_mask"],
            inputs_embeds=inputs["inputs_embeds"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=return_dict,
            training=inputs["training"],
        )

        output = transformer_outputs[0]
        outputs = self.pred_layer(output)

        if not return_dict:
            return (outputs,) + transformer_outputs[1:]

        return TFXLMWithLMHeadModelOutput(
            logits=outputs, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions
        )


@add_start_docstrings(
    """
    XLM Model with a sequence classification/regression head on top (a linear layer on top of the pooled output) e.g.
    for GLUE tasks.
    """,
    XLM_START_DOCSTRING,
)
class TFXLMForSequenceClassification(TFXLMPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        self.transformer = TFXLMMainLayer(config, name="transformer")
        self.sequence_summary = TFSequenceSummary(config, initializer_range=config.init_std, name="sequence_summary")

    @add_start_docstrings_to_model_forward(XLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="xlm-mlm-en-2048",
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        langs=None,
        token_type_ids=None,
        position_ids=None,
        lengths=None,
        cache=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        training=False,
        **kwargs,
    ):
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in ``[0, ...,
            config.num_labels - 1]``. If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
        """
        inputs = input_processing(
            func=self.call,
            input_ids=input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
            training=training,
            kwargs_call=kwargs,
        )
        return_dict = inputs["return_dict"] if inputs["return_dict"] is not None else self.transformer.return_dict
        transformer_outputs = self.transformer(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            langs=inputs["langs"],
            token_type_ids=inputs["token_type_ids"],
            position_ids=inputs["position_ids"],
            lengths=inputs["lengths"],
            cache=inputs["cache"],
            head_mask=inputs["head_mask"],
            inputs_embeds=inputs["inputs_embeds"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=return_dict,
            training=inputs["training"],
        )
        output = transformer_outputs[0]

        logits = self.sequence_summary(output)

        loss = None if inputs["labels"] is None else self.compute_loss(inputs["labels"], logits)

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@add_start_docstrings(
    """
    XLM Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    XLM_START_DOCSTRING,
)
class TFXLMForMultipleChoice(TFXLMPreTrainedModel, TFMultipleChoiceLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.transformer = TFXLMMainLayer(config, name="transformer")
        self.sequence_summary = TFSequenceSummary(config, initializer_range=config.init_std, name="sequence_summary")
        self.logits_proj = tf.keras.layers.Dense(
            1, kernel_initializer=get_initializer(config.initializer_range), name="logits_proj"
        )

    @property
    def dummy_inputs(self):
        """
        Dummy inputs to build the network.

        Returns:
            tf.Tensor with dummy inputs
        """
        return {
            "input_ids": tf.constant(MULTIPLE_CHOICE_DUMMY_INPUTS),
            "langs": tf.constant(MULTIPLE_CHOICE_DUMMY_INPUTS),
        }

    @add_start_docstrings_to_model_forward(XLM_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="xlm-mlm-en-2048",
        output_type=TFMultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        langs=None,
        token_type_ids=None,
        position_ids=None,
        lengths=None,
        cache=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        training=False,
        **kwargs,
    ):
        inputs = input_processing(
            func=self.call,
            input_ids=input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
            training=training,
            kwargs_call=kwargs,
        )
        return_dict = inputs["return_dict"] if inputs["return_dict"] is not None else self.transformer.return_dict

        if inputs["input_ids"] is not None:
            num_choices = shape_list(inputs["input_ids"])[1]
            seq_length = shape_list(inputs["input_ids"])[2]
        else:
            num_choices = shape_list(inputs["inputs_embeds"])[1]
            seq_length = shape_list(inputs["inputs_embeds"])[2]

        flat_input_ids = tf.reshape(inputs["input_ids"], (-1, seq_length)) if inputs["input_ids"] is not None else None
        flat_attention_mask = (
            tf.reshape(inputs["attention_mask"], (-1, seq_length)) if inputs["attention_mask"] is not None else None
        )
        flat_token_type_ids = (
            tf.reshape(inputs["token_type_ids"], (-1, seq_length)) if inputs["token_type_ids"] is not None else None
        )
        flat_position_ids = (
            tf.reshape(inputs["position_ids"], (-1, seq_length)) if inputs["position_ids"] is not None else None
        )
        flat_langs = tf.reshape(inputs["langs"], (-1, seq_length)) if inputs["langs"] is not None else None
        flat_inputs_embeds = (
            tf.reshape(inputs["inputs_embeds"], (-1, seq_length, shape_list(inputs["inputs_embeds"])[3]))
            if inputs["inputs_embeds"] is not None
            else None
        )

        if inputs["lengths"] is not None:
            logger.warn(
                "The `lengths` parameter cannot be used with the XLM multiple choice models. Please use the "
                "attention mask instead.",
            )
            inputs["lengths"] = None

        transformer_outputs = self.transformer(
            flat_input_ids,
            flat_attention_mask,
            flat_langs,
            flat_token_type_ids,
            flat_position_ids,
            inputs["lengths"],
            inputs["cache"],
            inputs["head_mask"],
            flat_inputs_embeds,
            inputs["output_attentions"],
            inputs["output_hidden_states"],
            return_dict=return_dict,
            training=inputs["training"],
        )
        output = transformer_outputs[0]
        logits = self.sequence_summary(output)
        logits = self.logits_proj(logits)
        reshaped_logits = tf.reshape(logits, (-1, num_choices))

        loss = None if inputs["labels"] is None else self.compute_loss(inputs["labels"], reshaped_logits)

        if not return_dict:
            output = (reshaped_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TFMultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@add_start_docstrings(
    """
    XLM Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    XLM_START_DOCSTRING,
)
class TFXLMForTokenClassification(TFXLMPreTrainedModel, TFTokenClassificationLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        self.transformer = TFXLMMainLayer(config, name="transformer")
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.classifier = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.init_std), name="classifier"
        )

    @add_start_docstrings_to_model_forward(XLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="xlm-mlm-en-2048",
        output_type=TFTokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        langs=None,
        token_type_ids=None,
        position_ids=None,
        lengths=None,
        cache=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        training=False,
        **kwargs,
    ):
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        inputs = input_processing(
            func=self.call,
            input_ids=input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
            training=training,
            kwargs_call=kwargs,
        )
        return_dict = inputs["return_dict"] if inputs["return_dict"] is not None else self.transformer.return_dict
        transformer_outputs = self.transformer(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            langs=inputs["langs"],
            token_type_ids=inputs["token_type_ids"],
            position_ids=inputs["position_ids"],
            lengths=inputs["lengths"],
            cache=inputs["cache"],
            head_mask=inputs["head_mask"],
            inputs_embeds=inputs["inputs_embeds"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=return_dict,
            training=inputs["training"],
        )

        sequence_output = transformer_outputs[0]

        sequence_output = self.dropout(sequence_output, training=inputs["training"])
        logits = self.classifier(sequence_output)

        loss = None if inputs["labels"] is None else self.compute_loss(inputs["labels"], logits)

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TFTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@add_start_docstrings(
    """
    XLM Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layer
    on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    XLM_START_DOCSTRING,
)
class TFXLMForQuestionAnsweringSimple(TFXLMPreTrainedModel, TFQuestionAnsweringLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFXLMMainLayer(config, name="transformer")
        self.qa_outputs = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.init_std), name="qa_outputs"
        )

    @add_start_docstrings_to_model_forward(XLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="xlm-mlm-en-2048",
        output_type=TFQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        langs=None,
        token_type_ids=None,
        position_ids=None,
        lengths=None,
        cache=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        start_positions=None,
        end_positions=None,
        training=False,
        **kwargs,
    ):
        r"""
        start_positions (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        inputs = input_processing(
            func=self.call,
            input_ids=input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            start_positions=start_positions,
            end_positions=end_positions,
            training=training,
            kwargs_call=kwargs,
        )
        return_dict = inputs["return_dict"] if inputs["return_dict"] is not None else self.transformer.return_dict
        transformer_outputs = self.transformer(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            langs=inputs["langs"],
            token_type_ids=inputs["token_type_ids"],
            position_ids=inputs["position_ids"],
            lengths=inputs["lengths"],
            cache=inputs["cache"],
            head_mask=inputs["head_mask"],
            inputs_embeds=inputs["inputs_embeds"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=return_dict,
            training=inputs["training"],
        )

        sequence_output = transformer_outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)

        loss = None
        if inputs["start_positions"] is not None and inputs["end_positions"] is not None:
            labels = {"start_position": inputs["start_positions"]}
            labels["end_position"] = inputs["end_positions"]
            loss = self.compute_loss(labels, (start_logits, end_logits))

        if not return_dict:
            output = (start_logits, end_logits) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TFQuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
