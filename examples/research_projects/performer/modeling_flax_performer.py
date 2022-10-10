# coding=utf-8
# Copyright 2018 The Google Flax Team Authors and The HuggingFace Inc. team.
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

from typing import Callable, Dict, Tuple

import numpy as np

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from modeling_flax_performer_utils import make_fast_softmax_attention
from transformers.file_utils import add_start_docstrings
from transformers.modeling_flax_utils import ACT2FN
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_flax_bert import FlaxBertOnlyMLMHead, FlaxBertPreTrainedModel
from transformers.utils import logging


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "BertConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"

BERT_START_DOCSTRING = r"""

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.BertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

BERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.BertTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`({0}, hidden_size)`, `optional`):
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
"""


class FlaxPerformerLayerNorm(nn.Module):
    """
    Layer normalization (https://arxiv.org/abs/1607.06450). Operates on the last axis of the input data.
    """

    epsilon: float = 1e-6
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    bias: bool = True  # If True, bias (beta) is added.
    scale: bool = True  # If True, multiply by scale (gamma). When the next layer is linear
    # (also e.g. nn.relu), this can be disabled since the scaling will be
    # done by the next layer.
    bias_init: jnp.ndarray = nn.initializers.zeros
    scale_init: jnp.ndarray = nn.initializers.ones

    @nn.compact
    def __call__(self, x):
        """
        Applies layer normalization on the input. It normalizes the activations of the layer for each given example in
        a batch independently, rather than across a batch like Batch Normalization. i.e. applies a transformation that
        maintains the mean activation within each example close to 0 and the activation standard deviation close to 1

        Args:
          x: the inputs

        Returns:
          Normalized inputs (the same shape as inputs).
        """
        features = x.shape[-1]
        mean = jnp.mean(x, axis=-1, keepdims=True)
        mean2 = jnp.mean(jax.lax.square(x), axis=-1, keepdims=True)
        var = mean2 - jax.lax.square(mean)
        mul = jax.lax.rsqrt(var + self.epsilon)
        if self.scale:
            mul = mul * jnp.asarray(self.param("gamma", self.scale_init, (features,)), self.dtype)
        y = (x - mean) * mul
        if self.bias:
            y = y + jnp.asarray(self.param("beta", self.bias_init, (features,)), self.dtype)
        return y


class FlaxPerformerEmbedding(nn.Module):
    """
    Specify a new class for doing the embedding stuff as Flax's one use 'embedding' for the parameter name and PyTorch
    use 'weight'
    """

    vocab_size: int
    hidden_size: int
    emb_init: Callable[..., np.ndarray] = nn.initializers.normal(stddev=0.1)

    @nn.compact
    def __call__(self, inputs):
        embedding = self.param("weight", self.emb_init, (self.vocab_size, self.hidden_size))
        return jnp.take(embedding, inputs, axis=0)


class FlaxPerformerEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    vocab_size: int
    hidden_size: int
    type_vocab_size: int
    max_length: int

    @nn.compact
    def __call__(self, input_ids, token_type_ids, position_ids, attention_mask):
        # Embed
        w_emb = FlaxPerformerEmbedding(self.vocab_size, self.hidden_size, name="word_embeddings")(
            jnp.atleast_2d(input_ids.astype("i4"))
        )
        p_emb = FlaxPerformerEmbedding(self.max_length, self.hidden_size, name="position_embeddings")(
            jnp.atleast_2d(position_ids.astype("i4"))
        )
        t_emb = FlaxPerformerEmbedding(self.type_vocab_size, self.hidden_size, name="token_type_embeddings")(
            jnp.atleast_2d(token_type_ids.astype("i4"))
        )

        # Sum all embeddings
        summed_emb = w_emb + jnp.broadcast_to(p_emb, w_emb.shape) + t_emb

        # Layer Norm
        layer_norm = FlaxPerformerLayerNorm(name="layer_norm")(summed_emb)

        return layer_norm


class FlaxPerformerAttention(nn.Module):
    num_heads: int
    head_size: int

    @nn.compact
    def __call__(self, hidden_state, attention_mask):
        single_head_dim = self.head_size // self.num_heads
        fast_softmax_attention = make_fast_softmax_attention(qkv_dim=single_head_dim)
        self_att = nn.attention.SelfAttention(
            num_heads=self.num_heads, qkv_features=self.head_size, name="self", attention_fn=fast_softmax_attention
        )(hidden_state, attention_mask)

        layer_norm = FlaxPerformerLayerNorm(name="layer_norm")(self_att + hidden_state)
        return layer_norm


class FlaxPerformerIntermediate(nn.Module):
    output_size: int
    hidden_act: str = "gelu"

    @nn.compact
    def __call__(self, hidden_state):
        # TODO: Add ACT2FN reference to change activation function
        dense = nn.Dense(features=self.output_size, name="dense")(hidden_state)
        return ACT2FN[self.hidden_act](dense)


class FlaxPerformerOutput(nn.Module):
    @nn.compact
    def __call__(self, intermediate_output, attention_output):
        hidden_state = nn.Dense(attention_output.shape[-1], name="dense")(intermediate_output)
        hidden_state = FlaxPerformerLayerNorm(name="layer_norm")(hidden_state + attention_output)
        return hidden_state


class FlaxPerformerLayer(nn.Module):
    num_heads: int
    head_size: int
    intermediate_size: int
    hidden_act: str = "gelu"

    @nn.compact
    def __call__(self, hidden_state, attention_mask):
        attention = FlaxPerformerAttention(self.num_heads, self.head_size, name="attention")(
            hidden_state, attention_mask
        )
        intermediate = FlaxPerformerIntermediate(
            self.intermediate_size, name="intermediate", hidden_act=self.hidden_act
        )(attention)
        output = FlaxPerformerOutput(name="output")(intermediate, attention)

        return output


class FlaxPerformerLayerCollection(nn.Module):
    """
    Stores N BertLayer(s)
    """

    num_layers: int
    num_heads: int
    head_size: int
    intermediate_size: int
    hidden_act: str = "gelu"

    @nn.compact
    def __call__(self, inputs, attention_mask):
        assert self.num_layers > 0, f"num_layers should be >= 1, got ({self.num_layers})"

        # Initialize input / output
        input_i = inputs

        # Forward over all encoders
        for i in range(self.num_layers):
            layer = FlaxPerformerLayer(
                self.num_heads, self.head_size, self.intermediate_size, hidden_act=self.hidden_act, name=f"{i}"
            )
            input_i = layer(input_i, attention_mask)
        return input_i


class FlaxPerformerEncoder(nn.Module):
    num_layers: int
    num_heads: int
    head_size: int
    intermediate_size: int
    hidden_act: str = "gelu"

    @nn.compact
    def __call__(self, hidden_state, attention_mask):
        layer = FlaxPerformerLayerCollection(
            self.num_layers,
            self.num_heads,
            self.head_size,
            self.intermediate_size,
            name="layer",
            hidden_act=self.hidden_act,
        )(hidden_state, attention_mask)
        return layer


class FlaxPerformerPooler(nn.Module):
    @nn.compact
    def __call__(self, hidden_state):
        cls_token = hidden_state[:, 0]
        out = nn.Dense(hidden_state.shape[-1], name="dense")(cls_token)
        return jax.lax.tanh(out)


class FlaxPerformerModule(nn.Module):
    vocab_size: int
    hidden_size: int
    type_vocab_size: int
    max_length: int
    num_encoder_layers: int
    num_heads: int
    head_size: int
    intermediate_size: int
    hidden_act: str = "gelu"
    add_pooling_layer: bool = True

    @nn.compact
    def __call__(self, input_ids, token_type_ids, position_ids, attention_mask):
        # Embedding
        embeddings = FlaxPerformerEmbeddings(
            self.vocab_size, self.hidden_size, self.type_vocab_size, self.max_length, name="embeddings"
        )(input_ids, token_type_ids, position_ids, attention_mask)

        # N stacked encoding layers
        encoder = FlaxPerformerEncoder(
            self.num_encoder_layers,
            self.num_heads,
            self.head_size,
            self.intermediate_size,
            hidden_act=self.hidden_act,
            name="encoder",
        )(embeddings, attention_mask)

        if not self.add_pooling_layer:
            return encoder

        pooled = FlaxPerformerPooler(name="pooler")(encoder)
        return encoder, pooled


@add_start_docstrings(
    "The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",
    BERT_START_DOCSTRING,
)
class FlaxPerformerModel(FlaxBertPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    """

    model_class = FlaxPerformerModule
    config_class = BertConfig
    base_model_prefix = "bert"

    @staticmethod
    def convert_from_pytorch(pt_state: Dict, config: BertConfig) -> Dict:
        jax_state = dict(pt_state)

        # Need to change some parameters name to match Flax names so that we don't have to fork any layer
        for key, tensor in pt_state.items():
            # Key parts
            key_parts = set(key.split("."))

            # Every dense layer has "kernel" parameters instead of "weight"
            if "dense.weight" in key:
                del jax_state[key]
                key = key.replace("weight", "kernel")
                jax_state[key] = tensor

            # SelfAttention needs also to replace "weight" by "kernel"
            if {"query", "key", "value"} & key_parts:

                # Flax SelfAttention decomposes the heads (num_head, size // num_heads)
                if "bias" in key:
                    jax_state[key] = tensor.reshape((config.num_attention_heads, -1))
                elif "weight":
                    del jax_state[key]
                    key = key.replace("weight", "kernel")
                    tensor = tensor.reshape((config.num_attention_heads, -1, config.hidden_size)).transpose((2, 0, 1))
                    jax_state[key] = tensor

            # SelfAttention output is not a separate layer, remove one nesting
            if "attention.output.dense" in key:
                del jax_state[key]
                key = key.replace("attention.output.dense", "attention.self.out")
                jax_state[key] = tensor

            # SelfAttention output is not a separate layer, remove nesting on layer norm
            if "attention.output.LayerNorm" in key:
                del jax_state[key]
                key = key.replace("attention.output.LayerNorm", "attention.LayerNorm")
                jax_state[key] = tensor

            # There are some transposed parameters w.r.t their PyTorch counterpart
            if "intermediate.dense.kernel" in key or "output.dense.kernel" in key:
                jax_state[key] = tensor.T

            # Self Attention output projection needs to be transposed
            if "out.kernel" in key:
                jax_state[key] = tensor.reshape((config.hidden_size, config.num_attention_heads, -1)).transpose(
                    1, 2, 0
                )

            # Pooler needs to transpose its kernel
            if "pooler.dense.kernel" in key:
                jax_state[key] = tensor.T

            # Handle LayerNorm conversion
            if "LayerNorm" in key:
                del jax_state[key]

                # Replace LayerNorm by layer_norm
                new_key = key.replace("LayerNorm", "layer_norm")

                if "weight" in key:
                    new_key = new_key.replace("weight", "gamma")
                elif "bias" in key:
                    new_key = new_key.replace("bias", "beta")

                jax_state[new_key] = tensor

        return jax_state

    def __init__(
        self, config: BertConfig, input_shape: Tuple = (1, 1), seed: int = 0, dtype: jnp.dtype = jnp.float32, **kwargs
    ):
        module = FlaxPerformerModule(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            type_vocab_size=config.type_vocab_size,
            max_length=config.max_position_embeddings,
            num_encoder_layers=config.num_hidden_layers,
            num_heads=config.num_attention_heads,
            head_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            dropout_rate=config.hidden_dropout_prob,
            hidden_act=config.hidden_act,
        )

        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype)

    @property
    def module(self) -> nn.Module:
        return self._module

    def __call__(
        self, input_ids, token_type_ids=None, position_ids=None, dropout_rng: PRNGKey = None, attention_mask=None
    ):

        input_ids, attention_mask, token_type_ids, position_ids = self._check_inputs(
            input_ids, attention_mask, token_type_ids, position_ids
        )

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        return self.module.apply(
            {"params": self.params},
            jnp.array(input_ids, dtype="i4"),
            jnp.array(token_type_ids, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            rng=rngs,
        )


class FlaxPerformerForMaskedLM(FlaxBertPreTrainedModel):
    def __init__(
        self, config: BertConfig, input_shape: Tuple = (1, 1), seed: int = 0, dtype: jnp.dtype = jnp.float32, **kwargs
    ):
        module = FlaxPerformerForMaskedLMModule(
            vocab_size=config.vocab_size,
            type_vocab_size=config.type_vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            head_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_encoder_layers=config.num_hidden_layers,
            max_length=config.max_position_embeddings,
            hidden_act=config.hidden_act,
            **kwargs,
        )

        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype)

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        params: dict = None,
        train: bool = False,
        dropout_rng: PRNGKey = None,
    ):
        input_ids, attention_mask, token_type_ids, position_ids = self._check_inputs(
            input_ids, attention_mask, token_type_ids, position_ids
        )

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        return self.module.apply(
            {"params": params or self.params},
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(token_type_ids, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            not train,
            rngs=rngs,
        )


class FlaxPerformerForMaskedLMModule(nn.Module):
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    head_size: int
    num_heads: int
    num_encoder_layers: int
    type_vocab_size: int
    max_length: int
    hidden_act: str
    dropout_rate: float = 0.0
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, deterministic: bool = True
    ):
        # Model
        encoder = FlaxPerformerModule(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            type_vocab_size=self.type_vocab_size,
            max_length=self.max_length,
            num_encoder_layers=self.num_encoder_layers,
            num_heads=self.num_heads,
            head_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            add_pooling_layer=False,
            name="bert",
        )(input_ids, attention_mask, token_type_ids, position_ids)

        # Compute the prediction scores
        encoder = nn.Dropout(rate=self.dropout_rate)(encoder, deterministic=deterministic)
        logits = FlaxBertOnlyMLMHead(
            vocab_size=self.vocab_size, hidden_act=self.hidden_act, name="cls", dtype=self.dtype
        )(encoder)

        return (logits,)
