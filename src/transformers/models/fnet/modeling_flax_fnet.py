# coding=utf-8
# Copyright 2021 Google AI and The HuggingFace Inc. team.
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

import functools
from typing import Callable, Optional, Tuple

import numpy as np

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from flax.linen.attention import dot_product_attention_weights
from jax import lax

from ...file_utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_flax_outputs import (
    FlaxBaseModelOutput,
    FlaxBaseModelOutputWithPooling,
    FlaxMaskedLMOutput,
    FlaxMultipleChoiceModelOutput,
    FlaxNextSentencePredictorOutput,
    FlaxQuestionAnsweringModelOutput,
    FlaxSequenceClassifierOutput,
    FlaxTokenClassifierOutput,
)
from ...modeling_flax_utils import (
    ACT2FN,
    FlaxPreTrainedModel,
    append_call_sample_docstring,
    append_replace_return_docstrings,
    overwrite_call_docstring,
)
from ...utils import logging
from .configuration_fnet import FNetConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "google/fnet-base"
_CONFIG_FOR_DOC = "FNetConfig"
_TOKENIZER_FOR_DOC = "FNetTokenizer"

# Copied from https://github.com/google-research/google-research/blob/master/f_net/fourier.py
def two_dim_matmul(x, matrix_dim_one, matrix_dim_two, precision=lax.Precision.DEFAULT):
    """
    Applies 2D matrix multiplication to 2D input arrays.

    Args:
      x: Input of shape [MAX_SEQ_LEN, HIDDEN_DIM]
      matrix_dim_one: [MAX_SEQ_LEN, MAX_SEQ_LEN] matrix to apply to first
        (sequence) dimension of input.
      matrix_dim_two: [HIDDEN_DIM, HIDDEN_DIM] matrix to apply to second (hidden)
        dimension of input.
      precision: XLA precision for matrix multiplication operation.

    Returns:
      [MAX_SEQ_LEN, HIDDEN_DIM] array resulting from application of two consecutive matrix multiplications.
    """
    return _two_dim_matmul(x, matrix_dim_one, matrix_dim_two, precision)


# Copied from https://github.com/google-research/google-research/blob/master/f_net/fourier.py
@functools.partial(jax.jit, static_argnums=3)
def _two_dim_matmul(x, matrix_dim_one, matrix_dim_two, precision):
    """Applies 2D matrix multiplication to 2D input arrays."""
    return jnp.einsum("ij,jk,ni->nk", x, matrix_dim_two, matrix_dim_one, optimize=True, precision=precision)


# Copied from


@flax.struct.dataclass
class FlaxFNetForPreTrainingOutput(ModelOutput):
    """
    Output type of :class:`~transformers.FNetForPreTraining`.

    Args:
        prediction_logits (:obj:`jnp.ndarray` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (:obj:`jnp.ndarray` of shape :obj:`(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (:obj:`tuple(jnp.ndarray)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of
            shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(jnp.ndarray)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`jnp.ndarray` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    prediction_logits: jnp.ndarray = None
    seq_relationship_logits: jnp.ndarray = None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    attentions: Optional[Tuple[jnp.ndarray]] = None


FNET_START_DOCSTRING = r"""

    This model inherits from :class:`~transformers.FlaxPreTrainedModel`. Check the superclass documentation for the
    generic methods the library implements for all its model (such as downloading, saving and converting weights from
    PyTorch models)

    This model is also a Flax Linen `flax.linen.Module
    <https://flax.readthedocs.io/en/latest/flax.linen.html#module>`__ subclass. Use it as a regular Flax linen Module
    and refer to the Flax documentation for all matter related to general usage and behavior.

    Finally, this model supports inherent JAX features such as:

    - `Just-In-Time (JIT) compilation <https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit>`__
    - `Automatic Differentiation <https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation>`__
    - `Vectorization <https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap>`__
    - `Parallelization <https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap>`__

    Parameters:
        config (:class:`~transformers.FNetConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.FlaxPreTrainedModel.from_pretrained` method to load the
            model weights.
"""

FNET_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`numpy.ndarray` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.FNetTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :func:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`numpy.ndarray` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`numpy.ndarray` of shape :obj:`({0})`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`__
        position_ids (:obj:`numpy.ndarray` of shape :obj:`({0})`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.

"""


class FlaxFNetEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    config: FNetConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.word_embeddings = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        self.position_embeddings = nn.Embed(
            self.config.max_position_embeddings,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        self.token_type_embeddings = nn.Embed(
            self.config.type_vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, input_ids, token_type_ids, position_ids, deterministic: bool = True):
        # Embed
        inputs_embeds = self.word_embeddings(input_ids.astype("i4"))
        position_embeds = self.position_embeddings(position_ids.astype("i4"))
        token_type_embeddings = self.token_type_embeddings(token_type_ids.astype("i4"))

        # Sum all embeddings
        hidden_states = inputs_embeds + token_type_embeddings + position_embeds

        # Layer Norm
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states


class FlaxFNetSelfOutput(nn.Module):
    config: FNetConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range, self.dtype),
            dtype=self.dtype,
        )
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, hidden_states, input_tensor, deterministic: bool = True):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class FlaxFNetIntermediate(nn.Module):
    config: FNetConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.dense = nn.Dense(
            self.config.intermediate_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range, self.dtype),
            dtype=self.dtype,
        )
        self.activation = ACT2FN[self.config.hidden_act]

    def __call__(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class FlaxFNetOutput(nn.Module):
    config: FNetConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range, self.dtype),
            dtype=self.dtype,
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    def __call__(self, hidden_states, attention_output, deterministic: bool = True):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.LayerNorm(hidden_states + attention_output)
        return hidden_states


class FlaxFNetLayer(nn.Module):
    config: FNetConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.intermediate = FlaxFNetIntermediate(self.config, dtype=self.dtype)
        self.output = FlaxFNetOutput(self.config, dtype=self.dtype)

    def __call__(self, hidden_states, deterministic: bool = True, output_attentions: bool = False):
        hidden_states = self.intermediate(fnet_output)
        hidden_states = self.output(hidden_states, fnet_output, deterministic=deterministic)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (fnet_output[1],)
        return outputs


class FlaxFNetLayerCollection(nn.Module):
    config: FNetConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.layers = [
            FlaxFNetLayer(self.config, name=str(i), dtype=self.dtype) for i in range(self.config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer(hidden_states, deterministic=deterministic, output_attentions=output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        outputs = (hidden_states,)

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


class FlaxFNetEncoder(nn.Module):
    config: FNetConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.layer = FlaxFNetLayerCollection(self.config, dtype=self.dtype)

    def __call__(
        self,
        hidden_states,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        return self.layer(
            hidden_states,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class FlaxFNetPooler(nn.Module):
    config: FNetConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range, self.dtype),
            dtype=self.dtype,
        )

    def __call__(self, hidden_states):
        cls_hidden_state = hidden_states[:, 0]
        cls_hidden_state = self.dense(cls_hidden_state)
        return nn.tanh(cls_hidden_state)


class FlaxFNetPredictionHeadTransform(nn.Module):
    config: FNetConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.dense = nn.Dense(self.config.hidden_size, dtype=self.dtype)
        self.activation = ACT2FN[self.config.hidden_act]
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    def __call__(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return self.LayerNorm(hidden_states)


class FlaxFNetLMPredictionHead(nn.Module):
    config: FNetConfig
    dtype: jnp.dtype = jnp.float32
    bias_init: Callable[..., np.ndarray] = jax.nn.initializers.zeros

    def setup(self):
        self.transform = FlaxFNetPredictionHeadTransform(self.config, dtype=self.dtype)
        self.decoder = nn.Dense(self.config.vocab_size, dtype=self.dtype, use_bias=False)
        self.bias = self.param("bias", self.bias_init, (self.config.vocab_size,))

    def __call__(self, hidden_states, shared_embedding=None):
        hidden_states = self.transform(hidden_states)

        if shared_embedding is not None:
            hidden_states = self.decoder.apply({"params": {"kernel": shared_embedding.T}}, hidden_states)
        else:
            hidden_states = self.decoder(hidden_states)

        hidden_states += self.bias
        return hidden_states


class FlaxFNetOnlyMLMHead(nn.Module):
    config: FNetConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.predictions = FlaxFNetLMPredictionHead(self.config, dtype=self.dtype)

    def __call__(self, hidden_states, shared_embedding=None):
        hidden_states = self.predictions(hidden_states, shared_embedding=shared_embedding)
        return hidden_states


class FlaxFNetOnlyNSPHead(nn.Module):
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.seq_relationship = nn.Dense(2, dtype=self.dtype)

    def __call__(self, pooled_output):
        return self.seq_relationship(pooled_output)


class FlaxFNetPreTrainingHeads(nn.Module):
    config: FNetConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.predictions = FlaxFNetLMPredictionHead(self.config, dtype=self.dtype)
        self.seq_relationship = nn.Dense(2, dtype=self.dtype)

    def __call__(self, hidden_states, pooled_output, shared_embedding=None):
        prediction_scores = self.predictions(hidden_states, shared_embedding=shared_embedding)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class FlaxFNetPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = FNetConfig
    base_model_prefix = "fnet"
    module_class: nn.Module = None

    def __init__(
        self, config: FNetConfig, input_shape: Tuple = (1, 1), seed: int = 0, dtype: jnp.dtype = jnp.float32, **kwargs
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple) -> FrozenDict:
        # init input tensors
        input_ids = jnp.zeros(input_shape, dtype="i4")
        token_type_ids = jnp.zeros_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        attention_mask = jnp.ones_like(input_ids)

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        return self.module.init(rngs, input_ids, token_type_ids, position_ids, return_dict=False)["params"]

    @add_start_docstrings_to_model_forward(FNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        params: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # init input tensors if not passed
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)

        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        return self.module.apply(
            {"params": params or self.params},
            jnp.array(input_ids, dtype="i4"),
            jnp.array(token_type_ids, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            not train,
            output_attentions,
            output_hidden_states,
            return_dict,
            rngs=rngs,
        )


class FlaxFNetModule(nn.Module):
    config: FNetConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    add_pooling_layer: bool = True

    def setup(self):
        self.embeddings = FlaxFNetEmbeddings(self.config, dtype=self.dtype)
        self.encoder = FlaxFNetEncoder(self.config, dtype=self.dtype)
        self.pooler = FlaxFNetPooler(self.config, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        token_type_ids: Optional[np.ndarray] = None,
        position_ids: Optional[np.ndarray] = None,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # make sure `token_type_ids` is correctly initialized when not passed
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)

        # make sure `position_ids` is correctly initialized when not passed
        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        hidden_states = self.embeddings(input_ids, token_type_ids, position_ids, deterministic=deterministic)
        outputs = self.encoder(
            hidden_states,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        pooled = self.pooler(hidden_states) if self.add_pooling_layer else None

        if not return_dict:
            # if pooled is None, don't return it
            if pooled is None:
                return (hidden_states,) + outputs[1:]
            return (hidden_states, pooled) + outputs[1:]

        return FlaxBaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=pooled,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    "The bare FNet Model transformer outputting raw hidden-states without any specific head on top.",
    FNET_START_DOCSTRING,
)
class FlaxFNetModel(FlaxFNetPreTrainedModel):
    module_class = FlaxFNetModule


append_call_sample_docstring(
    FlaxFNetModel, _TOKENIZER_FOR_DOC, _CHECKPOINT_FOR_DOC, FlaxBaseModelOutputWithPooling, _CONFIG_FOR_DOC
)


class FlaxFNetForPreTrainingModule(nn.Module):
    config: FNetConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.fnet = FlaxFNetModule(config=self.config, dtype=self.dtype)
        self.cls = FlaxFNetPreTrainingHeads(config=self.config, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        token_type_ids,
        position_ids,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):

        # Model
        outputs = self.fnet(
            input_ids,
            token_type_ids,
            position_ids,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.config.tie_word_embeddings:
            shared_embedding = self.fnet.variables["params"]["embeddings"]["word_embeddings"]["embedding"]
        else:
            shared_embedding = None

        hidden_states = outputs[0]
        pooled_output = outputs[1]

        prediction_scores, seq_relationship_score = self.cls(
            hidden_states, pooled_output, shared_embedding=shared_embedding
        )

        if not return_dict:
            return (prediction_scores, seq_relationship_score) + outputs[2:]

        return FlaxFNetForPreTrainingOutput(
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    FNet Model with two heads on top as done during the pretraining: a `masked language modeling` head and a `next
    sentence prediction (classification)` head.
    """,
    FNET_START_DOCSTRING,
)
class FlaxFNetForPreTraining(FlaxFNetPreTrainedModel):
    module_class = FlaxFNetForPreTrainingModule


FLAX_FNET_FOR_PRETRAINING_DOCSTRING = """
    Returns:

    Example::

        >>> from transformers import FNetTokenizer, FlaxFNetForPreTraining

        >>> tokenizer = FNetTokenizer.from_pretrained('google/fnet-base')
        >>> model = FlaxFNetForPreTraining.from_pretrained('google/fnet-base')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="np")
        >>> outputs = model(**inputs)

        >>> prediction_logits = outputs.prediction_logits
        >>> seq_relationship_logits = outputs.seq_relationship_logits
"""

overwrite_call_docstring(
    FlaxFNetForPreTraining,
    FNET_INPUTS_DOCSTRING.format("batch_size, sequence_length") + FLAX_FNET_FOR_PRETRAINING_DOCSTRING,
)
append_replace_return_docstrings(
    FlaxFNetForPreTraining, output_type=FlaxFNetForPreTrainingOutput, config_class=_CONFIG_FOR_DOC
)


class FlaxFNetForMaskedLMModule(nn.Module):
    config: FNetConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.fnet = FlaxFNetModule(config=self.config, add_pooling_layer=False, dtype=self.dtype)
        self.cls = FlaxFNetOnlyMLMHead(config=self.config, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        token_type_ids,
        position_ids,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # Model
        outputs = self.fnet(
            input_ids,
            token_type_ids,
            position_ids,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.tie_word_embeddings:
            shared_embedding = self.fnet.variables["params"]["embeddings"]["word_embeddings"]["embedding"]
        else:
            shared_embedding = None

        # Compute the prediction scores
        logits = self.cls(hidden_states, shared_embedding=shared_embedding)

        if not return_dict:
            return (logits,) + outputs[1:]

        return FlaxMaskedLMOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings("""FNet Model with a `language modeling` head on top. """, FNET_START_DOCSTRING)
class FlaxFNetForMaskedLM(FlaxFNetPreTrainedModel):
    module_class = FlaxFNetForMaskedLMModule


append_call_sample_docstring(
    FlaxFNetForMaskedLM, _TOKENIZER_FOR_DOC, _CHECKPOINT_FOR_DOC, FlaxMaskedLMOutput, _CONFIG_FOR_DOC
)


class FlaxFNetForNextSentencePredictionModule(nn.Module):
    config: FNetConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.fnet = FlaxFNetModule(config=self.config, dtype=self.dtype)
        self.cls = FlaxFNetOnlyNSPHead(dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        token_type_ids,
        position_ids,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # Model
        outputs = self.fnet(
            input_ids,
            token_type_ids,
            position_ids,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        seq_relationship_scores = self.cls(pooled_output)

        if not return_dict:
            return (seq_relationship_scores,) + outputs[2:]

        return FlaxNextSentencePredictorOutput(
            logits=seq_relationship_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """FNet Model with a `next sentence prediction (classification)` head on top. """,
    FNET_START_DOCSTRING,
)
class FlaxFNetForNextSentencePrediction(FlaxFNetPreTrainedModel):
    module_class = FlaxFNetForNextSentencePredictionModule


FLAX_FNET_FOR_NEXT_SENT_PRED_DOCSTRING = """
    Returns:

    Example::

        >>> from transformers import FNetTokenizer, FlaxFNetForNextSentencePrediction

        >>> tokenizer = FNetTokenizer.from_pretrained('google/fnet-base')
        >>> model = FlaxFNetForNextSentencePrediction.from_pretrained('google/fnet-base')

        >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
        >>> encoding = tokenizer(prompt, next_sentence, return_tensors='jax')

        >>> outputs = model(**encoding)
        >>> logits = outputs.logits
        >>> assert logits[0, 0] < logits[0, 1] # next sentence was random
"""


overwrite_call_docstring(
    FlaxFNetForNextSentencePrediction,
    FNET_INPUTS_DOCSTRING.format("batch_size, sequence_length") + FLAX_FNET_FOR_NEXT_SENT_PRED_DOCSTRING,
)
append_replace_return_docstrings(
    FlaxFNetForNextSentencePrediction, output_type=FlaxNextSentencePredictorOutput, config_class=_CONFIG_FOR_DOC
)


class FlaxFNetForSequenceClassificationModule(nn.Module):
    config: FNetConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.fnet = FlaxFNetModule(config=self.config, dtype=self.dtype)
        classifier_dropout = (
            self.config.classifier_dropout
            if self.config.classifier_dropout is not None
            else self.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(rate=classifier_dropout)
        self.classifier = nn.Dense(
            self.config.num_labels,
            dtype=self.dtype,
        )

    def __call__(
        self,
        input_ids,
        token_type_ids,
        position_ids,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # Model
        outputs = self.fnet(
            input_ids,
            token_type_ids,
            position_ids,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output, deterministic=deterministic)
        logits = self.classifier(pooled_output)

        if not return_dict:
            return (logits,) + outputs[2:]

        return FlaxSequenceClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    FNet Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    FNET_START_DOCSTRING,
)
class FlaxFNetForSequenceClassification(FlaxFNetPreTrainedModel):
    module_class = FlaxFNetForSequenceClassificationModule


append_call_sample_docstring(
    FlaxFNetForSequenceClassification,
    _TOKENIZER_FOR_DOC,
    _CHECKPOINT_FOR_DOC,
    FlaxSequenceClassifierOutput,
    _CONFIG_FOR_DOC,
)


class FlaxFNetForMultipleChoiceModule(nn.Module):
    config: FNetConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.fnet = FlaxFNetModule(config=self.config, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        self.classifier = nn.Dense(1, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        token_type_ids,
        position_ids,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        num_choices = input_ids.shape[1]
        input_ids = input_ids.reshape(-1, input_ids.shape[-1]) if input_ids is not None else None
        token_type_ids = token_type_ids.reshape(-1, token_type_ids.shape[-1]) if token_type_ids is not None else None
        position_ids = position_ids.reshape(-1, position_ids.shape[-1]) if position_ids is not None else None

        # Model
        outputs = self.fnet(
            input_ids,
            token_type_ids,
            position_ids,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output, deterministic=deterministic)
        logits = self.classifier(pooled_output)

        reshaped_logits = logits.reshape(-1, num_choices)

        if not return_dict:
            return (reshaped_logits,) + outputs[2:]

        return FlaxMultipleChoiceModelOutput(
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    FNet Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    FNET_START_DOCSTRING,
)
class FlaxFNetForMultipleChoice(FlaxFNetPreTrainedModel):
    module_class = FlaxFNetForMultipleChoiceModule


overwrite_call_docstring(
    FlaxFNetForMultipleChoice, FNET_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
)
append_call_sample_docstring(
    FlaxFNetForMultipleChoice, _TOKENIZER_FOR_DOC, _CHECKPOINT_FOR_DOC, FlaxMultipleChoiceModelOutput, _CONFIG_FOR_DOC
)


class FlaxFNetForTokenClassificationModule(nn.Module):
    config: FNetConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.fnet = FlaxFNetModule(config=self.config, dtype=self.dtype, add_pooling_layer=False)
        classifier_dropout = (
            self.config.classifier_dropout
            if self.config.classifier_dropout is not None
            else self.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(rate=classifier_dropout)
        self.classifier = nn.Dense(self.config.num_labels, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        token_type_ids,
        position_ids,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # Model
        outputs = self.fnet(
            input_ids,
            token_type_ids,
            position_ids,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        logits = self.classifier(hidden_states)

        if not return_dict:
            return (logits,) + outputs[1:]

        return FlaxTokenClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    FNet Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    FNET_START_DOCSTRING,
)
class FlaxFNetForTokenClassification(FlaxFNetPreTrainedModel):
    module_class = FlaxFNetForTokenClassificationModule


append_call_sample_docstring(
    FlaxFNetForTokenClassification, _TOKENIZER_FOR_DOC, _CHECKPOINT_FOR_DOC, FlaxTokenClassifierOutput, _CONFIG_FOR_DOC
)


class FlaxFNetForQuestionAnsweringModule(nn.Module):
    config: FNetConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.fnet = FlaxFNetModule(config=self.config, dtype=self.dtype, add_pooling_layer=False)
        self.qa_outputs = nn.Dense(self.config.num_labels, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        token_type_ids,
        position_ids,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # Model
        outputs = self.fnet(
            input_ids,
            token_type_ids,
            position_ids,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        logits = self.qa_outputs(hidden_states)
        start_logits, end_logits = logits.split(self.config.num_labels, axis=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if not return_dict:
            return (start_logits, end_logits) + outputs[1:]

        return FlaxQuestionAnsweringModelOutput(
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    FNet Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    FNET_START_DOCSTRING,
)
class FlaxFNetForQuestionAnswering(FlaxFNetPreTrainedModel):
    module_class = FlaxFNetForQuestionAnsweringModule


append_call_sample_docstring(
    FlaxFNetForQuestionAnswering,
    _TOKENIZER_FOR_DOC,
    _CHECKPOINT_FOR_DOC,
    FlaxQuestionAnsweringModelOutput,
    _CONFIG_FOR_DOC,
)
