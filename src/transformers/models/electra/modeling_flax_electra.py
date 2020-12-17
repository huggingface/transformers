from typing import Callable, Dict, Tuple

import numpy as np

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from flax.core import freeze
from jax.random import PRNGKey

from ...file_utils import add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel
from ...utils import logging
from .configuration_electra import ElectraConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "ElectraConfig"
_TOKENIZER_FOR_DOC = "ElectraTokenizer"


ELECTRA_START_DOCSTRING = r"""

    This model inherits from :class:`~transformers.FlaxPreTrainedModel`. Check the superclass documentation for the
    generic methods the library implements for all its model (such as downloading, saving and converting weights from
    PyTorch models)

    This model is also a Flax Linen `flax.nn.Module
    <https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html>`__ subclass. Use it as a regular Flax
    Module and refer to the Flax documentation for all matter related to general usage and behavior.

    Finally, this model supports inherent JAX features such as:

    - `Just-In-Time (JIT) compilation <https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit>`__
    - `Automatic Differentiation <https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation>`__
    - `Vectorization <https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap>`__
    - `Parallelization <https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap>`__

    Parameters:
        config (:class:`~transformers.ElectraConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

ELECTRA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`numpy.ndarray` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.ElectraTokenizer`. See
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

# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertLayerNorm with Bert->ELECTRA
class FlaxElectraLayerNorm(nn.Module):
    """
    Layer normalization (https://arxiv.org/abs/1607.06450). Operates on the last axis of the input data.
    """

    epsilon: float = 1e-12
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    bias: bool = True  # If True, bias (beta) is added.
    scale: bool = True  # If True, multiply by scale (gamma). When the next layer is linear
    # (also e.g. nn.relu), this can be disabled since the scaling will be
    # done by the next layer.
    scale_init: Callable[..., np.ndarray] = jax.nn.initializers.ones
    bias_init: Callable[..., np.ndarray] = jax.nn.initializers.zeros

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
            mul = mul * jnp.asarray(self.param("gamma", self.scale_init, (features,)))
        y = (x - mean) * mul

        if self.bias:
            y = y + jnp.asarray(self.param("beta", self.bias_init, (features,)))
        return y

# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertEmbedding with Bert->Electra
class FlaxElectraEmbedding(nn.Module):
    """
    Specify a new class for doing the embedding stuff as Flax's one use 'embedding' for the parameter name and PyTorch
    use 'weight'
    """

    vocab_size: int
    hidden_size: int
    kernel_init_scale: float = 0.2
    emb_init: Callable[..., np.ndarray] = jax.nn.initializers.normal(stddev=kernel_init_scale)
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    @nn.compact
    def __call__(self, inputs):
        embedding = self.param("weight", self.emb_init, (self.vocab_size, self.hidden_size))
        return jnp.take(embedding, inputs, axis=0)


class FlaxElectraEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    vocab_size: int
    hidden_size: int
    type_vocab_size: int
    max_length: int
    kernel_init_scale: float = 0.2
    dropout_rate: float = 0.0
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    @nn.compact
    def __call__(self, input_ids, token_type_ids, position_ids, attention_mask, deterministic: bool = True):

        # Embed
        w_emb = FlaxElectraEmbedding(
            self.vocab_size,
            self.hidden_size,
            kernel_init_scale=self.kernel_init_scale,
            name="word_embeddings",
            dtype=self.dtype,
        )(jnp.atleast_2d(input_ids.astype("i4")))
        p_emb = FlaxElectraEmbedding(
            self.max_length,
            self.hidden_size,
            kernel_init_scale=self.kernel_init_scale,
            name="position_embeddings",
            dtype=self.dtype,
        )(jnp.atleast_2d(position_ids.astype("i4")))
        t_emb = FlaxElectraEmbedding(
            self.type_vocab_size,
            self.hidden_size,
            kernel_init_scale=self.kernel_init_scale,
            name="token_type_embeddings",
            dtype=self.dtype,
        )(jnp.atleast_2d(token_type_ids.astype("i4")))

        # Sum all embeddings
        summed_emb = w_emb + jnp.broadcast_to(p_emb, w_emb.shape) + t_emb

        # Layer Norm
        layer_norm = FlaxElectraLayerNorm(name="layer_norm", dtype=self.dtype)(summed_emb)
        embeddings = nn.Dropout(rate=self.dropout_rate)(layer_norm, deterministic=deterministic)
        return embeddings


# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertAttention with Bert->Electra
class FlaxElectraAttention(nn.Module):
    num_heads: int
    head_size: int
    dropout_rate: float = 0.0
    kernel_init_scale: float = 0.2
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    @nn.compact
    def __call__(self, hidden_states, attention_mask, deterministic: bool = True):
        # Attention mask comes in as attention_mask.shape == (*batch_sizes, kv_length)
        # FLAX expects: attention_mask.shape == (*batch_sizes, 1, 1, kv_length) such that it is broadcastable
        # with attn_weights.shape == (*batch_sizes, num_heads, q_length, kv_length)
        attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
        self_att = nn.attention.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.head_size,
            dropout_rate=self.dropout_rate,
            deterministic=deterministic,
            kernel_init=jax.nn.initializers.normal(self.kernel_init_scale, self.dtype),
            bias_init=jax.nn.initializers.zeros,
            name="self",
            dtype=self.dtype,
        )(hidden_states, attention_mask)

        layer_norm = FlaxElectraLayerNorm(name="layer_norm", dtype=self.dtype)(self_att + hidden_states)
        return layer_norm


# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertIntermediate with Bert->Electra
class FlaxElectraIntermediate(nn.Module):
    output_size: int
    hidden_act: str = "gelu"
    kernel_init_scale: float = 0.2
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    @nn.compact
    def __call__(self, hidden_states):
        hidden_states = nn.Dense(
            features=self.output_size,
            kernel_init=jax.nn.initializers.normal(self.kernel_init_scale, self.dtype),
            name="dense",
            dtype=self.dtype,
        )(hidden_states)
        hidden_states = ACT2FN[self.hidden_act](hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertOutput with Bert->Electra
class FlaxElectraOutput(nn.Module):
    dropout_rate: float = 0.0
    kernel_init_scale: float = 0.2
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    @nn.compact
    def __call__(self, intermediate_output, attention_output, deterministic: bool = True):
        hidden_states = nn.Dense(
            attention_output.shape[-1],
            kernel_init=jax.nn.initializers.normal(self.kernel_init_scale, self.dtype),
            name="dense",
            dtype=self.dtype,
        )(intermediate_output)
        hidden_states = nn.Dropout(rate=self.dropout_rate)(hidden_states, deterministic=deterministic)
        hidden_states = FlaxElectraLayerNorm(name="layer_norm", dtype=self.dtype)(hidden_states + attention_output)
        return hidden_states


# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertLayer with Bert->Electra
class FlaxElectraLayer(nn.Module):
    num_heads: int
    head_size: int
    intermediate_size: int
    hidden_act: str = "gelu"
    dropout_rate: float = 0.0
    kernel_init_scale: float = 0.2
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    @nn.compact
    def __call__(self, hidden_states, attention_mask, deterministic: bool = True):
        attention = FlaxElectraAttention(
            self.num_heads,
            self.head_size,
            kernel_init_scale=self.kernel_init_scale,
            dropout_rate=self.dropout_rate,
            name="attention",
            dtype=self.dtype,
        )(hidden_states, attention_mask, deterministic=deterministic)
        intermediate = FlaxElectraIntermediate(
            self.intermediate_size,
            kernel_init_scale=self.kernel_init_scale,
            hidden_act=self.hidden_act,
            name="intermediate",
            dtype=self.dtype,
        )(attention)
        output = FlaxElectraOutput(
            kernel_init_scale=self.kernel_init_scale, dropout_rate=self.dropout_rate, name="output", dtype=self.dtype
        )(intermediate, attention, deterministic=deterministic)

        return output


# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertLayerCollection with Bert->Electra
class FlaxElectraLayerCollection(nn.Module):
    """
    Stores N ElectraLayer(s)
    """

    num_layers: int
    num_heads: int
    head_size: int
    intermediate_size: int
    hidden_act: str = "gelu"
    dropout_rate: float = 0.0
    kernel_init_scale: float = 0.2
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    @nn.compact
    def __call__(self, inputs, attention_mask, deterministic: bool = True):
        assert self.num_layers > 0, f"num_layers should be >= 1, got ({self.num_layers})"

        # Initialize input / output
        input_i = inputs

        # Forward over all encoders
        for i in range(self.num_layers):
            layer = FlaxElectraLayer(
                self.num_heads,
                self.head_size,
                self.intermediate_size,
                kernel_init_scale=self.kernel_init_scale,
                dropout_rate=self.dropout_rate,
                hidden_act=self.hidden_act,
                name=f"{i}",
                dtype=self.dtype,
            )
            input_i = layer(input_i, attention_mask, deterministic=deterministic)
        return input_i


# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertEncoder with Bert->Electra
class FlaxElectraEncoder(nn.Module):
    num_layers: int
    num_heads: int
    head_size: int
    intermediate_size: int
    hidden_act: str = "gelu"
    dropout_rate: float = 0.0
    kernel_init_scale: float = 0.2
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    @nn.compact
    def __call__(self, hidden_states, attention_mask, deterministic: bool = True):
        layer = FlaxElectraLayerCollection(
            self.num_layers,
            self.num_heads,
            self.head_size,
            self.intermediate_size,
            hidden_act=self.hidden_act,
            kernel_init_scale=self.kernel_init_scale,
            dropout_rate=self.dropout_rate,
            name="layer",
            dtype=self.dtype,
        )(hidden_states, attention_mask, deterministic=deterministic)
        return layer


# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertPooler with Bert->Electra
class FlaxElectraPooler(nn.Module):
    kernel_init_scale: float = 0.2
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    @nn.compact
    def __call__(self, hidden_states):
        cls_token = hidden_states[:, 0]
        out = nn.Dense(
            hidden_states.shape[-1],
            kernel_init=jax.nn.initializers.normal(self.kernel_init_scale, self.dtype),
            name="dense",
            dtype=self.dtype,
        )(cls_token)
        return nn.tanh(out)


class FlaxElectraPredictionHeadTransform(nn.Module):
    hidden_act: str = "gelu"
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, hidden_states):
        hidden_states = nn.Dense(hidden_states.shape[-1], name="dense", dtype=self.dtype)(hidden_states)
        hidden_states = ACT2FN[self.hidden_act](hidden_states)
        return FlaxElectraLayerNorm(name="layer_norm", dtype=self.dtype)(hidden_states)


class FlaxElectraLMPredictionHead(nn.Module):
    vocab_size: int
    hidden_act: str = "gelu"
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, hidden_states):
        # TODO: The output weights are the same as the input embeddings, but there is
        #   an output-only bias for each token.
        #   Need a link between the two variables so that the bias is correctly
        #   resized with `resize_token_embeddings`

        hidden_states = FlaxElectraPredictionHeadTransform(
            name="transform", hidden_act=self.hidden_act, dtype=self.dtype
        )(hidden_states)
        hidden_states = nn.Dense(self.vocab_size, name="decoder", dtype=self.dtype)(hidden_states)
        return hidden_states


class FlaxElectraGeneratorPredictions(nn.Module):
    embedding_size: int
    hidden_act: str = "gelu"
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.dense = nn.Dense(self.embedding_size, dtype=self.dtype)
        self.layer_norm = FlaxElectraLayerNorm(dtype=self.dtype)

    def __call__(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = ACT2FN[self.hidden_act](hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class FlaxElectraDiscriminatorPredictions(nn.Module):
    """Prediction module for the discriminator, made up of two dense layers."""
    hidden_size: int
    hidden_act: str = "gelu"
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.dense = nn.Dense(self.hidden_size, dtype=self.dtype)
        self.dense_prediction = nn.Dense(1, dtype=self.dtype)

    def __call__(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = ACT2FN[self.hidden_act](hidden_states)
        hidden_states = self.dense_prediction(hidden_states).squeeze(-1)
        return hidden_states


class FlaxElectraPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ElectraConfig
    base_model_prefix = "electra"

    def _check_inputs(self, input_ids, attention_mask, token_type_ids, position_ids):
        if token_type_ids is None:
            token_type_ids = jnp.ones_like(input_ids)

        if position_ids is None:
            position_ids = jnp.arange(jnp.atleast_2d(input_ids).shape[-1])

        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        return input_ids, attention_mask, token_type_ids, position_ids

    def init(self, rng: jax.random.PRNGKey, input_shape: Tuple) -> FrozenDict:
        input_ids, attention_mask, token_type_ids, position_ids = self._check_inputs(
            jnp.zeros(input_shape, dtype="i4"), None, None, None
        )

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        return self.module.init(rngs, input_ids, attention_mask, token_type_ids, position_ids)["params"]

    @staticmethod
    def convert_from_pytorch(pt_state: Dict, config: ElectraConfig) -> Dict:
        jax_state = dict(pt_state)

        # Need to change some parameters name to match Flax names so that we don't have to fork any layer
        for key, tensor in pt_state.items():
            # Key parts
            key_parts = set(key.split("."))

            # Every dense layer has "kernel" parameters instead of "weight"
            if "dense.weight" in key:
                del jax_state[key]
                key = key.replace("weight", "kernel")
                # Need to transpose "weight" parameter to work correctly
                if "generator_predictions" in key or "discriminator_predictions" in key:
                    jax_state[key] = tensor.T
                else:
                    jax_state[key] = tensor

            # Need to transpose "weight" parameter to work correctly
            if "embeddings_project.weight" in key or "generator_lm_head.weight" in key or "dense_prediction.weight" in key:
                del jax_state[key]
                key = key.replace("weight", "kernel")
                jax_state[key] = tensor.T

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


class FlaxElectraModule(nn.Module):
    vocab_size: int
    embedding_size: int
    hidden_size: int
    type_vocab_size: int
    max_length: int
    num_encoder_layers: int
    num_heads: int
    head_size: int
    intermediate_size: int
    hidden_act: str = "gelu"
    kernel_init_scale: float = 0.2
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.embeddings = FlaxElectraEmbeddings(
            vocab_size=self.vocab_size, 
            hidden_size=self.embedding_size, 
            type_vocab_size=self.type_vocab_size, 
            max_length=self.max_length,
            kernel_init_scale=self.kernel_init_scale,
            dropout_rate=self.hidden_dropout_prob
        )
        
        if self.embedding_size != self.hidden_size:
            self.embeddings_project = nn.Dense(self.hidden_size)

        self.encoder = FlaxElectraEncoder(
            num_layers=self.num_encoder_layers,
            num_heads=self.num_heads,
            head_size=self.head_size,
            intermediate_size=self.intermediate_size,
            dropout_rate=self.attention_probs_dropout_prob,
            hidden_act=self.hidden_act,
            kernel_init_scale=self.kernel_init_scale
        )
        
    def __call__(self, input_ids, attention_mask, token_type_ids, position_ids, deterministic: bool = True):
        embeddings = self.embeddings(input_ids, token_type_ids, position_ids, attention_mask, deterministic=deterministic)

        if self.embedding_size != self.hidden_size:
            embeddings = self.embeddings_project(embeddings)

        # N stacked encoding layers
        encoder = self.encoder(embeddings, attention_mask, deterministic=deterministic)

        return encoder


@add_start_docstrings(
    "The bare Electra Model transformer outputting raw hidden-states without any specific head on top.",
    ELECTRA_START_DOCSTRING,
)
class FlaxElectraModel(FlaxElectraPreTrainedModel):
    
    def __init__(
        self, config: ElectraConfig, input_shape: Tuple = (1, 1), seed: int = 0, dtype: jnp.dtype = jnp.float32, **kwargs
    ):
        module = FlaxElectraModule(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            embedding_size=config.embedding_size,
            type_vocab_size=config.type_vocab_size,
            max_length=config.max_position_embeddings,
            num_encoder_layers=config.num_hidden_layers,
            num_heads=config.num_attention_heads,
            head_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            hidden_act=config.hidden_act,
            kernel_init_scale=config.initializer_range,
            dtype=dtype,
            **kwargs,
        )

        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype)

    @add_start_docstrings_to_model_forward(ELECTRA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        params: dict = None,
        dropout_rng: PRNGKey = None,
        train: bool = False,
    ):
        input_ids, attention_mask, token_type_ids, position_ids = self._check_inputs(
            input_ids, attention_mask, token_type_ids, position_ids
        )

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        logits = self.module.apply(
            {"params": params or self.params},
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(token_type_ids, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            not train,
            rngs=rngs,
        )
        return (logits, )


class FlaxElectraForMaskedLMModule(nn.Module):
    vocab_size: int
    embedding_size: int
    hidden_size: int
    intermediate_size: int
    head_size: int
    num_heads: int
    num_encoder_layers: int
    type_vocab_size: int
    max_length: int
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    kernel_init_scale: float = 0.2
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.electra = FlaxElectraModule(
            vocab_size=self.vocab_size,
            embedding_size=self.embedding_size,
            hidden_size=self.hidden_size,
            type_vocab_size=self.type_vocab_size,
            max_length=self.max_length,
            num_encoder_layers=self.num_encoder_layers,
            num_heads=self.num_heads,
            head_size=self.head_size, 
            intermediate_size=self.intermediate_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            hidden_act=self.hidden_act,
            kernel_init_scale=self.kernel_init_scale,
            dtype=self.dtype
        )

        self.generator_predictions = FlaxElectraGeneratorPredictions(embedding_size=self.embedding_size)
        self.generator_lm_head = nn.Dense(self.vocab_size)

    def __call__(
        self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, deterministic: bool = True
    ):
        generator_hidden_states = self.electra(input_ids, attention_mask, token_type_ids, position_ids, deterministic=deterministic)
        generator_sequence_output = generator_hidden_states

        prediction_scores = self.generator_predictions(generator_sequence_output)
        prediction_scores = self.generator_lm_head(prediction_scores)

        return prediction_scores


class FlaxElectraForMaskedLM(FlaxElectraPreTrainedModel):
    def __init__(self, config: ElectraConfig, input_shape: Tuple = (1, 1), seed: int = 0, dtype: jnp.dtype = jnp.float32, **kwargs):
        module = FlaxElectraForMaskedLMModule(
            vocab_size=config.vocab_size,
            type_vocab_size=config.type_vocab_size,
            embedding_size=config.embedding_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            head_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_encoder_layers=config.num_hidden_layers,
            max_length=config.max_position_embeddings,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            hidden_act=config.hidden_act,
            kernel_init_scale=config.initializer_range,
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
        dropout_rng: PRNGKey = None,
        train: bool = False,
    ):
        input_ids, attention_mask, token_type_ids, position_ids = self._check_inputs(
            input_ids, attention_mask, token_type_ids, position_ids
        )

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        logits = self.module.apply(
            {"params": freeze(params or self.params)},
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(token_type_ids, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            not train,
            rngs=rngs,
        )

        return (logits, )
    
    
class FlaxElectraForPreTrainingModule(nn.Module):
    vocab_size: int
    embedding_size: int
    hidden_size: int
    intermediate_size: int
    head_size: int
    num_heads: int
    num_encoder_layers: int
    type_vocab_size: int
    max_length: int
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    kernel_init_scale: float = 0.2
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.electra = FlaxElectraModule(
            vocab_size=self.vocab_size,
            embedding_size=self.embedding_size,
            hidden_size=self.hidden_size,
            type_vocab_size=self.type_vocab_size,
            max_length=self.max_length,
            num_encoder_layers=self.num_encoder_layers,
            num_heads=self.num_heads,
            head_size=self.head_size, 
            intermediate_size=self.intermediate_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            hidden_act=self.hidden_act,
            kernel_init_scale=self.kernel_init_scale,
            dtype=self.dtype
        )

        self.discriminator_predictions = FlaxElectraDiscriminatorPredictions(hidden_size=self.hidden_size)

    def __call__(
        self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, deterministic: bool = True
    ):
        hidden_states = self.electra(input_ids, attention_mask, token_type_ids, position_ids, deterministic=deterministic)
        logits = self.discriminator_predictions(hidden_states)
        
        return logits
    

class FlaxElectraForPreTraining(FlaxElectraPreTrainedModel):
    
    def __init__(self, config: ElectraConfig, input_shape: Tuple = (1, 1), seed: int = 0, dtype: jnp.dtype = jnp.float32, **kwargs):
        module = FlaxElectraForPreTrainingModule(
            vocab_size=config.vocab_size,
            type_vocab_size=config.type_vocab_size,
            embedding_size=config.embedding_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            head_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_encoder_layers=config.num_hidden_layers,
            max_length=config.max_position_embeddings,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            hidden_act=config.hidden_act,
            kernel_init_scale=config.initializer_range,
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
        dropout_rng: PRNGKey = None,
        train: bool = False,
    ):
        input_ids, attention_mask, token_type_ids, position_ids = self._check_inputs(
            input_ids, attention_mask, token_type_ids, position_ids
        )

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        logits = self.module.apply(
            {"params": freeze(params or self.params)},
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(token_type_ids, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            not train,
            rngs=rngs,
        )

        return (logits, )