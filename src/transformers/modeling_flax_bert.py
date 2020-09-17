from typing import Callable, Dict

import numpy as np

import flax.linen as nn
from flax.linen import compact
import jax
import jax.numpy as jnp
from transformers import BertConfig
from transformers.modeling_flax_utils import FlaxPreTrainedModel


@jax.jit
def gelu(x):
    r"""Gaussian error linear unit activation function.

    Computes the element-wise function:

    .. math::
      \mathrm{gelu}(x) = \frac{x}{2} \left(1 + \mathrm{tanh} \left(
        \sqrt{\frac{2}{\pi}} \left(x + 0.044715 x^3 \right) \right) \right)

    We explicitly use the approximation rather than the exact formulation for
    speed. For more information, see `Gaussian Error Linear Units (GELUs)
    <https://arxiv.org/abs/1606.08415>`_, section 2.
    """
    return x * 0.5 * (1.0 + jax.lax.erf(x / jnp.sqrt(2.0)))


ACT2FN = {
    "gelu": gelu,
    "relu": nn.relu,
    "swish": nn.swish,
    "gelu_new": gelu,
}


class BertLayerNorm(nn.Module):
    """Layer normalization (https://arxiv.org/abs/1607.06450).
    Operates on the last axis of the input data.
    """
    epsilon: float = 1e-6
    dtype: jnp.dtype = jnp.float32
    bias: bool = True
    scale: bool = True
    bias_init: jnp.ndarray = nn.initializers.zeros
    scale_init: jnp.ndarray = nn.initializers.ones

    @compact
    def __call__(self, x):
        """Applies layer normalization on the input.
        It normalizes the activations of the layer for each given example in a
        batch independently, rather than across a batch like Batch Normalization.
        i.e. applies a transformation that maintains the mean activation within
        each example close to 0 and the activation standard deviation close to 1.
        Args:
          x: the inputs
          epsilon: A small float added to variance to avoid dividing by zero.
          dtype: the dtype of the computation (default: float32).
          bias:  If True, bias (beta) is added.
          scale: If True, multiply by scale (gamma). When the next layer is linear
            (also e.g. nn.relu), this can be disabled since the scaling will be done
            by the next layer.
          bias_init: Initializer for bias, by default, zero.
          scale_init: Initializer for scale, by default, one.
        Returns:
          Normalized inputs (the same shape as inputs).
        """
        features = x.shape[-1]
        mean = jnp.mean(x, axis=-1, keepdims=True)
        mean2 = jnp.mean(jax.lax.square(x), axis=-1, keepdims=True)
        var = mean2 - jax.lax.square(mean)
        mul = jax.lax.rsqrt(var + self.epsilon)
        if self.scale:
            mul = mul * jnp.asarray(self.param("gamma", self.scale_init, (features,)), 
                                    self.dtype)
        y = (x - mean) * mul
        if self.bias:
            y = y + jnp.asarray(self.param("beta", self.bias_init, (features,)),
                                self.dtype)
        return y


class BertEmbedding(nn.Module):
    """
    Specify a new class for doing the embedding stuff
    as Flax's one use 'embedding' for the parameter name
    and PyTorch use 'weight'
    """
    vocab_size: int
    hidden_size: int
    emb_init: Callable[..., np.ndarray] = nn.initializers.normal(stddev=0.1)

    @compact
    def __call__(self, inputs):
        embedding = self.param("weight", self.emb_init, (self.vocab_size, self.hidden_size))
        return jnp.take(embedding, inputs, axis=0)


class BertEmbeddings(nn.Module):
    vocab_size: int
    hidden_size: int
    type_vocab_size: int
    max_length: int

    @compact
    def __call__(self, input_ids, token_type_ids, position_ids, attention_mask):

        # Embed
        w_emb = BertEmbedding(self.vocab_size, self.hidden_size, name="word_embeddings")(jnp.atleast_2d(input_ids.astype("i4")))
        p_emb = BertEmbedding(self.max_length, self.hidden_size, name="position_embeddings")(jnp.atleast_2d(position_ids.astype("i4")))
        t_emb = BertEmbedding(self.type_vocab_size, self.hidden_size, name="token_type_embeddings")(jnp.atleast_2d(token_type_ids.astype("i4")))

        # Sum all embeddings
        summed_emb = w_emb + jnp.broadcast_to(p_emb, w_emb.shape) + t_emb

        # Layer Norm
        layer_norm = BertLayerNorm(name="layer_norm")(summed_emb)

        return layer_norm


class BertAttention(nn.Module):
    num_heads: int
    head_size: int

    @compact
    def __call__(self, hidden_state, attention_mask):
        self_att = nn.attention.SelfAttention(
            num_heads=self.num_heads, qkv_features=self.head_size, name="self")(hidden_state, attention_mask)

        layer_norm = BertLayerNorm(name="layer_norm")(self_att + hidden_state)
        return layer_norm


class BertIntermediate(nn.Module):
    output_size: int

    @compact
    def __call__(self, hidden_state):
        # TODO: Add ACT2FN reference to change activation function
        dense = nn.Dense(features=self.output_size, name="dense")(hidden_state)
        return gelu(dense)


class BertOutput(nn.Module):
    @compact
    def __call__(self, intermediate_output, attention_output):
        h = nn.Dense(attention_output.shape[-1], name="dense")(intermediate_output)
        h = BertLayerNorm(name="layer_norm")(h + attention_output)
        return h


class BertLayer(nn.Module):
    num_heads: int
    head_size: int
    intermediate_size: int

    @compact
    def __call__(self, hidden_state, attention_mask):
        attention = BertAttention(self.num_heads, self.head_size, name="attention")(hidden_state, attention_mask)
        intermediate = BertIntermediate(self.intermediate_size, name="intermediate")(attention)
        output = BertOutput(name="output")(intermediate, attention)

        return output


class BertLayerCollection(nn.Module):
    """
    Stores N BertLayer(s)
    """
    num_layers: int
    num_heads: int
    head_size: int
    intermediate_size: int

    @compact
    def __call__(self, inputs, attention_mask):
        assert self.num_layers > 0, "num_layers should be >= 1, got ({})".format(self.num_layers)

        # Initialize input / output
        input_i = output_i = inputs

        # Forward over all encoders
        for i in range(self.num_layers):
            layer = BertLayer(self.num_heads, self.head_size, self.intermediate_size, name="{}".format(i))
            output_i = layer(input_i, attention_mask)
            input_i = output_i
        return output_i


class BertEncoder(nn.Module):
    num_layers: int
    num_heads: int
    head_size: int
    intermediate_size: int

    @compact
    def __call__(self, hidden_state, attention_mask):
        layer = BertLayerCollection(self.num_layers, self.num_heads, self.head_size, 
            self.intermediate_size, name="layer")(hidden_state, attention_mask)
        return layer


class BertPooler(nn.Module):
    @compact
    def __call__(self, hidden_state):
        first_token = hidden_state[:, 0]
        out = nn.Dense(hidden_state.shape[-1], name="dense")(first_token)
        return jax.lax.tanh(out)


class BertModel(nn.Module):
    vocab_size: int
    hidden_size: int
    type_vocab_size: int
    max_length: int
    num_encoder_layers: int
    num_heads: int
    head_size: int
    intermediate_size: int
    padding_idx: int
    emb_init: Callable[..., np.ndarray] = nn.initializers.normal(stddev=0.1)

    @compact
    def __call__(self, input_ids, token_type_ids, position_ids, attention_mask):

        # Embedding
        embeddings = BertEmbeddings(self.vocab_size, self.hidden_size, self.type_vocab_size,
            self.max_length, name="embeddings")(input_ids, token_type_ids, position_ids, attention_mask)

        # N stacked encoding layers
        encoder = BertEncoder(self.num_encoder_layers, self.num_heads, self.head_size, 
            self.intermediate_size, name="encoder")(embeddings, attention_mask)

        pooled = BertPooler(name="pooler")(encoder)
        return encoder, pooled


class FlaxBertModel(FlaxPreTrainedModel):
    """
    BERT implementation using JAX/Flax as backend
    """

    model_class = BertModel
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

    def __init__(self, config: BertConfig, state: dict, seed: int = 0, **kwargs):
        model = BertModel(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            type_vocab_size=config.type_vocab_size,
            max_length=config.max_position_embeddings,
            num_encoder_layers=config.num_hidden_layers,
            num_heads=config.num_attention_heads,
            head_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            padding_idx=config.pad_token_id,
        )

        super().__init__(config, model, state, seed)

    @property
    def module(self) -> BertModel:
        return self._module

    @property
    def config(self) -> BertConfig:
        return self._config

    def __call__(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None):
        @jax.jit
        def predict(input_ids, token_type_ids=None, position_ids=None, attention_mask=None):
            return self.model.apply(
                {"param": self.params},
                jnp.array(input_ids, dtype="i4"),
                jnp.array(token_type_ids, dtype="i4"),
                jnp.array(position_ids, dtype="i4"),
                jnp.array(attention_mask, dtype="i4")
            )

        if token_type_ids is None:
            token_type_ids = np.ones_like(input_ids)

        if position_ids is None:
            position_ids = np.arange(np.atleast_2d(input_ids).shape[-1])

        if attention_mask is None:
            attention_mask = np.ones_like(input_ids)

        return predict(input_ids, token_type_ids, position_ids, attention_mask)


if __name__ == "__main__":
    model = FlaxBertModel.from_pretrained("bert-base-cased")
    input()
