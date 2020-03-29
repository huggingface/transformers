"""
TODO: add dropout where appropriate and a training indicator
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import torch

from typing import Callable

import haiku as hk
from haiku.initializers import Constant
from jax.random import PRNGKey

from transformers import BertModel as PTBertModel, BertTokenizerFast, BertConfig


def gelu(x):
    """
    Less efficient but more precise than 
    the approximation used by jax.nn.gelu
    """
    return x * 0.5 * (1.0 + jax.lax.erf(x / jnp.sqrt(2.0)))


ACT2FN = {
    "gelu": gelu,
    "relu": jax.nn.relu,
    "swish": jax.nn.swish,
    "gelu_new": jax.nn.gelu,
}


class PretrainedModule(hk.Module):
    """
    Lightweight wrapper around hk.Module to expose 
    pre-trained parameters when provided
    """

    def __init__(self, name=None, state=None, **settings):
        super().__init__(name=name)
        self._state = state or {}
        for setting_key, setting_value in settings.items():
            setattr(self, setting_key, setting_value)

    def state(self, prefix):
        prefix = prefix + "."
        matching_states = {}
        for key in self._state.keys():
            if key.startswith(prefix):
                trimmed_key = key[len(prefix) :]
                matching_states[trimmed_key] = self._state[key]
        return matching_states

    def pretrained(self, key):
        try:
            return Constant(self._state[key])
        except KeyError:
            print(f"Failed to find {key} in `{self.__class__.__name__}({self.name})` state: {self._state.keys()}")
            return None


class BertLayerNorm(PretrainedModule):
    """
    Layer normalization (https://arxiv.org/abs/1607.06450).
    Operates on the last axis of the input data.
    """

    def __init__(
        self,
        name="LayerNorm",
        state=None,
        epsilon=1e-6,
        bias=True,
        scale=True,
        bias_init=hk.initializers.Constant(0.0),
        scale_init=hk.initializers.Constant(1.0),
    ):
        super().__init__(
            name=name, state=state, epsilon=epsilon, bias=bias, scale=scale, bias_init=bias_init, scale_init=scale_init
        )

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
        return hk.LayerNorm(
            axis=-1,
            eps=self.epsilon,
            create_scale=self.scale,
            create_offset=self.bias,
            scale_init=self.pretrained("gamma") or self.scale_init,
            offset_init=self.pretrained("beta") or self.bias_init,
        )(x)


class BertEmbedding(PretrainedModule):
    """
    Handles word_embedding / position embedding and token type embedding
    """

    def __call__(self, x):
        flat_token_ids = jnp.reshape(x, [1, x.shape[0] * x.shape[1]])
        flat_token_embeddings = hk.Embed(
            vocab_size=self.vocab_size,
            embed_dim=self.hidden_size,
            # TODO: get actual weight stddev value
            w_init=self.pretrained("weight") or hk.initializers.RandomNormal(0.1),
        )(flat_token_ids)
        token_embeddings = jnp.reshape(flat_token_embeddings, [x.shape[0], x.shape[1], flat_token_embeddings.shape[-1]])
        return token_embeddings


class BertEmbeddings(PretrainedModule):
    def __call__(self, input_ids, token_type_ids):
        # Embed
        w_emb = BertEmbedding(
            name="word_embeddings",
            state=self.state("word_embeddings"),
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
        )(jnp.atleast_2d(input_ids.astype("i4")))
        p_emb = BertEmbedding(
            name="position_embeddings",
            state=self.state("position_embeddings"),
            vocab_size=self.max_length,
            hidden_size=self.hidden_size,
        )(jnp.atleast_2d(np.arange(input_ids.shape[-1])))
        t_emb = BertEmbedding(
            name="token_type_embeddings",
            state=self.state("token_type_embeddings"),
            vocab_size=self.type_vocab_size,
            hidden_size=self.hidden_size,
        )(jnp.atleast_2d(token_type_ids.astype("i4")))

        # Sum all embeddings
        summed_emb = w_emb + jnp.broadcast_to(p_emb, w_emb.shape) + t_emb

        # Layer Norm
        norm = BertLayerNorm(name="LayerNorm", state=self.state("LayerNorm"))(summed_emb)
        return norm


class Linear(PretrainedModule):
    def __call__(self, x):
        return hk.Linear(
            name=self.name,
            output_size=self.output_size,
            w_init=self.pretrained("weight") or hk.initializers.RandomNormal(0.1),
            b_init=self.pretrained("bias") or hk.initializers.Constant(0.0),
        )(x)


class SelfAttention(PretrainedModule):
    def _split_into_heads(self, x):
        return jnp.reshape(x, [x.shape[0], x.shape[1], self.num_heads, self.head_size])

    def _join_heads(self, x):
        return jnp.reshape(x, [x.shape[0], x.shape[1], self.num_heads * self.head_size])

    def __call__(self, x, mask):
        # Project to queries, keys, and values
        # Shapes are all [batch, sequence_length, hidden_size]
        hidden_size = self.num_heads * self.head_size

        queries = Linear(name="query", state=self.state("query"), output_size=hidden_size)(x)
        keys = Linear(name="key", state=self.state("key"), output_size=hidden_size)(x)
        values = Linear(name="value", state=self.state("value"), output_size=hidden_size)(x)

        # Reshape our hidden state to group into heads
        # New shapes are:
        # [batch, sequence_length, n_heads, size_per_head]
        queries = self._split_into_heads(queries)
        keys = self._split_into_heads(keys)
        values = self._split_into_heads(values)

        # Compute per head attention weights
        # b: batch
        # s: source sequence
        # t: target sequence
        # n: number of heads
        # h: per-head hidden state
        attention_logits = jnp.einsum("bsnh,btnh->bnst", queries, keys)
        attention_logits /= np.sqrt(queries.shape[-1])
        # Add logits of mask tokens with a large negative number
        # to prevent attending to those terms.
        attention_logits += jnp.reshape((1 - mask) * -(2 ** 32), [mask.shape[0], 1, 1, mask.shape[1]])
        attention_weights = jax.nn.softmax(attention_logits, axis=-1)
        per_head_attention_output = jnp.einsum("btnh,bnst->bsnh", values, attention_weights)
        attention_output = self._join_heads(per_head_attention_output)

        return attention_output


class BertAttention(PretrainedModule):
    def __call__(self, hidden_state, attention_mask):
        attention_output = SelfAttention(
            name="self", state=self.state("self"), num_heads=self.num_heads, head_size=self.head_size,
        )(hidden_state, attention_mask)
        output = Linear(name="dense", state=self.state("output.dense"), output_size=self.num_heads * self.head_size,)(
            attention_output
        )
        output = BertLayerNorm(name="LayerNorm", state=self.state("output.LayerNorm"))(hidden_state + output)
        return output


class BertIntermediate(PretrainedModule):
    def __call__(self, hidden_state):
        h = Linear(name="dense", state=self.state("dense"), output_size=self.intermediate_size)(hidden_state)
        return ACT2FN[self.activation_fn](h)


class BertOutput(PretrainedModule):
    def __call__(self, intermediate_output, attention_output):
        h = Linear(name="dense", state=self.state("dense"), output_size=self.output_size)(intermediate_output)
        h = BertLayerNorm(name="LayerNorm", state=self.state("LayerNorm"),)(h + attention_output)
        return h


class BertLayer(PretrainedModule):
    def __call__(self, hidden_state, attention_mask):
        attention = BertAttention(
            name="attention", state=self.state("attention"), num_heads=self.num_heads, head_size=self.head_size
        )(hidden_state, attention_mask)
        intermediate = BertIntermediate(
            name="intermediate",
            state=self.state("intermediate"),
            intermediate_size=self.intermediate_size,
            activation_fn=self.activation_fn,
        )(attention)
        output = BertOutput(name="output", state=self.state("output"), output_size=self.num_heads * self.head_size)(
            intermediate, attention
        )
        return output


class BertEncoder(PretrainedModule):
    def __call__(self, x, mask):
        assert self.num_layers > 0, "num_layers should be >= 1, got ({})".format(self.num_layers)
        # Forward over all encoders
        for i in range(self.num_layers):
            layer_name = "layer_{}".format(i)
            x = BertLayer(
                name=layer_name,
                state=self.state(layer_name.replace("_", ".")),
                num_heads=self.num_heads,
                head_size=self.head_size,
                intermediate_size=self.intermediate_size,
                activation_fn=self.activation_fn,
            )(x, mask)
        return x


class BertPooler(PretrainedModule):
    def __call__(self, x):
        first_token = x[:, 0]
        out = Linear(name="dense", state=self.state("dense"), output_size=x.shape[-1])(first_token)
        return jax.lax.tanh(out)


class BertModel(PretrainedModule):
    def __call__(self, input_ids, token_type_ids, attention_mask):

        # Embedding
        embeddings = BertEmbeddings(
            name="embeddings",
            state=self.state("embeddings"),
            vocab_size=self.vocab_size,
            type_vocab_size=self.type_vocab_size,
            hidden_size=self.hidden_size,
            max_length=self.max_length,
        )(input_ids, token_type_ids)

        # N stacked encoding layers
        encoder = BertEncoder(
            name="encoder",
            state=self.state("encoder"),
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_size=self.head_size,
            intermediate_size=self.intermediate_size,
            activation_fn=self.activation_fn,
        )(embeddings, jnp.atleast_2d(attention_mask))

        pooled = BertPooler(name="pooler", state=self.state("pooler"),)(encoder)
        return encoder, pooled


class HaikuBertModel:
    """
    BERT implementation using JAX/Haiku as backend
    """

    def __init__(self, config: BertConfig, state: dict, **kwargs):
        self.config = config
        self.rng = PRNGKey(0)
        self.pretrained_state = state
        self.params = None
        self.predict_fn = None

    def __call__(self, input_ids, token_type_ids, attention_mask):

        if self.params is None or self.predict_fn is None:
            # Lazily initialize parameters and JIT compile
            def predict(input_ids, token_type_ids, attention_mask):
                model = BertModel(
                    name="bert",
                    state=self.pretrained_state,
                    vocab_size=self.config.vocab_size,
                    hidden_size=self.config.hidden_size,
                    type_vocab_size=self.config.type_vocab_size,
                    max_length=self.config.max_position_embeddings,
                    num_layers=self.config.num_hidden_layers,
                    num_heads=self.config.num_attention_heads,
                    head_size=self.config.hidden_size // self.config.num_attention_heads,
                    intermediate_size=self.config.intermediate_size,
                    activation_fn=self.config.hidden_act,
                )
                return model(
                    jnp.atleast_2d(jnp.asarray(input_ids, dtype="i4")),
                    jnp.atleast_2d(jnp.asarray(token_type_ids, dtype="i4")),
                    jnp.atleast_2d(jnp.asarray(attention_mask, dtype="i4")),
                )

            predict_module = hk.transform(predict, apply_rng=True)
            self.params = predict_module.init(self.rng, input_ids, token_type_ids, attention_mask)
            self.predict_fn = jax.jit(predict_module.apply)
        return self.predict_fn(self.params, self.rng, input_ids, token_type_ids, attention_mask)

    @staticmethod
    def from_pretrained(config: BertConfig, state: dict):
        # TODO: subclass huggingface models properly so this can also
        # take in models by name
        state = dict(state)
        for key, value in dict(state).items():
            value = value.numpy()
            if key.endswith("weight") and not "embeddings" in key:
                value = value.T
            del state[key]
            state[key.replace("bert.", "")] = value
        return HaikuBertModel(config, state)

    def save_pretrained(self, folder):
        # TODO: find an elegant way to convert stored params dictionary into expected format
        raise NotImplementedError()


if __name__ == "__main__":
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    model_pt = PTBertModel.from_pretrained("bert-base-cased")

    with open("/home/m/Downloads/bert-base-cased-pytorch_model.bin", "rb") as model_f:
        state = torch.load(model_f)
        model = HaikuBertModel.from_pretrained(model_pt.config, state)

    # Inputs
    haiku_input = tokenizer.batch_encode_plus(["My name is Morgan"] * 2)
    pt_input = tokenizer.encode_plus("My name is Morgan", return_tensors="pt")

    # Forward
    model_pt.eval()
    pt_enc = model_pt(pt_input["input_ids"], pt_input["attention_mask"])
    pt_seq_features = pt_enc[0].detach().numpy()
    pt_pooled_features = pt_enc[1].detach().numpy()
    haiku_seq_features, haiku_pooled_features = model(**haiku_input)
    assert np.allclose(haiku_seq_features, pt_seq_features, atol=1e-3)
    assert np.allclose(haiku_pooled_features, pt_pooled_features, atol=1e-3)
