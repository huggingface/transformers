import jax
import jax.numpy as jnp
import flax.nn as nn
import numpy as np
import torch

from dataclasses import dataclass
from typing import Tuple, Callable

from flax.nn import Model
from flax.serialization import from_state_dict
from flax.traverse_util import unflatten_dict
from jax.random import PRNGKey

from transformers import BertModel as PTBertModel, BertTokenizerFast


ACT2FN = {
    "gelu": nn.gelu,
    "relu": nn.relu,
    "swish": nn.swish,
    "gelu_new": nn.gelu,
}


@dataclass
class BertConfig:
    max_length: int
    vocab_size: int
    hidden_size: int
    initializer_range: Tuple[int, int] = (-1, 1)


class BertLayerNorm(nn.Module):
    """Layer normalization (https://arxiv.org/abs/1607.06450).
    Operates on the last axis of the input data.
    """

    def apply(self,
              x,
              epsilon=1e-6,
              dtype=jnp.float32,
              bias=True,
              scale=True,
              bias_init=nn.initializers.zeros,
              scale_init=nn.initializers.ones):
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
        mean2 = jnp.mean(jnp.lax.square(x), axis=-1, keepdims=True)
        var = mean2 - jnp.lax.square(mean)
        mul = jnp.lax.rsqrt(var + epsilon)
        if scale:
            mul = mul * jnp.asarray(self.param('gamma', (features,), scale_init), dtype)
        y = (x - mean) * mul
        if bias:
            y = y + jnp.asarray(self.param('beta', (features,), bias_init), dtype)
        return y


class BertEmbedding(nn.Module):
    """
    Specify a new class for doing the embedding stuff
    as Flax's one use 'embedding' for the parameter name
    and PyTorch use 'weight'
    """

    def apply(self, input, vocab_size: int, hidden_size: int,
              emb_init: Callable[..., np.ndarray] = nn.initializers.normal(stddev=0.1)):

        embedding = self.param('weight', (vocab_size, hidden_size), emb_init)
        return jnp.take(embedding, input, axis=0)


class BertEmbeddings(nn.Module):
    def apply(self, input_ids, token_type_ids, attention_mask, vocab_size: int,
              hidden_size: int, type_vocab_size: int, max_length: int):

        # Embed
        w_emb = BertEmbedding(jnp.atleast_2d(input_ids.astype('i4')), vocab_size, hidden_size, name="word_embeddings")
        p_emb = BertEmbedding(jnp.atleast_2d(np.arange(input_ids.shape[-1])), max_length, hidden_size, name="position_embeddings")
        t_emb = BertEmbedding(jnp.atleast_2d(token_type_ids.astype('i4')), 2, hidden_size, name="token_type_embeddings")

        # Sum all embeddings
        summed_emb = w_emb + jnp.broadcast_to(p_emb, w_emb.shape) + t_emb

        # Layer Norm
        norm = BertLayerNorm(summed_emb, name="LayerNorm")

        return norm


class BertModel(nn.Module):

    def apply(self, input_ids, token_type_ids, attention_mask,
              vocab_size: int, hidden_size: int, type_vocab_size: int, max_length: int,
              emb_init: Callable[..., np.ndarray] = nn.initializers.normal(stddev=0.1)):

        embeddings = BertEmbeddings(
            input_ids, token_type_ids, attention_mask,
            vocab_size, hidden_size, type_vocab_size, max_length,
            name="embeddings"
        )

        return embeddings


class FXBertModel:
    """
    BERT implementation using JAX/Flax as backend
    """

    def __init__(self, config: BertConfig, **kwargs):
        self.config = config
        self.key = PRNGKey(0)
        self.state = None

    def __call__(self, input_ids, token_type_ids, attention_mask):
        model_def = BertModel.partial(
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.hidden_size,
            type_vocab_size=2,
            max_length=self.config.max_length,
        )

        # inputs_shape = [(1, len(input_ids))] * 3
        # _ = model_def.init_by_shape(self.key, inputs_shape)
        bert = Model(model_def, self.state)

        @jax.jit
        def predict(input_ids, token_type_ids, attention_mask):
            return bert(
                jnp.array(input_ids, dtype='i4'),
                jnp.array(token_type_ids, dtype='i4'),
                jnp.array(attention_mask, dtype='i4')
            )

        return predict(input_ids, token_type_ids, attention_mask)

    def from_pretrained(self, state):
        self.state = from_state_dict(BertModel, state)
        return self.state


if __name__ == '__main__':
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    model = FXBertModel(BertConfig(512, 28996, 768))
    model_pt = PTBertModel.from_pretrained('bert-base-cased')
    model_pt.eval()

    with open("/data/Downloads/bert-base-cased-pytorch_model.bin", 'rb') as model_f:
        state = torch.load(model_f)
        state = unflatten_dict({tuple(k.split('.')[1:]): v.numpy() for k, v in state.items()})
        model.from_pretrained(state)

    out = model(**tokenizer.encode_plus("My name is Morgan"))
    out_pt = model_pt.embeddings(tokenizer.encode_plus("My name is Morgan", return_tensors="pt")['input_ids'])
    input()