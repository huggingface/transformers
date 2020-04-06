import os

import jax
import jax.numpy as jnp
import flax.nn as nn
import numpy as np

from typing import Callable

from flax.serialization import to_bytes
from jax.random import PRNGKey

from transformers import BertTokenizerFast, BertConfig, AutoModel
from transformers.modeling_jax_utils import JaxPreTrainedModel

ACT2FN = {
    "gelu": nn.gelu,
    "relu": nn.relu,
    "swish": nn.swish,
    "gelu_new": nn.gelu,
}


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
    return x * 0.5 * (1. + jax.lax.erf(x / jnp.sqrt(2.)))


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


class BertAttention(nn.Module):

    def apply(self, hidden_state, attention_mask, num_heads: int, head_size: int, output_size: int):
        self_att = nn.attention.SelfAttention(
            hidden_state,
            num_heads=num_heads,
            qkv_features=head_size,
            name="self"
        )

        return BertLayerNorm(self_att + hidden_state, name="layer_norm")


class BertIntermediate(nn.Module):

    def apply(self, hidden_state, output_size: int):
        # TODO: Had ACT2FN reference to change activation function
        h = nn.Dense(hidden_state, features=output_size, name="dense")
        return gelu(h)


class BertOutput(nn.Module):
    def apply(self, intermediate_output, attention_output):
        h = nn.Dense(intermediate_output, attention_output.shape[-1], name="dense")
        h = BertLayerNorm(h + attention_output, name="LayerNorm")

        return h


class BertLayer(nn.Module):

    def apply(self, hidden_state, attention_mask, num_heads: int, head_size: int, intermediate_size: int):
        attention = BertAttention(hidden_state, attention_mask, num_heads, head_size, intermediate_size, name="attention")
        intermediate = BertIntermediate(attention, intermediate_size, name="intermediate")
        output = BertOutput(intermediate, attention, name="output")

        return output


class BertLayerCollection(nn.Module):
    """
    Stores N BertLayer(s)
    """
    def apply(self, input, attention_mask, num_layers: int, num_heads: int, head_size: int, intermediate_size: int):
        assert num_layers > 0, "num_layers should be >= 1, got ({})".format(num_layers)

        # Initialize input / output
        input_i = output_i = input

        # Forward over all encoders
        for i in range(num_layers):
            output_i = BertLayer(input_i, attention_mask, num_heads, head_size, intermediate_size, name="{}".format(i))
            input_i = output_i
        return output_i


class BertEncoder(nn.Module):

    def apply(self, hidden_state, attention_mask, num_layers: int, num_heads: int, head_size: int, intermediate_size: int):
        encodings = BertLayerCollection(hidden_state, attention_mask, num_layers, num_heads, head_size, intermediate_size, name="layer")
        return encodings


class BertPooler(nn.Module):

    def apply(self, hidden_state):
        first_token = hidden_state[:, 0]
        out = nn.Dense(first_token, hidden_state.shape[-1], name="dense")
        return jax.lax.tanh(out)


class BertModel(nn.Module):

    def apply(self, input_ids, token_type_ids, attention_mask,
              vocab_size: int, hidden_size: int, type_vocab_size: int,
              max_length: int, num_encoder_layers: int, num_heads: int,
              head_size: int, intermediate_size: int,
              emb_init: Callable[..., np.ndarray] = nn.initializers.normal(stddev=0.1)):

        # Embedding
        embeddings = BertEmbeddings(
            input_ids, token_type_ids, attention_mask,
            vocab_size, hidden_size, type_vocab_size, max_length,
            name="embeddings"
        )

        # N stacked encoding layers
        encoder = BertEncoder(
            embeddings, attention_mask, num_encoder_layers,
            num_heads, head_size, intermediate_size,
            name="encoder"
        )

        pooled = BertPooler(encoder, name="pooler")
        return encoder, pooled


class FlaxBertModel(JaxPreTrainedModel):
    """
    BERT implementation using JAX/Flax as backend
    """

    MODEL_CLASS = BertModel
    config_class = BertConfig

    def __init__(self, config: BertConfig, state: dict, **kwargs):
        self.config = config
        self.key = PRNGKey(0)
        self.state = state

        self._model_def = BertModel.partial(
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.hidden_size,
            type_vocab_size=self.config.type_vocab_size,
            max_length=self.config.max_position_embeddings,
            num_encoder_layers=self.config.num_hidden_layers,
            num_heads=self.config.num_attention_heads,
            head_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size
        )

        self._bert = nn.Model(self._model_def, self.state)

    def __call__(self, input_ids, token_type_ids, attention_mask):

        @jax.jit
        def predict(input_ids, token_type_ids, attention_mask):
            return self._bert(
                jnp.array(input_ids, dtype='i4'),
                jnp.array(token_type_ids, dtype='i4'),
                jnp.array(attention_mask, dtype='i4')
            )

        return predict(input_ids, token_type_ids, attention_mask)

    def save_pretrained(self, folder):
        folder_abs = os.path.abspath(folder)

        if not os.path.exists(folder_abs):
            os.mkdir(folder_abs)

        with open(os.path.join(folder_abs, 'model.flax'), 'wb') as f:
            model_bytes = to_bytes(self._bert)
            f.write(model_bytes)


if __name__ == '__main__':
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    model_pt = AutoModel.from_pretrained('bert-base-cased')
    model_pt.eval()

    # with open("/data/Downloads/bert-base-cased-pytorch_model.bin", 'rb') as model_f:
    #     state = torch.load(model_f)
    #     state = {k: v.numpy() for k, v in state.items()}
    #
    #     # Need to change some parameters name to match Flax names so that we don't have to fork any layer
    #     state = {k.replace("weight", "kernel") if {"dense", "query", "key", "value"} & set(k.split('.')) else k: v
    #              for k, v in state.items()}
    #
    #     # SelfAttention output is not a separate layer, remove one nesting
    #     state = {k.replace("attention.output.dense", "attention.self.out"): v for k, v in state.items()}
    #     state = {k.replace("attention.output.LayerNorm", "attention.layer_norm"): v for k, v in state.items()}
    #
    #     # SelfAttention.out
    #     state = {k: v.T if v.shape == (3072, 768) or v.shape == (768, 3072) else v for k, v in state.items()}
    #     state = {k: v.reshape((12, 64, 768)).transpose((2, 0, 1)) if v.shape == (768, 768) and "out" not in k and "pooler" not in k else v for k, v in state.items()}
    #
    #     # Bias
    #     state = {k: v.reshape((12, -1)) if v.shape == (768, ) and ("bias" in k) and ("out" not in k) and ("pooler" not in k) else v for k, v in state.items()}
    #
    #     # Self Attention output projection
    #     state = {k: v.reshape((768, 12, 64)).transpose(1, 2, 0) if "out.kernel" in k else v for k, v in state.items()}
    #
    #     # Pooler
    #     state = {k: v.T if "pooler.dense.kernel" in k else v for k, v in state.items()}
    #     state = unflatten_dict({tuple(k.split('.')[1:]): v for k, v in state.items()})
    #     model = FlaxBertModel.from_pretrained(model_pt.config, state)

    model = FlaxBertModel.from_pretrained(
        '/data/Workspace/transformers/src/transformers/bert-base-cased/bert-base-cased.bin',
        config=model_pt.config,
    )

    # Inputs
    flax_input = tokenizer.encode_plus("My name is Morgan")
    pt_input = tokenizer.encode_plus("My name is Morgan", return_tensors="pt")

    # Forward
    model_pt.eval()
    pt_enc = model_pt(pt_input['input_ids'], pt_input['attention_mask'])
    flax_enc = model(**flax_input)

    input()