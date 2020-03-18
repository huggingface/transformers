import jax.numpy as jnp
import flax.nn as nn


ACT2FN = {
    "gelu": nn.gelu,
    "relu": nn.relu,
    "swish": nn.swish,
    "gelu_new": nn.gelu,
}


class BertEmbeddings(nn.Module):
    def apply(self, inputs, token_type_ids, attention_mask, vocab_size, embedding_dim):
        x = nn.Embed(inputs, num_embeddings=vocab_size, features=embedding_dim, name='embed')


class BertModel(nn.Module):


class FXBertModel:
    """
    BERT implementation using JAX/Flax as backend
    """

    def __init__(self, config, **kwargs):
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.initializer_range = config.initializer_range


