import jax.numpy as jnp
import flax.nn as nn
import numpy as np
import torch

from dataclasses import dataclass
from typing import Tuple, Callable

from flax.nn import Model
from flax.nn.normalization import LayerNorm
from flax.serialization import from_state_dict
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


# class BertEmbeddings(nn.Module):
#     def apply(self, inputs, vocab_size, embedding_dim, emb_init: Callable[..., np.ndarray] = nn.initializers.normal(stddev=0.1)):
#         word_embeddings = self.param('bert.embeddings.word_embeddings.weight', (vocab_size, embedding_dim), emb_init)
#         return word_embeddings[inputs.astype(np.int32)]
#
#         # return nn.Embed(inputs, num_embeddings=vocab_size, features=embedding_dim, name='')


class BertModel(nn.Module):

    def apply(self, input_ids, token_type_ids, attention_mask,
              vocab_size: int, hidden_size: int, type_vocab_size: int, max_length: int,
              emb_init: Callable[..., np.ndarray] = nn.initializers.normal(stddev=0.1)):

        # Embed
        word_embeddings = self.param('bert.embeddings.word_embeddings.weight', (vocab_size, hidden_size), emb_init)
        pos_embeddings  = self.param('bert.embeddings.position_embeddings.weight', (max_length, hidden_size), emb_init)
        type_embeddings = self.param('bert.embeddings.token_type_embeddings.weight', (type_vocab_size, hidden_size), emb_init)

        w_emb = word_embeddings[jnp.atleast_2d(input_ids.astype('i4'))]
        p_emb = pos_embeddings[jnp.atleast_2d(np.arange(input_ids.shape[-1]))]
        t_emb = type_embeddings[jnp.atleast_2d(token_type_ids.astype('i4'))]

        # Sum all embeddings
        summed_emb = w_emb + jnp.broadcast_to(p_emb, w_emb.shape) + t_emb

        # Layer Norm
        norm_gamma = self.param('bert.embeddings.LayerNorm.gamma', (hidden_size, ), emb_init)
        norm_beta  = self.param('bert.embeddings.LayerNorm.beta', (hidden_size, ), emb_init)
        norm = LayerNorm.call({"scale": norm_gamma, "bias": norm_beta, "epsilon": 1e-12}, summed_emb)

        return norm


class FXBertModel:
    """
    BERT implementation using JAX/Flax as backend
    """

    def __init__(self, config: BertConfig, **kwargs):
        self.config = config
        self.key = PRNGKey(0)
        self.state = None

    def __call__(self, input_ids, token_type_ids, attention_mask):
        inputs_shape = [(1, len(input_ids))] * 3

        model_def = BertModel.partial(
            name="bert",
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.hidden_size,
            type_vocab_size=2,
            max_length=self.config.max_length
        )

        _ = model_def.init_by_shape(self.key, inputs_shape)
        bert = Model(model_def, self.state)
        return bert(
            jnp.array(input_ids, dtype='i4'),
            jnp.array(token_type_ids, dtype='i4'),
            jnp.array(attention_mask, dtype='i4')
        )

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
        state = {k: v.numpy() for k, v in state.items()}
        model.from_pretrained(state)

    out = model(**tokenizer.encode_plus("My name is Morgan"))
    out_pt = model_pt.embeddings(tokenizer.encode_plus("My name is Morgan", return_tensors="pt")['input_ids'])
    input()