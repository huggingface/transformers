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


class FXBertModel:
    """
    BERT implementation using JAX/Flax as backend
    """

    def __init__(self, config, **kwargs):
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.initializer_range = config.initializer_range

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