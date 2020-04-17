import math
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor, nn

from transformers.modeling_utils import create_position_ids_from_input_ids


def create_sinusoidal_embeddings(n_pos, dim, out):
    position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()
    out.requires_grad = False


class SinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = create_sinusoidal_embeddings(init_size, embedding_dim)
        self.max_positions = init_size

    @torch.no_grad()
    def forward(
        self,
        input_ids,
        use_cache=False,
        # timestep: Optional[Tensor] = None,
        # positions: Optional[Any] = None,
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input_ids.shape[:2]
        # max_pos = self.padding_idx + 1 + seq_len
        # if self.weights is None or max_pos > self.weights.size(0):
        #    # recompute/expand embeddings if needed
        #    self.weights = self.get_embedding(max_pos, self.embedding_dim, self.padding_idx
        # )
        # self.weights = self.weights.to(self._float_tensor)
        if use_cache:
            assert seq_len != 1, "Remove me"
            return self.weights[seq_len].expand(bsz, 1, -1)
        else:
            positions = create_position_ids_from_input_ids(input_ids, 0, 1)
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()


def create_position_ids_from_input_ids(input_ids, padding_idx, offset=1):
    """ Replace non-padding symbols with their position numbers. Position numbers begin at
    padding_idx+1. Padding symbols are ignored. This is modified from fairseq's
    `utils.make_positions`.

    :param torch.Tensor x:
    :return torch.Tensor:
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
    return incremental_indices.long() + offset


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(
        self, num_embeddings: int, embedding_dim: int, padding_idx: int,
    ):
        # if padding_idx is specified then offset the embedding ids by
        # this index and adjust num_embeddings appropriately
        assert padding_idx is not None
        num_embeddings += padding_idx + 1  # WHY?
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def forward(self, input, use_cache=False):
        """Input is expected to be of size [bsz x seqlen]."""
        if use_cache:  # the position is our current step in the decoded sequence
            pos = int(self.padding_idx + input.size(1))
            positions = input.data.new(1, 1).fill_(pos)  # called before slicing.
        else:
            positions = create_position_ids_from_input_ids(input, self.padding_idx, 1)
        return super().forward(positions)
