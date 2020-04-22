import numpy as np
import torch
from torch import nn

from transformers.modeling_utils import create_position_ids_from_input_ids


class SinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, init_size, embedding_dim, padding_idx):
        super().__init__(init_size, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter):
        """Identical to the XLM create_sinusoidal_embeddings except features are not interleaved.
            The cos features are in the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        out[:, 0 : dim // 2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, dim // 2 :] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        out.requires_grad = False
        return out

    @torch.no_grad()
    def forward(
        self, input_ids, use_cache=False,
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input_ids.shape[:2]
        if use_cache:
            positions = input_ids.data.new(1, 1).fill_(seq_len - 1)  # called before slicing.
        else:
            positions = torch.arange(seq_len, dtype=torch.long)  # starts at 0, ends at 1-seq_len
        return super().forward(positions)


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
        super().__init__(num_embeddings + 2, embedding_dim, padding_idx=padding_idx)

    def forward(self, input, use_cache=False):
        """Input is expected to be of size [bsz x seqlen]."""
        if use_cache:  # the position is our current step in the decoded sequence
            pos = int(self.padding_idx + input.size(1))
            positions = input.data.new(1, 1).fill_(pos)  # called before slicing.
        else:
            positions = create_position_ids_from_input_ids(input, self.padding_idx)
        return super().forward(positions)
