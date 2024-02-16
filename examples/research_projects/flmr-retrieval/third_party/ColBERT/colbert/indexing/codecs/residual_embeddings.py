import os
import torch
import ujson
import tqdm

from colbert.indexing.codecs.residual_embeddings_strided import ResidualEmbeddingsStrided
from colbert.utils.utils import print_message


class ResidualEmbeddings:
    Strided = ResidualEmbeddingsStrided

    def __init__(self, codes, residuals):
        """
            Supply the already compressed residuals.
        """

        # assert isinstance(residuals, bitarray), type(residuals)
        assert codes.size(0) == residuals.size(0), (codes.size(), residuals.size())
        assert codes.dim() == 1 and residuals.dim() == 2, (codes.size(), residuals.size())
        assert residuals.dtype == torch.uint8

        self.codes = codes.to(torch.int32)  # (num_embeddings,) int32
        self.residuals = residuals   # (num_embeddings, compressed_dim) uint8

    @classmethod
    def load_chunks(cls, index_path, chunk_idxs, num_embeddings):
        num_embeddings += 512  # pad for access with strides

        dim, nbits = get_dim_and_nbits(index_path)

        codes = torch.empty(num_embeddings, dtype=torch.int32)
        residuals = torch.empty(num_embeddings, dim // 8 * nbits, dtype=torch.uint8)

        codes_offset = 0

        print_message("#> Loading codes and residuals...")

        for chunk_idx in tqdm.tqdm(chunk_idxs):
            chunk = cls.load(index_path, chunk_idx)

            codes_endpos = codes_offset + chunk.codes.size(0)

            # Copy the values over to the allocated space
            codes[codes_offset:codes_endpos] = chunk.codes
            residuals[codes_offset:codes_endpos] = chunk.residuals

            codes_offset = codes_endpos

        # codes, residuals = codes.cuda(), residuals.cuda()  # FIXME: REMOVE THIS LINE!

        return cls(codes, residuals)

    @classmethod
    def load(cls, index_path, chunk_idx):
        codes = cls.load_codes(index_path, chunk_idx)
        residuals = cls.load_residuals(index_path, chunk_idx)

        return cls(codes, residuals)

    @classmethod
    def load_codes(self, index_path, chunk_idx):
        codes_path = os.path.join(index_path, f'{chunk_idx}.codes.pt')
        return torch.load(codes_path, map_location='cpu')

    @classmethod
    def load_residuals(self, index_path, chunk_idx):
        residuals_path = os.path.join(index_path, f'{chunk_idx}.residuals.pt')  # f'{chunk_idx}.residuals.bn'
        # return _load_bitarray(residuals_path)

        return torch.load(residuals_path, map_location='cpu')

    def save(self, path_prefix):
        codes_path = f'{path_prefix}.codes.pt'
        residuals_path = f'{path_prefix}.residuals.pt'  # f'{path_prefix}.residuals.bn'

        torch.save(self.codes, codes_path)
        torch.save(self.residuals, residuals_path)
        # _save_bitarray(self.residuals, residuals_path)

    def __len__(self):
        return self.codes.size(0)


def get_dim_and_nbits(index_path):
    # TODO: Ideally load this using ColBERTConfig.load_from_index!
    with open(os.path.join(index_path, 'metadata.json')) as f:
        metadata = ujson.load(f)['config']

    dim = metadata['dim']
    nbits = metadata['nbits']

    assert (dim * nbits) % 8 == 0, (dim, nbits, dim * nbits)

    return dim, nbits
