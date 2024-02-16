import torch
import numpy as np

from bitarray import bitarray


class IndexManager():
    def __init__(self, dim):
        self.dim = dim

    def save(self, tensor, path_prefix):
        torch.save(tensor, path_prefix)

    def save_bitarray(self, bitarray, path_prefix):
        with open(path_prefix, "wb") as f:
            bitarray.tofile(f)


def load_index_part(filename, verbose=True):
    part = torch.load(filename)

    if type(part) == list:  # for backward compatibility
        part = torch.cat(part)

    return part


def load_compressed_index_part(filename, dim, bits):
    a = bitarray()

    with open(filename, "rb") as f:
        a.fromfile(f)

    n = len(a) // dim // bits
    part = torch.tensor(np.frombuffer(a.tobytes(), dtype=np.uint8))  # TODO: isn't from_numpy(.) faster?
    part = part.reshape((n, int(np.ceil(dim * bits / 8))))

    return part
