import torch
import random

import numpy as np

from colbert.utils.utils import flatten


"""
import line_profiler
import atexit
profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats)
"""


class StridedTensorCore:
    # # @profile
    def __init__(self, packed_tensor, lengths, dim=None, use_gpu=True):
        self.dim = dim
        self.tensor = packed_tensor
        self.inner_dims = self.tensor.size()[1:]
        self.use_gpu = use_gpu

        self.lengths = lengths.long() if torch.is_tensor(lengths) else torch.LongTensor(lengths)

        self.strides = _select_strides(self.lengths, [.5, .75, .9, .95]) + [self.lengths.max().item()]
        self.max_stride = self.strides[-1]

        zero = torch.zeros(1, dtype=torch.long, device=self.lengths.device)
        self.offsets = torch.cat((zero, torch.cumsum(self.lengths, dim=0)))

        if self.offsets[-2] + self.max_stride > self.tensor.size(0):
            # if self.tensor.size(0) > 10_000_000:
            #     print("#> WARNING: StridedTensor has to add padding, internally, to a large tensor.")
            #     print("#> WARNING: Consider doing this padding in advance to save memory!")

            padding = torch.zeros(self.max_stride, *self.inner_dims, dtype=self.tensor.dtype, device=self.tensor.device)
            self.tensor = torch.cat((self.tensor, padding))

        self.views = {stride: _create_view(self.tensor, stride, self.inner_dims) for stride in self.strides}

    @classmethod
    def from_packed_tensor(cls, tensor, lengths):
        return cls(tensor, lengths)

    @classmethod
    def from_padded_tensor(cls, tensor, mask):
        pass

    @classmethod
    def from_nested_list(cls, lst):
        flat_lst = flatten(lst)

        tensor = torch.Tensor(flat_lst)
        lengths = [len(sublst) for sublst in lst]

        return cls(tensor, lengths, dim=0)

    @classmethod
    def from_tensors_list(cls, tensors):
        # torch.cat(tensors)
        # lengths.
        # cls(tensor, lengths)
        raise NotImplementedError()

    def as_packed_tensor(self, return_offsets=False):
        unpadded_packed_tensor = self.tensor  # [:self.offsets[-1]]

        return_vals = [unpadded_packed_tensor, self.lengths]

        if return_offsets:
            return_vals.append(self.offsets)

        return tuple(return_vals)

    # # @profile
    def as_padded_tensor(self):
        if self.use_gpu:
            view = _create_view(self.tensor.cuda(), self.max_stride, self.inner_dims)[self.offsets[:-1]]
            mask = _create_mask(self.lengths.cuda(), self.max_stride, like=view, use_gpu=self.use_gpu)
        else:
            #import pdb
            #pdb.set_trace()
            view = _create_view(self.tensor, self.max_stride, self.inner_dims)
            view = view[self.offsets[:-1]]
            mask = _create_mask(self.lengths, self.max_stride, like=view, use_gpu=self.use_gpu)

        return view, mask

    def as_tensors_list(self):
        raise NotImplementedError()



def _select_strides(lengths, quantiles):
    if lengths.size(0) < 5_000:
        return _get_quantiles(lengths, quantiles)
    
    sample = torch.randint(0, lengths.size(0), size=(2_000,))

    return _get_quantiles(lengths[sample], quantiles)

def _get_quantiles(lengths, quantiles):
    return torch.quantile(lengths.float(), torch.tensor(quantiles, device=lengths.device)).int().tolist()


def _create_view(tensor, stride, inner_dims):
    outdim = tensor.size(0) - stride + 1
    size = (outdim, stride, *inner_dims)

    inner_dim_prod = int(np.prod(inner_dims))
    multidim_stride = [inner_dim_prod, inner_dim_prod] + [1] * len(inner_dims)

    return torch.as_strided(tensor, size=size, stride=multidim_stride)


def _create_mask(lengths, stride, like=None, use_gpu=True):
    if use_gpu:
        mask = torch.arange(stride).cuda() + 1
        mask = mask.unsqueeze(0) <= lengths.cuda().unsqueeze(-1)
    else:
        mask = torch.arange(stride) + 1
        mask = mask.unsqueeze(0) <= lengths.unsqueeze(-1)

    if like is not None:
        for _ in range(like.dim() - mask.dim()):
            mask = mask.unsqueeze(-1)

    return mask
