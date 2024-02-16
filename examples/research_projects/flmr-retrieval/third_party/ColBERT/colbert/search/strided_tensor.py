from struct import pack
import torch
from torch._C import device

from colbert.utils.utils import flatten, print_message

from .strided_tensor_core import StridedTensorCore, _create_mask, _create_view

import os
import pathlib
from torch.utils.cpp_extension import load


class StridedTensor(StridedTensorCore):
    def __init__(self, packed_tensor, lengths, dim=None, use_gpu=True):
        super().__init__(packed_tensor, lengths, dim=dim, use_gpu=use_gpu)
        StridedTensor.try_load_torch_extensions(use_gpu)

    @classmethod
    def try_load_torch_extensions(cls, use_gpu):
        if hasattr(cls, "loaded_extensions") or use_gpu:
            return

        print_message(f"Loading segmented_lookup_cpp extension (set COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True for more info)...")
        segmented_lookup_cpp = load(
            name="segmented_lookup_cpp",
            sources=[
                os.path.join(
                    pathlib.Path(__file__).parent.resolve(), "segmented_lookup.cpp"
                ),
            ],
            extra_cflags=["-O3"],
            verbose=os.getenv("COLBERT_LOAD_TORCH_EXTENSION_VERBOSE", "False") == "True",
        )
        cls.segmented_lookup = segmented_lookup_cpp.segmented_lookup_cpp

        cls.loaded_extensions = True

    @classmethod
    def pad_packed(cls, packed_tensor, lengths):
        assert False, "This seems to be incorrect but I can't see why. Is it the inner_dims in the views?"

        packed_tensor, lengths = packed_tensor.cuda().contiguous(), lengths.cuda()

        inner_dims = packed_tensor.size()[1:]
        stride = lengths.max().item()
        offsets = torch.cumsum(lengths, dim=0) - lengths[0]

        padding = torch.zeros(stride, *inner_dims, device=packed_tensor.device, dtype=packed_tensor.dtype)
        packed_tensor = torch.cat((packed_tensor, padding))

        view = _create_view(packed_tensor, stride, inner_dims)[offsets]
        mask = _create_mask(lengths, stride, like=view)

        return view, mask

    def _prepare_lookup(self, pids):
        if isinstance(pids, list):
            pids = torch.tensor(pids)
        
        pids = pids.cpu()

        assert pids.dim() == 1

        pids = pids.long()
        lengths = self.lengths[pids]
        offsets = self.offsets[pids]

        if self.use_gpu:
            pids = pids.cuda()

        if self.use_gpu:
            lengths = lengths.cuda()
        
        return pids, lengths, offsets

    def lookup(self, pids, output='packed'):
        pids, lengths, offsets = self._prepare_lookup(pids)

        if self.use_gpu:
            stride = lengths.max().item()
            stride = next(s for s in self.strides if stride <= s)

            tensor = self.views[stride][offsets]
            if self.use_gpu:
                tensor = tensor.cuda()

            mask = _create_mask(lengths, stride, use_gpu=self.use_gpu)

            if output == 'padded':
                return tensor, mask

            assert output == 'packed'

            tensor = tensor[mask]
        else:
            tensor = StridedTensor.segmented_lookup(self.tensor, pids, lengths, offsets)

        return tensor, lengths

    def lookup_staggered(self, pids, output='packed'):
        permute_idxs, unordered_tensors, unordered_lengths, unordered_masks = self.lookup_packed_unordered(pids)

        output_tensor = torch.empty(permute_idxs.size(0), self.max_stride, *self.inner_dims,
                                    dtype=unordered_tensors[0].dtype, device=unordered_tensors[0].device)

        output_mask = torch.zeros(permute_idxs.size(0), self.max_stride,
                                  dtype=unordered_masks[0].dtype, device=unordered_masks[0].device)

        offset = 0
        for tensor, mask in zip(unordered_tensors, unordered_masks):
            endpos = offset + tensor.size(0)
            output_tensor[offset:endpos, :tensor.size(1)] = tensor
            output_mask[offset:endpos, :mask.size(1)] = mask
            offset = endpos

        output_mask = output_mask[permute_idxs]
        output_tensor = output_tensor[permute_idxs]

        if output == 'padded':
            return output_tensor, output_mask

        assert output == 'packed'

        output_tensor = output_tensor[output_mask]

        return output_tensor, unordered_lengths[permute_idxs]

    def lookup_packed_unordered(self, pids):
        pids, lengths, offsets = self._prepare_lookup(pids)

        lengths2 = lengths.clone()
        sentinel = self.strides[-1] + 1
        order = torch.arange(pids.size(0), device='cuda' if self.use_gpu else 'cpu')

        all_orders = []
        all_tensors = []
        all_lengths = []
        all_masks = []

        for stride in self.strides:
            is_shorter = lengths2 <= stride

            if is_shorter.sum() == 0:
                continue

            order_ = order[is_shorter]
            tensor_, lengths_, mask_ = self._lookup_with_stride(stride, lengths[is_shorter], offsets[is_shorter])

            all_orders.append(order_)
            all_tensors.append(tensor_)
            all_lengths.append(lengths_)
            all_masks.append(mask_)

            lengths2[is_shorter] = sentinel

        assert lengths2.allclose(torch.tensor([sentinel], device='cuda' if self.use_gpu else 'cpu'))

        all_orders = torch.cat(all_orders)
        permute_idxs = torch.sort(all_orders).indices

        return permute_idxs, all_tensors, torch.cat(all_lengths), all_masks

    def _lookup_with_stride(self, stride, lengths, offsets):
        tensor = self.views[stride][offsets]
        if self.use_gpu:
            tensor = tensor.cuda()

        mask = _create_mask(lengths, stride, use_gpu=self.use_gpu)
        # tensor = tensor[mask]

        return tensor, lengths, mask


if __name__ == '__main__':
    # lst = []
    # for _ in range(10):
    #     lst.append(list(range(random.randint(0, 10))))

    # print(lst)

    # t = StridedTensor.from_nested_list(lst)
    # print(t.lookup([9]))

    import os
    import pickle

    index_path = '/future/u/okhattab/root/unit/indexes/2021/08/residual.NQ-micro'
    with open(os.path.join(index_path, "centroid_idx_to_embedding_ids.pickle"), "rb") as f:
        ivf_list = pickle.load(f)

    assert len(ivf_list) == max(ivf_list.keys()) + 1
    ivf_list = [ivf_list[i] for i in range(len(ivf_list))]

    for x in ivf_list:
        assert type(x) is list
        assert type(x[0]) is int

    ncentroids = len(ivf_list)

    ivf = StridedTensor.from_nested_list(ivf_list)

    import time

    torch.cuda.synchronize()
    t = time.time()

    N = 100
    for _ in range(N):
        probed_centroids = torch.randint(0, ncentroids, size=(32, 8)).flatten()
        emb_ids, emb_ids_lengths = ivf.lookup(probed_centroids).as_packed_tensor()

    torch.cuda.synchronize()
    print((time.time() - t) * 1000 / N, 'ms')

    print(emb_ids_lengths)

    slow_result = flatten([ivf_list[idx] for idx in probed_centroids.flatten().tolist()])
    print(emb_ids.size(), len(slow_result))

    for a, b in zip(slow_result, emb_ids.flatten().tolist()):
        assert a == b, (a, b)

    print("#> Done!")

    print(ivf.lookup(probed_centroids).as_padded_tensor()[0].size())
