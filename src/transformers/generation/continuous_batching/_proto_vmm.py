# [scratch, gap4 pause] CB_CACHE_VMM=1: the KV cache lives on a virtual address range of its own (CUDA virtual memory
# management), backed by physical chunks that `release_memory` unmaps and `restore_memory` maps again at the same
# addresses. The cache never moves, so the captured CUDA graphs stay valid across a pause, and the chunks that hold
# blocks of requests in flight stay mapped where they are: nothing goes to the host and back.
import torch


def _check(result):
    err = result[0]
    if int(err) != 0:
        raise RuntimeError(f"CUDA driver call failed: {err}")
    return result[1] if len(result) == 2 else result[1:]


class _ArrayInterface:
    """What `torch.as_tensor` reads to wrap a device pointer without a copy. The tensor keeps this object alive, and this
    object keeps the mapping alive: the range is unmapped and freed once the last view of the tensor is gone."""

    def __init__(self, owner, nbytes):
        self.owner = owner
        self.__cuda_array_interface__ = {
            "shape": (nbytes,), "typestr": "|u1", "data": (owner.va, False), "version": 3, "strides": None,
            "stream": None,
        }  # fmt: skip


class VmmBuffer:
    """`nbytes` of device memory at a fixed virtual address, as `num_chunks` physical chunks of `chunk_bytes`."""

    def __init__(self, nbytes: int, device: torch.device, chunk_bytes: int):
        from cuda.bindings import driver as cu

        self.cu = cu
        self.device = device
        index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_device(index)  # the driver calls below act on the current context
        prop = cu.CUmemAllocationProp()
        prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = index
        self.prop = prop
        access = cu.CUmemAccessDesc()
        access.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access.location.id = index
        access.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        self.access = access
        granularity = _check(
            cu.cuMemGetAllocationGranularity(prop, cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED)
        )
        self.chunk = max(granularity, (chunk_bytes // granularity) * granularity)
        self.num_chunks = -(-nbytes // self.chunk)
        self.total = self.num_chunks * self.chunk
        self.nbytes = nbytes
        self.va = int(_check(cu.cuMemAddressReserve(self.total, self.chunk, 0, 0)))
        self.handles = [None] * self.num_chunks
        self.map_chunks(range(self.num_chunks))

    def make_tensor(self) -> torch.Tensor:
        """The uint8 tensor over the whole range (it keeps this buffer alive, the buffer holds no reference to it)."""
        tensor = torch.as_tensor(_ArrayInterface(self, self.nbytes), device=self.device)
        assert tensor.data_ptr() == self.va and tensor.numel() == self.nbytes and tensor.dtype == torch.uint8
        return tensor

    def map_chunks(self, chunks) -> int:
        """Backs the given chunks with new physical memory (contents undefined). Returns the number of bytes mapped."""
        cu, mapped = self.cu, 0
        for i in chunks:
            if self.handles[i] is not None:
                continue
            handle = _check(cu.cuMemCreate(self.chunk, self.prop, 0))
            _check(cu.cuMemMap(self.va + i * self.chunk, self.chunk, 0, handle, 0))
            _check(cu.cuMemSetAccess(self.va + i * self.chunk, self.chunk, [self.access], 1))
            self.handles[i] = handle
            mapped += self.chunk
        return mapped

    def unmap_chunks(self, chunks) -> int:
        """Gives the physical memory of the given chunks back to the device. Nothing may touch them until they are mapped
        again: the caller synchronizes the device first. Returns the number of bytes unmapped."""
        cu, unmapped = self.cu, 0
        for i in chunks:
            if self.handles[i] is None:
                continue
            _check(cu.cuMemUnmap(self.va + i * self.chunk, self.chunk))
            _check(cu.cuMemRelease(self.handles[i]))
            self.handles[i] = None
            unmapped += self.chunk
        return unmapped

    def unmapped_chunks(self) -> list[int]:
        return [i for i, h in enumerate(self.handles) if h is None]

    def __del__(self):
        try:
            torch.cuda.synchronize(self.device)
            self.unmap_chunks(range(self.num_chunks))
            self.cu.cuMemAddressFree(self.va, self.total)
        except Exception:  # interpreter shutdown: the driver may be gone already
            pass
