import json
import os
import torch

import numpy as np
from itertools import accumulate
from functools import lru_cache
import struct

def read_longs(f, n):
    a = np.empty(n, dtype=np.int64)
    f.readinto(a)
    return a

dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float,
    7: np.double,
    8: np.uint16
}

def index_file_path(prefix_path):
    return prefix_path + '.idx'

def _warmup_mmap_file(path):
    with open(path, 'rb') as stream:
        while stream.read(100 * 1024 * 1024):
            pass

def data_file_path(prefix_path):
    return prefix_path + '.bin'

def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)

def exscan_from_cumsum_(arr):
    # given an array holding the result of an inclusive scan (cumsum),
    # convert to an exclusive scan (shift to the right)
    # [10, 30, 35, 50] --> [0, 10, 30, 35]
    if arr.size > 1:
        arr[1:] = arr[:-1]
    if arr.size > 0:
        arr[0] = 0


def get_pointers_with_total(sizes, elemsize, dtype):
    """Return a numpy array of type np.dtype giving the byte offsets.

    Multiplies values in the sizes array by elemsize (bytes),
    and then computes an exclusive scan to get byte offsets.
    Returns the total number of bytes as second item in a tuple.
    """

    # scale values in sizes array by elemsize to get sizes in bytes
    pointers = np.array(sizes, dtype=dtype)
    pointers *= elemsize
    np.cumsum(pointers, axis=0, out=pointers)

    # get total number of bytes from all sizes (last element)
    bytes_last = pointers[-1] if len(sizes) > 0 else 0

    # convert to byte offsets
    exscan_from_cumsum_(pointers)

    return pointers, bytes_last

def code(dtype):
    for k in dtypes.keys():
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)

class MMapIndexedDataset(torch.utils.data.Dataset):
    class Index(object):
        _HDR_MAGIC = b'MMIDIDX\x00\x00'

        @staticmethod
        def write_header(fout, dtype, numsizes, numdocs):
            """Writes header for mmap indexed dataset to given file handle, return number of bytes written."""
            startpos = fout.tell()

            fout.write(MMapIndexedDataset.Index._HDR_MAGIC)
            fout.write(struct.pack('<Q', 1))
            fout.write(struct.pack('<B', code(dtype)))
            fout.write(struct.pack('<Q', numsizes))
            fout.write(struct.pack('<Q', numdocs))

            endpos = fout.tell()
            return endpos - startpos

        @classmethod
        def writer(cls, path, dtype):
            class _Writer(object):
                def __enter__(self):
                    self._file = open(path, 'wb')
                    return self

                @staticmethod
                def _get_pointers(sizes, npdtype):
                    """Return a numpy array of byte offsets given a list of sizes.

                    Multiplies values in the sizes array by dtype size (bytes),
                    and then computes an exclusive scan to get byte offsets.
                    """

                    # compute element sizes in bytes
                    pointers, _ = get_pointers_with_total(sizes, dtype().itemsize, npdtype)
                    return pointers

                def write(self, sizes, doc_idx):
                    MMapIndexedDataset.Index.write_header(self._file, dtype, len(sizes), len(doc_idx))

                    sizes32 = np.array(sizes, dtype=np.int32)
                    self._file.write(sizes32.tobytes(order='C'))
                    del sizes32

                    pointers = self._get_pointers(sizes, np.int64)
                    self._file.write(pointers.tobytes(order='C'))
                    del pointers

                    doc_idx = np.array(doc_idx, dtype=np.int64)
                    self._file.write(doc_idx.tobytes(order='C'))

                def __exit__(self, exc_type, exc_val, exc_tb):
                    self._file.close()

            return _Writer()

        def __init__(self, path, skip_warmup=False):
            with open(path, 'rb') as stream:
                magic_test = stream.read(9)
                assert self._HDR_MAGIC == magic_test, (
                    'Index file doesn\'t match expected format. '
                    'Make sure that --dataset-impl is configured properly.'
                )
                version = struct.unpack('<Q', stream.read(8))
                assert (1,) == version

                dtype_code, = struct.unpack('<B', stream.read(1))
                self._dtype = dtypes[dtype_code]
                self._dtype_size = self._dtype().itemsize

                self._len = struct.unpack('<Q', stream.read(8))[0]
                self._doc_count = struct.unpack('<Q', stream.read(8))[0]
                offset = stream.tell()

            if not skip_warmup:
                print_rank_0("    warming up index mmap file...")
                _warmup_mmap_file(path)

            self._bin_buffer_mmap = np.memmap(path, mode='r', order='C')
            self._bin_buffer = memoryview(self._bin_buffer_mmap)
            print_rank_0("    reading sizes...")
            self._sizes = np.frombuffer(
                self._bin_buffer,
                dtype=np.int32,
                count=self._len,
                offset=offset)
            print_rank_0("    reading pointers...")
            self._pointers = np.frombuffer(self._bin_buffer, dtype=np.int64, count=self._len,
                                           offset=offset + self._sizes.nbytes)
            print_rank_0("    reading document index...")
            self._doc_idx = np.frombuffer(self._bin_buffer, dtype=np.int64, count=self._doc_count,
                                          offset=offset + self._sizes.nbytes + self._pointers.nbytes)

        def __del__(self):
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap

        @property
        def dtype(self):
            return self._dtype

        @property
        def sizes(self):
            return self._sizes

        @property
        def doc_idx(self):
            return self._doc_idx

        @lru_cache(maxsize=8)
        def __getitem__(self, i):
            return self._pointers[i], self._sizes[i]

        def __len__(self):
            return self._len

    def __init__(self, path, skip_warmup=False):
        super().__init__()

        self._path = None
        self._index = None
        self._bin_buffer = None

        self._do_init(path, skip_warmup)

    def __getstate__(self):
        return self._path

    def __setstate__(self, state):
        self._do_init(state)

    def _do_init(self, path, skip_warmup):
        self._path = path
        self._index = self.Index(index_file_path(self._path), skip_warmup)

        if not skip_warmup:
            print_rank_0("    warming up data mmap file...")
            _warmup_mmap_file(data_file_path(self._path))
        print_rank_0("    creating numpy buffer of mmap...")
        self._bin_buffer_mmap = np.memmap(data_file_path(self._path), mode='r', order='C')
        print_rank_0("    creating memory view of numpy buffer...")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def __del__(self):
        self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap
        del self._index

    def __len__(self):
        return len(self._index)

    # @lru_cache(maxsize=8)
    def __getitem__(self, idx):
        if isinstance(idx, int):
            ptr, size = self._index[idx]
            np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype,
                                     count=size, offset=ptr)
            return np_array
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")
            ptr = self._index._pointers[start]
            sizes = self._index._sizes[idx]
            offsets = list(accumulate(sizes))
            total_size = sum(sizes)
            np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype,
                                     count=total_size, offset=ptr)
            sents = np.split(np_array, offsets[:-1])
            return sents

    def get(self, idx, offset=0, length=None):
        """ Retrieves a single item from the dataset with the option to only
        return a portion of the item.

        get(idx) is the same as [idx] but get() does not support slicing.
        """
        ptr, size = self._index[idx]
        if length is None:
            length = size - offset
        ptr += offset * np.dtype(self._index.dtype).itemsize
        np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype,
                                 count=length, offset=ptr)
        return np_array

    @property
    def sizes(self):
        return self._index.sizes

    @property
    def doc_idx(self):
        return self._index.doc_idx

    def get_doc_idx(self):
        return self._index._doc_idx

    def set_doc_idx(self, doc_idx_):
        self._index._doc_idx = doc_idx_

    @property
    def supports_prefetch(self):
        return False

    @staticmethod
    def exists(path):
        return (
            os.path.exists(index_file_path(path)) and os.path.exists(data_file_path(path))
        )

    @property
    def dtype(self):
        return self._index.dtype