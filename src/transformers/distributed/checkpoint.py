# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DCP planners for DTensors whose local storage contains disjoint global regions."""

from itertools import product

import torch
from torch.distributed.checkpoint import DefaultLoadPlanner, DefaultSavePlanner
from torch.distributed.checkpoint.metadata import ChunkStorageMetadata, MetadataIndex, TensorProperties
from torch.distributed.checkpoint.planner import TensorWriteData, WriteItem, WriteItemType
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor.placement_types import _StridedShard


def _slice_regions(regions, start, length):
    """Slice a concatenation of (global offset, length) intervals without allocating tensor data."""
    result = []
    local_offset = 0
    for offset, size in regions:
        left, right = max(start, local_offset), min(start + length, local_offset + size)
        if left < right:
            result.append((offset + left - local_offset, right - left))
        local_offset += size
    return result


class _CheckpointView:
    """
    A checkpoint view of a DTensor for saving and loading that exposes its local storage as a set of disjoint global
    regions when the DTensor is sharded with `_StridedShard` placements.

    Example:
        Context:
            - Global tensor: [10, 11, 12, 13, 14, 15, 16, 17]
            - Placement: _StridedShard(dim=0, split_factor=2)
            - Mesh size:    2
            - Rank 0 local: [10, 11, 14, 15]

        The DTensor hooks would normally expose the local storage as a single contiguous region, which would be saved
        as a single chunk:
        ```
            __create_write_items__:
                [WriteItem(name="weight", global_shape=[8], offset=[0], size=[4])]

            __create_chunk_list__:
                [ChunkStorageMetadata(offsets=[0], sizes=[4])]

            __get_tensor_shard__(MetadataIndex("weight", offset=[0])):
                [10, 11, 14, 15]
        ```
        While the data is correct, the chunk metadata is misleading because it implies that the local storage
        corresponds to a single contiguous region of the global tensor, which is not the case.

        The `_CheckpointView` exposes the local storage as two disjoint regions, which will be saved as two chunks:
        ```
            __create_write_items__:
                [WriteItem(name="weight", global_shape=[8], offset=[0], size=[2]),
                 WriteItem(name="weight", global_shape=[8], offset=[4], size=[2])]
            __create_chunk_list__:
                [ChunkStorageMetadata(offsets=[0], sizes=[2]),
                 ChunkStorageMetadata(offsets=[4], sizes=[2])]
            __get_tensor_shard__(MetadataIndex("weight", offset=[0])):
                [10, 11]
            __get_tensor_shard__(MetadataIndex("weight", offset=[4])):
                [14, 15]
        ```

    The view preserves the tensor's placements and exposes slices of its existing local storage.
    """

    def __init__(self, tensor):
        self.tensor = tensor
        self.chunks = []
        self.views = {}
        coordinate = tensor.device_mesh.get_coordinate()
        if coordinate is None:
            return
        regions = [[(0, size)] for size in tensor.shape]
        for axis, placement in enumerate(tensor.placements):
            if placement.is_partial():
                raise ValueError("Checkpointing DTensors with Partial placements is unsupported.")
            if not isinstance(placement, (Shard, _StridedShard)):
                continue
            dim = placement.dim
            size = sum(length for _, length in regions[dim])
            splits = placement.split_factor if isinstance(placement, _StridedShard) else 1
            split_size = (size + splits - 1) // splits
            selected = []
            for split in range(splits):
                split_start = split * split_size
                length = max(0, min(split_size, size - split_start))
                shard_size = (length + tensor.device_mesh.size(axis) - 1) // tensor.device_mesh.size(axis)
                start = min(coordinate[axis] * shard_size, length)
                selected.extend(_slice_regions(regions[dim], split_start + start, min(shard_size, length - start)))
            regions[dim] = selected

        local = tensor.to_local()
        expected_shape = tuple(sum(length for _, length in intervals) for intervals in regions)
        if tuple(local.shape) != expected_shape:
            raise ValueError(f"Unsupported DTensor layout: expected local shape {expected_shape}, got {local.shape}.")
        dimensions = []
        for intervals in regions:
            local_offset = 0
            dimension = []
            for offset, length in intervals:
                dimension.append((offset, length, local_offset))
                local_offset += length
            dimensions.append(dimension)
        for region in product(*dimensions):
            offsets = torch.Size(item[0] for item in region)
            sizes = torch.Size(item[1] for item in region)
            view = local[tuple(slice(start, start + length) for _, length, start in region)]
            self.chunks.append(ChunkStorageMetadata(offsets, sizes))
            self.views[offsets] = view

    def size(self):
        return self.tensor.size()

    def __create_write_items__(self, fqn, object):
        return [
            WriteItem(
                index=MetadataIndex(fqn, chunk.offsets),
                type=WriteItemType.SHARD,
                tensor_data=TensorWriteData(
                    chunk=chunk, properties=TensorProperties.create_from_tensor(self.tensor), size=self.tensor.size()
                ),
            )
            for chunk in self.chunks
        ]

    def __create_chunk_list__(self):
        return self.chunks

    def __get_tensor_shard__(self, index):
        return self.views[index.offset]


def _checkpoint_views(state_dict):
    return {
        name: _CheckpointView(value)
        if isinstance(value, DTensor) and any(isinstance(p, _StridedShard) for p in value.placements)
        else value
        for name, value in state_dict.items()
    }


class HuggingFaceSavePlanner(DefaultSavePlanner):
    """
    Extend the default DCP save planner with support for `_StridedShard` placements.
    """

    def create_local_plan(self):
        original = self.state_dict
        self._views = _checkpoint_views(original)
        self.state_dict = self._views
        try:
            return super().create_local_plan()
        finally:
            self.state_dict = original

    def lookup_object(self, index):
        value = self._views[index.fqn]
        if isinstance(value, _CheckpointView):
            return value.__get_tensor_shard__(index)
        return super().lookup_object(index)


class HuggingFaceLoadPlanner(DefaultLoadPlanner):
    """
    Extend the default DCP load planner with support for `_StridedShard` placements.
    """

    def create_local_plan(self):
        original = self.state_dict
        self._views = _checkpoint_views(original)
        self.state_dict = self._views
        try:
            return super().create_local_plan()
        finally:
            self.state_dict = original

    def lookup_tensor(self, index):
        value = self._views[index.fqn]
        if isinstance(value, _CheckpointView):
            return value.__get_tensor_shard__(index)
        return super().lookup_tensor(index)
