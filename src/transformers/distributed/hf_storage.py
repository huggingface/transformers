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
"""Safetensors DCP storage supporting multiple chunks of a parameter per file."""

import json
from dataclasses import dataclass

import torch
from safetensors import safe_open
from safetensors.torch import _getdtype, save
from torch.distributed.checkpoint import DefaultLoadPlanner
from torch.distributed.checkpoint._hf_utils import (
    CUSTOM_METADATA_KEY,
    SAVED_OFFSETS_KEY,
    _gen_file_name,
    _HFStorageInfo,
    _metadata_fn,
)
from torch.distributed.checkpoint.hf_storage import (
    HuggingFaceStorageReader as TorchHuggingFaceStorageReader,
)
from torch.distributed.checkpoint.hf_storage import (
    HuggingFaceStorageWriter as TorchHuggingFaceStorageWriter,
)
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    StorageMeta,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import WriteItemType
from torch.distributed.checkpoint.storage import WriteResult
from torch.futures import Future


_CHUNK_VERSION = "transformers_dcp_chunk_version"


@dataclass
class _ChunkStorageInfo(_HFStorageInfo):
    tensor_key: str


class HuggingFaceStorageWriter(TorchHuggingFaceStorageWriter):
    """
    Write unique physical chunk keys while retaining logical names in file metadata.

    Unconsolidated files require the matching Transformers reader. Consolidation produces
    standard Hugging Face safetensors files and an index with the original parameter names.
    """

    def write_data(self, plan, planner):
        storage_plan = plan.storage_data.get("fqn_to_index_mapping")
        buckets = self._split_by_storage_plan(storage_plan, plan.items)
        highest_index = max(storage_plan.values()) if storage_plan else 1
        results = []
        for file_index, items in buckets.items():
            if not items:
                continue
            file_name = _gen_file_name(file_index, highest_index, plan.storage_data.get("shard_index"))
            tensors, chunks = {}, {}
            for number, item in enumerate(items):
                if item.type == WriteItemType.BYTE_IO:
                    raise ValueError("Safetensors checkpoints only support tensor values.")
                key = f"chunk_{number}"
                tensor = planner.resolve_data(item).detach().to("cpu").contiguous()
                tensors[key] = tensor
                chunks[key] = {
                    "fqn": item.index.fqn,
                    "shape": list(item.tensor_data.size),
                    SAVED_OFFSETS_KEY: list(item.tensor_data.chunk.offsets),
                }
                results.append(
                    WriteResult(
                        index=item.index,
                        size_in_bytes=tensor.numel() * tensor.element_size(),
                        storage_data=_ChunkStorageInfo(file_name, tensor.size(), tensor.dtype, key),
                    )
                )
            with self.fs.create_stream(self.fs.concat_path(self.path, file_name), "wb") as stream:
                stream.write(
                    save(
                        tensors,
                        metadata={"format": "pt", _CHUNK_VERSION: "1", CUSTOM_METADATA_KEY: json.dumps(chunks)},
                    )
                )
        future = Future()
        future.set_result(results)
        return future

    def finish(self, metadata, results):
        if self.save_distributed and not self.enable_consolidation:
            return
        output_path = self.consolidated_output_path or str(self.path)
        mapping = self.fqn_to_index_mapping or dict.fromkeys(metadata.state_dict_metadata, 1)
        reader = HuggingFaceStorageReader(str(self.path))
        saved_metadata = reader.read_metadata()
        reader.set_up_storage_reader(saved_metadata, is_coordinator=True)
        weight_map, total_size = {}, 0
        # Read one output file's tensors at a time. Only the coordinator executes finish().
        for file_index in sorted(set(mapping.values())):
            tensors = {
                name: torch.empty(info.size, dtype=info.properties.dtype)
                for name, info in metadata.state_dict_metadata.items()
                if mapping[name] == file_index
            }
            planner = DefaultLoadPlanner()
            planner.set_up_planner(tensors, saved_metadata, is_coordinator=True)
            reader.read_data(planner.create_local_plan(), planner).wait()
            file_name = _gen_file_name(file_index, max(mapping.values()))
            with self.fs.create_stream(self.fs.concat_path(output_path, file_name), "wb") as stream:
                stream.write(save(tensors, metadata={"format": "pt"}))
            weight_map.update(dict.fromkeys(tensors, file_name))
            total_size += sum(t.numel() * t.element_size() for t in tensors.values())
        with self.fs.create_stream(self.fs.concat_path(output_path, _metadata_fn), "w") as stream:
            json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, stream, indent=2)


class HuggingFaceStorageReader(TorchHuggingFaceStorageReader):
    """Read chunk-key files, legacy PyTorch DCP safetensors, and ordinary safetensors."""

    def read_metadata(self):
        tensors, storage = {}, {}
        for path in self.fs.ls(self.path):
            if not path.endswith(".safetensors"):
                continue
            with safe_open(path, framework="pt") as file:
                extra = file.metadata() or {}
                version = extra.get(_CHUNK_VERSION)
                if version is not None and version != "1":
                    raise ValueError(f"Unsupported safetensors chunk metadata version: {version}.")
                chunks = json.loads(extra.get(CUSTOM_METADATA_KEY, "{}"))
                for key in file.keys():
                    view = file.get_slice(key)
                    shape, dtype = torch.Size(view.get_shape()), _getdtype(view.get_dtype())
                    info = chunks.get(key, {})
                    name = info["fqn"] if version else key
                    offset = torch.Size(info.get(SAVED_OFFSETS_KEY, [0] * len(shape)))
                    global_shape = torch.Size(info["shape"] if version else [o + s for o, s in zip(offset, shape)])
                    chunk = ChunkStorageMetadata(offset, shape)
                    if name not in tensors:
                        tensors[name] = TensorStorageMetadata(TensorProperties(dtype=dtype), global_shape, [])
                    else:
                        tensors[name].size = torch.Size(max(a, b) for a, b in zip(tensors[name].size, global_shape))
                    tensors[name].chunks.append(chunk)
                    storage[MetadataIndex(name, offset)] = _ChunkStorageInfo(path, shape, dtype, key)
        return Metadata(tensors, storage_data=storage, storage_meta=StorageMeta(load_id=self.load_id))

    def _process_read_request(self, file, request, planner):
        info = self.storage_data[request.storage_index]
        slices = tuple(
            slice(offset, offset + length) for offset, length in zip(request.storage_offsets, request.lengths)
        )
        tensor = file.get_slice(info.tensor_key)[slices]
        target = planner.resolve_tensor(request).detach()
        if target.shape != tensor.shape:
            raise ValueError(f"Checkpoint slice shape {tensor.shape} does not match destination {target.shape}.")
        target.copy_(tensor)
        planner.commit_tensor(request, target)
