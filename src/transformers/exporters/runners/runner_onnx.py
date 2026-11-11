# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Runs an ONNX program in ONNX Runtime."""

from __future__ import annotations

import os
import tempfile
import weakref
from typing import Any

from ...utils.import_utils import is_onnxruntime_available, is_onnxscript_available, is_torch_available
from ..metadata import ExportMetadata
from ..utils import get_leaf_tensors
from .base import ModelRunner


if is_torch_available():
    import torch

if is_onnxruntime_available():
    import onnxruntime as ort

if is_onnxscript_available():
    import onnx_ir


# Past this much weight a graph opens from a file rather than serialized: protobuf caps a model at 2GB (torch's own
# threshold, leaving room for the graph itself).
_SERIALIZED_WEIGHTS_LIMIT = int(1.5 * 2**30)


def _execution_providers(device=None) -> list:
    """The ONNX Runtime providers for `device`: CUDA when asked for, or by default when this build and a GPU have it."""
    available = ort.get_available_providers()
    if device is None:
        device = "cuda" if torch.cuda.is_available() and "CUDAExecutionProvider" in available else "cpu"
    device = torch.device(device)
    if device.type == "cpu":
        return ["CPUExecutionProvider"]
    if device.type == "cuda" and "CUDAExecutionProvider" in available:
        return [("CUDAExecutionProvider", {"device_id": device.index or 0}), "CPUExecutionProvider"]
    raise ValueError(f"This ONNX Runtime build cannot run on {device} (providers: {available}).")


class OnnxModelRunner(ModelRunner):
    """Runs an ONNX graph in an ONNX Runtime session opened for the device, its inputs and outputs bound in place."""

    def __init__(self, metadata: ExportMetadata, session: ort.InferenceSession):
        super().__init__(metadata)
        self.session = session

    @property
    def device(self) -> torch.device:
        provider, options = self.session.get_providers()[0], self.session.get_provider_options()
        if provider in ("CUDAExecutionProvider", "TensorrtExecutionProvider"):
            return torch.device("cuda", int(options[provider]["device_id"]))
        return torch.device("cpu")

    @classmethod
    def from_program(
        cls,
        program: onnx_ir.Model,
        metadata: ExportMetadata,
        device=None,
        providers: list | None = None,
        session_options: ort.SessionOptions | None = None,
    ) -> OnnxModelRunner:
        """A loaded graph opens its saved file (`meta["path"]`), ONNX Runtime reading the weights from beside it; a
        fresh one opens serialized, or from a temporary file past protobuf's size limit. `providers`, ONNX Runtime's
        own list (TensorRT with its options, say), replaces the ones picked for `device`; `session_options` open the
        session (`intra_op_num_threads`: ONNX Runtime sizes its thread pool from every core of the machine, not the
        ones the process may run on)."""
        if device is not None and providers is not None:
            raise ValueError(
                "Pass either `device` or `providers`: the providers choose the device (`device_id` in the CUDA or "
                "TensorRT provider's options)."
            )
        providers = providers or _execution_providers(device)
        if "path" in program.meta:
            return cls(metadata, ort.InferenceSession(program.meta["path"], session_options, providers=providers))

        weights = sum(
            value.const_value.nbytes for value in program.graph.initializers.values() if value.const_value is not None
        )
        if weights < _SERIALIZED_WEIGHTS_LIMIT:
            serialized = onnx_ir.to_proto(program).SerializeToString()
            return cls(metadata, ort.InferenceSession(serialized, session_options, providers=providers))
        directory = tempfile.TemporaryDirectory()
        path = os.path.join(directory.name, "model.onnx")
        onnx_ir.save(program, path, external_data="model.onnx.data")
        session = ort.InferenceSession(path, session_options, providers=providers)
        # The CPU provider maps the weights file for as long as the session lives.
        weakref.finalize(session, directory.cleanup)
        return cls(metadata, session)

    def __call__(self, share_inputs: bool = True, share_outputs: bool = False, **kwargs) -> Any:
        """Run with the inputs and outputs bound in place on the session's device (IO binding), instead of copied
        through host arrays: sharing either way already. Inputs bind by name, the leaf's path (with an `input.` prefix
        on a mutated one): the export prunes the ones the graph doesn't read (a linear-attention layer's mask)."""
        from torch.onnx._internal.exporter._core import torch_dtype_to_onnx_dtype

        device = self.device
        on_gpu = device.type == "cuda"
        leaves = get_leaf_tensors(kwargs)
        specs = self.session.get_inputs()
        missing = [spec.name for spec in specs if spec.name.removeprefix("input.") not in leaves]
        if missing:
            raise ValueError(f"The graph takes {missing}, which {list(kwargs)} don't hold.")
        binding, bound_buffers = self.session.io_binding(), []
        for spec in specs:
            leaf = leaves[spec.name.removeprefix("input.")]
            # Integers bind on host: ORT reads them on its CPU, where a GPU pointer is an illegal access.
            on_host = not (on_gpu and leaf.is_floating_point())
            tensor = (leaf.cpu() if on_host else leaf.to(device)).contiguous()
            # ORT refuses the null pointer of an empty tensor; a one-element stand-in serves.
            buffer = tensor if tensor.numel() else torch.empty(1, dtype=tensor.dtype, device=tensor.device)
            bound_buffers.append(buffer)
            binding.bind_input(
                spec.name,
                "cpu" if on_host else "cuda",
                0 if on_host else device.index,
                int(torch_dtype_to_onnx_dtype(tensor.dtype)),
                list(tensor.shape),
                buffer.data_ptr(),
            )
        for spec in self.session.get_outputs():
            binding.bind_output(spec.name, device.type, device.index or 0)
        # ORT runs on its own stream: sync both ways, or it reads half-written inputs.
        if on_gpu:
            torch.cuda.current_stream(device).synchronize()
        self.session.run_with_iobinding(binding)
        if on_gpu:
            binding.synchronize_outputs()
        outputs = [torch.from_dlpack(value._ortvalue.to_dlpack()) for value in binding.get_outputs()]
        return self.unflatten(outputs)
